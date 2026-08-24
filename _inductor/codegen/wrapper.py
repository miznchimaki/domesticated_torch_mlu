from typing import (
    List,
    Optional,
    Tuple,
    Union,
)
import contextlib
import sympy

import torch
from torch._inductor import async_compile, config
from torch._inductor.codegen.wrapper import PythonWrapperCodegen

from ...utils import gorilla


def write_header(self) -> None:
    """Write the header section of the generated Python wrapper code."""
    context = torch._guards.TracingContext.try_get()
    aot_config_comment = ""
    if context is not None and context.aot_graph_name is not None:
        aot_config_comment = f"# AOT ID: {context.aot_graph_name}"
    inductor_debug_utils = ""
    if int(config.aot_inductor.debug_intermediate_value_printer) > 0:
        inductor_debug_utils = "from torch._inductor.codegen.debug_utils import _print_debugging_tensor_value_info"
    elif torch._inductor.config.test_configs.track_memory_lifecycle:
        inductor_debug_utils = (
            "from torch._inductor.runtime.debug_utils import tracked_empty_strided\n"
        )

    self.imports.splice(
        f"""
            {aot_config_comment}
            from ctypes import c_void_p, c_long, c_int
            import torch
            # Modify by CAMBRICON
            import torch_mlu
            from contextlib import suppress
            # end Modify by CAMBRICON
            import math
            import random
            import os
            import tempfile
            from math import inf, nan
            from cmath import nanj
            from torch._inductor.hooks import run_intermediate_hooks
            from torch._inductor.utils import maybe_profile
            from torch._inductor.codegen.memory_planning import _align as align
            from torch import device, empty_strided
            from {async_compile.__name__} import AsyncCompile
            from torch._inductor.select_algorithm import extern_kernels
            {inductor_debug_utils}

            # Modify by CAMBRICON
            with suppress(ImportError):
                import torch_mlu_ops
            # end Modify by CAMBRICON
        """,
        strip=True,
    )
    self.header.splice(
        """
            aten = torch.ops.aten
            inductor_ops = torch.ops.inductor
            _quantized = torch.ops._quantized
            assert_size_stride = torch._C._dynamo.guards.assert_size_stride
            assert_alignment = torch._C._dynamo.guards.assert_alignment
            empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
            empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
            empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
            empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
            empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
            reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
            alloc_from_pool = torch.ops.inductor._alloc_from_pool
            async_compile = AsyncCompile()
        """,
        strip=True,
    )
    try:
        # Only add empty_strided_p2p() if distributed and SymmetricMemory
        # is available
        from torch._C._distributed_c10d import _SymmetricMemory  # noqa: F401

        self.header.splice(
            """
            empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
            """,
            strip=True,
        )
    except (AttributeError, ImportError):
        pass
    if config.annotate_training:
        self.header.writeline("from torch.cuda import nvtx")
    if config.triton.proton_profiling:
        self.header.writeline("import triton.profiler as proton")
        self.header.writeline("import triton.profiler.language as pl")
        self.header.writeline(
            "from triton.profiler.hooks import HookManager as _ProtonHookManager"
        )
        self.header.writeline("import triton")
        self.header.writeline("import atexit")
        self.header.writeline("import os")
        self.header.writeline(
            "triton.set_allocator(lambda size, align, stream: "
            "torch.empty(size, dtype=torch.uint8, device='cuda'))"
        )
        output_dir = config.triton.proton_output_dir or os.path.join(
            get_debug_dir(), "proton"
        )
        self.header.writeline(f'os.makedirs("{output_dir}", exist_ok=True)')
        proton_name = f'os.path.join("{output_dir}", "inductor")'
        trace_path = f'os.path.join("{output_dir}", "inductor.chrome_trace")'
        group_by_sm = config.triton.proton_group_by_sm
        split_invocations = config.triton.proton_split_invocations
        per_cta_occupancy = config.triton.proton_per_cta_occupancy
        self.header.writeline(
            "from torch._inductor.runtime.proton_utils import process_proton_trace as _proton_process_trace"
        )
        self.header.splice(
            f"""
            def _proton_finalize_and_postprocess():
                proton.finalize()
                _trace_path = {trace_path}
                if os.path.exists(_trace_path):
                    _proton_process_trace(
                        _trace_path,
                        group_by_sm={group_by_sm},
                        split_invocations={split_invocations},
                        per_cta_occupancy={per_cta_occupancy},
                    )
            """
        )
        # Start proton before kernel compilation (instrumentation backend needs to hook JIT)
        self.header.writeline(
            "if not _ProtonHookManager.active_hooks: "
            f'proton.start({proton_name}, backend="instrumentation", data="trace"); '
            "atexit.register(_proton_finalize_and_postprocess)"
        )
        self.header.writeline('pl.enable_semantic("triton")')


patch = gorilla.Patch(PythonWrapperCodegen, "write_header", write_header)
gorilla.apply(patch)


def write_kernel_autotune_defs_header(self) -> None:
    self.kernel_autotune_defs.splice(
        f"""
            import torch
            # Modify by CAMBRICON
            import torch_mlu
            # end Modify by CAMBRICON
            from torch._dynamo.testing import rand_strided
            from torch._dynamo.utils import preserve_rng_state
            from torch._inductor.select_algorithm import AlgorithmSelectorCache
            from {async_compile.__name__} import AsyncCompile

            async_compile = AsyncCompile()
            generate_example_value = AlgorithmSelectorCache.generate_example_value
            empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
            empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
        """
    )

    try:
        # Modify by Cambricon
        # from torch._C import _cuda_getCurrentRawStream  # noqa: F401
        # self.kernel_autotune_defs.splice(
        #     """
        #     get_raw_stream = torch._C._cuda_getCurrentRawStream
        #     """,
        #     strip=True,
        # )
        from torch_mlu._MLUC import _mlu_getCurrentRawStream

        self.kernel_autotune_defs.splice(
            """
            get_raw_stream = torch_mlu._MLUC._mlu_getCurrentRawStream
            """,
            strip=True,
        )
        # end Modify by Cambricon
    except (ImportError, AttributeError):
        pass


patch = gorilla.Patch(
    PythonWrapperCodegen,
    "write_kernel_autotune_defs_header",
    write_kernel_autotune_defs_header,
)
gorilla.apply(patch)
