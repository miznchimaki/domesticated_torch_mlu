from typing import Union, Optional
import sympy
import itertools

import torch
from torch.utils._ordered_set import OrderedSet
from torch._inductor.codegen import triton_combo_kernel
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.triton_combo_kernel import ComboKernel
from torch._inductor.runtime.hints import ReductionHint
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import TritonKernel
from ...utils import gorilla

# Modify by CAMBRICON: SequentialDispatch accuracy error
# only use RoundRobinDispatch as combo kernel dispatch_strategy
triton_combo_kernel.BLOCK_UTILIZATION = 0.0


@classmethod
def SequentialDispatch_codegen_pid_range(
    cls, kernel: "ComboKernel", num: int, code: IndentedBuffer
) -> None:
    if num == 0:
        cls._calculate_xblocks(kernel, code)
        code.splice(f"if pid < num_xblocks_{num}:")
        with code.indent():
            code.splice("pid_offset = pid")
    else:
        code.splice(f"elif pid < num_xblocks_{num}:")
        with code.indent():
            code.splice(f"pid_offset = pid - num_xblocks_{num - 1}")
    # Add by CAMBRICON
    with code.indent():
        code.splice(f"block_step = tl.cdiv({kernel.x_numels_list[num]}, XBLOCK)")
    # end Add by CAMBRICON


patch = gorilla.Patch(
    triton_combo_kernel.ComboKernel.SequentialDispatch,
    "codegen_pid_range",
    SequentialDispatch_codegen_pid_range,
)
gorilla.apply(patch)


@classmethod
def RoundRobinDispatch_codegen_pid_range(
    cls, kernel: "ComboKernel", num: int, code: IndentedBuffer
) -> None:
    num_kernels = len(kernel.sub_kernels)
    if num == 0:
        cond = "if"
    else:
        cond = "elif"
    code.splice(f"{cond} pid % {num_kernels} == {num}:")
    with code.indent():
        code.splice(f"pid_offset = pid // {num_kernels}")
        # Add by CAMBRICON
        code.splice(f"block_step = tl.num_programs(0) // {num_kernels}")
        # end Add by CAMBRICON


patch = gorilla.Patch(
    triton_combo_kernel.ComboKernel.RoundRobinDispatch,
    "codegen_pid_range",
    RoundRobinDispatch_codegen_pid_range,
)
gorilla.apply(patch)


def __init__(
    self,
    triton_kernel_cls: type[TritonKernel],
    enable_autotune: bool = False,
    mixed_sizes: bool = False,
) -> None:
    # Modify by CAMBRICON
    # super().__init__()
    super(ComboKernel, self).__init__()
    # end Modify by CAMBRICON
    self.triton_kernel_cls = triton_kernel_cls
    self.sub_kernels: list[TritonKernel] = []
    self.iter_vars_count = itertools.count()
    self.grids: list[list[int]] = []
    self.min_x_blocks_list: list[Union[int, str]] = []
    self.x_numels_list: list[Union[int, str]] = []
    self.y_tree_list: list = []
    self.enable_autotune = enable_autotune
    self.mixed_sizes = mixed_sizes
    self.dispatch_class: Optional[
        type[
            Union[
                ComboKernel.SequentialDispatch,
                ComboKernel.SequentialFlattenGridDispatch,
                ComboKernel.RoundRobinDispatch,
            ]
        ]
    ] = None
    self.block_args: list[str] = []
    # there following are used when autotuning is disabled
    self.block_size_1d = 1024  # Try tuning this value
    self.block_size_2d = 32

    # Modify by Cambricon: we only support num_warps=1 right now
    # self.num_warps = 8
    self.num_warps = 1
    # end Modify by CAMBRICON

    self.block_size_reduce = 256
    self.dynamic_shape_args: list[str] = []


patch = gorilla.Patch(
    torch._inductor.codegen.triton_combo_kernel.ComboKernel,
    "__init__",
    __init__,
    settings=gorilla.Settings(use_replace_references=True),
)
gorilla.apply(patch)
