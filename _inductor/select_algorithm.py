from typing import Any

import torch
from torch._inductor import config
from torch._inductor.codegen.triton_utils import (
    config_of,
    equal_1_arg_indices,
    signature_to_meta,
)
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.select_algorithm import TritonTemplateKernel
from torch._inductor.utils import Placeholder
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.runtime.triton_heuristics import FixedGrid
from torch._inductor.runtime.triton_compat import HAS_WARP_SPEC

from ..utils import gorilla


def jit_lines(self):
    if self.use_jit:
        return "@triton.jit"

    argdefs, _, signature, _ = self.args.python_argdefs()
    triton_meta: dict[str, Any] = {
        "signature": signature_to_meta(
            signature,
            size_dtype=self.index_dtype,
            argdefs=argdefs,
            is_template=True,
        ),
        "device": DeviceProperties.create(self.output_node.get_device()),
        "constants": {},
    }
    # fix by CAMBRICON: deal with cpu scalar triton compile bug, here signature_to_meta func
    # will take cpu scalar tensor as ptr type, change it to value type here.
    for node_name, node in {x.get_name(): x for x in self.input_nodes}.items():
        layout = node.layout
        arg_name = self.args.input_buffers.get(node_name, None)
        if not arg_name or arg_name not in triton_meta["signature"]:
            continue
        if layout.device.type == "cpu" and not layout.size:
            tye = triton_meta["signature"][arg_name]
            new_tye = tye.lstrip("*")
            if new_tye in ["fp16", "bf16"]:
                new_tye = "fp32"
            triton_meta["signature"][arg_name] = new_tye

    triton_meta["configs"] = [config_of(signature)]
    for arg_num in equal_1_arg_indices(signature):  # type: ignore[index]
        # Modify by CAMBRICON
        # triton_meta["constants"][signature[arg_num].name] = 1  # type: ignore[index,union-attr]
        from collections.abc import Iterable

        if isinstance(arg_num, Iterable):
            triton_meta["constants"][signature[arg_num[0]].name] = 1
        else:
            triton_meta["constants"][signature[arg_num].name] = 1  # type: ignore[index]
        # end Modify by CAMBRICON
    matrix_instr_nonkdim = self.meta.get("matrix_instr_nonkdim", None)
    waves_per_eu = self.meta.get("waves_per_eu", None)
    kpack = self.meta.get("kpack", None)
    if matrix_instr_nonkdim:
        triton_meta["matrix_instr_nonkdim"] = matrix_instr_nonkdim
    if waves_per_eu:
        triton_meta["waves_per_eu"] = waves_per_eu
    if kpack:
        triton_meta["kpack"] = kpack

    for k in tlx_only_cuda_options():
        if v := self.meta.get(k, None):
            triton_meta[k] = v

    if self.triton_meta is None:
        self.triton_meta = triton_meta
    else:
        self.triton_meta.update(triton_meta)

    inductor_meta = {
        "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
        # Modify by CAMBRICON
        # **self.inductor_meta_common(),
        **TritonKernel.inductor_meta_common(),
        # end Modify by CAMBRICON
        **FixedGrid.setup_grid_as_args(),
    }
    if config.profile_bandwidth or config.benchmark_kernel:
        num_gb = self.estimate_kernel_num_bytes() / 1e9
        inductor_meta["kernel_num_gb"] = num_gb
    if config.benchmark_kernel:
        flops = self.estimate_flops()
        inductor_meta["kernel_flop"] = flops

    inductor_meta["config_args"] = self.meta

    template_args = f"""
        num_stages={self.num_stages},
        num_warps={self.num_warps},
        triton_meta={self.triton_meta!r},
        inductor_meta={inductor_meta!r},
    """

    if HAS_WARP_SPEC:
        template_args += f"""
        num_consumer_groups={self.num_consumer_groups},
        num_buffers_warp_spec={self.num_buffers_warp_spec},
    """

    for k in tlx_only_cuda_options():
        if v := self.meta.get(k, None):
            template_args += f"""
                {k}={v},
            """
            self.triton_meta[k] = v

    return f"""
        @triton_heuristics.template(
            {template_args}
        )
        @triton.jit
    """


patch = gorilla.Patch(
    TritonTemplateKernel,
    "jit_lines",
    jit_lines,
)
gorilla.apply(patch)
