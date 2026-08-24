from __future__ import annotations

import dataclasses
import re
from itertools import chain, count, zip_longest
from typing import Any, Optional, Union
from typing_extensions import Self

import sympy
import textwrap

import torch
from torch import dtype as torch_dtype

from torch._inductor.runtime.runtime_utils import dynamo_timed

from torch._inductor import config
from torch._inductor.codecache import CudaKernelParamCache
from torch._inductor.ir import GraphPartitionSignature, TensorBox

from torch_mlu._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch_mlu._inductor.codegen import cpp_wrapper_gpu
from torch._inductor.virtualized import V
from torch._inductor.codegen.aoti_hipify_utils import maybe_hipify_code_wrapper
from torch._inductor.codegen.common import get_device_op_overrides
from torch._inductor.codegen.cpp_utils import cexpr
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.codegen.triton_utils import should_unwrap_unspec_arg
from torch._inductor.codegen.wrapper import PythonWrapperCodegen, SymbolicCallArg
from torch._inductor.codegen.cpp_wrapper_gpu import (
    DeferredTritonCallWrapper,
    CppWrapperGpu,
)

DEVICE_TO_ATEN = {
    "meta": "at::kMeta",
    "cpu": "at::kCPU",
    "cuda": "at::kCUDA",
    "xpu": "at::kXPU",
    "mlu": "at::kPrivateUse1",
}

_cpp_string_literal_escapes = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\t": "\\t",
    "\r": "\\r",
}
_cpp_string_literal_pattern = re.compile(r'["\\\n\t\r]')


def cpp_string_literal(s: str) -> str:
    escaped = _cpp_string_literal_pattern.sub(
        lambda match: _cpp_string_literal_escapes[match.group(0)], s
    )
    return f'"{escaped}"'


class CppWrapperMlu(CppWrapperGpu, CppWrapperCpu):
    """
    Generates cpp wrapper for running on MLU and calls MLU kernels
    """

    def __init__(self) -> None:
        self.device = "mlu"
        self.device_codegen = get_device_op_overrides(self.device)
        self._kernel_name_to_body: dict[str, str] = {}
        self._triton_call_wrappers: dict[str, DeferredTritonCallWrapper] = {}
        self.autotune_input_prefix = "_REAL_AUTOTUNE_INPUT"
        super(CppWrapperGpu, self).__init__()
        self.grid_id = count()

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        # TODO - support subgraph codegen by lifting functions. Check the
        # comment at CppWrapperCpu `codegen_subgraph` function.
        return CppWrapperMlu()

    def generate(self, is_inference):
        with dynamo_timed("CppWrapperMlu.generate", log_pt2_compile_event=True):
            return super().generate(is_inference)

    def codegen_device(self, device):
        assert device.type in DEVICE_TO_ATEN, (
            device.type + " not found in DEVICE_TO_ATEN"
        )
        device_str = DEVICE_TO_ATEN[device.type][5:].lower()  # remove "at::k"
        self.used_cached_devices.add(device_str)
        return f"cached_torch_device_type_{device_str}, {device.index if device.index else 0}"

    @staticmethod
    def get_device_include_path(device: str) -> str:
        include_path = """
        #include <torch/torch.h>
        #include "cn_api.h"
        #include <cnrt.h>
        #include "framework/core/MLUStream.h"
        #include "framework/core/device.h"
        #include "framework/inductor/aoti_runtime/utils_mlu.h"
        #include "framework/inductor/aoti_torch/generated/c_shim_mlu.h"
        #include "framework/inductor/aoti_torch/generated/c_shim_extra_mlu.h"
        #include "framework/inductor/cpp_wrapper/common.h"
        """
        return include_path
