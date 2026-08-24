import os
import sys

import warnings
from typing import Literal, Optional
import torch
from torch.utils._config_module import Config, get_tristate_env, install_config_module


def _warn_custom_op_config(stacklevel=2):
    def is_first_time():
        if not hasattr(_warn_custom_op_config, "has_warned"):
            return True
        else:
            return not _warn_custom_op_config.__dict__["has_warned"]

    if is_first_time():
        message = (
            "torch_mlu._inductor.config.aot_inductor.custom_op_libs will be deprecated, "
            "and replaced by torch._inductor.config.aot_inductor.custom_op_libs. "
            "torch_mlu._inductor.config.aot_inductor.custom_ops_to_c_shims will be deprecated, "
            "and replaced by torch._inductor.config.aot_inductor.custom_ops_to_c_shims."
        )
        warnings.warn(message, UserWarning, stacklevel=stacklevel + 1)
        _warn_custom_op_config.__dict__["has_warned"] = True


class aot_inductor:
    # Custom ops that have implemented C shim wrappers, defined as an op to C shim declaration dict
    custom_ops_to_c_shims: dict[torch._ops.OpOverload, list[str]] = {}
    # custom op libs that have implemented C shim wrappers
    custom_op_libs: Optional[list[str]] = None


torch._inductor.config.combo_kernels_autotune = 2
torch._inductor.config.unroll_reductions_threshold = 1
torch._inductor.config.split_reductions = False
torch._inductor.config.fallback_random = True
torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.inplace_buffers = False
torch._inductor.config.optimize_scatter_upon_const_tensor = (
    os.environ.get("TORCHINDUCTOR_OPTIMIZE_SCATTER_UPON_CONST_TENSOR", "0") == "1"
)
torch._inductor.config.size_asserts = (
    os.environ.get("TORCHINDUCTOR_SIZE_ASSERTS", "1") == "1"
)
torch._inductor.config.triton.max_tiles = 3
torch._inductor.config.triton.use_block_ptr = True
torch._inductor.config.triton.persistent_reductions = True
torch._inductor.config.triton.prefer_nd_tiling = True
torch._inductor.config.triton.codegen_upcast_to_fp32 = (
    os.environ.get("TORCHINDUCTOR_CODEGEN_UPCAST_TO_FP32", "1") == "1"
)
torch._inductor.config.layout_optimization = (
    os.environ.get("TORCHINDUCTOR_LAYOUT_OPTIMIZATION", "0") == "1"
)
torch._inductor.config.max_autotune_gemm_backends = "ATEN"

# mlu triton not support scalar.item() value dtype is fp64.
torch._inductor.config._use_fp64_for_unbacked_floats = False

# Currently do not support this feature, temporarily turn off by default to prevent opened incorrectly by gpu migration.
torch._inductor.config.shape_padding = (
    os.environ.get("TORCHINDUCTOR_SHAPE_PADDING", "0") == "1"
)
# by default, disbale in mlu
torch._inductor.config.comprehensive_padding = (
    os.environ.get("TORCHINDUCTOR_COMPREHENSIVE_PADDING", "0") == "1"
)
torch._inductor.config.alignment_asserts = (
    os.environ.get("TORCHINDUCTOR_ALIGNMENT_ASSERTS", "0") == "1"
)
torch._inductor.config.loop_ordering_after_fusion: bool = (
    os.environ.get("TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION", "0") == "1"
)

# by default, enable in mlu
# autotune_fallback_to_aten is deprecated, mlu will remove using the config in cat lowering func
torch._inductor.config.autotune_fallback_to_aten = True

torch._inductor.config.max_fusion_size = os.environ.get(
    "TORCHINDUCTOR_MLU_MAX_FUSION_SIZE", 32
)

# used for debug
torch._inductor.config.triton.unique_kernel_names = (
    os.environ.get("TORCHINDUCTOR_UNIQUE_KERNEL_NAMES", "1") == "1"
)

filter_configs = (
    os.environ.get("TORCHINDUCTOR_MLU_FILTER_PRECOMPILE_CONFIGS", "1") == "1"
)
debug_tunning = os.environ.get("TORCHINDUCTOR_MLU_DEBUG_TUNNING", "0") == "1"
enable_post_grad = os.environ.get("TORCHINDUCTOR_MLU_ENABLE_POSTGRAD", "1") == "1"
emulate_precision_casts_ops = [
    x.strip()
    for x in os.environ.get("TORCHINDUCTOR_MLU_EMULATE_PRECISION_CASTS_OPS", "").split(
        ","
    )
    if x
]


# FX passes to skip during compilation.
# Use TORCHINDUCTOR_MLU_SKIPPED_FX_PASSES environment variable to specify passes to skip
# as a comma-separated string. For example: "fold_cat,normalization"
#
# To get the list of valid pass names, use the API:
#   from torch_mlu._inductor.fx_passes import get_skippable_fx_passes
#   passes = get_skippable_fx_passes()
#   print("Valid passes:", ", ".join(passes))
skipped_fx_passes = [
    x.strip()
    for x in os.environ.get("TORCHINDUCTOR_MLU_SKIPPED_FX_PASSES", "").split(",")
    if x
]

# FX passes to explicitly enable during compilation.
# Use TORCHINDUCTOR_MLU_ENABLED_FX_PASSES environment variable to specify passes to enable
# as a comma-separated string. When unset, passes keep their default behavior.
# For example: "combo_matmul"
#
# To get the list of valid pass names, use the API:
#   from torch_mlu._inductor.fx_passes import get_enableable_fx_passes
#   passes = get_enableable_fx_passes()
#   print("Valid passes:", ", ".join(passes))
enabled_fx_passes = [
    x.strip()
    for x in os.environ.get("TORCHINDUCTOR_MLU_ENABLED_FX_PASSES", "").split(",")
    if x
]

min_combine_mm_width = max(
    int(os.environ.get("TORCHINDUCTOR_MLU_PASS_COMBINE_MM_WIDTH", 4)), 2
)

min_combine_poi_width = max(
    int(os.environ.get("TORCHINDUCTOR_MLU_PASS_COMBINE_POINTWISE_WIDTH", 3)), 2
)


class tritonfusion:
    skipped_fusing_ops = [
        x.strip()
        for x in os.environ.get(
            "TORCHINDUCTOR_MLU_TRITONFUSION_SKIPPING_OPS", ""
        ).split(",")
        if x
    ]


add_lowering_list = [
    x.strip()
    for x in os.environ.get("TORCHINDUCTOR_MLU_ADD_LOWERING_DENYLIST", "").split(",")
    if x
]
remove_lowering_list = [
    x.strip()
    for x in os.environ.get("TORCHINDUCTOR_MLU_REMOVE_LOWERING_DENYLIST", "").split(",")
    if x
]
add_decomp_list = [
    x.strip()
    for x in os.environ.get("TORCHINDUCTOR_MLU_ADD_DECOMP_DENYLIST", "").split(",")
    if x
]
remove_decomp_list = [
    x.strip()
    for x in os.environ.get("TORCHINDUCTOR_MLU_REMOVE_DECOMP_DENYLIST", "").split(",")
    if x
]

enable_triton_fusion = (
    os.environ.get("TORCHINDUCTOR_MLU_ENABLE_TRITON_FUSION", "0") == "1"
)

# By default, libdevice uses kernels matching eager‐mode precision.
# The “use_ultra_math” flag enables an ultra‐fast (but lower‐precision) code path,
# and is only effective when both “use_ultra_math” is true and the operator’s
# ULTRA environment variable is enabled.
use_ultra_math = os.environ.get("TORCHINDUCTOR_MLU_USE_ULTRA_MATH", "0") == "1"
use_ultra_gelu = os.environ.get("TORCHINDUCTOR_MLU_USE_ULTRA_GELU", "1") == "1"
use_ultra_tanh = os.environ.get("TORCHINDUCTOR_MLU_USE_ULTRA_TANH", "1") == "1"
use_ultra_sigmoid = os.environ.get("TORCHINDUCTOR_MLU_USE_ULTRA_SIGMOID", "1") == "1"
use_ultra_silu = os.environ.get("TORCHINDUCTOR_MLU_USE_ULTRA_SILU", "1") == "1"
use_ultra_erf = os.environ.get("TORCHINDUCTOR_MLU_USE_ULTRA_ERF", "1") == "1"

use_fast_div = os.environ.get("TORCHINDUCTOR_MLU_USE_FAST_DIV", "0") == "1"

# torch._inductor.config.triton.debug_sync_kernel = True
# torch._inductor.config.triton.debug_sync_graph = True
# torch._inductor.config.autotune_fallback_to_aten = False
# torch._inductor.config.max_autotune_gemm_search_space = "EXHAUSTIVE"
# torch._inductor.config.max_autotune_gemm_backends = "ATEN,triton"

# NOTE(luohaizhao): we use this env to control codecache lock timeout value
# to avoid compile failed cuz timeout(PYTORCH-16627)
#
# The current codecache logic, designed to support caching across multiple
# processes, works roughly like this:
#
# 1. Generate a hash value from the code.
# 2. Use the hash value to acquire a file lock.
# 3. Compile the code into a .so using g++.
# 4. Release the file lock.
#
# When multiple processes share the same hash:
#
#  i.   Process A acquires the lock first and proceeds to step 2.
#  ii.  Process B, being slower, waits to acquire the lock.
#  iii. If Process A takes too long, Process B will eventually timeout.

codecache_lock_ori_timeout = torch._inductor.codecache.LOCK_TIMEOUT
try:
    timeout_str = os.environ.get(
        "TORCHINDUCTOR_MLU_CODECACHE_LOCK_TIMEOUT", str(codecache_lock_ori_timeout)
    )
    new_timeout = int(timeout_str)
except ValueError:
    new_timeout = codecache_lock_ori_timeout
    import logging

    log = logging.getLogger(__name__)
    log.warning(
        f"TORCHINDUCTOR_MLU_CODECACHE_LOCK_TIMEOUT env contains invalid digits str: {timeout_str}"
    )

# Backend to use for mlu
mlu_backend: Literal["triton"] = "triton"


torch._inductor.codecache.LOCK_TIMEOUT = new_timeout
install_config_module(sys.modules[__name__])

# set pass config
from torch_mlu._inductor.fx_passes.mlu_pre_grad_pass import mlu_pre_grad_pass
from torch_mlu._inductor.fx_passes.mlu_post_pass import MLUPostPass
from torch_mlu._inductor.fx_passes.joint_graph_pass import MLUJointPass

if not isinstance(torch._inductor.config.joint_custom_post_pass, MLUJointPass):
    torch._inductor.config.joint_custom_post_pass = MLUJointPass(
        torch._inductor.config.joint_custom_post_pass
    )
if not isinstance(torch._inductor.config.post_grad_custom_post_pass, MLUPostPass):
    torch._inductor.config.post_grad_custom_post_pass = MLUPostPass(
        torch._inductor.config.post_grad_custom_post_pass
    )

torch._inductor.config.pre_grad_custom_pass = mlu_pre_grad_pass
