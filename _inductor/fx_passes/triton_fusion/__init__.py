"""
@Copyright (C) [2022-2025] by Cambricon.
@File    :   __init__.py
"""
import os
from . import config

from .transforms import (
    TransformTorch2Triton,
    get_triton_inductor_config,
    triton_fusion_config_prune,
    get_triton_inductor_grid,
    get_triton_inductor_grid_fn,
    get_tensor_strided,
)
from .processors import (
    REGISTERED_PROCESSOR,
    infer_tiledim_front,
    infer_tiledim_back,
    infer_tiledim_back_all,
    infer_tiledim_front_all,
    is_supported_operation,
    get_externkernelchoice,
)

from .auto_fusion import find_subgraphs

from .triton_fusion import triton_fusion_pass
from .common import (
    get_total_core_num,
    TRITONFUSION_ENABLE,
    torch_compile_without_cache,
    get_max_grid_sizes,
    get_isa_version,
    get_max_shared_mem,
    get_max_wram_size,
    get_max_nram_size,
    get_cluster_num,
)

__all__ = [
    "TransformTorch2Triton",
    "get_triton_inductor_config",
    "triton_fusion_config_prune",
    "get_triton_inductor_grid",
    "get_triton_inductor_grid_fn",
    "REGISTERED_PROCESSOR",
    "infer_tiledim_front",
    "infer_tiledim_back",
    "infer_tiledim_back_all",
    "infer_tiledim_front_all",
    "is_supported_operation",
    "find_subgraphs",
    "triton_fusion_pass",
    "get_tensor_strided",
    "get_total_core_num",
    "get_max_grid_sizes",
    "get_isa_version",
    "get_max_shared_mem",
    "get_max_wram_size",
    "get_max_nram_size",
    "get_cluster_num",
    "TRITONFUSION_ENABLE",
    "get_externkernelchoice",
    "torch_compile_without_cache",
]
