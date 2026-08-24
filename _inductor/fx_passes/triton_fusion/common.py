"""
@Copyright (C) [2022-2025] by Cambricon.
@File    :   common.py
"""

import functools, sys, operator
from enum import Enum
import torch, sympy
from torch.fx.node import Node, map_aggregate
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx.immutable_collections import immutable_list
import triton.backends.mlu.driver as driver
from contextlib import contextmanager
import torch._inductor.config as inductor_config
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch_mlu._inductor import config as torch_mlu_config
from torch_mlu._inductor.fx_passes.triton_fusion import config as tt_config


class MEMTYPE(Enum):
    NRAM = 101
    WRAM = 102
    SM = 3
    DDR = 1


class ALLOCTYPE(Enum):
    ADVANCED = 1  # pre allocated mem, like cat
    NORMAL = 2  # normal allocated mem
    INVARIANT = 3  # loop invariant mem, like weights for mm without tile


DEVPROB = None


@functools.lru_cache()
def get_dev_attr(attr_str: str, default=None):
    global DEVPROB
    if DEVPROB is None:
        DEVPROB = driver.BangUtils().get_device_properties(torch.mlu.current_device())
    return DEVPROB.get(attr_str, default)


def get_cluster_num():
    return get_dev_attr("cluster_num")


def get_core_num_per_cluster():
    return get_dev_attr("core_num_per_cluster")


def get_max_nram_size():
    return get_dev_attr("max_nram_size", 0)


def get_max_wram_size():
    return get_dev_attr("max_wram_size", 0)


def get_max_shared_mem():
    return get_dev_attr("max_shared_mem", 0)


def get_isa_version():
    return get_dev_attr("isa_version")


def get_total_core_num():
    return get_cluster_num() * get_core_num_per_cluster()


def get_max_grid_sizes():
    return [
        get_dev_attr("max_block_task_dim_x", sys.maxsize),
        get_dev_attr("max_block_task_dim_y", sys.maxsize),
        get_dev_attr("max_block_task_dim_z", sys.maxsize),
    ]


# Promote weight to sm first to save nram.
FORCE_USE_SM = True


BATCHBLOCKNAME = "BS_BLOCK"
TILEDIMNAME = "TILE_DIM"
EVENBSBLOCKNAME = "EVEN_BS_BLOCK"
NUMTASKSNAME = "num_tasks"
TENSORMETANAME = "tensor_meta"
VALMETANAME = "val"
DEVICENAME = "mlu"
TRITONFUSIONDEBUGNAME = "TRITONFUSION_DEBUG"
TRITONFUSION_SKIP_FUSION_FLAG = "TRITONFUSION_SKIP_FUSION_FLAG"
TRITONFUSION_ENABLE = "TORCHINDUCTOR_MLU_ENABLE_TRITON_FUSION"
TRITONFUSION_SAVE_TENSOR_ENV = tt_config.TRITONFUSION_SAVE_TENSOR_ENV
TRITONFUSION_LOAD_TENSOR_ENV = tt_config.TRITONFUSION_LOAD_TENSOR_ENV
MAX_BANDWIDTH_BYTES_PER_LINE = 256
TRITONFUSION_CUSTOM_OPS_NAME = "tritonfusion_custom_ops"

TORCH2TRITON_DTYPE_STR = {
    "torch.float32": "tl.float32",
    "torch.float16": "tl.float16",
    "torch.int32": "tl.int32",
    "torch.int64": "tl.int64",
    "torch.int8": "tl.int8",
    "torch.uint8": "tl.uint8",
    "torch.int16": "tl.int16",
    "torch.bool": "tl.int1",
    "torch.float64": "tl.float64",
    "torch.bfloat16": "tl.bfloat16",
    "torch.complex64": "tl.complex64",
    "torch.complex128": "tl.complex128",
}

TORCH2TRITON_LOAD_STORE_DTYPE_STR = TORCH2TRITON_DTYPE_STR.copy()
TORCH2TRITON_LOAD_STORE_DTYPE_STR["torch.bool"] = "tl.int8"

TORCH_INT_TYPES = [
    torch.int32,
    torch.int64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.bool,
]

TORCH_FLOAT_TYPES = [
    torch.float32,
    torch.float16,
    torch.float64,
    torch.bfloat16,
]

PROPAGATE_NAN = "tl.PropagateNan.ALL"


def get_user(cur_node: Node):
    if cur_node.target != operator.getitem:
        return list(cur_node.users)
    ret = []
    inp_node = cur_node.args[0]
    for usr_node in inp_node.users:
        ret += list(usr_node.users)
    return ret


def check_loop_invariant(node: Node, tile_dim_all: dict[Node:set], ind=0):
    tiledims = tile_dim_all.get(node, None)
    return not (tiledims and tiledims[ind])


def is_shape_dynamic(shape):
    # Be aware, can't process nesting situation.
    if hasattr(shape, "__iter__") and not isinstance(shape, type):
        return any([is_shape_dynamic(x) for x in shape])
    # If type is sympy.Expr/integer/torch.SymInt etc.
    if isinstance(shape, (torch.SymInt, torch.SymFloat)):
        shape = shape._sympy_()
    if isinstance(shape, sympy.Expr):
        return len(shape.free_symbols) > 0
    return False


def get_shape_exprs(shape):
    if hasattr(shape, "__iter__") and not isinstance(shape, type):
        return [get_shape_exprs(x) for x in shape]
    if isinstance(shape, (torch.SymInt, torch.SymFloat)):
        shape = shape._sympy_()
    return shape


def is_tensor_node(node: Node):
    if not isinstance(node, Node):
        return False
    if (VALMETANAME not in node.meta) or isinstance(
        node.meta.get(VALMETANAME), torch.SymInt
    ):
        return False
    return True


def get_tensor_metas(node: Node):
    if not is_tensor_node(node):
        return []
    if TENSORMETANAME not in node.meta:
        node.meta[TENSORMETANAME] = map_aggregate(
            node.meta[VALMETANAME], _extract_tensor_metadata
        )
    meta = node.meta[TENSORMETANAME]
    if isinstance(meta, TensorMetadata):
        return [meta]
    elif isinstance(meta, (tuple, list, immutable_list)):
        return list(meta)
    assert 0, f"get unsupported meta type: {type(meta)} {meta}"


def align(val: int, align: int) -> int:
    """
    Aligns `val` to the nearest multiple of `align`.
    If `val` is already a multiple of `align`, returns `val`;
    otherwise, returns the smallest multiple of `align` greater than `val`.
    """
    if align == 0:
        raise ValueError("align must not be 0")
    return (val + align - 1) // align * align


def get_torch_dtype_bytes(dty: torch.dtype):
    if dty.is_floating_point:
        return torch.finfo(dty).bits / 8
    if dty == torch.bool:
        return 1
    return torch.iinfo(dty).bits / 8


def get_target_name(node: Node):
    if node.op == "placeholder" or node.op == "output":
        return node.op
    return str(node.target)


@contextmanager
def torch_compile_without_cache(
    enable_tritonfusion=None,
    tritonfusion_pre_perf=None,
    tritonfusion_pre_check=None,
):
    kep_cache_env = inductor_config.force_disable_caches
    kepenv_tt = torch_mlu_config.enable_triton_fusion
    kep_pre_perf = tt_config.pre_test_perf_eager
    kep_pre_check = tt_config.pre_check_triton_kernel
    try:
        inductor_config.force_disable_caches = True
        if enable_tritonfusion is not None:
            torch_mlu_config.enable_triton_fusion = enable_tritonfusion
        if tritonfusion_pre_perf is not None:
            tt_config.pre_test_perf_eager = tritonfusion_pre_perf
        if tritonfusion_pre_check is not None:
            tt_config.pre_check_triton_kernel = tritonfusion_pre_check
        yield
    finally:
        torch.compiler.reset()
        torch_mlu_config.enable_triton_fusion = kepenv_tt
        inductor_config.force_disable_caches = kep_cache_env
        tt_config.pre_test_perf_eager = kep_pre_perf
        tt_config.pre_check_triton_kernel = kep_pre_check
