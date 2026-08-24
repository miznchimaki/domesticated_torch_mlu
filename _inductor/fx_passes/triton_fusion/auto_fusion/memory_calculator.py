"""
@Copyright (C) [2022-2025] by Cambricon.
@File    :   memory_calculator.py
"""

import math
import torch
from torch.fx.node import Node
from .allocator import BFCAllocator
from ..transforms import get_inputs_outputs, can_promote_shared
from ..processors import (
    get_memory_require,
    SUPPORTED_MM_OPS,
    REGISTERED_PROCESSOR,
    get_op_processor,
)
from ..config import *
from ..common import (
    MEMTYPE,
    ALLOCTYPE,
    check_loop_invariant,
    get_core_num_per_cluster,
    get_torch_dtype_bytes,
    get_target_name,
    TENSORMETANAME,
    is_tensor_node,
    FORCE_USE_SM,
    get_tensor_metas,
    get_user,
)

logger = get_simple_logger(__name__)


def check_is_filter(nod: Node, nodes: list[Node]):
    nod_mm_users = [
        x
        for x in get_user(nod)
        if (x in nodes) and (get_target_name(x) in SUPPORTED_MM_OPS)
    ]
    is_filter = False
    if nod_mm_users:
        for nod_mm in nod_mm_users:
            if get_op_processor(nod_mm).get_filter(nod_mm) == nod:
                is_filter = True
                break
    return is_filter


def check_is_advance_alloc(nod: Node):
    return get_target_name(nod) in ["aten.cat.default"]


def check_promote_sm(start_node, tile_dim_all):
    use_promote_sm = False
    start_tiledims = tile_dim_all.get(start_node, None)
    if FORCE_USE_SM and start_tiledims:
        for ind, meta in enumerate(get_tensor_metas(start_node)):
            if not start_tiledims[ind]:
                continue
            tiledim_ind = list(start_tiledims[ind])
            st_shape = meta.shape[tiledim_ind[0]]
            use_promote_sm = can_promote_shared(st_shape)
            break
    return use_promote_sm


def get_memory_total_size(
    start_node: Node,
    subgraph_fused_nodes: dict[Node:set] | list[Node],
    tile_dim_all: dict[Node:set],
):
    nodes = sorted(subgraph_fused_nodes)
    logger.debug(
        f"Begin compute memory size for {len(nodes)} nodes: {subgraph_fused_nodes}"
    )
    ins, outs = get_inputs_outputs(nodes)
    logger.debug(
        f"Compute memory size get {len(ins)} inputs: { {x:tile_dim_all.get(x, None) for x in ins} }"
    )
    nram_alloc = BFCAllocator()
    wram_alloc = BFCAllocator()
    sm_alloc = BFCAllocator()

    def get_total_sizes():
        return {
            MEMTYPE.NRAM: nram_alloc.total_size(),
            MEMTYPE.WRAM: wram_alloc.total_size(),
            MEMTYPE.SM: sm_alloc.total_size(),
        }

    total_nodes = nodes + ins
    node_du_counter = {}
    # Init all nodes with du count in nodes.
    for inode in total_nodes:
        cnt = 0
        for u in inode.users:
            # Don't count input nodes' outside users.
            if inode in ins and u not in subgraph_fused_nodes:
                continue
            # Advanced mem alloc won't count.
            if check_is_advance_alloc(u):
                continue
            cnt += 1
        node_du_counter[inode] = cnt
    node_mem_alloc = {x: [] for x in total_nodes}
    # First init inputs mem.
    for nod in ins:
        if not is_tensor_node(nod):
            continue
        tiledims = tile_dim_all.get(nod, None)
        nod_metas = get_tensor_metas(nod)
        assert not (
            tiledims and len(nod_metas) != len(tiledims)
        ), f"node {nod} get different len of meta and tiledims: {len(nod_metas)} vs {len(tiledims)}"
        for ind, meta in enumerate(nod_metas):
            out_shape = list(meta.shape)
            out_dtype = meta.dtype
            if tiledims:
                for dim in tiledims[ind]:
                    out_shape[dim] = 1
            nram = math.prod(out_shape) * get_torch_dtype_bytes(out_dtype)
            wram = 0
            sm = 0
            alloctype = ALLOCTYPE.NORMAL

            # Process weight.
            if check_is_filter(nod, nodes):
                if check_promote_sm(start_node, tile_dim_all):
                    # Now prompote sm is each core has it's weight mem.
                    sm = nram * get_core_num_per_cluster()
                    nram = 0
            if check_loop_invariant(nod, tile_dim_all, ind):
                alloctype = ALLOCTYPE.INVARIANT

            logger.debug(
                f"Compute memory alloc for input nod: {nod} 's output {ind} with nram: {nram}  wram: {wram}  sm: {sm}  alloctype: {alloctype}"
            )
            node_mem_alloc[nod].append(
                {
                    MEMTYPE.NRAM: nram_alloc.alloc(nram),
                    MEMTYPE.WRAM: wram_alloc.alloc(wram),
                    MEMTYPE.SM: sm_alloc.alloc(sm),
                    ALLOCTYPE: alloctype,
                }
            )
        logger.debug(
            f"after alloc for input: {nod} the total sizes are: {get_total_sizes()}"
        )
    # Process advance mem first.
    cat_nodes = [x for x in nodes if check_is_advance_alloc(x)]
    if cat_nodes:
        for nod in cat_nodes:
            mems_all = get_memory_require(nod, tile_dim_all)
            logger.debug(f"First alloc for cat node: {nod} with memory: {mems_all}")
            for mems in mems_all:
                node_mem_alloc[nod].append(
                    {
                        MEMTYPE.NRAM: nram_alloc.alloc(mems[MEMTYPE.NRAM]),
                        MEMTYPE.WRAM: wram_alloc.alloc(mems[MEMTYPE.WRAM]),
                        MEMTYPE.SM: sm_alloc.alloc(mems[MEMTYPE.SM]),
                        ALLOCTYPE: ALLOCTYPE.ADVANCED,
                    }
                )

    # Process inner nodes.
    for nod in nodes:
        # Skip advance allocated op.
        if check_is_advance_alloc(nod):
            continue
        # Skip advance allocated op's inputs alloc.
        if node_du_counter[nod] > 0:
            # Begin process inner nodes' alloc.
            mems_all = get_memory_require(nod, tile_dim_all)
            logger.debug(
                f"Compute memory alloc for inner node: {nod} with memory: {mems_all}"
            )
            for mems in mems_all:
                node_mem_alloc[nod].append(
                    {
                        MEMTYPE.NRAM: nram_alloc.alloc(mems[MEMTYPE.NRAM]),
                        MEMTYPE.WRAM: wram_alloc.alloc(mems[MEMTYPE.WRAM]),
                        MEMTYPE.SM: sm_alloc.alloc(mems[MEMTYPE.SM]),
                        ALLOCTYPE: mems[ALLOCTYPE],
                    }
                )
            logger.debug(
                f"after alloc for inner node: {nod} the total sizes are: {get_total_sizes()}"
            )
        # Free node's inputs.
        for inp_nod in nod.all_input_nodes:
            node_du_counter[inp_nod] -= 1
            assert (
                node_du_counter[inp_nod] >= 0
            ), f"Get error user count with node: {inp_nod}"
            if node_du_counter[inp_nod] == 0:
                logger.debug(
                    f"Compute memory will free inputs for inner node: {nod} with {node_mem_alloc[inp_nod]}"
                )
                for node_mem in node_mem_alloc[inp_nod]:
                    # Skip advaced/invariant free.
                    if node_mem[ALLOCTYPE] in [ALLOCTYPE.INVARIANT]:
                        continue
                    assert (
                        node_mem[MEMTYPE.NRAM] is not None
                    ), f"Get error double free with node: {inp_nod}"
                    nram_alloc.free(node_mem[MEMTYPE.NRAM])
                    node_mem[MEMTYPE.NRAM] = None

                    assert (
                        node_mem[MEMTYPE.WRAM] is not None
                    ), f"Get error double free with node: {inp_nod}"
                    wram_alloc.free(node_mem[MEMTYPE.WRAM])
                    node_mem[MEMTYPE.WRAM] = None

                    assert (
                        node_mem[MEMTYPE.SM] is not None
                    ), f"Get error double free with node: {inp_nod}"
                    sm_alloc.free(node_mem[MEMTYPE.SM])
                    node_mem[MEMTYPE.SM] = None
    ret = get_total_sizes()
    logger.debug(
        f"Finally alloc for {len(subgraph_fused_nodes)} nodes get total sizes: {ret}"
    )
    return ret
