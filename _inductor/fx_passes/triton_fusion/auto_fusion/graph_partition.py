"""
@Copyright (C) [2022-2025] by Cambricon.
@File    :   graph_partition.py
"""

import math
import sys
import torch
from torch import fx
import torch_mlu
from collections import deque
from ..processors import (
    infer_tiledim_front,
    infer_tiledim_back,
    GetitemProcessor,
    get_op_processor,
)
from .memory_calculator import get_memory_total_size
from .. import config as tt_config
from ..config import get_simple_logger
from ..common import (
    MEMTYPE,
    is_shape_dynamic,
    TENSORMETANAME,
    get_target_name,
    is_tensor_node,
    get_torch_dtype_bytes,
    TRITONFUSION_SKIP_FUSION_FLAG,
    MAX_BANDWIDTH_BYTES_PER_LINE,
    get_max_nram_size,
    get_max_wram_size,
    get_max_shared_mem,
    get_isa_version,
    get_total_core_num,
    get_tensor_metas,
    get_user,
)

logger = get_simple_logger(__name__)


def check_memory_size(
    start_node: fx.Node,
    subgraph_fused_nodes: dict[fx.Node : set] | list[fx.Node],
    tile_dim_all: dict[fx.Node : list[set]],
):
    mems = get_memory_total_size(start_node, subgraph_fused_nodes, tile_dim_all)
    isa_v = get_isa_version()
    if isa_v >= 600:
        # The wram and nram use same mem space on arch 6xx.
        mem_req = mems[MEMTYPE.NRAM] + mems[MEMTYPE.WRAM]
        mem_avail = get_max_nram_size() + get_max_wram_size()
        if mem_req > mem_avail:
            logger.debug(f"nram+wram required: {mem_req} VS available: {mem_avail}")
            return False
    elif isa_v >= 500 and isa_v < 600:
        if (
            mems[MEMTYPE.NRAM] > get_max_nram_size()
            or mems[MEMTYPE.WRAM] > get_max_wram_size()
        ):
            logger.debug(
                f"nram required: {mems[MEMTYPE.NRAM]} VS available: {get_max_nram_size()} and\
                wram required: {mems[MEMTYPE.WRAM]} VS available: {get_max_wram_size()}"
            )
            return False
    else:
        AssertionError(f"Unsupported ISA arch num: {get_isa_version()}")
    # SM check.
    if mems[MEMTYPE.SM] > get_max_shared_mem():
        logger.debug(
            f"sm required: {mems[MEMTYPE.SM]} VS available: {get_max_shared_mem()}"
        )
        return False
    return mems


def has_ring_fuse(subgraph_nodes: list[fx.Node], curnode: fx.Node, fuse_front=True):
    if not subgraph_nodes:
        return False
    limit_node = min(subgraph_nodes) if fuse_front else max(subgraph_nodes)
    step_fn = (lambda x: x.all_input_nodes) if fuse_front else (lambda x: x.users)
    worklist = [n for n in step_fn(curnode) if n not in subgraph_nodes]
    visited = set()
    while worklist:
        node = worklist.pop(0)
        if node in visited:
            continue
        visited.add(node)
        if (fuse_front and node < limit_node) or (
            (not fuse_front) and node > limit_node
        ):
            continue
        if node in subgraph_nodes:
            return True
        worklist.extend(step_fn(node))
    return False


def fuse_along_dim(
    start_node: fx.Node,
    visited_graph_nodes: set[fx.Node],
    tile_dim=None,
    max_nodes_limit=None,
    max_inputs_limits=120,
):
    """
    Greedily fuse nodes in the graph starting from start_node along a specified dimension.
    """
    assert tile_dim is not None, "tile_dim must be specified for fuse_along_dim"
    tile_dim_all = {}
    subgraph_fused_nodes = {}
    if start_node in visited_graph_nodes:
        return subgraph_fused_nodes

    def check_nodes_limits(cur_node: fx.Node):
        # Input limit check, in case very big cat with large inputs num,
        # this leads very long time compiling.
        if len(cur_node.all_input_nodes) > max_inputs_limits:
            logger.debug(
                f"check_nodes_limits failed for node: {cur_node}, too many inputs: {len(cur_node.all_input_nodes)} more than limit {max_inputs_limits}"
            )
            return False
        # Num limit check.
        if max_nodes_limit is not None and len(subgraph_fused_nodes) >= max_nodes_limit:
            logger.debug(
                f"check_nodes_limits failed for node: {cur_node}, subgraph nodes num more than max_nodes_limit: {max_nodes_limit}"
            )
            return False
        # Already fused check.
        if cur_node in visited_graph_nodes or cur_node in subgraph_fused_nodes:
            return False
        # Skipped nodes check.
        if get_target_name(cur_node) in tt_config.skipped_fusing_ops:
            logger.debug(
                f"check_nodes_limits failed for node: {cur_node}, {get_target_name(cur_node)} is in skipping list"
            )
            return False
        # Reserved for inductor to fuse
        if cur_node.op == "call_function" and cur_node.meta.get(
            TRITONFUSION_SKIP_FUSION_FLAG
        ):
            logger.debug(
                f"check_nodes_limits failed for node: {cur_node}, reserved for inductor skipping node: {get_target_name(cur_node)}"
            )
            return False
        return True

    def check_fuse_limits(cur_node: fx.Node, self_and_input_tiledims: dict):
        for node in cur_node.all_input_nodes + [cur_node]:
            if not is_tensor_node(cur_node):
                continue
            tiles = self_and_input_tiledims.get(node, None)
            metas = get_tensor_metas(node)
            for ind, meta in enumerate(metas):
                shape = list(meta.shape)
                # Other shape should not be dynamic.
                for dim, x in enumerate(shape):
                    if tiles and dim in tiles[ind]:
                        continue
                    if is_shape_dynamic(x):
                        logger.debug(
                            f"check_fuse_limits find non tiledim dynamic node: {node} {shape} tiledims: {tiles}"
                        )
                        return False
        return True

    def check_multi_user_back_infer_same(cur_node: fx.Node, tiledims: list):
        user_infered_dims = [
            infer_tiledim_back(user, tile_dim_all).get(cur_node, None)
            for user in cur_node.users
            if user.op != "output"
        ]
        if not all(tiledims == x for x in user_infered_dims):
            logger.debug(
                f"back fuse failed: {cur_node} check multi user infer back same failed: \
                {tiledims} vs {user_infered_dims}"
            )
            return False
        return True

    # init
    start_op_processor = get_op_processor(start_node)
    tile_dim_all[start_node] = start_op_processor.get_tiledims_from_dim(
        start_node, tile_dim
    )

    has_fusion = True
    while has_fusion:
        has_fusion = False
        # FIRST FUSE BACK
        for cur_node, tiledims in list(tile_dim_all.items()):
            if not check_nodes_limits(cur_node):
                logger.debug(
                    f"back fuse failed: check_nodes_limits failed for new node {cur_node}"
                )
                continue

            # check if all users of cur_node are added(except start node).
            real_users = [u for u in get_user(cur_node) if u.op != "output"]
            if cur_node != start_node:
                if not all(user in subgraph_fused_nodes for user in real_users):
                    logger.debug(
                        f"back fuse failed: check user limits failed for new node {cur_node}"
                    )
                    continue

                # check if has conflict tile_dim(except start node).
                if len([x for x in cur_node.users if x.op != "output"]) > 1:
                    if not check_multi_user_back_infer_same(cur_node, tiledims):
                        continue
            # try infer back.
            updated_nodes = infer_tiledim_back(cur_node, tile_dim_all)
            if updated_nodes is None:
                continue
            if not check_fuse_limits(cur_node, updated_nodes | {cur_node: tiledims}):
                logger.debug(
                    f"back fuse failed: check_fuse_limits failed for new node {cur_node}"
                )
                continue

            cur_node_all = []
            # Special for getitem op.
            if get_target_name(cur_node) in [GetitemProcessor.opname]:
                inp_node = cur_node.args[0]
                for sub_node in [inp_node] + list(inp_node.users):
                    assert (
                        sub_node in updated_nodes
                    ), f"back fuse error: for getitem op {cur_node} should return input and all subops' tiledims,\
                        but can't find {sub_node}"
                    cur_node_all.append(sub_node)
            else:
                cur_node_all = [cur_node]

            # check nram usage
            if not check_memory_size(
                start_node,
                list(subgraph_fused_nodes) + cur_node_all,
                tile_dim_all | updated_nodes,
            ):
                logger.debug(
                    f"back fuse failed: check_memory_size failed for new node {cur_node}"
                )
                continue

            # Add nodes to subgraph.
            for sub_node in cur_node_all:
                if sub_node == cur_node:
                    subgraph_fused_nodes[sub_node] = tiledims
                else:
                    assert (
                        sub_node in updated_nodes
                    ), f"back fuse error: for getitem op should return input and all subops' tiledims"
                    subgraph_fused_nodes[sub_node] = updated_nodes[sub_node]

            has_fusion = True
            # expand tile_dim_all with all inputs.
            tile_dim_all.update(updated_nodes)

        # THEN FUSE FRONT
        for cur_node, tiledims in list(subgraph_fused_nodes.items()):
            # skip fusion if already flag

            for user_node in cur_node.users:
                if not check_nodes_limits(user_node):
                    logger.debug(
                        f"front fuse failed: check_nodes_limits failed for new node {user_node}"
                    )
                    continue

                # TODO: check if has conflict tile_dim
                updated_nodes = infer_tiledim_front(user_node, tile_dim_all)
                if not updated_nodes:
                    continue

                for updated_node, update_tiledims in updated_nodes.items():
                    if not (update_tiledims and any(update_tiledims)):
                        continue

                    if has_ring_fuse(
                        list(subgraph_fused_nodes.keys()), updated_node, True
                    ):
                        logger.debug(
                            f"front fuse failed: get ring fuse for new node {updated_node}"
                        )
                        continue

                    inferback = infer_tiledim_back(
                        updated_node, {updated_node: update_tiledims}
                    )
                    if not inferback:
                        continue
                    if not check_fuse_limits(
                        updated_node, inferback | {updated_node: update_tiledims}
                    ):
                        logger.debug(
                            f"front fuse failed: check_fuse_limits failed for new node {updated_node}"
                        )
                        continue
                    # Check nram usage.
                    if not check_memory_size(
                        start_node,
                        list(subgraph_fused_nodes) + [updated_node],
                        tile_dim_all | inferback | {updated_node: update_tiledims},
                    ):
                        logger.debug(
                            f"front fuse failed: check_memory_size failed for new node {updated_node}"
                        )
                        continue
                    tile_dim_all[updated_node] = update_tiledims
                    tile_dim_all.update(inferback)
                    subgraph_fused_nodes[updated_node] = update_tiledims
                    has_fusion = True
    logger.debug(
        f"fuse start from {start_node} along dim: {tile_dim} finally fused {len(subgraph_fused_nodes)} nodes with memory: {check_memory_size(start_node, subgraph_fused_nodes, tile_dim_all)}"
    )
    return subgraph_fused_nodes


def analyze_fuse_profit(start_node, tiledim_old, old_graph, tiledim_new, new_graph):
    # First time update.
    if tiledim_old is None or not old_graph:
        return sys.maxsize

    if len(new_graph) > len(old_graph):
        return len(new_graph) - len(old_graph)
    elif len(new_graph) == len(old_graph):
        start_op_processor = get_op_processor(start_node)
        start_shape = start_op_processor.get_output_shape(start_node)
        if (
            start_shape[tiledim_new] > start_shape[tiledim_old]
            and tiledim_new < len(start_shape) - 1
        ):
            return start_shape[tiledim_new] - start_shape[tiledim_old]
    return -1


def fuse_greedily(
    start_node: fx.Node,
    visited_graphs: list[list[fx.Node]],
    max_nodes_limit=None,
    max_inputs_limits=120,
):
    start_op_processor = get_op_processor(start_node)
    start_shape = start_op_processor.get_output_shape(start_node)
    max_nodes = None
    best_dim_val = None
    best_tiledims_subgraph = {}

    visited_nodes = set()
    for vg in visited_graphs:
        visited_nodes.update(vg)
    # check if start_node is already visited
    if start_node in visited_nodes:
        return best_tiledims_subgraph

    tiledim_find_list = []
    dynamic_dims = [
        (ind, x) for ind, x in enumerate(start_shape) if is_shape_dynamic(x)
    ]
    logger.debug(
        f"fuse_greedily: find dynamic dims: {dynamic_dims} for start node: {start_node} with shape: {start_shape}"
    )

    if len(dynamic_dims) > 1:
        logger.error(
            f"fuse_greedily: get too many dynamic dims for node: {start_node}  with shape: {start_shape}"
        )
        return best_tiledims_subgraph

    if dynamic_dims:
        logger.debug(
            f"fuse_greedily: begin find tiledim in dynamic dims: {dynamic_dims}"
        )
        tiledim_find_list = dynamic_dims
    else:
        logger.debug(f"fuse_greedily: begin find tiledim in all dims: {start_shape}")
        tiledim_find_list = [(ind, x) for ind, x in enumerate(start_shape)]

    for dim, st_shape in tiledim_find_list:
        # Skip if tile dim is cat's lowest dim, this is not io efficient.
        if (
            max_nodes is not None
            and start_node.op == "call_function"
            and str(start_node.target) == "aten.cat.default"
            and dim == len(start_shape) - 1
        ):
            continue
        logger.debug(
            f"fuse_greedily: start_node: {start_node} try fuse along dim: {dim}"
        )
        subgraph_tiledims = fuse_along_dim(
            start_node,
            visited_nodes,
            dim,
            max_nodes_limit,
            max_inputs_limits=max_inputs_limits,
        )
        if not subgraph_tiledims:
            continue
        fusion_num = len(subgraph_tiledims)
        logger.debug(
            f"fuse_greedily: start_node: {start_node} along dim: {dim} find {fusion_num} nodes: {subgraph_tiledims}"
        )
        # Update best tile result.
        if (
            analyze_fuse_profit(
                start_node,
                best_dim_val,
                best_tiledims_subgraph,
                dim,
                subgraph_tiledims,
            )
            > 0
        ):
            max_nodes = fusion_num
            best_dim_val = dim
            best_tiledims_subgraph = subgraph_tiledims

    logger.debug(
        f"fuse_greedily: start_node: {start_node} finally find {best_tiledims_subgraph}"
    )
    return best_tiledims_subgraph


def get_start_nodes(graph: fx.graph.Graph, node_types: list[str]) -> list[fx.Node]:
    targets = [
        node
        for node in reversed(graph.nodes)
        if node.op == "call_function" and (get_target_name(node) in node_types)
    ]
    return targets


# Entry function
def find_subgraphs(
    graph,
    min_nodes=1,
    max_nodes=None,
    visited_subgraphs=[],
    graphnum=None,
    max_inputs_limits=120,
):
    subgraphs = []
    kep_subgraph_nodes = []
    start_nodes = get_start_nodes(graph, tt_config.fusing_start_ops)

    for start_node in start_nodes:
        subgraph = fuse_greedily(
            start_node,
            kep_subgraph_nodes + visited_subgraphs,
            max_nodes,
            max_inputs_limits=max_inputs_limits,
        )
        if len(subgraph) < min_nodes or (
            max_nodes is not None and len(subgraph) > max_nodes
        ):
            continue

        subgraphs.append((start_node, subgraph))
        kep_subgraph_nodes.append(subgraph.keys())
        if graphnum is not None and len(subgraphs) >= graphnum:
            break

    return subgraphs
