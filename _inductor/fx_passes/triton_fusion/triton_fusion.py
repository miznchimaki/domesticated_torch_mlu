"""
@Copyright (C) [2022-2025] by Cambricon.
@File    :   triton_fusion.py
"""

import operator
import torch
import traceback
import copy
import torch.fx as fx

from .auto_fusion import find_subgraphs
from .transforms import TransformTorch2Triton
from . import config as tt_config
from .common import TRITONFUSION_CUSTOM_OPS_NAME
from .config import get_simple_logger

logger = get_simple_logger(__name__)


def move_dependent_nodes_before(tagert_node, move_point):
    if tagert_node <= move_point:
        return

    def get_backward_slice(start_node):
        visited = set()
        worklist = [start_node]
        while worklist:
            node = worklist.pop()
            if node in visited:
                continue
            visited.add(node)
            for input_node in node.all_input_nodes:
                if input_node not in visited and input_node > move_point:
                    worklist.append(input_node)
        return sorted(visited)

    moved_ops = get_backward_slice(tagert_node)
    for moved_op in moved_ops:
        move_point.prepend(moved_op)


def prepare_fusion_insertion(subgraph: dict[fx.Node, set]):
    """
    Adjusts the positions of external inputs and users of a fusion subgraph
    to ensure that the fusion insertion point is topologically valid.
    Specifically:
        - Moves external input nodes (not in the subgraph but used by it) that are
          located between the first and last subgraph nodes to just before the first node,
          if they can be safely moved (i.e., all their inputs appear before the subgraph).
        - Moves external user nodes (not in the subgraph but using its outputs) that are
          located between the first and last subgraph nodes to just after the last node,
          if they can be safely moved (i.e., all their inputs appear before or within the subgraph).
    The operation is performed in-place and preserves topological correctness.
    """
    # Simulate an ordered set using a dictionary.
    input_nodes = {}
    output_nodes = set()
    for node in subgraph:
        for inp in node.all_input_nodes:
            if inp not in subgraph:
                input_nodes[inp] = True
        for out in list(node.users):
            if out not in subgraph:
                output_nodes.add(node)
                break
    top_output_node = min(output_nodes)
    for node in list(input_nodes):
        if node < top_output_node:
            continue
        move_dependent_nodes_before(node, top_output_node)


def triton_fusion_pass(graph: fx.graph.Graph) -> bool:
    changed = False
    subgraphs = find_subgraphs(
        graph,
        min_nodes=tt_config.min_fused_nodes_num,
        max_nodes=tt_config.max_fused_nodes_num,
        graphnum=1,
    )
    total_fused_ops = 0
    total_failed_ops = 0
    ind = 0
    succeeded_cnt = 0
    visited_subgraphs = []
    graph_id = str(id(graph))
    while len(subgraphs) > 0:
        graph_name = f"graph{ind:0>5}_{graph_id}"
        start_node, fused_subgraph = subgraphs[0]
        logger.info(f"fusing {graph_name}: {len(fused_subgraph)}")
        logger.info(fused_subgraph)

        try:
            ite = TransformTorch2Triton(
                graph, fused_subgraph, graph_name, start_node=start_node
            )
            # Pre test perf with eager and local triton to avoid some bad fuse.
            if not tt_config.save_in_out_tensors and not tt_config.load_in_tensors:
                inp_fn = ite.get_input_func()
                if tt_config.pre_test_perf_eager:
                    inputs_t = inp_fn()
                    eager_perf_fn = ite.get_perf_test_func_eager()
                    eager_perf_ms = eager_perf_fn(inputs_t)
                    localtriton_fn = ite.get_perf_test_func_localtriton()
                    logger.debug("tritonfusion begin pre perf triton kernel")
                    localtriton_perf_ms = localtriton_fn(inputs_t)
                    logger.debug(f"tritonfusion get eager time(ms): {eager_perf_ms}")
                    logger.debug(
                        f"tritonfusion get localtriton time(ms): {localtriton_perf_ms}"
                    )
                    assert (
                        eager_perf_ms >= localtriton_perf_ms
                    ), f"tritonfusion perf test failed: eager time(ms): {eager_perf_ms}  localtriton time(ms): {localtriton_perf_ms}"
                elif tt_config.pre_check_triton_kernel:
                    inputs_t = inp_fn()
                    local_triton_wraper_fn = ite.get_triton_wraper_func()
                    logger.debug("tritonfusion begin pre check triton kernel")
                    local_triton_wraper_fn(inputs_t)
                    logger.debug("tritonfusion finish pre check triton kernel")
        except Exception as e:
            logger.info(f"Generate triton kernel {graph_name} failed:\n{e}")
            traceback.print_exc()
            visited_subgraphs.append(list(fused_subgraph))
            total_failed_ops += len(fused_subgraph)
            ind += 1
            subgraphs = find_subgraphs(
                graph,
                min_nodes=tt_config.min_fused_nodes_num,
                max_nodes=tt_config.max_fused_nodes_num,
                visited_subgraphs=visited_subgraphs,
                graphnum=1,
            )
            continue

        prepare_fusion_insertion(fused_subgraph)
        total_fused_ops += len(fused_subgraph)

        input_nodes = ite.input_nodes
        output_nodes = ite.output_nodes

        # Use custom ops mode to save each tritonfusion kernel in/out tensors
        # to tritonfusion cache dir, this is only for debug.
        if tt_config.save_in_out_tensors:
            fn = getattr(
                getattr(torch.ops, TRITONFUSION_CUSTOM_OPS_NAME),
                ite.graphname_wrapper_custom,
            ).default
        else:  # Use inductor mode to get best perf.
            fn = ite.get_inductor_func()
            fn._inductor_lowering_function = True

        with graph.inserting_before(output_nodes[0]):
            new_node = graph.call_function(fn, tuple([input_nodes]))

            for ind2, output_node in enumerate(output_nodes):
                if len(output_nodes) == 1:
                    new_output = new_node
                else:
                    new_output = graph.call_function(
                        operator.getitem, args=(new_node, ind2)
                    )
                new_output.meta = copy.copy(output_node.meta)
                output_node.replace_all_uses_with(new_output)
                changed = True
            # Set fused subgraph name with all fused op names.
            new_node.meta["fused_op_kernel_name"] = ite.fused_kernel_name_ops
            new_node.meta["keep_transform_triton"] = ite

        for unused_node in sorted(fused_subgraph, reverse=True):
            graph.erase_node(unused_node)

        ind += 1
        succeeded_cnt += 1
        subgraphs = find_subgraphs(
            graph,
            min_nodes=tt_config.min_fused_nodes_num,
            max_nodes=tt_config.max_fused_nodes_num,
            visited_subgraphs=visited_subgraphs,
            graphnum=1,
        )

    logger.info(
        f"triton_fusion summary: success_fused_ops: {total_fused_ops}   success_fused_kernels: {succeeded_cnt}\
            failed_fused_kernels: {ind - succeeded_cnt}   failed_fused_ops: {total_failed_ops}"
    )

    return changed
