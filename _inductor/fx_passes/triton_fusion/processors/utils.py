"""
@Copyright (C) [2022-2025] by Cambricon.
@File    :   utils.py
"""

import torch, triton
import sys
from torch.fx.node import Node
import math
from ..common import get_target_name, TENSORMETANAME, MEMTYPE, get_tensor_metas
from torch._inductor.select_algorithm import ExternKernelChoice, ExternKernelCaller
from ..config import get_simple_logger


REGISTERED_PROCESSOR = {}
REGISTERED_EXTERNKERNELCHOICE = {}

logger = get_simple_logger(__name__)


class TritonFusionExternKernelChoice(ExternKernelChoice):
    def __init__(self, kernel, cpp_kernel=None, **kwargs) -> None:
        tt_fn = self.make_tritonfusion_callable(kernel)
        super().__init__(tt_fn, cpp_kernel, **kwargs)
        self.torch_fn = kernel

    def make_tritonfusion_callable(self, torch_fn):
        return lambda *args, **kwargs: torch_fn(**kwargs)

    def to_callable(self):
        fn = super().to_callable()
        return self.make_tritonfusion_callable(fn)

    def bind(
        tt_self,
        tt_input_nodes,
        tt_layout,
        tt_ordered_kwargs_for_cpp_kernel=(),
        **kwargs,
    ):
        tt_self.ordered_kwargs_for_cpp_kernel = tt_ordered_kwargs_for_cpp_kernel
        return ExternKernelCaller(
            tt_self,
            tt_input_nodes,
            tt_layout,
            kwargs,
            has_out_variant=tt_self.has_out_variant,
        )


SUPPORTED_MM_OPS = [
    "aten.mm.default",
    "aten.bmm.default",
    "aten.addmm.default",
]


def is_supported_operation(node: Node) -> bool:
    """
    Check if the operation name is supported by any registered processor.
    """
    return get_target_name(node) in REGISTERED_PROCESSOR


def convert_to_triton(Node: Node) -> list[str]:
    """
    Convert a PyTorch FX node to Triton code.
    This function will look up the registered processor for the node's operation name.
    """
    opname = get_target_name(Node)
    if opname in REGISTERED_PROCESSOR:
        return REGISTERED_PROCESSOR[opname].generate_triton(Node)
    raise NotImplementedError(f"No processor registered for operation: {opname}")


def get_externkernelchoice(node: Node | str) -> TritonFusionExternKernelChoice:
    """get TritonFusionExternKernelChoice for this node.
       if not supported, raise error

    Args:
        node (Node): input node
    """
    opname = node
    if isinstance(opname, Node):
        opname = get_target_name(opname)
    opname = opname.replace(".", "_")
    if opname in REGISTERED_EXTERNKERNELCHOICE:
        return REGISTERED_EXTERNKERNELCHOICE[opname]
    raise NotImplementedError(
        f"No TritonFusionExternKernelChoice registered for operation: {opname}"
    )


def get_memory_require(node: Node, tiledims_all: dict[Node:set]) -> list[int]:
    """

    Args:
        Node (Node): the node who's mem require will be returned.

    Returns:
        list[int]: the list of different ram type require in bytes.
    """
    opname = get_target_name(node)
    if opname in REGISTERED_PROCESSOR:
        res = REGISTERED_PROCESSOR[opname].get_require_mem(node, tiledims_all)
        logger.debug(
            f"get memory required result for operation: {opname} node: {node}\
            shape: {[x.shape for x in get_tensor_metas(node)]}\
            dtype: {[x.dtype for x in get_tensor_metas(node)]}\
            tiledims: {tiledims_all.get(node, None)}  result: {res}"
        )
        return res
    else:
        logger.debug(f"no processor registered for operation: {opname}, node: {node}")
    return {
        MEMTYPE.NRAM: sys.maxsize,
        MEMTYPE.WRAM: sys.maxsize,
        MEMTYPE.SM: sys.maxsize,
    }


def update_tiledims(
    nodes: list[Node],
    tiledim_all: dict,
    update_dims: dict,
    cur_node: Node = None,
):
    """
    Update the tile dimensions for the given nodes.
    Args:
    nodes: list of all subgraph nodes.
    tiledim_all: dict containing all tile dimensions for all nodes, which key is node.
        It's value is set of tiledims for inner nodes, but dict of {tile dims set: list of users nodes} for input nodes.
    update_dims: dist containing nodes and their tile dimensions to update.
    cur_node: the current node being processed for back infer input nodes.
    """
    updated_nodes = {}
    if update_dims is None:
        return None
    for node, tiledim in update_dims.items():
        if not tiledim:
            continue
        # has multi tile dim, only for input nodes which no in nodes.
        if node in tiledim_all:
            if tiledim_all[node] == tiledim:
                continue
            # TODO: support multi tile dim for input nodes.
            logger.debug(
                f"find duplicated tile dim for node: {node.format_node()} \
                with tile dim: {tiledim_all[node]} vs {tiledim} \
                which in nodes: {node in nodes}"
            )
            # if node in nodes which means this node is in the graph,
            # it should not have multi tile dim lists.
            return None
        else:
            tiledim_all[node] = tiledim
            updated_nodes[node] = tiledim
    # debug infos.
    logger.debug(f"updated tile dims once: ")
    for node, tiledim in updated_nodes.items():
        logger.debug(f"  {node} with tile dim: {tiledim}")
    return updated_nodes


def infer_tiledim_back_recursive(nodes: list[Node], node: Node, tiledim_all: dict):
    """
    Infer the tile dimension for the given nodes from result to inputs recursively.
    """
    updated_nodes = {}
    if node not in nodes:
        return updated_nodes
    tiledim = tiledim_all.get(node, None)
    if not tiledim:
        return updated_nodes
    opname = get_target_name(node)
    if opname in REGISTERED_PROCESSOR:
        res = REGISTERED_PROCESSOR[opname].infer_tiledim_back(node, tiledim_all)
        logger.debug(
            f"infer_tiledim_back_recursive inferred node: {str(node)} {opname} result: {res}"
        )
        if res is None:
            return res
        if (once_update := update_tiledims(nodes, tiledim_all, res, node)) is None:
            return None
        updated_nodes.update(once_update)

        for inpnode in once_update:
            res = infer_tiledim_back_recursive(nodes, inpnode, tiledim_all)
            if res is None:
                return res
            updated_nodes.update(res)
    else:
        raise NotImplementedError(f"No processor registered for operation: {opname}")

    return updated_nodes


def infer_tiledim_front(node: Node, tiledim_all: dict):
    """
    Infer the tile dimension for a single node from inputs to result.
    This function will look up the registered processor for the node's operation name.
    """
    opname = get_target_name(node)
    if opname in REGISTERED_PROCESSOR:
        node_metas = get_tensor_metas(node)
        logger.debug(
            f"begin infer_tiledim_front for node: {node.format_node()} with shapes: {[x.shape for x in node_metas]}  strides: {[x.stride for x in node_metas]}"
        )
        res = REGISTERED_PROCESSOR[opname].infer_tiledim_front(node, tiledim_all)
        logger.debug(f"infer_tiledim_front for {str(node)} end, get result: {res}")
        return res
    else:
        logger.debug(f"no processor registered for operation: {opname}, node: {node}")
    return None


def infer_tiledim_front_all(nodes: list[Node], tiledim_all: dict):
    """
    Infer the tile dimension for all nodes from input to result.
    """
    updatecnt = {}
    for node in nodes:
        res = infer_tiledim_front(node, tiledim_all)
        if res is None:
            return res
        if (once_update := update_tiledims(nodes, tiledim_all, res, node)) is None:
            return None
        updatecnt.update(once_update)
    return updatecnt


def infer_tiledim_back(node: Node, tiledim_all: dict):
    """
    Infer the tile dimension for a single node from result to inputs.
    This function will look up the registered processor for the node's operation name.
    """
    opname = get_target_name(node)
    if opname in REGISTERED_PROCESSOR:
        node_metas = get_tensor_metas(node)
        logger.debug(
            f"begin infer_tiledim_back for node: {node.format_node()} with shapes: {[x.shape for x in node_metas]}  strides: {[x.stride for x in node_metas]}"
        )
        res = REGISTERED_PROCESSOR[opname].infer_tiledim_back(node, tiledim_all)
        logger.debug(f"infer_tiledim_back for {str(node)} end, get result: {res}")
        return res
    else:
        logger.debug(f"no processor registered for operation: {opname}, node: {node}")
    return None


def infer_tiledim_back_all(nodes: list[Node], tiledim_all: dict):
    """
    Infer the tile dimension for all nodes from result to inputs.
    """
    updatecnt = {}
    for node in reversed(nodes):
        res = infer_tiledim_back(node, tiledim_all)
        if res is None:
            return res
        if (once_update := update_tiledims(nodes, tiledim_all, res, node)) is None:
            return None
        updatecnt.update(once_update)
    return updatecnt
