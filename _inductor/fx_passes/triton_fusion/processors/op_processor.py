"""
@Copyright (C) [2022-2025] by Cambricon.
@File    :   op_processor.py
"""

import sys
import torch
import operator
from torch.fx.node import Node
from ..common import (
    TORCH2TRITON_DTYPE_STR,
    BATCHBLOCKNAME,
    TILEDIMNAME,
    get_target_name,
    TORCH_FLOAT_TYPES,
    TENSORMETANAME,
    PROPAGATE_NAN,
    get_torch_dtype_bytes,
    align,
    MEMTYPE,
    ALLOCTYPE,
    check_loop_invariant,
    is_tensor_node,
    get_tensor_metas,
)
from torch_mlu._inductor import config
from .utils import (
    REGISTERED_PROCESSOR,
    REGISTERED_EXTERNKERNELCHOICE,
    TritonFusionExternKernelChoice,
    is_supported_operation,
)
from torch._inductor.select_algorithm import extern_kernels
from ..config import get_simple_logger

import math

logger = get_simple_logger(__name__)


def register_op_processor(cls):
    """
    Decorator to register a class as a processor.
    """
    if not issubclass(cls, TritonCodeConverterBase):
        raise TypeError("Processor must be a subclass of TritonCodeConverterBase")
    REGISTERED_PROCESSOR[cls.opname] = cls

    # Fallback process.
    try:
        torch_func_name = "torch.ops." + cls.opname
        if cls.opname == str(operator.getitem):
            torch_func = operator.getitem
        else:
            torch_func = eval(torch_func_name)
        extern_kernel_name = cls.opname.replace(".", "_")
        if callable(torch_func) and not hasattr(extern_kernels, extern_kernel_name):
            REGISTERED_EXTERNKERNELCHOICE[
                extern_kernel_name
            ] = TritonFusionExternKernelChoice(
                torch_func, name=extern_kernel_name, has_out_variant=False
            )
    except AttributeError:
        logger.debug(f"Get error torch func call name: {torch_func_name}")
    return cls


def get_op_processor(node: Node | str):
    if isinstance(node, Node):
        node = get_target_name(node)
    return REGISTERED_PROCESSOR.get(node, None)


class MemoryRequireBase:
    @classmethod
    def get_require_mem(cls, node: Node, tiledims_all: dict[Node:set]):
        """get nram/wram bytes required

        Args:
            node (Node): which node to compute ram require on.
            tiledims_all (dict): this node's tiledims set.

        Returns:
            list[int]: nram, wram, shared mem required in bytes.
        """
        ret = []
        tiledims = tiledims_all.get(node)
        for ind, meta in enumerate(get_tensor_metas(node)):
            out_shape = list(meta.shape)
            out_dtype = meta.dtype

            if tiledims:
                for dim in tiledims[ind]:
                    out_shape[dim] = 1
            nram = math.prod(out_shape) * get_torch_dtype_bytes(out_dtype)
            wram = 0
            sm = 0
            ret.append(
                {
                    MEMTYPE.NRAM: nram,
                    MEMTYPE.WRAM: wram,
                    MEMTYPE.SM: sm,
                    ALLOCTYPE: ALLOCTYPE.NORMAL,
                }
            )
        return ret


class TileDimInferBase:
    """
    Base class for tile dimension inference rules.
    """

    @classmethod
    def infer_tiledim_back(cls, node: Node, tiledim_all: dict):
        """
        Infer input tile dim from result
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement infer_tiledim_back method."
        )

    @classmethod
    def infer_tiledim_front(cls, node: Node, tiledim_all: dict):
        """
        Infer result tile dim from inputs
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement infer_tiledim_front method."
        )


class TileDimAlignRight(TileDimInferBase):
    """
    Infer tile dim as right align rule
    """

    @classmethod
    def infer_tiledim_back(cls, node: Node, tiledim_all: dict):
        """
        Infer input tile dim from result
        """
        updated_nodes = {}
        tiledims = tiledim_all.get(node, None)
        if tiledims is None:
            return updated_nodes
        metas = get_tensor_metas(node)
        for ind, meta in enumerate(metas):
            if not tiledims[ind]:
                continue
            node_shape = meta.shape
            for tiledim in tiledims[ind]:
                for inp in node.all_input_nodes:
                    if not is_tensor_node(inp):
                        continue
                    # Input must not be multi output op.
                    inp_shape = get_tensor_metas(inp)[0].shape
                    # align right, the input don't contain tile dim.
                    if len(inp_shape) + tiledim < len(node_shape):
                        continue

                    inp_tiledim = tiledim - (len(node_shape) - len(inp_shape))

                    # maybe broadcast, so check the shape.
                    if inp_shape[inp_tiledim] != node_shape[tiledim]:
                        logger.debug(
                            f"skipped dim infer: input shape {inp_shape} in tiledim: {inp_tiledim} != result shape {node_shape} in tiledim: {tiledim} for node: {node.format_node()}"
                        )
                        continue
                    if inp not in updated_nodes:
                        updated_nodes[inp] = [set()]
                    updated_nodes[inp][0].add(inp_tiledim)
                    logger.debug(
                        f"updated {str(inp)} with tile dim: {updated_nodes[inp]}"
                    )
            break
        return updated_nodes

    @classmethod
    def infer_tiledim_front(cls, node: Node, tiledim_all: dict):
        """
        Infer result tile dim from inputs
        """
        updated_nodes = {node: []}
        metas = get_tensor_metas(node)

        for inp in node.all_input_nodes:
            if not is_tensor_node(inp):
                continue
            inptiledims = tiledim_all.get(inp, None)
            if inptiledims is None:
                continue
            inp_shape = get_tensor_metas(inp)[0].shape
            for ind, meta in enumerate(metas):
                node_shape = meta.shape
                for inp_tiledim in inptiledims[0]:
                    tiledim_offset = len(inp_shape) - inp_tiledim
                    resdim = len(node_shape) - tiledim_offset
                    assert (
                        resdim >= 0
                    ), f"TileDimAlignRight get error dim: {resdim}, dimoffset: {tiledim_offset}  \
                        which shape is {node_shape}, op is: {node.format_node()}"
                    if len(updated_nodes[node]) <= ind:
                        updated_nodes[node].append(set())
                    updated_nodes[node][ind].add(resdim)
                    logger.debug(
                        f"updated {str(node)} with tile dim: {updated_nodes[node]}"
                    )
            break
        return updated_nodes


class TileDimReduce(TileDimInferBase):
    """
    Infer tile dim as reduce rule
    """

    @classmethod
    def get_dims(cls, node: Node) -> list[int]:
        """
        Get the dimensions to reduce over.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement get_dims method.")

    @classmethod
    def continue_if_tiledim_equal(cls, node: Node) -> bool:
        """
        If continue dim infer if the tile dim is in reduce dims and the input and result tile dim's shape is equal,
        this should be false for cumsum like op, which input and output's shape is equal but can't be tiled, but for
        slice op, this could be true.
        """
        return False

    @classmethod
    def infer_tiledim_back(cls, node: Node, tiledim_all: dict):
        updated_nodes = {}
        dims = cls.get_dims(node)
        metas = get_tensor_metas(node)
        # reduce all dims, the infer won't continue.
        node_shapes = [x.shape for x in metas]
        if dims is None:
            return None
        if isinstance(dims, int):
            dims = [dims]
        tiledims = tiledim_all.get(node, None)
        logger.debug(
            f"begin infer tiledim back for {str(node)} with dims: {dims}\
            tiledim: {tiledims} node_shapes: {node_shapes}"
        )
        if tiledims is None:
            return updated_nodes

        for ind, meta in enumerate(metas):
            if not tiledims[ind]:
                continue
            node_shape = node_shapes[ind]
            for tiledim in tiledims[ind]:
                for inp in node.all_input_nodes:
                    if not is_tensor_node(inp):
                        continue
                    inp_shape = get_tensor_metas(inp)[0].shape
                    inp_tiledim = tiledim_all.get(inp, None)
                    tepdims = [x % len(inp_shape) for x in dims]
                    logger.debug(
                        f"trying with input: {str(inp)} shape: {inp_shape}  tiledim: {inp_tiledim}  dims: {tepdims}"
                    )
                    # keep dim
                    if len(inp_shape) == len(node_shape):
                        if tiledim in tepdims:
                            if not (
                                cls.continue_if_tiledim_equal(node)
                                and inp_shape[tiledim] == node_shape[tiledim]
                            ):
                                logger.debug(
                                    f"failed: tile dim in reduce dims and reduce dim is not equal,\
                                    dims: {tepdims} inpushape:{inp_shape}  resultshape: {node_shape} \
                                    tiledim: {tiledim}"
                                )
                                return None

                        if inp not in updated_nodes:
                            updated_nodes[inp] = [set()]
                        updated_nodes[inp][0].add(tiledim)
                        logger.debug(
                            f"updated {str(inp)} with tile dim: {updated_nodes[inp]}"
                        )
                    else:
                        if len(tepdims) + len(node_shape) == len(inp_shape):
                            inputtiledim = tiledim
                            for d in sorted(tepdims):
                                if d <= inputtiledim:
                                    inputtiledim += 1

                            if inputtiledim >= len(inp_shape):
                                logger.debug(f"failed: all dim is reduced")
                                return None
                            if inp not in updated_nodes:
                                updated_nodes[inp] = [set()]
                            updated_nodes[inp][0].add(inputtiledim)
                            logger.debug(
                                f"updated {str(inp)} with tile dim: {updated_nodes[inp]}"
                            )
                        else:
                            AssertionError(
                                f"reduce infer dim error: {node.format_node()}  \
                                with shape: {node_shape} get dims {tepdims} but with input shape: {inp_shape}"
                            )
            break
        return updated_nodes

    @classmethod
    def infer_tiledim_front(cls, node: Node, tiledim_all: dict):
        updated_nodes = {node: []}
        dims = cls.get_dims(node)
        # reduce all dims, the infer won't continue.
        if dims is None:
            return None
        if isinstance(dims, int):
            dims = [dims]
        tiledims = tiledim_all.get(node, None)
        metas = get_tensor_metas(node)
        node_shapes = [x.shape for x in metas]
        logger.debug(
            f"begin infer tiledim front for {str(node)} with dims: {dims}  origin tiledims: {tiledims}  shape: {node_shapes}"
        )

        for inp in node.all_input_nodes:
            if not is_tensor_node(inp):
                continue
            inp_shape = get_tensor_metas(inp)[0].shape
            inptiledims = tiledim_all.get(inp, None)
            if inptiledims is None:
                continue
            inptiledims = inptiledims[0]
            logger.debug(
                f"trying with input: {str(inp)} shape: {inp_shape}  tiledim: {inptiledims}"
            )
            for ind, meta in enumerate(metas):
                node_shape = node_shapes[ind]
                if len(updated_nodes[node]) <= ind:
                    updated_nodes[node].append(set())
                for inp_tiledim in inptiledims:
                    if len(node_shape) == len(inp_shape):
                        if inp_tiledim in dims:
                            if not (
                                cls.continue_if_tiledim_equal(node)
                                and inp_shape[inp_tiledim] == node_shape[inp_tiledim]
                            ):
                                logger.debug(
                                    f"failed: {inp} tile dim {inp_tiledim} in reduce dims {dims} \
                                    and reduce dim is not equal: {inp_shape} vs {node_shape}"
                                )
                                return None
                        updated_nodes[node][ind].add(inp_tiledim)
                    elif len(dims) + len(node_shape) == len(inp_shape):
                        if inp_tiledim in dims:
                            logger.debug(
                                f"failed: {inp} tile dim {inp_tiledim} in reduce dims {dims} \
                                which shape is {inp_shape} vs {node_shape}"
                            )
                            return None
                        outdim = inp_tiledim - sum([1 for x in dims if x < inp_tiledim])
                        updated_nodes[node][ind].add(outdim)
            logger.debug(
                f"updated {str(node)} with tile dim: {updated_nodes[node]} from input: {str(inp)}"
            )
            break
        return updated_nodes


class TileDimPermute(TileDimInferBase):
    """
    Infer tile dim as permute rule
    """

    @classmethod
    def get_permute(cls, node: Node) -> list[int]:
        """
        Get the permutations.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement get_permute method."
        )

    @classmethod
    def infer_tiledim_back(cls, node: Node, tiledim_all: dict):
        permute = cls.get_permute(node)
        updated_nodes = {}
        tiledims = tiledim_all.get(node, None)
        if tiledims is None:
            return updated_nodes
        metas = get_tensor_metas(node)
        node_shapes = [x.shape for x in metas]
        logger.debug(
            f"begin infer tiledim back for {str(node)} with permute: {permute}  tiledim: {tiledims}  shape: {node_shapes}"
        )
        for ind, meta in enumerate(metas):
            if not tiledims[ind]:
                continue
            node_shape = node_shapes[ind]
            for tiledim in tiledims[ind]:
                assert tiledim < len(permute) and len(permute) == len(
                    node_shape
                ), f"permute infer dim error: {node.format_node()}  \
                    with shape: {node_shape} get tile dim: {tiledim} and permute {permute}"
                inp_tiledim = permute[tiledim]
                logger.debug(f"get tile dim for result: {inp_tiledim}")
                # if permute leads to expand shape, the expanded dim's permute should be -1
                if inp_tiledim < 0:
                    return None

                for inp in node.all_input_nodes:
                    if not is_tensor_node(inp):
                        continue
                    inp_shape = get_tensor_metas(inp)[0].shape

                    assert (
                        len(inp_shape) > inp_tiledim
                    ), f"permute infer dim error: input: {inp.format_node()} shape: {inp_shape} VS tile dim: {inp_tiledim}"
                    if inp not in updated_nodes:
                        updated_nodes[inp] = [set()]
                    updated_nodes[inp][0].add(inp_tiledim)
                    logger.debug(
                        f"updated {str(inp)} with tile dim: {updated_nodes[inp]}"
                    )
            break
        return updated_nodes

    @classmethod
    def infer_tiledim_front(cls, node: Node, tiledim_all: dict):
        permute = cls.get_permute(node)
        updated_nodes = {node: []}
        tiledims = tiledim_all.get(node, None)
        metas = get_tensor_metas(node)
        node_shapes = [x.shape for x in metas]
        logger.debug(
            f"begin infer tiledim front for {str(node)} with permute: {permute}  tiledim: {tiledims}  shape: {node_shapes}"
        )

        for inp in node.all_input_nodes:
            if not is_tensor_node(inp):
                continue
            inptiledimteps = tiledim_all.get(inp, None)
            if inptiledimteps is None:
                continue
            inptiledimteps = inptiledimteps[0]
            for ind, meta in enumerate(metas):
                if len(updated_nodes[node]) <= ind:
                    updated_nodes[node].append(set())
                for inp_tiledim in inptiledimteps:
                    if inp_tiledim not in permute:
                        return None
                    outtiledim = permute.index(inp_tiledim)
                    updated_nodes[node][ind].add(outtiledim)
                    logger.debug(
                        f"updated {str(node)} with tile dim: {updated_nodes[node]}"
                    )
            break

        return updated_nodes


class TileDimMatmul(TileDimInferBase):
    """
    Infer tile dim as permute rule
    """

    @classmethod
    def get_input(cls, node: Node) -> Node:
        """
        Get input1.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement get_input method."
        )

    @classmethod
    def get_filter(cls, node: Node) -> Node:
        """
        Get input2.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement get_filter method."
        )

    @classmethod
    def get_bias(cls, node: Node) -> Node | None:
        """
        Get the bias.
        """
        return None

    @classmethod
    def get_residual(cls, node: Node) -> Node | None:
        return None

    @classmethod
    def get_trans_a(cls, node: Node) -> bool:
        return False

    @classmethod
    def get_trans_b(cls, node: Node) -> bool:
        return False

    @classmethod
    def infer_tiledim_back(cls, node: Node, tiledim_all: dict):
        input1 = cls.get_input(node)
        input2 = cls.get_filter(node)
        bias = cls.get_bias(node)
        residual = cls.get_residual(node)
        trans_a = cls.get_trans_a(node)
        trans_b = cls.get_trans_b(node)

        updated_nodes = {}
        tiledims = tiledim_all.get(node, None)
        if tiledims is None:
            return updated_nodes
        metas = get_tensor_metas(node)
        node_shapes = [x.shape for x in metas]

        input1_shape = input1.meta[TENSORMETANAME].shape
        input2_shape = input2.meta[TENSORMETANAME].shape
        residual_shape = residual.meta[TENSORMETANAME].shape if residual else None
        bias_shape = bias.meta[TENSORMETANAME].shape if bias else None
        logger.debug(
            f"class: {cls.__name__}  begin infer tiledim back for {str(node)} with tiledim: {tiledims}  shape: {node_shapes} input1: {input1}\
                shape: {input1_shape}  input2: {input2} shape: {input2_shape}   bias: {bias} shape: {bias_shape}    residual: {residual}   \
                    shape: {residual_shape}    trans_a: {trans_a}   trans_b: {trans_b}"
        )
        for ind, meta in enumerate(metas):
            node_shape = node_shapes[ind]
            assert (
                len(node_shape) >= 2
            ), f"get wrong rank of {node}, which shape is: {node_shape}"
            assert (
                len(input1_shape) == len(input2_shape) == len(node_shape)
            ), f"get different rank for input1: {input1_shape}  input2: {input2_shape}   result: {node_shape}"
            if residual:
                assert len(residual_shape) == len(
                    node_shape
                ), f"get different rank for residual: {residual_shape}   result: {node_shape}"
            if bias:
                assert (
                    bias_shape[-1] == node_shape[-1]
                ), f"get different rank for bias: {bias_shape}   result: {node_shape}"
            for tiledim in tiledims[ind]:
                if tiledim >= len(node_shape) - 2:
                    # tile m
                    if tiledim == len(node_shape) - 2:
                        # input1 process
                        if input1 not in updated_nodes:
                            updated_nodes[input1] = [set()]
                        if not trans_a:
                            updated_nodes[input1][0].add(tiledim)
                        else:
                            updated_nodes[input1][0].add(tiledim + 1)
                        if bias and len(node_shape) == len(bias_shape):
                            if bias not in updated_nodes:
                                updated_nodes[bias] = [set()]
                            bias_tile_dim = len(bias_shape) - 2
                            if bias_shape[bias_tile_dim] > 1:
                                updated_nodes[bias][0].add(bias_tile_dim)
                    # tile n
                    else:
                        # input2 process
                        if input2 not in updated_nodes:
                            updated_nodes[input2] = [set()]

                        if not trans_b:
                            updated_nodes[input2][0].add(tiledim)
                        else:
                            updated_nodes[input2][0].add(tiledim - 1)
                        # tile n for bias
                        if bias:
                            if bias not in updated_nodes:
                                updated_nodes[bias] = [set()]
                            if bias_shape[-1] > 1:
                                updated_nodes[bias][0].add(len(bias_shape) - 1)
                else:
                    for inp in [input1, input2]:
                        if not inp:
                            continue
                        if inp not in updated_nodes:
                            updated_nodes[inp] = [set()]
                        updated_nodes[inp][0].add(tiledim)
                # residual tile
                if residual:
                    if residual not in updated_nodes:
                        updated_nodes[residual] = [set()]
                    updated_nodes[residual][0].add(tiledim)

        return updated_nodes

    @classmethod
    def infer_tiledim_front(cls, node: Node, tiledim_all: dict):
        input1 = cls.get_input(node)
        input2 = cls.get_filter(node)
        bias = cls.get_bias(node)
        residual = cls.get_residual(node)
        trans_a = cls.get_trans_a(node)
        trans_b = cls.get_trans_b(node)

        updated_nodes = {node: []}
        tiledims = tiledim_all.get(node, None)
        metas = get_tensor_metas(node)
        node_shapes = [x.shape for x in metas]
        input1_shape = input1.meta[TENSORMETANAME].shape
        input2_shape = input2.meta[TENSORMETANAME].shape
        residual_shape = residual.meta[TENSORMETANAME].shape if residual else None
        bias_shape = bias.meta[TENSORMETANAME].shape if bias else None

        logger.debug(
            f"class: {cls.__name__}  begin infer tiledim front for {str(node)} with tiledim: {tiledims}  shapes: {node_shapes} input1: {input1}\
                shape: {input1_shape}  input2: {input2} shape: {input2_shape}   bias: {bias} shape: {bias_shape}    residual: {residual}   \
                    shape: {residual_shape}    trans_a: {trans_a}   trans_b: {trans_b}"
        )
        for ind, meta in enumerate(metas):
            node_shape = node_shapes[ind]
            if len(updated_nodes[node]) <= ind:
                updated_nodes[node].append(set())
            assert (
                len(input1_shape) == len(input2_shape) == len(node_shape)
            ), f"get different rank for input1: {input1_shape}  input2: {input2_shape}   result: {node_shape}"
            if residual:
                assert len(residual_shape) == len(
                    node_shape
                ), f"get different rank for residual: {residual_shape}   result: {node_shape}"
            if bias:
                assert (
                    bias_shape[-1] == node_shape[-1]
                ), f"get different rank for bias: {bias_shape}   result: {node_shape}"

            # infer from input1
            input1_tiledims = tiledim_all.get(input1, None)
            input2_tiledims = tiledim_all.get(input2, None)
            residual_tiledims = tiledim_all.get(residual, None) if residual else None
            bias_tiledims = tiledim_all.get(bias, None) if bias else None
            if input1_tiledims:
                for input1_tiledim in input1_tiledims[0]:
                    if input1_tiledim >= len(input1_shape) - 2:
                        if (
                            not trans_a and input1_tiledim == len(input1_shape) - 2
                        ) or (trans_a and input1_tiledim == len(input1_shape) - 1):
                            updated_nodes[node][ind].add(len(node_shape) - 2)
                        else:
                            return None
                    else:
                        updated_nodes[node][ind].add(input1_tiledim)
            # infer from input2
            elif input2_tiledims:
                for input2_tiledim in input2_tiledims[0]:
                    if input2_tiledim >= len(input2_shape) - 2:
                        if (
                            not trans_b and input2_tiledim == len(input2_shape) - 1
                        ) or (trans_b and input2_tiledim == len(input2_shape) - 2):
                            updated_nodes[node][ind].add(len(node_shape) - 1)
                        else:
                            return None
                    else:
                        updated_nodes[node][ind].add(input2_tiledim)
            # infer from residual
            elif residual_tiledims:
                for residual_tiledim in residual_tiledims[0]:
                    updated_nodes[node][ind].add(residual_tiledim)
            elif bias_tiledims:
                for bias_tiledim in bias_tiledims[0]:
                    if bias_tiledim == len(bias_shape) - 1:
                        updated_nodes[node][ind].add(len(node_shape) - 1)
                    elif bias_shape == node_shape:
                        updated_nodes[node][ind].add(bias_tiledim)
                    else:
                        return None
        if updated_nodes[node] and any(updated_nodes[node]):
            logger.debug(f"updated {str(node)} with tile dim: {updated_nodes[node]}")
        else:
            logger.debug(f"not update {str(node)} with tile dim")
        return updated_nodes


class TritonCodeConverterBase:
    opname = "base"

    def __init__(self):
        pass

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        """
        Generate Triton code for the given node.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement generate_triton method."
        )

    @classmethod
    def get_output_num(cls, node: Node = None) -> int:
        return 1

    @classmethod
    def get_output_shape(cls, node: Node) -> list:
        node_metas = get_tensor_metas(node)
        assert (
            len(node_metas) == 1
        ), f"multi output nodes processor class: {cls.__name__} should rewrite get_output_shape func"
        return list(node_metas[0].shape)

    @classmethod
    def get_tiledims_from_dim(cls, node: Node, tile_dim: int):
        node_metas = get_tensor_metas(node)
        assert (
            len(node_metas) == 1
        ), f"multi output nodes processor class: {cls.__name__} should rewrite get_tiledims_from_dim func"
        return [set([tile_dim])]


class MultiOutputInterfaceBase:
    @classmethod
    def infer_sibling_and_self_nodes(cls, node: Node, tiledim_all: dict[Node:list]):
        updated_nodes = {}
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else 0
        inp_shape = list(get_tensor_metas(input_node)[0].shape)
        tiledims_node = tiledim_all.get(node, None)
        if tiledims_node:
            updated_nodes[node] = tiledims_node
            for ind, usr_node in enumerate(node.users):
                assert (
                    usr_node.target == operator.getitem
                ), f"{node}'s user should be getitem but get {usr_node}"
                index = usr_node.args[1]
                updated_nodes[usr_node] = [tiledims_node[index]]
        else:
            for ind, usr_node in enumerate(node.users):
                # First find one tiled user.
                tiledims = tiledim_all.get(usr_node, None)
                if not tiledims:
                    continue
                logger.debug(
                    f"begin infer sibling tiledims for {str(node)} from {usr_node}  tiledim: {tiledims}  input shape: {inp_shape}  unbind dim: {dim}"
                )
                # Add other getitem tiledims.
                for usr_node_other in node.users:
                    updated_nodes[usr_node_other] = tiledims.copy()

                ubind_tiles = []
                for meta in get_tensor_metas(node):
                    ubind_tiles.append(tiledims[0])
                updated_nodes[node] = ubind_tiles
                break
        return updated_nodes


class UnaryPointWiseProcessor(
    TritonCodeConverterBase, TileDimAlignRight, MemoryRequireBase
):
    opname = "unarypointwise"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for pointwise operations.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement get_opstr method."
        )

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        """
        Generate Triton code for pointwise operations.
        """
        assert (
            len([x for x in node.all_input_nodes if is_tensor_node(x)]) == 1
        ), "UnaryPointWise operations should have exactly one input node."

        code_lines = []
        code_lines.append(f"{str(node)} = {cls.get_opstr(node) % (str(node.args[0]))}")
        return code_lines, []


class BinaryPointWiseProcessor(
    TritonCodeConverterBase, TileDimAlignRight, MemoryRequireBase
):
    opname = "binary_pointwise"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for binary pointwise operations.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement get_opstr method."
        )

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        """
        Generate Triton code for binary pointwise operations.
        """
        assert (
            len(node.args) == 2
        ), "BinaryPointWise operations should have exactly two input nodes."

        code_lines = []
        code_lines.append(
            f"{str(node)} = {cls.get_opstr(node) % (str(node.args[0]), str(node.args[1]))}"
        )
        return code_lines, []


class ReduceProcessor(TritonCodeConverterBase, TileDimReduce, MemoryRequireBase):
    opname = "reduce"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for binary pointwise operations.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement get_opstr method."
        )

    @classmethod
    def get_dims(cls, node: Node) -> list[int]:
        """
        Get the dimensions to reduce over.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement get_dims method.")

    @classmethod
    def expand_bitwidth(cls, node: Node):
        return False

    @classmethod
    def post_process(cls, node: Node, code_lines: list[str], extra_lines: list[str]):
        return code_lines, extra_lines

    @classmethod
    def generate_triton(cls, node: Node) -> list[list[str], list[str]]:
        """
        Generate Triton code for reduce operations.
        This method should be overridden by subclasses.
        """
        assert (
            len(node.args) >= 1
        ), "Reduce operations should have at least one input node."

        code_lines = []
        reducefnname = f"{str(node)}_reduce_fn"
        extra_lines = [
            f"# Reduce operation: {str(node)}",
            f"@triton.jit",
            f"def {reducefnname}(a, b):",
            f"    return {cls.get_opstr(node) % ('a', 'b')}",
        ]
        input_node = node.args[0]
        dims = cls.get_dims(node)

        node_metas = get_tensor_metas(node)
        node_dtype = node_metas[0].dtype
        node_shape = list(node_metas[0].shape)

        input_metas = get_tensor_metas(input_node)
        input_dtype = input_metas[0].dtype
        input_shape = list(input_metas[0].shape)
        keepdim = len(input_shape) == len(node_shape)
        compute_dtype = node_dtype
        # some reduce need upper bitwidth
        if cls.expand_bitwidth(node):
            if input_dtype in [torch.float16, torch.bfloat16]:
                compute_dtype = torch.float32
            elif input_dtype in [torch.int8, torch.int16]:
                compute_dtype = torch.int16
            elif input_dtype in [torch.uint8, torch.uint16]:
                compute_dtype = torch.uint32

        if compute_dtype != input_dtype:
            code_lines.append(
                f"{str(input_node)} = {str(input_node)}.to({TORCH2TRITON_DTYPE_STR[str(compute_dtype)]})"
            )
        compute_dtype_triton = TORCH2TRITON_DTYPE_STR[str(compute_dtype)]
        keepdimstr = f", keep_dims = {keepdim}" if keepdim else ""
        if isinstance(dims, int):
            dims = [dims]
        # special change to avoid transpose
        tiledims = node.meta.get(TILEDIMNAME, [None])[0]
        inp_tiledims = input_node.meta.get(TILEDIMNAME, [None])[0]
        looptiledim = (
            (dims is not None)
            and (len(dims) == 1)
            and (dims[0] == 1 and dims[0] < len(input_shape) - 1)
            and (tiledims is not None and len(tiledims) == 1)
            and (inp_tiledims is not None and len(inp_tiledims) == 1)
            and (inp_tiledims[0] != dims[0])
        )

        # reduce process for cambricon genesis
        if looptiledim:
            tiledim = tiledims[0]
            inp_tiledim = inp_tiledims[0]
            node_shape[tiledim] = BATCHBLOCKNAME
            buffer_shape_str = ",".join([str(x) for x in node_shape])
            code_lines.append(
                f"{str(node)} = tl.empty(({buffer_shape_str}), dtype={compute_dtype_triton})"
            )
            code_lines.append(f"for bs_ind in tl.range({BATCHBLOCKNAME}):")
            inp_slicestr = [":"] * len(input_shape)
            inp_slicestr[inp_tiledim] = "bs_ind"
            inp_slicestr = ",".join(inp_slicestr)

            out_slicestr = [":"] * len(node_shape)
            out_slicestr[tiledim] = "bs_ind"
            out_slicestr = ",".join(out_slicestr)
            code_lines.append(
                f"    {str(node)}[{out_slicestr}] = tl.reduce({str(input_node)}[{inp_slicestr}], \
                axis = {dims[0] - int(dims[0] > inp_tiledim)}, combine_fn = {reducefnname}{keepdimstr})"
            )
        # common reduce process
        else:
            if dims is None or isinstance(dims, int):
                code_lines.append(
                    f"{str(node)} = tl.reduce({str(input_node)}, axis = {dims}, combine_fn = {reducefnname}{keepdimstr})"
                )
            else:
                code_lines.append(f"{str(node)} = {str(input_node)}")
                for ind, dim in enumerate(sorted(dims)):
                    code_lines.append(
                        f"{str(node)} = tl.reduce({str(node)}, axis = {dim - (0 if keepdim else ind)}, combine_fn = {reducefnname}{keepdimstr})"
                    )

        # post cvt and others
        if compute_dtype != node_dtype:
            code_lines.append(
                f"{str(node)} = {str(node)}.to({TORCH2TRITON_DTYPE_STR[str(node_dtype)]})"
            )
        code_lines, extra_lines = cls.post_process(node, code_lines, extra_lines)

        return code_lines, extra_lines


@register_op_processor
class PlaceHolderProcessor(
    TritonCodeConverterBase, TileDimInferBase, MemoryRequireBase
):
    opname = "placeholder"

    def __init__(self):
        pass

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        return [], []

    @classmethod
    def infer_tiledim_back(cls, node: Node, tiledim_all: dict):
        return None

    @classmethod
    def infer_tiledim_front(cls, node: Node, tiledim_all: dict):
        return None


@register_op_processor
class FullProcessor(TritonCodeConverterBase, TileDimInferBase, MemoryRequireBase):
    opname = "aten.full.default"

    def __init__(self):
        pass

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        shape, num = list(node.args[0]), node.args[1]
        torch_dtype = node.kwargs.get("dtype", torch.float32)
        if torch_dtype is torch.int64:
            torch_dtype = torch.int32
        triton_dtype = TORCH2TRITON_DTYPE_STR[str(torch_dtype)]
        # device = node.kwargs.get("device", torch.device("mlu")).type
        tiledim = list(node.meta.get(TILEDIMNAME, [None]))[0]
        if tiledim:
            shape[tiledim[0]] = BATCHBLOCKNAME
        code_lines = [
            f"{str(node)} = tl.full([{','.join([str(x) for x in shape])}], {num}, dtype={triton_dtype})"
        ]
        return code_lines, []

    @classmethod
    def infer_tiledim_back(cls, node: Node, tiledim_all: dict):
        return {}

    @classmethod
    def infer_tiledim_front(cls, node: Node, tiledim_all: dict):
        return {}


@register_op_processor
class RepeatProcessor(TritonCodeConverterBase, TileDimInferBase, MemoryRequireBase):
    opname = "aten.repeat.default"

    def __init__(self):
        pass

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        inp, repeat_num = node.args[0], list(node.args[1])
        inp_shape = list(inp.meta[TENSORMETANAME].shape)
        node_shape = list(node.meta[TENSORMETANAME].shape)
        tiledim = node.meta.get(TILEDIMNAME, [None])[0]

        assert len(repeat_num) == len(node_shape) and len(repeat_num) >= len(
            inp_shape
        ), f"{cls.__name__} get wrong args: repeat_num: {repeat_num}  inp_shape: {inp_shape}  node_shape: {node_shape}"
        st_ind = len(repeat_num) - len(inp_shape)
        if tiledim:
            for tiled in tiledim:
                assert (
                    tiled >= st_ind
                ), f"repeat gencode get error tiledim: {tiled} outshape: {node_shape}  inpshape: {inp_shape}"
                inp_shape[tiled - st_ind] = BATCHBLOCKNAME
                node_shape[tiled] = BATCHBLOCKNAME
        view_shape = []
        broadcast_shape = []
        for ind, rnum in enumerate(repeat_num):
            if ind < st_ind:
                view_shape.append(1)
                broadcast_shape.append(rnum)
            else:
                if rnum > 1:
                    view_shape.append(1)
                    broadcast_shape.append(rnum)
                if inp_shape[ind - st_ind] != 1:
                    in_sha = inp_shape[ind - st_ind]
                    view_shape.append(in_sha)
                    broadcast_shape.append(in_sha)
        view_shape_str = ",".join([str(x) for x in view_shape])
        broadcast_shape_str = ",".join([str(x) for x in broadcast_shape])
        node_shape_str = ",".join([str(x) for x in node_shape])
        code_lines = [
            f"# Generate code for repeat op: {node}",
            f"{str(node)} = tl.reshape({str(inp)}, [{view_shape_str}], can_reorder=True)",
            f"{str(node)} = tl.broadcast_to({str(node)}, [{broadcast_shape_str}])",
            f"{str(node)} = tl.reshape({str(node)}, [{node_shape_str}], can_reorder=True)",
        ]
        return code_lines, []

    @classmethod
    def infer_tiledim_back(cls, node: Node, tiledim_all: dict):
        updated_nodes = {}
        tiledims = tiledim_all.get(node, None)
        if tiledims is None:
            return updated_nodes
        tiledims = tiledims[0]
        inp, repeat_num = node.args[0], list(node.args[1])
        node_shape = node.meta[TENSORMETANAME].shape
        inp_shape = inp.meta[TENSORMETANAME].shape
        assert len(node_shape) >= len(
            inp_shape
        ), f"{node} infer_tiledim_back get wrong args: inpshape: {inp_shape}  outputshape: {node_shape}  repeat_num: {repeat_num}"

        logger.debug(
            f"begin infer tiledim back for {str(node)} with repeat_num: {repeat_num}  tiledim: {tiledims}  node_shape: {node_shape} inp_shape: {inp_shape}"
        )
        st_ind = len(node_shape) - len(inp_shape)
        for tiledim in tiledims:
            if tiledim < st_ind or node_shape[tiledim] != inp_shape[tiledim - st_ind]:
                logger.debug(
                    f"{node} backinfer failed: repeat num {repeat_num} at tile dim {tiledim} can't infer, shape: {node_shape} vs {inp_shape}"
                )
                return None
            if inp not in updated_nodes:
                updated_nodes[inp] = [set()]
            updated_nodes[inp][0].add(tiledim)
        logger.debug(
            f"infer back updated {str(inp)} with tile dim: {updated_nodes[inp]}"
        )
        return updated_nodes

    @classmethod
    def infer_tiledim_front(cls, node: Node, tiledim_all: dict):
        updated_nodes = {node: [set()]}
        inp, repeat_num = node.args[0], list(node.args[1])
        inptiledims = tiledim_all.get(inp, None)
        node_shape = node.meta[TENSORMETANAME].shape
        inp_shape = inp.meta[TENSORMETANAME].shape
        if inptiledims is None:
            return updated_nodes
        inptiledims = inptiledims[0]
        assert len(inp_shape) <= len(
            node_shape
        ), f"{node} infer_tiledim_front get wrong args: inpshape: {inp_shape}  outputshape: {node_shape}  repeat_num: {repeat_num}"
        logger.debug(
            f"begin infer tiledim front for {str(node)} with repeat_num: {repeat_num}  tiledim: {inptiledims}  node_shape: {node_shape} inp_shape: {inp_shape}"
        )
        st_ind = len(node_shape) - len(inp_shape)
        for inp_tiledim in inptiledims:
            tile_dim_out = inp_tiledim + st_ind
            if repeat_num[tile_dim_out] != 1:
                logger.debug(
                    f"failed: repeat num {repeat_num} at tile dim {tile_dim_out} is not 1"
                )
                return None
            updated_nodes[node][0].add(tile_dim_out)
        logger.debug(
            f"infer front updated {str(node)} with tile dim: {updated_nodes[node]}"
        )
        return updated_nodes


@register_op_processor
class GatherProcessor(TritonCodeConverterBase, TileDimInferBase, MemoryRequireBase):
    opname = "aten.gather.default"

    def __init__(self):
        pass

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        inp, dim, index = node.args
        inp_shape = inp.meta[TENSORMETANAME].shape
        dim %= len(inp_shape)
        code_lines = [
            f"# Gen code for gather op: {node}",
            f"{str(node)} = tl.gather({inp}, {index}, {dim})",
        ]
        return code_lines, []

    @classmethod
    def infer_tiledim_back(cls, node: Node, tiledim_all: dict):
        updated_nodes = {}
        tiledims = tiledim_all.get(node, None)
        if tiledims is None:
            return updated_nodes
        tiledims = tiledims[0]
        inp, dim, index = node.args
        inp_shape = inp.meta[TENSORMETANAME].shape
        logger.debug(
            f"{cls.__name__} begin infer tiledim back for {str(node)} tiledim: {tiledims}, with dim: {dim}  input: {inp}   index: {index}"
        )
        dim %= len(inp_shape)

        if dim in tiledims:
            return None
        updated_nodes[inp] = [tiledims.copy()]
        updated_nodes[index] = [tiledims.copy()]
        logger.debug(f"{cls.__name__} updated tiledims: {updated_nodes}")
        return updated_nodes

    @classmethod
    def infer_tiledim_front(cls, node: Node, tiledim_all: dict):
        updated_nodes = {node: [set()]}
        input, dim, index = node.args
        inp_tile = tiledim_all.get(input, None)
        inp_shape = input.meta[TENSORMETANAME].shape
        dim %= len(inp_shape)

        index_tile = tiledim_all.get(index, None)
        index_shape = index.meta[TENSORMETANAME].shape
        logger.debug(
            f"{cls.__name__} begin infer tiledim front for {str(node)} with dim: {dim}  input: {input} {inp_shape}  index: {index} {index_shape}"
        )
        if index_tile:
            for tile in index_tile[0]:
                if tile == dim:
                    return None
                updated_nodes[node][0].add(tile)
        elif inp_tile:
            for tile in inp_tile[0]:
                if tile == dim:
                    return None
                # Index shape[i] must <= input shape[i] for i != dim, if infer from input,
                # we should check shape is same.
                if inp_shape[tile] != index_shape[tile]:
                    return None
                updated_nodes[node][0].add(tile)
        logger.debug(f"{cls.__name__} updated tiledims: {updated_nodes}")
        return updated_nodes


@register_op_processor
class TanhProcessor(UnaryPointWiseProcessor):
    opname = "aten.tanh.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for tanh.
        """
        if config.use_ultra_tanh:
            return "tl.extra.mlu.libdevice.ultra_tanh(%s)"
        return "tl.extra.mlu.libdevice.fast_tanh(%s)"


@register_op_processor
class SigmoidProcessor(UnaryPointWiseProcessor):
    opname = "aten.sigmoid.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for sigmoid.
        """
        nodedtype = node.meta[TENSORMETANAME].dtype
        return f"tl.extra.mlu.libdevice.fast_sigmoid(%s.to(tl.float32)).to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]})"


@register_op_processor
class GeluProcessor(UnaryPointWiseProcessor):
    opname = "aten.gelu.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for gelu.
        gelu support f32.
        """
        nodedtype = node.meta[TENSORMETANAME].dtype
        if config.use_ultra_gelu:
            return f"tl.extra.mlu.libdevice.ultra_gelu(%s.to(tl.float32)).to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]})"
        return f"tl.extra.mlu.libdevice.fast_gelu(%s.to(tl.float32)).to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]})"


@register_op_processor
class SiluProcessor(UnaryPointWiseProcessor):
    opname = "aten.silu.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for silu.
        ultra_silu support f16/f32/bf16
        """
        nodedtype = node.meta[TENSORMETANAME].dtype
        if nodedtype in TORCH_FLOAT_TYPES:
            return f"tl.extra.mlu.libdevice.ultra_silu(%s)"
        return f"tl.extra.mlu.libdevice.ultra_silu(%s.to(tl.float32)).to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]})"


@register_op_processor
class ReluProcessor(UnaryPointWiseProcessor):
    opname = "aten.relu.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for relu.
        """
        nodedtype = node.meta[TENSORMETANAME].dtype
        if nodedtype in TORCH_FLOAT_TYPES:
            return f"tl.maximum(%s, c0.to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]}), propagate_nan={PROPAGATE_NAN})"
        return f"tl.maximum(%s, 0, propagate_nan={PROPAGATE_NAN})"


@register_op_processor
class CloneProcessor(UnaryPointWiseProcessor):
    opname = "aten.clone.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        return "%s"


@register_op_processor
class ExpandProcessor(UnaryPointWiseProcessor):
    opname = "aten.expand.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        node_shape = list(node.meta[TENSORMETANAME].shape)
        tiledims = node.meta.get(TILEDIMNAME, None)
        if tiledims:
            tiledim = list(tiledims[0])[0]
            node_shape[tiledim] = BATCHBLOCKNAME
        out_shape = ",".join([str(x) for x in node_shape])
        return f"%s.broadcast_to({out_shape})"


@register_op_processor
class PowTensorScalarProcessor(UnaryPointWiseProcessor):
    opname = "aten.pow.Tensor_Scalar"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for pow.
        """
        assert (
            len(node.args) == 2
        ), f"pow tensor scalar get {len(node.args)} inputs for node: {node.format_node()}"
        nodedtype = node.meta[TENSORMETANAME].dtype
        exponent = node.args[1]
        if exponent == 0.5:
            return f"tl.extra.mlu.libdevice.sqrt(%s)"
        return f"tl.extra.mlu.libdevice.ultra_pow(%s.to(tl.float32), {float(exponent)}).to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]})"


@register_op_processor
class SqrtProcessor(UnaryPointWiseProcessor):
    opname = "aten.sqrt.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for sqrt.
        """
        assert (
            len(node.args) == 1
        ), f"sqrt get {len(node.args)} inputs for node: {node.format_node()}"
        node_dtype = node.meta[TENSORMETANAME].dtype
        arg0 = node.args[0]
        is_float_t = (
            isinstance(arg0, Node) and arg0.meta[TENSORMETANAME].dtype.is_floating_point
        )
        if is_float_t:
            return f"tl.extra.mlu.libdevice.sqrt(%s)"
        return f"tl.extra.mlu.libdevice.sqrt(%s.to(tl.float32)).to({TORCH2TRITON_DTYPE_STR[str(node_dtype)]})"


@register_op_processor
class IsNanProcessor(UnaryPointWiseProcessor):
    opname = "aten.isnan.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for isnan.
        """
        assert (
            len(node.args) == 1
        ), f"isnan tensor get {len(node.args)} inputs for node: {node.format_node()}"
        return f"tl.extra.mlu.libdevice.isnan(%s)"


@register_op_processor
class NegProcessor(UnaryPointWiseProcessor):
    opname = "aten.neg.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for neg.
        """
        assert (
            len(node.args) == 1
        ), f"neg tensor get {len(node.args)} inputs for node: {node.format_node()}"
        return f"(-%s)"


@register_op_processor
class ReciprocalProcessor(UnaryPointWiseProcessor):
    opname = "aten.reciprocal.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for reciprocal.
        """
        assert (
            len(node.args) == 1
        ), f"reciprocal tensor get {len(node.args)} inputs for node: {node.format_node()}"
        nodedtype = node.meta[TENSORMETANAME].dtype
        return f"(c1.to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]}) / %s)"


@register_op_processor
class ExpProcessor(UnaryPointWiseProcessor):
    opname = "aten.exp.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for exp.
        """
        assert (
            len(node.args) == 1
        ), f"exp tensor get {len(node.args)} inputs for node: {node.format_node()}"
        nodedtype = node.meta[TENSORMETANAME].dtype
        return f"tl.extra.mlu.libdevice.fast_expf(%s.to(tl.float32)).to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]})"


@register_op_processor
class LogProcessor(UnaryPointWiseProcessor):
    opname = "aten.log.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for log.
        """
        assert (
            len(node.args) == 1
        ), f"log tensor get {len(node.args)} inputs for node: {node.format_node()}"
        nodedtype = node.meta[TENSORMETANAME].dtype
        return f"tl.extra.mlu.libdevice.fast_log(%s.to(tl.float32)).to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]})"


@register_op_processor
class EqualScalarProcessor(UnaryPointWiseProcessor):
    opname = "aten.eq.Scalar"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for equal scalar.
        """
        assert (
            len(node.args) == 2
        ), f"equal scalar get {len(node.args)} inputs for node: {node.format_node()}"
        scal = node.args[1]
        return f"(%s == {scal})"


@register_op_processor
class RSqrtProcessor(UnaryPointWiseProcessor):
    opname = "aten.rsqrt.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for rsqrt.
        """
        return f"tl.extra.mlu.libdevice.rsqrt(%s)"


@register_op_processor
class SignProcessor(UnaryPointWiseProcessor):
    opname = "aten.sign.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for sign.
        """
        return f"tl.extra.mlu.libdevice.sign(%s)"


@register_op_processor
class SumProcessor(ReduceProcessor):
    opname = "aten.sum.dim_IntList"

    def __init__(self):
        super().__init__()

    @classmethod
    def expand_bitwidth(cls, node: Node):
        return True

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        return "%s + %s"

    @classmethod
    def get_dims(cls, node: Node) -> list[int]:
        """
        Get the dimensions to reduce over for sum operations.
        """
        if len(node.args) > 1:
            dims = node.args[1]
        else:
            dims = node.kwargs.get("dim", None)
        if isinstance(dims, int):
            dims = [dims]
        if dims is None:
            return dims
        input_node = node.args[0]
        input_shape = list(input_node.meta[TENSORMETANAME].shape)
        dims = [x % len(input_shape) for x in dims]
        return list(dims)


@register_op_processor
class MeanProcessor(SumProcessor):
    opname = "aten.mean.dim"

    def __init__(self):
        super().__init__()

    @classmethod
    def expand_bitwidth(cls, node: Node):
        return True

    @classmethod
    def post_process(cls, node: Node, code_lines: list[str], extra_lines: list[str]):
        dims = cls.get_dims(node)
        input_node = node.args[0]
        inp_shape = input_node.meta[TENSORMETANAME].shape
        if dims is None:
            dims = list(range(len(inp_shape)))
        cnt = 1
        for dim in dims:
            cnt *= inp_shape[dim]
        if cnt != 1:
            code_lines.append(f"{str(node)} = {str(node)}/{float(cnt)}")
        return code_lines, extra_lines


@register_op_processor
class VarProcessor(ReduceProcessor):
    opname = "aten.var.correction"

    def __init__(self):
        super().__init__()

    @classmethod
    def expand_bitwidth(cls, node: Node):
        return True

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        return "%s + %s"

    @classmethod
    def get_dims(cls, node: Node) -> list[int]:
        """
        Get the dimensions to reduce over for var operations.
        """
        if len(node.args) > 1:
            dims = node.args[1]
        else:
            dims = node.kwargs.get("dim", None)
        if isinstance(dims, int):
            dims = [dims]
        if dims is None:
            return dims
        input_node = node.args[0]
        input_shape = list(get_tensor_metas(input_node)[0].shape)
        dims = [x % len(input_shape) for x in dims]
        return list(dims)

    @classmethod
    def get_correction(cls, node: Node) -> int:
        return node.kwargs.get("correction", 1)

    @classmethod
    def get_keepdim(cls, node: Node) -> bool:
        return node.kwargs.get("keepdim", False)

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        """
        Override generate triton code for var operations.
        """
        assert (
            len(node.args) >= 1
        ), "Var operations should have at least one input node."

        code_lines = []
        reducefnname = f"{str(node)}_reduce_fn"
        extra_lines = [
            f"# Reduce operation: {str(node)}",
            f"@triton.jit",
            f"def {reducefnname}(a, b):",
            f"    return {cls.get_opstr(node) % ('a', 'b')}",
        ]
        input_node = node.args[0]
        dims = cls.get_dims(node)
        correction = cls.get_correction(node)

        node_metas = get_tensor_metas(node)
        node_dtype = node_metas[0].dtype
        node_shape = list(node_metas[0].shape)

        input_metas = get_tensor_metas(input_node)
        input_dtype = input_metas[0].dtype
        input_shape = list(input_metas[0].shape)

        keepdim = cls.get_keepdim(node)
        compute_dtype = node_dtype
        # some reduce need upper bitwidth
        if cls.expand_bitwidth(node):
            if input_dtype in [torch.float16, torch.bfloat16]:
                compute_dtype = torch.float32
            elif input_dtype in [torch.int8, torch.int16]:
                compute_dtype = torch.int16
            elif input_dtype in [torch.uint8, torch.uint16]:
                compute_dtype = torch.uint32

        if compute_dtype != input_dtype:
            code_lines.append(
                f"{str(input_node)} = {str(input_node)}.to({TORCH2TRITON_DTYPE_STR[str(compute_dtype)]})"
            )

        code_lines.append("\n")
        code_lines.append(f"# generate code for aten.var.correction: {node}")
        # get numel.
        reduce_dims = dims
        if dims is None:
            reduce_dims = list(range(len(input_shape)))
        cnt = 1
        for dim in reduce_dims:
            cnt *= input_shape[dim]
        code_lines.append(f"{str(node)}_numel = {cnt}")

        # first get mean.
        if dims is None or isinstance(dims, int):
            # sum(xi)
            code_lines.append(
                f"{str(node)}_sum = tl.reduce({str(input_node)}, axis = {dims}, combine_fn = {reducefnname}, keep_dims = True)"
            )
        else:
            code_lines.append(f"{str(node)}_sum = {str(input_node)}")
            for ind, dim in enumerate(sorted(dims)):
                code_lines.append(
                    f"{str(node)}_sum = tl.reduce({str(node)}_sum, axis = {dim}, combine_fn = {reducefnname}, keep_dims = True)"
                )
        # inp_mean = sum(xi) / N
        code_lines.append(f"{str(node)}_mean = {str(node)}_sum / {str(node)}_numel")
        code_lines.append(
            f"{str(node)}_mean_diff = {str(input_node)} - {str(node)}_mean"
        )
        code_lines.append(
            f"{str(node)}_var = {str(node)}_mean_diff * {str(node)}_mean_diff"
        )
        # inp_var_mean = sum(inp * inp) / (N - correction)
        if dims is None or isinstance(dims, int):
            code_lines.append(
                f"{str(node)}_var_sum = tl.reduce({str(node)}_var, axis = {dims}, combine_fn = {reducefnname}, keep_dims = True)"
            )
        else:
            code_lines.append(f"{str(node)}_var_sum = {str(node)}_var")
            for ind, dim in enumerate(sorted(dims)):
                code_lines.append(
                    f"{str(node)}_var_sum = tl.reduce({str(node)}_var_sum, axis = {dim}, combine_fn = {reducefnname}, keep_dims = True)"
                )
        code_lines.append(
            f"{str(node)} = {str(node)}_var_sum / ({str(node)}_numel - {correction})"
        )

        # post cvt and others
        if compute_dtype != node_dtype:
            code_lines.append(
                f"{str(node)} = {str(node)}.to({TORCH2TRITON_DTYPE_STR[str(node_dtype)]})"
            )
        if not keepdim:
            out_shape_tiled = node_shape.copy()
            tiledims = node.meta.get(TILEDIMNAME, None)
            if tiledims:
                for tiled in tiledims[0]:
                    if out_shape_tiled[tiled] > 1:
                        out_shape_tiled[tiled] = BATCHBLOCKNAME
            out_shape = ",".join([str(x) for x in out_shape_tiled])
            code_lines.append(
                f"{str(node)} = tl.reshape({str(node)}, [{out_shape}], can_reorder=True)"
            )

        # Post process.
        code_lines, extra_lines = cls.post_process(node, code_lines, extra_lines)
        return code_lines, extra_lines


@register_op_processor
class MaxDimProcessor(ReduceProcessor, MultiOutputInterfaceBase):
    opname = "aten.max.dim"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_dims(cls, node: Node) -> list[int]:
        """
        Get the dimensions to reduce over for max operations.
        """
        if len(node.args) > 1:
            dims = node.args[1]
        else:
            dims = node.kwargs.get("dim", None)
        if isinstance(dims, int):
            dims = [dims]
        assert (
            dims is not None
        ), f"MaxDimProcessor get dim {dims} for node: {node.format_node()}"
        return list(dims)

    @classmethod
    def get_output_num(cls, node: Node) -> int:
        return 2

    @classmethod
    def get_output_shape(cls, node: Node) -> list:
        node_metas = get_tensor_metas(node)
        # All output's shape is same, return first.
        return list(node_metas[0].shape)

    @classmethod
    def get_tiledims_from_dim(cls, node: Node, tile_dim: int):
        node_metas = get_tensor_metas(node)
        return [set([tile_dim])] * len(node_metas)

    @classmethod
    def generate_triton(cls, node: Node) -> list[list[str], list[str]]:
        code_lines = []
        input_node = node.args[0]
        dims = cls.get_dims(node)

        node_metas = get_tensor_metas(node)
        node_shape = list(node_metas[0].shape)

        input_metas = get_tensor_metas(input_node)
        input_shape = list(input_metas[0].shape)

        keepdim = len(input_shape) == len(node_shape)

        index_user = False
        for user in node.users:
            assert (
                user.target == operator.getitem
            ), f"{node} get non getitem user: {user}"
            index = user.args[1]
            if index == 1 and len(user.users) > 0:
                index_user = True
                break
        ret_str = f"{node}_item_0, {node}_item_1" if index_user else f"{node}_item_0"
        code_lines.append(
            f"{ret_str} = tl.max({input_node}, {dims if dims is None else dims[0]}, return_indices={index_user}, keep_dims={keepdim})"
        )
        return code_lines, []


@register_op_processor
class MaxDefaultProcessor(ReduceProcessor):
    opname = "aten.max.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        nodedtype = node.meta[TENSORMETANAME].dtype
        return f"tl.maximum(%s, %s, propagate_nan={PROPAGATE_NAN}).to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]})"

    @classmethod
    def get_dims(cls, node: Node) -> list[int]:
        """
        Get the dimensions to reduce over for max operations.
        """
        return None


@register_op_processor
class MinDimProcessor(ReduceProcessor, MultiOutputInterfaceBase):
    opname = "aten.min.dim"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_dims(cls, node: Node) -> list[int]:
        """
        Get the dimensions to reduce over for min operations.
        """
        if len(node.args) > 1:
            dims = node.args[1]
        else:
            dims = node.kwargs.get("dim", None)
        if isinstance(dims, int):
            dims = [dims]
        assert (
            dims is not None
        ), f"MinDimProcessor get dim {dims} for node: {node.format_node()}"
        return list(dims)

    @classmethod
    def get_output_num(cls, node: Node) -> int:
        return 2

    @classmethod
    def get_output_shape(cls, node: Node) -> list:
        node_metas = get_tensor_metas(node)
        # All output's shape is same, return first.
        return list(node_metas[0].shape)

    @classmethod
    def get_tiledims_from_dim(cls, node: Node, tile_dim: int):
        node_metas = get_tensor_metas(node)
        return [set([tile_dim])] * len(node_metas)

    @classmethod
    def generate_triton(cls, node: Node) -> list[list[str], list[str]]:
        code_lines = []
        input_node = node.args[0]
        dims = cls.get_dims(node)

        node_metas = get_tensor_metas(node)
        node_shape = list(node_metas[0].shape)

        input_metas = get_tensor_metas(input_node)
        input_shape = list(input_metas[0].shape)

        keepdim = len(input_shape) == len(node_shape)

        index_user = False
        for user in node.users:
            assert (
                user.target == operator.getitem
            ), f"{node} get non getitem user: {user}"
            index = user.args[1]
            if index == 1 and len(user.users) > 0:
                index_user = True
                break
        ret_str = f"{node}_item_0, {node}_item_1" if index_user else f"{node}_item_0"
        code_lines.append(
            f"{ret_str} = tl.min({input_node}, {dims if dims is None else dims[0]}, return_indices={index_user}, keep_dims={keepdim})"
        )
        return code_lines, []


@register_op_processor
class MinDefaultProcessor(ReduceProcessor):
    opname = "aten.min.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        nodedtype = node.meta[TENSORMETANAME].dtype
        return f"tl.minimum(%s, %s, propagate_nan={PROPAGATE_NAN}).to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]})"

    @classmethod
    def get_dims(cls, node: Node) -> list[int]:
        """
        Get the dimensions to reduce over for min operations.
        """
        return None


@register_op_processor
class MultiplyProcessor(BinaryPointWiseProcessor):
    opname = "aten.mul.Tensor"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for multiplication.
        """
        return "%s * %s"


@register_op_processor
class AddProcessor(BinaryPointWiseProcessor):
    opname = "aten.add.Tensor"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for addition.
        """
        return "%s + %s"


@register_op_processor
class SubProcessor(BinaryPointWiseProcessor):
    opname = "aten.sub.Tensor"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for subtraction.
        """
        return "%s - %s"


@register_op_processor
class DivProcessor(BinaryPointWiseProcessor):
    opname = "aten.div.Tensor"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for division.
        """
        assert len(node.args) >= 2, "BinaryPointWise div should have at least 2 inputs."
        rval = node.args[1]
        if isinstance(rval, (int, float)):
            rval = 1.0 / rval
            return f"%s * {rval} # div %s change to reciprocal mul"
        lval = node.args[0]
        l_float = (
            isinstance(lval, Node) and lval.meta[TENSORMETANAME].dtype.is_floating_point
        )
        r_float = (
            isinstance(rval, Node) and rval.meta[TENSORMETANAME].dtype.is_floating_point
        )

        if config.use_fast_div and l_float and r_float:
            return "tl.extra.mlu.libdevice.fast_dividef(%s, %s)"
        return "%s / %s"


@register_op_processor
class LogicalAndProcessor(BinaryPointWiseProcessor):
    opname = "aten.logical_and.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for logical and.
        """
        return "(%s.to(tl.int1) and %s.to(tl.int1))"


@register_op_processor
class LogicalOrProcessor(BinaryPointWiseProcessor):
    opname = "aten.logical_or.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for logical or.
        """
        return "(%s.to(tl.int1) or %s.to(tl.int1))"


@register_op_processor
class LogicalNotProcessor(UnaryPointWiseProcessor):
    opname = "aten.logical_not.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for logical not.
        """
        return f"(not %s.to(tl.int1))"


@register_op_processor
class BitwiseAndProcessor(BinaryPointWiseProcessor):
    opname = "aten.bitwise_and.Tensor"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for bitwise and.
        """
        return "(%s & %s)"


@register_op_processor
class EqualTensorProcessor(BinaryPointWiseProcessor):
    opname = "aten.eq.Tensor"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for equal tensor.
        """
        return f"(%s == %s)"


@register_op_processor
class ConcatProcessor(TritonCodeConverterBase, TileDimReduce, MemoryRequireBase):
    opname = "aten.cat.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_dims(cls, node: Node) -> list[int]:
        if len(node.args) > 1:
            dims = node.args[1]
        else:
            dims = node.kwargs.get("dim", 0)
        if isinstance(dims, int):
            dims = [dims]
        if dims is None:
            dims = [0]
        node_shape = node.meta[TENSORMETANAME].shape
        dims = [dim % len(node_shape) for dim in dims]
        return list(dims)

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        """
        Generate Triton code for concatenation operations.
        """
        code_lines = []
        inputlist = node.args[0]
        dim = 0
        if len(node.args) > 1:
            dim = node.args[1]
        else:
            dim = node.kwargs.get("dim", dim)

        assert (
            len(inputlist) > 0
        ), "Concat operation should have at least one input node."
        if len(inputlist) == 1:
            code_lines.append(f"{str(node)} = {str(inputlist[0])}")
        else:
            code_lines.append(f"# Concat operation: {str(node)}")
            out_shape = node.meta[TENSORMETANAME].shape
            outdtype = node.meta[TENSORMETANAME].dtype
            tritonoutdtype = TORCH2TRITON_DTYPE_STR[str(outdtype)]
            tiledims = node.meta.get(TILEDIMNAME, None)

            dim = dim % len(out_shape)
            concatshape = [str(k) for k in out_shape]
            # Set tiledims to buffer.
            if tiledims:
                for tiledim in tiledims[0]:
                    if out_shape[tiledim] > 1:
                        concatshape[tiledim] = BATCHBLOCKNAME
            code_lines.append(
                f"{str(node)} = tl.empty(({','.join(concatshape)}), dtype={tritonoutdtype})"
            )
            offset = 0
            for inpnode in inputlist:
                inp_shape = inpnode.meta[TENSORMETANAME].shape
                slices = (
                    [":"] * dim
                    + [str(offset) + ":" + str(inp_shape[dim] + offset)]
                    + [":"] * (len(inp_shape) - dim - 1)
                )
                slices_str = ", ".join(slices)
                code_lines.append(
                    f"{str(node)}[{slices_str}] = {str(inpnode)}.to({tritonoutdtype})"
                )
                offset += inp_shape[dim]
            code_lines.append(f"")
        return code_lines, []


@register_op_processor
class WhereProcessor(TritonCodeConverterBase, TileDimAlignRight, MemoryRequireBase):
    opname = "aten.where.self"

    def __init__(self):
        super().__init__()

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        """
        Generate Triton code for where operations.
        """
        assert (
            len(node.args) == 3
        ), "Where operation should have exactly three input nodes."

        code_lines = []
        condition, true_value, false_value = node.args

        code_lines.append(
            f"{str(node)} = tl.where({str(condition)}.to(tl.int1), {str(true_value)}, {str(false_value)})"
        )

        return code_lines, []


@register_op_processor
class ConvertDtypeProcessor(
    TritonCodeConverterBase, TileDimAlignRight, MemoryRequireBase
):
    opname = "prims.convert_element_type.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        """
        Generate Triton code for type conversion operations.
        """
        assert (
            len(node.args) == 2
        ), "Convert operation should have exactly one input node."

        input_node = node.args[0]
        output_dtype = TORCH2TRITON_DTYPE_STR[str(get_tensor_metas(node)[0].dtype)]

        code_lines = []
        code_lines.append(f"{str(node)} = {str(input_node)}.to({output_dtype})")

        return code_lines, []


@register_op_processor
class ConvertToDtypeProcessor(ConvertDtypeProcessor):
    opname = "aten.to.dtype"

    def __init__(self):
        super().__init__()


@register_op_processor
class GtScalarProcessor(BinaryPointWiseProcessor):
    opname = "aten.gt.Scalar"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for greater than scalar.
        """
        return "%s > %s"


@register_op_processor
class GtTensorProcessor(GtScalarProcessor):
    opname = "aten.gt.Tensor"

    def __init__(self):
        super().__init__()


@register_op_processor
class LtScalarProcessor(BinaryPointWiseProcessor):
    opname = "aten.lt.Scalar"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for less than scalar.
        """
        return "%s < %s"


@register_op_processor
class LtTensorProcessor(LtScalarProcessor):
    opname = "aten.lt.Tensor"

    def __init__(self):
        super().__init__()


@register_op_processor
class GeScalarProcessor(BinaryPointWiseProcessor):
    opname = "aten.ge.Scalar"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for greater equal than scalar.
        """
        return "%s >= %s"


@register_op_processor
class GeTensorProcessor(GeScalarProcessor):
    opname = "aten.ge.Tensor"

    def __init__(self):
        super().__init__()


@register_op_processor
class LeScalarProcessor(BinaryPointWiseProcessor):
    opname = "aten.le.Scalar"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for less equal than scalar.
        """
        return "%s <= %s"


@register_op_processor
class LeTensorProcessor(LeScalarProcessor):
    opname = "aten.le.Tensor"

    def __init__(self):
        super().__init__()


@register_op_processor
class ClampMaxProcessor(BinaryPointWiseProcessor):
    opname = "aten.clamp_max.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for ClampMax.
        """
        nodedtype = node.meta[TENSORMETANAME].dtype
        return f"tl.minimum(%s, %s, propagate_nan={PROPAGATE_NAN}).to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]})"


@register_op_processor
class ClampMinProcessor(BinaryPointWiseProcessor):
    opname = "aten.clamp_min.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_opstr(cls, node: Node) -> str:
        """
        Get the operation string for ClampMax.
        """
        nodedtype = node.meta[TENSORMETANAME].dtype
        return f"tl.maximum(%s, %s, propagate_nan={PROPAGATE_NAN}).to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]})"


@register_op_processor
class PermuteProcessor(TritonCodeConverterBase, TileDimPermute, MemoryRequireBase):
    opname = "aten.permute.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_permute(cls, node: Node) -> list[int]:
        assert (
            len(node.args) == 2
        ), "Permute operation should have exactly one input node and one permutation argument."
        permute = node.args[1]
        return list(permute)

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        """
        Generate Triton code for permute operations.
        """
        assert (
            len(node.args) == 2
        ), "Permute operation should have exactly one input node and one permutation argument."

        input_node = node.args[0]
        order = node.args[1]

        code_lines = []
        code_lines.append(f"{str(node)} = tl.permute({str(input_node)}, {order})")

        return code_lines, []


@register_op_processor
class UnsqueezeProcessor(TritonCodeConverterBase, TileDimPermute, MemoryRequireBase):
    opname = "aten.unsqueeze.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_dims(cls, node: Node) -> list[int]:
        dims = node.kwargs.get("dim", None)
        if dims is None:
            dims = node.kwargs.get("dims", None)
        if len(node.args) >= 2:
            dims = node.args[1]
        if isinstance(dims, int):
            dims = [dims]
        assert dims is not None, "error: get None dims for unsqueeze/squeeze"
        return list(dims)

    @classmethod
    def get_permute(cls, node: Node) -> list[int]:
        assert (
            len(node.args) >= 1
        ), "Unsqueeze operation should have exactly one input node and one dimension argument."
        input_node = node.args[0]
        dims = cls.get_dims(node)
        assert len(dims) == 1, f"Unsqueeze operation get len(dims) != 1, dims: {dims}"
        inp_node_shape = input_node.meta[TENSORMETANAME].shape
        node_shape = node.meta[TENSORMETANAME].shape
        permute = list(range(len(inp_node_shape)))
        for dim in dims:
            dim = dim % len(node_shape)
            permute.insert(dim, -1)
        return permute

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        """
        Generate Triton code for unsqueeze operations.
        """
        assert (
            len(node.args) >= 1
        ), "Unsqueeze operation should have exactly one input node and one dimension argument."

        input_node = node.args[0]
        dims = cls.get_dims(node)
        assert len(dims) == 1, f"Unsqueeze operation get len(dims) != 1, dims: {dims}"
        dim = dims[0]
        out_shape = list(node.meta[TENSORMETANAME].shape)
        assert out_shape[dim] == 1, (
            "Unsqueeze operation should have a dimension of size 1 in dim %d." % dim
        )
        tiledims = node.meta.get(TILEDIMNAME, None)
        if tiledims:
            for tiled in tiledims[0]:
                if out_shape[tiled] > 1:
                    out_shape[tiled] = BATCHBLOCKNAME
        out_shape = ",".join([str(x) for x in out_shape])

        code_lines = []
        code_lines.append(
            f"{str(node)} = tl.reshape({str(input_node)}, [{out_shape}], can_reorder=True)"
        )
        return code_lines, []


@register_op_processor
class SqueezeDimsProcessor(UnsqueezeProcessor):
    opname = "aten.squeeze.dims"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_permute(cls, node: Node) -> list[int]:
        assert (
            len(node.args) >= 1
        ), "Squeeze operation should have at least one input node."
        input_node = node.args[0]
        dims = cls.get_dims(node)
        inp_node_shape = input_node.meta[TENSORMETANAME].shape
        permute = list(range(len(inp_node_shape)))
        for dim in dims:
            dim = dim % len(inp_node_shape)
            assert (
                dim in permute
            ), f"error: get dims: {dims} not in permute: {permute}, node: {node.format_node()}"
            if inp_node_shape[dim] == 1:
                permute.remove(dim)
        return permute

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        """
        Generate Triton code for squeeze operations.
        """
        assert (
            len(node.args) >= 1
        ), "Squeeze operation should have exactly one input node and one dimension argument."

        input_node = node.args[0]
        dims = cls.get_dims(node)

        out_shape = list(node.meta[TENSORMETANAME].shape)
        inp_shape = list(input_node.meta[TENSORMETANAME].shape)
        for dim in dims:
            assert inp_shape[dim] == 1, (
                "Squeeze operation input should have a dimension of size 1 in dim %d."
                % dim
            )
        tiledims = node.meta.get(TILEDIMNAME, None)
        if tiledims:
            for tiled in tiledims[0]:
                if out_shape[tiled] > 1:
                    out_shape[tiled] = BATCHBLOCKNAME
        out_shape = ",".join([str(x) for x in out_shape])

        code_lines = []
        code_lines.append(
            f"{str(node)} = tl.reshape({str(input_node)}, [{out_shape}], can_reorder=True)"
        )

        return code_lines, []


@register_op_processor
class SqueezeDimProcessor(SqueezeDimsProcessor):
    opname = "aten.squeeze.dim"

    def __init__(self):
        super().__init__()


@register_op_processor
class SelectProcessor(TritonCodeConverterBase, TileDimReduce, MemoryRequireBase):
    opname = "aten.select.int"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_dims(cls, node: Node) -> list[int]:
        assert (
            len(node.args) == 3
        ), "Select operation should have exactly one input node ,one dim and one index argument."
        dim = node.args[1]
        node_shape = node.meta[TENSORMETANAME].shape
        dim = dim % len(node_shape)
        return [dim]

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        """
        Generate Triton code for select operations.
        """
        assert (
            len(node.args) == 3
        ), "Select operation should have exactly one input node ,one dim and one index argument."
        code_lines = []
        input_node = node.args[0]
        dim = node.args[1]
        index = node.args[2]

        inshape = list(input_node.meta[TENSORMETANAME].shape)
        slicestr = [":"] * dim + [str(index)] + [":"] * (len(inshape) - dim - 1)
        slicestr = ",".join(slicestr)
        code_lines.append(f"{str(node)} = {str(input_node)}[{slicestr}]")
        return code_lines, []


@register_op_processor
class SliceProcessor(TritonCodeConverterBase, TileDimReduce, MemoryRequireBase):
    opname = "aten.slice.Tensor"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_dims(cls, node: Node) -> list[int]:
        assert (
            len(node.args) >= 4
        ), f"Slice operation should have exactly 1 input node ,1 dim and 2 or more slice argument, but get {node.args}."
        dim = node.args[1]
        node_shape = node.meta[TENSORMETANAME].shape
        dim = dim % len(node_shape)
        return [dim]

    @classmethod
    def continue_if_tiledim_equal(cls, node: Node) -> bool:
        return True

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        """
        Generate Triton code for slice operations.
        """
        assert (
            len(node.args) >= 4
        ), f"Slice operation should have exactly 1 input node ,1 dim and 2 or more slice argument, but get {node.args}."
        code_lines = []
        input_node = node.args[0]
        input_node_shape = input_node.meta[TENSORMETANAME].shape
        dim = cls.get_dims(node)[0]
        index_st = node.args[2]
        if index_st < 0:
            index_st += input_node_shape[dim]
        index_ed = min(node.args[3], input_node_shape[dim])
        if index_ed < 0:
            index_ed += input_node_shape[dim]
        assert (
            index_st < index_ed and index_ed >= 0 and index_st >= 0
        ), f"Slice node: {node} has error start/end indices, which args are: {node.args}"
        index_step = 1
        if len(node.args) > 4:
            index_step = node.args[4]

        if index_st == 0 and index_ed == input_node_shape[dim] and index_step == 1:
            code_lines.append(f"{str(node)} = {str(input_node)}")
        else:
            slice_dim_str = f"{index_st}:{index_ed}" + (
                f":{index_step}" if index_step != 1 else ""
            )
            slicestr = (
                [":"] * dim
                + [slice_dim_str]
                + [":"] * (len(input_node_shape) - dim - 1)
            )
            slicestr = ",".join(slicestr)
            code_lines.append(f"{str(node)} = {str(input_node)}[{slicestr}]")
        return code_lines, []


@register_op_processor
class TorchMatmulProcessorBase(
    TritonCodeConverterBase, TileDimMatmul, MemoryRequireBase
):
    """
    output = active(alpha * matmul(input, filter) + bias * beta + sigma * residual)
    """

    opname = "matmul.base.template"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_input(cls, node: Node) -> Node:
        """
        Get input.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement get_input method."
        )

    @classmethod
    def get_filter(cls, node: Node) -> Node:
        """
        Get input2.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement get_filter method."
        )

    @classmethod
    def get_bias(cls, node: Node) -> Node | None:
        """
        Get the bias.
        """
        return None

    @classmethod
    def get_residual(cls, node: Node) -> Node | None:
        return None

    @classmethod
    def get_act_mode(cls, node: Node) -> str | None:
        return None

    @classmethod
    def get_alpha(cls, node: Node) -> float:
        return 1

    @classmethod
    def get_beta(cls, node: Node) -> float:
        return 1

    @classmethod
    def get_sigma(cls, node: Node) -> float:
        return 0

    @classmethod
    def get_use_fast(cls, node: Node) -> bool:
        return True

    @classmethod
    def get_approximate(cls, node: Node) -> bool:
        return False

    @classmethod
    def get_a_scale(cls, node: Node) -> float | None:
        """
        If input is INT8/FLOAT8 type, the corresponding quantization scale, currently only support per-tensor quantization.
        If input is int dtype and a_scale is float type, then a_scale=input_max/dtype_max, which pass in the original value;
        in other cases, a_scale=dtype_max/input_max, which pass in the reciprocal value.
        """
        return 1.0

    @classmethod
    def get_b_scale(cls, node: Node) -> float | None:
        """
        Same as a_scale, but for input2.
        """
        return 1.0

    @classmethod
    def get_trans_a(cls, node: Node) -> bool:
        return False

    @classmethod
    def get_trans_b(cls, node: Node) -> bool:
        return False

    @classmethod
    def get_compute_dtype(cls, node: Node) -> torch.dtype:
        """
        Get the compute dtype for matmul.
        """
        return torch.float32

    @classmethod
    def get_require_mem(cls, node: Node, tiledims_all: dict[Node:list]):
        """get nram/wram bytes required

        Args:
            node (Node): which node to compute ram require on.
            tiledims (set): this node's tiledims set.

        Returns:
            list[int]: nram, wram, shared mem required in bytes.
        """
        ret = []
        out_shape = list(node.meta[TENSORMETANAME].shape)
        out_dtype = node.meta[TENSORMETANAME].dtype
        cmp_dtype = cls.get_compute_dtype(node)
        tiledims = tiledims_all.get(node)
        if tiledims:
            for dim in tiledims[0]:
                out_shape[dim] = 1

        nram = math.prod(out_shape) * max(
            get_torch_dtype_bytes(out_dtype), get_torch_dtype_bytes(cmp_dtype)
        )
        wram = 0
        sm = 0

        filter_node = cls.get_filter(node)
        filter_shape = list(filter_node.meta[TENSORMETANAME].shape)
        filter_dtype = filter_node.meta[TENSORMETANAME].dtype
        co_align = 64
        ci_align = 64 // get_torch_dtype_bytes(filter_dtype)
        filter_tiledim = tiledims_all.get(filter_node)
        wram_type = (
            ALLOCTYPE.INVARIANT
            if check_loop_invariant(filter_node, tiledims_all)
            else ALLOCTYPE.NORMAL
        )
        if filter_tiledim:
            for dim in filter_tiledim[0]:
                filter_shape[dim] = 1
        filter_shape[-1] = align(filter_shape[-1], co_align)
        filter_shape[-2] = align(filter_shape[-2], ci_align)
        wram += math.prod(filter_shape) * get_torch_dtype_bytes(filter_dtype)

        ret.append(
            {
                MEMTYPE.NRAM: 0,
                MEMTYPE.WRAM: wram,
                MEMTYPE.SM: 0,
                ALLOCTYPE: wram_type,
            }
        )
        ret.append(
            {
                MEMTYPE.NRAM: nram,
                MEMTYPE.WRAM: 0,
                MEMTYPE.SM: sm,
                ALLOCTYPE: ALLOCTYPE.NORMAL,
            }
        )
        return ret

    @classmethod
    def gen_compute_code(
        cls,
        input1_node: Node,
        input2_node: Node,
        outnode: Node,
        compute_dtype: torch.dtype,
    ):
        code_lines = []
        code_lines.append(
            f"{str(outnode)} = tl.dot({str(input1_node)}, {str(input2_node)}, out_dtype={TORCH2TRITON_DTYPE_STR[str(compute_dtype)]}, allow_tf32=False)"
        )
        return code_lines

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        input1 = cls.get_input(node)
        input2 = cls.get_filter(node)
        bias = cls.get_bias(node)
        residual = cls.get_residual(node)
        trans_a = cls.get_trans_a(node)
        trans_b = cls.get_trans_b(node)
        alpha = cls.get_alpha(node)
        beta = cls.get_beta(node)
        sigma = cls.get_sigma(node)
        act_mode = cls.get_act_mode(node)

        node_shape = node.meta[TENSORMETANAME].shape
        nodedtype = node.meta[TENSORMETANAME].dtype
        input1_shape = input1.meta[TENSORMETANAME].shape
        input2_shape = input2.meta[TENSORMETANAME].shape

        compute_dtype = cls.get_compute_dtype(node)

        assert (
            len(node_shape) >= 2
        ), f"{cls.__name__} get wrong rank of result: {node}, which shape is: {node_shape}"

        code_lines = []
        code_lines.append(f"# generate code for {cls.opname}")
        if trans_a:
            perm = list(range(len(input1_shape)))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            code_lines.append(f"{str(input1)} = tl.trans({str(input1)}, {perm})")
        if trans_b:
            perm = list(range(len(input2_shape)))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            code_lines.append(f"{str(input2)} = tl.trans({str(input2)}, {perm})")
        # Generate compute code.
        code_lines.extend(cls.gen_compute_code(input1, input2, node, compute_dtype))

        if alpha != beta:
            # alpha scale
            if alpha != 1:
                code_lines.append(f"{str(node)} = {str(node)} * {alpha}")
            # bias
            if bias and beta != 0:
                if beta != 1:
                    code_lines.append(f"{str(bias)} = {str(bias)} * {beta}")
                code_lines.append(f"{str(node)} = {str(node)} + {str(bias)}")
        else:
            if bias and beta != 0:
                code_lines.append(f"{str(node)} = {str(node)} + {str(bias)}")
            if beta != 1:
                code_lines.append(f"{str(node)} = {str(node)} * {beta}")

        # residual
        if residual and sigma != 0:
            if sigma != 1:
                code_lines.append(f"{str(residual)} = {str(residual)} * {sigma}")
            code_lines.append(f"{str(node)} = {str(node)} + {str(residual)}")
        # active
        if act_mode:
            if act_mode == "relu":
                if nodedtype in TORCH_FLOAT_TYPES:
                    code_lines.append(
                        f"{str(node)} = tl.maximum({str(node)}, c0.to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]}), propagate_nan={PROPAGATE_NAN})"
                    )
                else:
                    code_lines.append(
                        f"{str(node)} = tl.maximum({str(node)}, 0, propagate_nan={PROPAGATE_NAN})"
                    )
            elif act_mode == "gelu":
                # gelu need compute in float32.
                code_lines.append(f"{str(node)} = {str(node)}.to(tl.float32)")
                compute_dtype = torch.float32
                if config.use_ultra_gelu:
                    code_lines.append(
                        f"{str(node)} = tl.extra.mlu.libdevice.ultra_gelu({str(node)})"
                    )
                else:
                    code_lines.append(
                        f"{str(node)} = tl.extra.mlu.libdevice.fast_gelu({str(node)})"
                    )
            elif act_mode == "silu":
                code_lines.append(
                    f"{str(node)} = tl.extra.mlu.libdevice.ultra_silu({str(node)})"
                )
        # cvt to target dtype
        code_lines.append(
            f"{str(node)} = {str(node)}.to({TORCH2TRITON_DTYPE_STR[str(nodedtype)]})"
        )

        return code_lines, []


@register_op_processor
class TorchBatchMatmulProcessor(TorchMatmulProcessorBase):
    """
    output = active(alpha * batch_matmul(input, filter) + bias * beta + sigma * residual)
    """

    opname = "aten.bmm.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_input(cls, node: Node) -> Node:
        """
        Get input.
        """
        return node.args[0]

    @classmethod
    def get_filter(cls, node: Node) -> Node:
        """
        Get input2.
        """
        return node.args[1]


@register_op_processor
class TorchAddMatmulProcessor(TorchMatmulProcessorBase):
    """
    output = alpha * matmul(input, filter) + bias * beta
    """

    opname = "aten.addmm.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_input(cls, node: Node) -> Node:
        """
        Get input.
        """
        return node.args[1]

    @classmethod
    def get_filter(cls, node: Node) -> Node:
        """
        Get input2.
        """
        return node.args[2]

    @classmethod
    def get_bias(cls, node: Node) -> Node | None:
        return node.args[0]

    @classmethod
    def get_alpha(cls, node: Node) -> float:
        return node.kwargs.get("alpha", 1)

    @classmethod
    def get_beta(cls, node: Node) -> float:
        return node.kwargs.get("beta", 1)

    @classmethod
    def get_trans_a(cls, node: Node) -> bool:
        return False

    @classmethod
    def get_trans_b(cls, node: Node) -> bool:
        return False

    @classmethod
    def get_compute_dtype(cls, node: Node) -> torch.dtype:
        """
        Get the compute dtype for matmul.
        """
        if len(node.args) > 3:
            return node.args[3]
        return torch.float32

    @classmethod
    def gen_compute_code(
        cls,
        input1_node: Node,
        input2_node: Node,
        outnode: Node,
        compute_dtype: torch.dtype,
    ):
        node_dtype = outnode.meta[TENSORMETANAME].dtype
        code_lines = []
        code_lines.append(
            f"{str(outnode)} = tl.dot({str(input1_node)}, {str(input2_node)}, out_dtype={TORCH2TRITON_DTYPE_STR[str(compute_dtype)]}, allow_tf32=False)"
        )
        # code_lines.append(
        #     f"{str(outnode)} = {str(outnode)}.to({TORCH2TRITON_DTYPE_STR[str(node_dtype)]})"
        # )
        return code_lines


@register_op_processor
class TorchMMProcessor(TorchMatmulProcessorBase):
    """
    output = matmul(input, filter)
    """

    opname = "aten.mm.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_input(cls, node: Node) -> Node:
        """
        Get input.
        """
        return node.args[0]

    @classmethod
    def get_filter(cls, node: Node) -> Node:
        """
        Get weight.
        """
        return node.args[1]


@register_op_processor
class TorchMatmulProcessor(TorchMatmulProcessorBase):
    """
    output = matmul(input, filter)
    """

    opname = "aten.matmul.default"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_input(cls, node: Node) -> Node:
        """
        Get input.
        """
        return node.args[0]

    @classmethod
    def get_filter(cls, node: Node) -> Node:
        """
        Get weight.
        """
        return node.args[1]


@register_op_processor
class ReshapeProcessor(TritonCodeConverterBase, TileDimInferBase, MemoryRequireBase):
    opname = "aten.reshape.default"

    def __init__(self):
        pass

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        input_node = node.args[0]
        out_shape = list(node.meta[TENSORMETANAME].shape)
        inp_shape = list(input_node.meta[TENSORMETANAME].shape)
        assert math.prod(out_shape) == math.prod(
            inp_shape
        ), f"ReshapeProcessor get wrong shape for node: {node}, input shape: {inp_shape}, output shape: {out_shape}"
        tiledims = node.meta.get(TILEDIMNAME, None)
        if tiledims:
            for tiled in tiledims[0]:
                if out_shape[tiled] > 1:
                    out_shape[tiled] = BATCHBLOCKNAME
        out_shape = ",".join([str(x) for x in out_shape])
        code_lines = []
        code_lines.append(
            f"{str(node)} = tl.reshape({str(input_node)}, [{out_shape}], can_reorder=True)"
        )
        return code_lines, []

    @classmethod
    def infer_tiledim_back(cls, node: Node, tiledim_all: dict):
        updated_nodes = {}
        input_node = node.args[0]
        inp_shape = list(input_node.meta[TENSORMETANAME].shape)

        node_shape = list(node.meta[TENSORMETANAME].shape)
        assert math.prod(node_shape) == math.prod(
            inp_shape
        ), f"Reshape infer_tiledim_back get wrong shape for node: {node}, input shape: {inp_shape}, output shape: {node_shape}"
        tiledims = tiledim_all.get(node, None)
        if tiledims is None:
            return updated_nodes
        kep_tiledims = set()
        for tiledim in tiledims[0]:
            target_num = math.prod(node_shape[:tiledim])
            cur_num = 1
            inp_ind = -1
            for i in range(len(inp_shape)):
                dim = inp_shape[i]
                if (
                    cur_num == target_num
                    and dim == node_shape[tiledim]
                    and (i not in kep_tiledims)
                ):
                    inp_ind = i
                    break
                elif cur_num > target_num:
                    break
                cur_num *= dim
            # check if find a valid input tile dim for tiledim.
            if inp_ind < 0 or inp_ind >= len(inp_shape):
                logger.debug(
                    f"Reshape infer_tiledim_back failed to find input tiledim for node: {node} with shape {node_shape}, get tiledim: {tiledims}, input node: {input_node} with shape: {inp_shape}"
                )
                return None
            kep_tiledims.add(inp_ind)
        updated_nodes[input_node] = [kep_tiledims]
        logger.debug(
            f"Reshape infer_tiledim_back infer from node: {node} shape: {node_shape} tiledim: {tiledims}, update input node: {input_node} shape: {inp_shape} with tiledim: {kep_tiledims}"
        )
        return updated_nodes

    @classmethod
    def infer_tiledim_front(cls, node: Node, tiledim_all: dict):
        updated_nodes = {node: [set()]}
        input_node = node.args[0]
        inp_shape = list(input_node.meta[TENSORMETANAME].shape)
        inptiledims = tiledim_all.get(input_node, None)
        tiledims = tiledim_all.get(node, None)
        node_shape = node.meta[TENSORMETANAME].shape
        logger.debug(
            f"Reshape infer_tiledim_front get node: {node}, with shape: {node_shape}, get tiledim: {tiledims}, input node: {input_node} with shape: {inp_shape}, get input tiledim: {inptiledims}"
        )
        assert math.prod(node_shape) == math.prod(
            inp_shape
        ), f"Reshape infer_tiledim_front get wrong shape for node: {node}, input shape: {inp_shape}, output shape: {node_shape}"

        if inptiledims is None:
            return updated_nodes

        for inptiledim in inptiledims[0]:
            target_num = math.prod(inp_shape[:inptiledim])
            cur_num = 1
            out_ind = -1
            for i in range(len(node_shape)):
                dim = node_shape[i]
                if (
                    cur_num == target_num
                    and dim == inp_shape[inptiledim]
                    and (i not in updated_nodes[node])
                ):
                    out_ind = i
                    break
                elif cur_num > target_num:
                    break
                cur_num *= dim
            # check if find a valid output tile dim for inptiledim.
            if out_ind < 0 or out_ind >= len(node_shape):
                logger.debug(
                    f"Reshape infer_tiledim_front failed to find output tiledim for node: {node} with shape {node_shape}, get tiledim: {tiledims}, input node: {input_node} with shape: {inp_shape}, get input tiledim: {inptiledims}"
                )
                return None
            updated_nodes[node][0].add(out_ind)
        logger.debug(
            f"Reshape infer_tiledim_front from input node: {input_node} shape: {inp_shape} tiledim: {inptiledims}, updated node: {node} shape: {node_shape} tiledim: {updated_nodes[node]}"
        )
        return updated_nodes


@register_op_processor
class GetitemProcessor(TritonCodeConverterBase, TileDimInferBase, MemoryRequireBase):
    opname = "<built-in function getitem>"

    def __init__(self):
        super().__init__()

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        input_node, index = node.args[0], node.args[1]
        code_lines = []
        code_lines.append(f"{str(node)} = {input_node}_item_{index}")
        return code_lines, []

    @classmethod
    def infer_tiledim_back(cls, node: Node, tiledim_all: dict):
        input_node, index = node.args[0], node.args[1]
        if not is_supported_operation(input_node):
            return None
        inp_processor = get_op_processor(input_node)
        assert issubclass(
            inp_processor, MultiOutputInterfaceBase
        ), f"getitem op: {node}'s input op {input_node}'s processor should has base class MultiOutputInterfaceBase, which is {inp_processor.__name__}"
        sibling_nodes = inp_processor.infer_sibling_and_self_nodes(
            input_node, tiledim_all
        )
        input_node = inp_processor.infer_tiledim_back(
            input_node, tiledim_all | sibling_nodes
        )
        return sibling_nodes | input_node

    @classmethod
    def infer_tiledim_front(cls, node: Node, tiledim_all: dict):
        input_node, index = node.args[0], node.args[1]
        updated_nodes = {}
        inp_tiledims = tiledim_all.get(input_node, None)
        if inp_tiledims is None:
            return updated_nodes
        assert isinstance(
            inp_tiledims, (tuple, list)
        ), f"Getitem require input node's tiledim to be list, but get: {type(inp_tiledims)} {inp_tiledims}"
        updated_nodes[node] = [inp_tiledims[index]]
        return updated_nodes

    @classmethod
    def get_require_mem(cls, node: Node, tiledims_all: dict[Node:set]):
        nram = 0
        wram = 0
        sm = 0
        return [
            {
                MEMTYPE.NRAM: nram,
                MEMTYPE.WRAM: wram,
                MEMTYPE.SM: sm,
                ALLOCTYPE: ALLOCTYPE.NORMAL,
            }
        ]


@register_op_processor
class UnbindProcessor(
    TritonCodeConverterBase,
    TileDimInferBase,
    MemoryRequireBase,
    MultiOutputInterfaceBase,
):
    opname = "aten.unbind.int"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_output_num(cls, node: Node) -> int:
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else 0
        inp_shape = list(input_node.meta[TENSORMETANAME].shape)
        return inp_shape[dim]

    @classmethod
    def get_output_shape(cls, node: Node) -> list:
        node_metas = get_tensor_metas(node)
        # All output's shape is same, return first.
        return list(node_metas[0].shape)

    @classmethod
    def get_tiledims_from_dim(cls, node: Node, tile_dim: int):
        node_metas = get_tensor_metas(node)
        return [set([tile_dim])] * len(node_metas)

    @classmethod
    def generate_triton(cls, node: Node) -> list[str]:
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else 0
        inp_shape = list(input_node.meta[TENSORMETANAME].shape)
        dim_len = inp_shape[dim]
        slice_list = [":"] * dim + [0] + [":"] * (len(inp_shape) - dim - 1)

        code_lines = []
        for ind in range(dim_len):
            slice_list[dim] = str(ind)
            slice_str = ", ".join(slice_list)
            code_lines.append(f"{str(node)}_item_{ind} = {input_node}[{slice_str}]")
        return code_lines, []

    @classmethod
    def infer_tiledim_back(cls, node: Node, tiledim_all: dict):
        updated_nodes = {}
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else 0
        inp_shape = list(get_tensor_metas(input_node)[0].shape)

        tiledims_list = tiledim_all.get(node, None)
        if tiledims_list is None:
            return updated_nodes
        logger.debug(
            f"begin infer back tiledims for {str(node)}  tiledim: {tiledims_list}  input shape: {inp_shape}  unbind dim: {dim}"
        )
        assert (
            isinstance(tiledims_list, (list, tuple)) and len(tiledims_list) > 0
        ), f"multioutput op's tiledim should be list, get {tiledims_list} {type(tiledims_list)}"
        # Process unbind tiledims, use first output's tiledims.
        tiledims = tiledims_list[0]
        # Add input node tile dims.
        for tiledim in tiledims:
            in_tiledim = tiledim + int(tiledim >= dim)
            if input_node not in updated_nodes:
                updated_nodes[input_node] = [set()]
            updated_nodes[input_node][0].add(in_tiledim)
        return updated_nodes

    @classmethod
    def infer_tiledim_front(cls, node: Node, tiledim_all: dict):
        updated_nodes = {node: []}
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else 0
        inp_shape = list(input_node.meta[TENSORMETANAME].shape)
        inp_tiledims = tiledim_all.get(input_node, None)
        logger.debug(
            f"begin infer tiledim back for {str(node)} with input tiledim: {inp_tiledims}  input shape: {inp_shape}"
        )
        if not inp_tiledims:
            return None
        for inp_tiledim in inp_tiledims[0]:
            if inp_tiledim == dim:
                return None
            for usr_node in node.users:
                if usr_node not in updated_nodes:
                    updated_nodes[usr_node] = [set()]
                updated_nodes[usr_node][0].add(inp_tiledim - int(inp_tiledim > dim))
        for usr_node in node.users:
            updated_nodes[node].append(updated_nodes[usr_node][0])
        return updated_nodes


@register_op_processor
class VarMeanProcessor(
    VarProcessor,
    MultiOutputInterfaceBase,
):
    opname = "aten.var_mean.correction"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_output_num(cls, node: Node) -> int:
        return 2

    @classmethod
    def get_output_shape(cls, node: Node) -> list:
        node_metas = get_tensor_metas(node)
        # All output's shape is same, return first.
        return list(node_metas[0].shape)

    @classmethod
    def get_tiledims_from_dim(cls, node: Node, tile_dim: int):
        node_metas = get_tensor_metas(node)
        return [set([tile_dim])] * len(node_metas)

    @classmethod
    def post_process(cls, node: Node, code_lines: list[str], extra_lines: list[str]):
        node_metas = get_tensor_metas(node)
        node_dtype1 = node_metas[1].dtype
        node_shape1 = list(node_metas[1].shape)
        keepdim = cls.get_keepdim(node)
        code_lines.append(f"{str(node)}_item_0 = {str(node)}")
        if not keepdim:
            out_shape_tiled = node_shape1.copy()
            tiledims = node.meta.get(TILEDIMNAME, None)
            if tiledims:
                for tiled in tiledims[1]:
                    if out_shape_tiled[tiled] > 1:
                        out_shape_tiled[tiled] = BATCHBLOCKNAME
            out_shape = ",".join([str(x) for x in out_shape_tiled])

            code_lines.append(
                f"{str(node)}_mean = tl.reshape({str(node)}_mean, [{out_shape}], can_reorder=True)"
            )
        code_lines.append(
            f"{str(node)}_item_1 = {str(node)}_mean.to({TORCH2TRITON_DTYPE_STR[str(node_dtype1)]})"
        )
        return code_lines, extra_lines
