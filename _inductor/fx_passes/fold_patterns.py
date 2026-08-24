from typing import Any, Union
import operator

import torch
from torch.utils._ordered_set import OrderedSet
from torch._decomp import register_decomposition
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    CallFunctionVarArgs,
    Ignored,
    KeywordArg,
    Match,
    register_graph_pattern,
    get_arg_value,
    PatternMatcherPass,
    ListOf,
    MULTIPLE,
    filter_nodes,
)
from torch._inductor.fx_utils import get_node_storage, get_fake_args_kwargs
from torch._inductor.fx_passes.post_grad import same_meta
from torch._inductor.fx_passes.split_cat import is_sorted_and_consecutive
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq
from torch._prims_common import is_integer_dtype


from .utils import (
    is_mlu_tensor_node,
    is_signed_integer_tensor,
)

aten = torch.ops.aten

fold_matmul_like_view_pointwise_view_pass = PatternMatcherPass(
    "mlu_fold_matmul_like_view_pointwise_view"
)

fold_passes = {
    "fold_binaryop": PatternMatcherPass("mlu_fold_binaryop"),
    "fold_cat": PatternMatcherPass("mlu_fold_cat"),
    "fold_expand": PatternMatcherPass("mlu_fold_expand"),
    "fold_reduce": PatternMatcherPass("mlu_fold_reduce"),
    "fold_nest_view": PatternMatcherPass("mlu_fold_nest_view"),
    "fold_matmul_like_view_pointwise_view": fold_matmul_like_view_pointwise_view_pass,
    "fold_where": PatternMatcherPass("mlu_fold_where"),
    "fold_stack": PatternMatcherPass("mlu_fold_stack"),
    "fold_abs": PatternMatcherPass("mlu_fold_abs"),
    "fold_maximini": PatternMatcherPass("mlu_fold_maximini"),
    "fold_neg": PatternMatcherPass("mlu_fold_neg"),
    "fold_logical_not": PatternMatcherPass("mlu_fold_logical_not"),
    "fold_log": PatternMatcherPass("mlu_fold_log"),
}


def should_fold_binaryop_scalar(match):
    arg0 = match.args[0].meta.get("val", None)
    if not isinstance(arg0, torch.Tensor) or not arg0.is_mlu:
        return False
    return True


@register_graph_pattern(
    CallFunction(
        [aten.add.Tensor, aten.add.Scalar, aten.sub.Tensor, aten.sub.Scalar],
        Arg(),
        0,
    ),
    extra_check=should_fold_binaryop_scalar,
    pass_dict=fold_passes["fold_binaryop"],
)
@register_graph_pattern(
    CallFunction(
        aten.add.Tensor,
        0,
        Arg(),
        alpha=1,
    ),
    extra_check=should_fold_binaryop_scalar,
    pass_dict=fold_passes["fold_binaryop"],
)
@register_graph_pattern(
    CallFunction(
        [aten.mul.Tensor, aten.mul.Scalar, aten.div.Tensor, aten.div.Scalar],
        Arg(),
        1,
    ),
    extra_check=should_fold_binaryop_scalar,
    pass_dict=fold_passes["fold_binaryop"],
)
@register_graph_pattern(
    CallFunction(
        aten.mul.Tensor,
        1,
        Arg(),
    ),
    extra_check=should_fold_binaryop_scalar,
    pass_dict=fold_passes["fold_binaryop"],
)
def fold_binaryop_scalar(match: Match, input):
    def repl(input, out_dtype):
        if input.dtype != out_dtype:
            return aten._to_copy(input, dtype=out_dtype)
        else:
            return input.clone()

    binaryop_node = match.output_node()
    match.replace_by_example(repl, (input, binaryop_node.meta["val"].dtype))


def should_fold_binaryop_broadcast(match):
    out_node = match.output_node()
    arg0 = match.args[0]
    if not isinstance(arg0, torch.fx.Node) or not hasattr(arg0, "meta"):
        return False
    arg0 = arg0.meta.get("val", None)
    if not isinstance(arg0, torch.Tensor) or not arg0.is_mlu:
        return False
    if statically_known_true(sym_eq(arg0.size(), out_node.meta["val"].size())):
        return False

    return True


@register_graph_pattern(
    CallFunction(
        [aten.add.Tensor, aten.sub.Tensor],
        Arg(),
        CallFunction(aten.full.default, Ignored(), 0),
    ),
    extra_check=should_fold_binaryop_broadcast,
    pass_dict=fold_passes["fold_binaryop"],
)
@register_graph_pattern(
    CallFunction(
        aten.add.Tensor,
        CallFunction(aten.full.default, Ignored(), 0),
        Arg(),
        alpha=1,
    ),
    extra_check=should_fold_binaryop_broadcast,
    pass_dict=fold_passes["fold_binaryop"],
)
@register_graph_pattern(
    CallFunction(
        [aten.mul.Tensor, aten.div.Tensor],
        Arg(),
        CallFunction(aten.full.default, Ignored(), 1),
    ),
    extra_check=should_fold_binaryop_broadcast,
    pass_dict=fold_passes["fold_binaryop"],
)
@register_graph_pattern(
    CallFunction(
        aten.mul.Tensor,
        CallFunction(aten.full.default, Ignored(), 1),
        Arg(),
    ),
    extra_check=should_fold_binaryop_broadcast,
    pass_dict=fold_passes["fold_binaryop"],
)
def fold_binaryop_broadcast(match: Match, input):
    def repl(input, out_shape, out_dtype):
        input = input.expand(out_shape)
        if input.dtype != out_dtype:
            return aten._to_copy(input, dtype=out_dtype)
        else:
            return input.clone()

    binaryop_node = match.output_node()
    match.replace_by_example(
        repl, (input, binaryop_node.meta["val"].shape, binaryop_node.meta["val"].dtype)
    )


def should_fold_cat_only_one_tensor(match):
    add_node = match.output_node()
    inputs = add_node.args[0]
    return len(inputs) == 1


def check_nested_cat(match):
    cat_node = match.output_node()
    for inp in cat_node.args[0]:
        if (
            isinstance(inp, torch.fx.Node)
            and inp.op == "call_function"
            and inp.target == aten.cat.default
            and len(inp.users) == 1
        ):
            return True
    return False


def _get_cat_dim(cat_node):
    cat_dim = get_arg_value(cat_node, 1, "dim")
    if cat_dim is None:
        cat_dim = 0
    if cat_dim < 0:
        cat_dim += cat_node.meta["val"].dim()
    return cat_dim


@register_graph_pattern(
    CallFunctionVarArgs(aten.cat.default),
    extra_check=should_fold_cat_only_one_tensor,
    pass_dict=fold_passes["fold_cat"],
)
def fold_cat_only_one_tensor(match: Match, *args, **kwargs):
    def repl(input):
        return input.clone()

    add_node = match.output_node()
    inputs = add_node.args[0]
    match.replace_by_example(repl, (inputs[0],))


@register_graph_pattern(
    CallFunctionVarArgs(aten.cat.default),
    extra_check=check_nested_cat,
    pass_dict=fold_passes["fold_cat"],
)
def fold_cat_cat(match: Match, *args, **kwargs):
    graph = match.graph
    for node in graph.find_nodes(
        op="call_function", target=aten.cat.default, sort=True
    ):
        cat_dim = _get_cat_dim(node)
        fuse_inputs = []
        should_fold = False
        wait_rm_inputs = []
        for inp in node.args[0]:
            if (
                isinstance(inp, torch.fx.Node)
                and inp.op == "call_function"
                and inp.target == aten.cat.default
                and len(inp.users) == 1
            ):
                arg_cat_dim = _get_cat_dim(inp)
                if cat_dim == arg_cat_dim:
                    should_fold = True
                    fuse_inputs.extend(inp.args[0])
                    wait_rm_inputs.append(inp)
                    continue
            fuse_inputs.append(inp)
        if should_fold:
            with graph.inserting_before(node):
                fuse_node = graph.call_function(
                    aten.cat.default, (fuse_inputs, cat_dim)
                )
                node.replace_all_uses_with(fuse_node)
                fuse_node.meta.update(node.meta)
                graph.erase_node(node)
                for rm_node in wait_rm_inputs:
                    graph.erase_node(rm_node)


def has_multiple_cats(match) -> bool:
    graph = match.graph
    all_cat_nodes = list(
        graph.find_nodes(op="call_function", target=aten.cat.default, sort=True)
    )
    return len(all_cat_nodes) > 1


def expand_tensor_sequence(node):
    if not isinstance(node, torch.fx.Node):
        return [node]
    if node.op == "call_function" and node.target == aten.cat.default:
        seq = []
        for inp in node.args[0]:
            seq.extend(expand_tensor_sequence(inp))
        return seq
    else:
        return [node]


@register_graph_pattern(
    CallFunctionVarArgs(aten.cat.default),
    pass_dict=fold_passes["fold_cat"],
    extra_check=has_multiple_cats,
)
def fold_sequential_cat(match: Match, *args, **kwargs):
    """
    This pass detects and replaces consecutive tensor segments in a cat node inputs with
    previously defined cat nodes, effectively folding redundant concatenations to reduce repeated computation.

    cat_1 = cat([a, b])
    cat_2 = cat([a, b, c]) -> cat([cat_1, c])
    cat_3 = cat([a, b, c, d]) -> cat([cat_2, d])
    cat_4 = cat([e, f, g, h])
    cat_5 = ...
    mul_1 = ...
    cat_5 = cat([a, b, c, mul_1, d, cat_5, e, f, g, h, i]) -> cat([cat_2, mul_1, d, cat_5, cat4, i])
    """
    graph = match.graph
    cat_node = match.output_node()
    cat_inputs = list(cat_node.args[0])
    node_dim = _get_cat_dim(cat_node)

    prev_cats = []
    all_cat_nodes = list(
        graph.find_nodes(op="call_function", target=aten.cat.default, sort=True)
    )

    node_pos = all_cat_nodes.index(cat_node)
    for n in all_cat_nodes[:node_pos]:
        if n.op == "call_function" and n.target == aten.cat.default:
            dim = _get_cat_dim(n)
            prev_cats.append((expand_tensor_sequence(n), n, dim))

    i = 0
    new_inputs = []
    while i < len(cat_inputs):
        inp = cat_inputs[i]
        if not (
            isinstance(inp, torch.fx.Node) and inp.op in ("placeholder", "get_attr")
        ):
            new_inputs.append(inp)
            i += 1
            continue
        best_match = None
        best_len = 0
        for tensor_seq, prev_cat_node, cat_dim in prev_cats:
            if cat_dim != node_dim:
                continue
            k = len(tensor_seq)

            if i + k <= len(cat_inputs) and cat_inputs[i : i + k] == tensor_seq:
                if k > best_len:
                    best_match = prev_cat_node
                    best_len = k
        if best_match:
            new_inputs.append(best_match)
            i += best_len
        else:
            new_inputs.append(inp)
            i += 1
    if new_inputs != cat_inputs:
        with graph.inserting_before(cat_node):
            new_cat = graph.call_function(aten.cat.default, (new_inputs, node_dim))
            new_cat.meta.update(cat_node.meta)
            cat_node.replace_all_uses_with(new_cat)
            graph.erase_node(cat_node)
            graph.lint()


def should_fold_expand(match):
    out_node = match.output_node()
    input_node = out_node.args[0]
    if statically_known_true(
        sym_eq(input_node.meta["val"].size(), out_node.meta["val"].size())
    ):
        return True
    else:
        return False


@register_graph_pattern(
    CallFunction(torch.ops.aten.expand.default, Ignored(), Ignored()),
    extra_check=should_fold_expand,
    pass_dict=fold_passes["fold_expand"],
)
def fold_redundant_expand(match: Match):
    graph = match.graph
    expand_node = match.output_node()
    input_node = expand_node.args[0]

    with graph.inserting_before(expand_node):
        expand_node.replace_all_uses_with(input_node)
        graph.erase_node(expand_node)


@register_graph_pattern(
    CallFunctionVarArgs(aten.sum.dim_IntList),
    pass_dict=fold_passes["fold_reduce"],
)
def fold_redundant_reduce(match: Match, *args, **kwargs):
    reduce_node = match.output_node()
    input = reduce_node.args[0].meta["val"]
    dims = reduce_node.args[1]
    if dims is None:
        dims = list(range(len(input.shape)))
    if not isinstance(dims, list):
        return
    if all(statically_known_true(sym_eq(input.shape[d], 1)) for d in dims):
        keepdim = reduce_node.args[2] if len(reduce_node.args) > 2 else False

        def repl(input, input_dtype, dims, keepdim, out_dtype):
            if input_dtype != out_dtype:
                output = aten._to_copy(input, dtype=out_dtype)
            else:
                output = input.clone()
            if not keepdim and len(dims) > 0:
                return output.squeeze(dim=dims)
            return output

        match.replace_by_example(
            repl,
            (
                reduce_node.args[0],
                input.dtype,
                dims,
                keepdim,
                reduce_node.meta["val"].dtype,
            ),
        )


_VIEW_LIKE_OPS = [
    aten.reshape.default,
    aten.squeeze.default,
    aten.squeeze.dim,
    aten.squeeze.dims,
    aten.unsqueeze.default,
]


@register_graph_pattern(
    CallFunction(
        aten.reshape.default,
        CallFunctionVarArgs(_VIEW_LIKE_OPS),
        Ignored(),
    ),
    pass_dict=fold_passes["fold_nest_view"],
)
def fold_nested_views(match: Match, *args):
    # Can not use match.replace_by_example, because aten.reshape will be traced to aten.view.
    graph = match.graph
    out_node = match.output_node()
    erase_nodes = [out_node, out_node.args[0]]
    input = out_node.args[0].args[0]
    while (
        isinstance(input, torch.fx.Node)
        and input.op == "call_function"
        and input.target in _VIEW_LIKE_OPS
        and len(input.users) == 1
    ):
        erase_nodes.append(input)
        input = input.args[0]
    with graph.inserting_before(out_node):
        new_node = graph.call_function(
            aten.reshape.default, (input, out_node.meta["val"].shape)
        )
        out_node.replace_all_uses_with(new_node)
        new_node.meta.update(out_node.meta)
        for node in erase_nodes:
            graph.erase_node(node)


_RESHAPE_VIEW_OPS = [
    aten.reshape.default,
    aten.view.default,
]

_MATMUL_VIEW_FOLD_ROOT_OPS = [
    aten.add.Tensor,
    aten.addmm.default,
    aten.baddbmm.default,
    aten.bmm.default,
    aten.mm.default,
]

_MATMUL_VIEW_FOLD_POINTWISE_OPS = [
    aten.gelu.default,
    aten.relu.default,
    aten.sigmoid.default,
]

_MATMUL_VIEW_FOLD_UNARY_POINTWISE_OPS = [
    aten.gelu.default,
    aten.relu.default,
    aten.sigmoid.default,
]

_FUSED_MM_ROOT_OPS = [
    aten.addmm.default,
    aten.mm.default,
]

_FUSED_BMM_ROOT_OPS = [
    aten.baddbmm.default,
    aten.bmm.default,
]

_FUSED_MM_ACTIVATION_OPS = [
    aten.gelu.default,
    aten.relu.default,
    aten.sigmoid.default,
]

_BMM_VIEW_FOLD_ACTIVATION_OPS = [
    aten.gelu.default,
    aten.relu.default,
    aten.sigmoid.default,
]


def _is_call_function_node(node, target=None):
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    if target is None:
        return True
    if isinstance(target, (list, tuple)):
        return node.target in target
    return node.target == target


def _shape_numel(shape):
    numel = 1
    for dim in shape:
        numel = numel * dim
    return numel


def _same_shape(lhs, rhs):
    return statically_known_true(sym_eq(lhs, rhs))


def _same_numel(lhs, rhs):
    return statically_known_true(sym_eq(_shape_numel(lhs), _shape_numel(rhs)))


def _get_meta_shape(node):
    if not isinstance(node, torch.fx.Node):
        return None
    val = node.meta.get("val", None)
    if not isinstance(val, torch.Tensor):
        return None
    return val.shape


def _get_matmul_pointwise_view_nodes(match):
    reshape2_node = match.output_node()
    if (
        not _is_call_function_node(reshape2_node, _RESHAPE_VIEW_OPS)
        or len(reshape2_node.args) < 1
    ):
        return None

    pointwise_node = reshape2_node.args[0]
    if (
        not _is_call_function_node(pointwise_node, _MATMUL_VIEW_FOLD_POINTWISE_OPS)
        or len(pointwise_node.args) < 1
    ):
        return None

    reshape1_node = pointwise_node.args[0]
    if (
        not _is_call_function_node(reshape1_node, _RESHAPE_VIEW_OPS)
        or len(reshape1_node.args) < 1
    ):
        return None

    matmul_node = reshape1_node.args[0]
    if not _is_call_function_node(matmul_node, _MATMUL_VIEW_FOLD_ROOT_OPS):
        return None

    return matmul_node, reshape1_node, pointwise_node, reshape2_node


def _has_single_user(node, user):
    return len(node.users) == 1 and next(iter(node.users)) is user


def _has_tensor_node_arg_after_first(node):
    for arg in list(node.args[1:]) + list(node.kwargs.values()):
        if isinstance(arg, torch.fx.Node) and isinstance(
            arg.meta.get("val"), torch.Tensor
        ):
            return True
    return False


def _get_add_tensor_matmul_kind(node):
    if not _is_call_function_node(node, aten.add.Tensor) or len(node.args) < 2:
        return None

    alpha = node.kwargs.get("alpha", node.args[2] if len(node.args) > 2 else 1)
    if alpha != 1:
        return None

    lhs, rhs = node.args[0], node.args[1]
    if _is_call_function_node(lhs, aten.mm.default) or _is_call_function_node(
        rhs, aten.mm.default
    ):
        return "mm"
    if _is_call_function_node(lhs, aten.bmm.default) or _is_call_function_node(
        rhs, aten.bmm.default
    ):
        return "bmm"
    return None


def _is_supported_matmul_pointwise(matmul_node, pointwise_node):
    if matmul_node.target in _FUSED_MM_ROOT_OPS:
        return pointwise_node.target in _FUSED_MM_ACTIVATION_OPS
    if matmul_node.target in _FUSED_BMM_ROOT_OPS:
        return pointwise_node.target in _BMM_VIEW_FOLD_ACTIVATION_OPS
    if matmul_node.target != aten.add.Tensor:
        return False

    matmul_kind = _get_add_tensor_matmul_kind(matmul_node)
    if matmul_kind == "mm":
        return pointwise_node.target in _FUSED_MM_ACTIVATION_OPS
    if matmul_kind == "bmm":
        return pointwise_node.target in _BMM_VIEW_FOLD_ACTIVATION_OPS
    return False


def _is_valid_matmul_pointwise_view(match):
    nodes = _get_matmul_pointwise_view_nodes(match)
    if nodes is None:
        return False
    matmul_node, reshape1_node, pointwise_node, reshape2_node = nodes

    if not _is_supported_matmul_pointwise(matmul_node, pointwise_node):
        return False

    if _has_tensor_node_arg_after_first(pointwise_node):
        return False

    if not (
        _has_single_user(matmul_node, reshape1_node)
        and _has_single_user(reshape1_node, pointwise_node)
        and _has_single_user(pointwise_node, reshape2_node)
    ):
        return False

    matmul_shape = _get_meta_shape(matmul_node)
    reshape1_shape = _get_meta_shape(reshape1_node)
    pointwise_shape = _get_meta_shape(pointwise_node)
    reshape2_shape = _get_meta_shape(reshape2_node)
    if (
        matmul_shape is None
        or reshape1_shape is None
        or pointwise_shape is None
        or reshape2_shape is None
    ):
        return False

    return (
        _same_numel(matmul_shape, reshape1_shape)
        and _same_shape(reshape1_shape, pointwise_shape)
        and _same_numel(pointwise_shape, reshape2_shape)
        and _same_shape(matmul_shape, reshape2_shape)
    )


_matmul_view_pattern = CallFunction(
    _RESHAPE_VIEW_OPS,
    CallFunctionVarArgs(_MATMUL_VIEW_FOLD_ROOT_OPS),
    Ignored(),
)


@register_graph_pattern(
    CallFunction(
        _RESHAPE_VIEW_OPS,
        CallFunction(
            _MATMUL_VIEW_FOLD_UNARY_POINTWISE_OPS,
            _matmul_view_pattern,
        ),
        Ignored(),
    ),
    pass_dict=fold_matmul_like_view_pointwise_view_pass,
    extra_check=_is_valid_matmul_pointwise_view,
)
@register_graph_pattern(
    CallFunction(
        _RESHAPE_VIEW_OPS,
        CallFunction(
            aten.gelu.default,
            _matmul_view_pattern,
            approximate=Ignored(),
        ),
        Ignored(),
    ),
    pass_dict=fold_matmul_like_view_pointwise_view_pass,
    extra_check=_is_valid_matmul_pointwise_view,
)
def fold_matmul_like_view_pointwise_view(match: Match, *args):
    graph = match.graph
    nodes = _get_matmul_pointwise_view_nodes(match)
    if nodes is None:
        return
    matmul_node, reshape1_node, pointwise_node, reshape2_node = nodes
    new_args = tuple(
        matmul_node if arg is reshape1_node else arg for arg in pointwise_node.args
    )
    new_kwargs = dict(pointwise_node.kwargs)

    with graph.inserting_before(reshape2_node):
        new_pointwise = graph.call_function(pointwise_node.target, new_args, new_kwargs)
        new_pointwise.meta.update(reshape2_node.meta)

    reshape2_node.replace_all_uses_with(new_pointwise)
    graph.erase_node(reshape2_node)
    graph.erase_node(pointwise_node)
    graph.erase_node(reshape1_node)


_same_where_input = Arg()


# torch.where(cond, x, x) → x
# torch.where(cond, ones, ones) → ones
# torch.where(cond, zeros, zeros) → zeros
@register_graph_pattern(
    CallFunction(
        aten.where.self,
        Ignored(),
        _same_where_input,
        _same_where_input,
    ),
    pass_dict=fold_passes["fold_where"],
)
@register_graph_pattern(
    CallFunction(
        aten.where.self,
        Ignored(),
        CallFunction(aten.full.default, Ignored(), 1),
        CallFunction(aten.full.default, Ignored(), 1),
    ),
    pass_dict=fold_passes["fold_where"],
)
@register_graph_pattern(
    CallFunction(
        aten.where.self,
        Ignored(),
        CallFunction(aten.full.default, Ignored(), 0),
        CallFunction(aten.full.default, Ignored(), 0),
    ),
    pass_dict=fold_passes["fold_where"],
)
def fold_redundant_where(match: Match, *args):
    def repl(input, out_shape, out_dtype):
        if not statically_known_true(sym_eq(input.shape, out_shape)):
            input = input.expand(out_shape)
        if input.dtype != out_dtype:
            return aten._to_copy(input, dtype=out_dtype)
        else:
            return input.clone()

    out_node = match.output_node()
    input = out_node.args[1]
    match.replace_by_example(
        repl, (input, out_node.meta["val"].shape, out_node.meta["val"].dtype)
    )


def _is_valid_unbind_getitem_cat(match):
    unbind_nodes = filter_nodes(match.nodes, aten.unbind.int)
    get_item_nodes = filter_nodes(match.nodes, operator.getitem)
    if len(unbind_nodes) != 1:
        return False
    unbind_node = unbind_nodes[0]
    unbind_cat_dim = get_arg_value(unbind_node, 1, "dim")
    if unbind_cat_dim < 0:
        unbind_cat_dim += unbind_node.meta["val"][0].dim() + 1
    # The dim of unbind、unsqueeze and cat should match for passthrough
    cat_dim = _get_cat_dim(filter_nodes(match.nodes, aten.cat)[0])
    if unbind_cat_dim is None or unbind_cat_dim != cat_dim:
        return False
    for get_item_node in get_item_nodes:
        unsqueeze_node = list(get_item_node.users.keys())[0]
        if unbind_cat_dim != get_arg_value(unsqueeze_node, 1, "dim"):
            return False
    get_item_args = [
        get_arg_value(get_item_node, 1) for get_item_node in get_item_nodes
    ]
    assert None not in get_item_args
    # All parts of unbind should be included in the cat and
    # the order of get_item_args should be same with unbind output.
    if get_item_args != list(range(len(unbind_node.meta["val"]))):
        return False

    return True


def _is_valid_select_cat(match, cat_dim):
    input = match.kwargs["input"].meta["val"]
    select_ids = []
    select_nodes = filter_nodes(match.nodes, aten.select.int)
    for select_node in select_nodes:
        select_dim = get_arg_value(select_node, 1, "dim")
        if select_dim < 0:
            select_dim += input.dim()
        if cat_dim != select_dim:
            return False
        select_index = get_arg_value(select_node, 2, "index")
        if statically_known_true(select_index < 0):
            select_index += input.shape[select_dim]
        select_ids.append(select_index)

    if not is_sorted_and_consecutive(select_ids) or not statically_known_true(
        len(select_ids) == input.shape[cat_dim]
    ):
        return False

    return True


def _is_valid_select_unsqueeze_cat(match):
    cat_dim = _get_cat_dim(filter_nodes(match.nodes, aten.cat)[0])
    if not _is_valid_select_cat(match, cat_dim):
        return False
    unsqueeze_nodes = filter_nodes(match.nodes, aten.unsqueeze.default)
    for unsqueeze_node in unsqueeze_nodes:
        unsqueeze_dim = get_arg_value(unsqueeze_node, 1, "dim")
        if unsqueeze_dim < 0:
            unsqueeze_dim += unsqueeze_node.meta["val"].dim()
        if cat_dim != unsqueeze_dim:
            return False

    return True


unbind_src = CallFunction(
    aten.unbind.int, KeywordArg("input"), Ignored(), _users=MULTIPLE
)
unbind_cat_tangent = KeywordArg("input")


@register_graph_pattern(
    CallFunction(
        aten.cat,
        ListOf(
            CallFunction(
                aten.unsqueeze,
                CallFunction(
                    operator.getitem,
                    unbind_src,
                    Ignored(),
                ),
                Ignored(),
            )
        ),
        Ignored(),
    ),
    pass_dict=fold_passes["fold_stack"],
    extra_check=_is_valid_unbind_getitem_cat,
)
@register_graph_pattern(
    CallFunction(
        aten.cat,
        ListOf(
            CallFunction(
                aten.unsqueeze,
                CallFunction(
                    aten.select.int,
                    unbind_cat_tangent,
                    Ignored(),
                    Ignored(),
                ),
                Ignored(),
            )
        ),
        Ignored(),
    ),
    pass_dict=fold_passes["fold_stack"],
    extra_check=_is_valid_select_unsqueeze_cat,
)
def fold_unbind_unsqueeze_cat(match, *, input):
    def repl(input):
        return input.clone()

    # TODO(cdzhan): propagate stride to users of output?
    match.replace_by_example(repl, (input,))


def _is_valid_unbind_cat_reshape(match):
    unbind_node = filter_nodes(match.nodes, aten.unbind.int)[0]
    input = get_arg_value(unbind_node, 0, "self")
    reshape_node = filter_nodes(match.nodes, aten.reshape)[0]
    if not statically_known_true(
        sym_eq(reshape_node.meta["val"].shape, input.meta["val"].shape)
    ):
        return False
    unbind_cat_dim = get_arg_value(unbind_node, 1, "dim")
    if unbind_cat_dim is None:
        unbind_cat_dim = 0
    if unbind_cat_dim < 0:
        unbind_cat_dim += unbind_node.meta["val"][0].dim() + 1
    cat_dim = _get_cat_dim(filter_nodes(match.nodes, aten.cat)[0])
    if unbind_cat_dim != cat_dim:
        return False
    get_item_nodes = filter_nodes(match.nodes, operator.getitem)
    get_item_args = [
        get_arg_value(get_item_node, 1) for get_item_node in get_item_nodes
    ]
    assert None not in get_item_args
    # All parts of unbind should be included in the cat and
    # the order of get_item_args should be same with unbind output.
    if get_item_args != list(range(len(unbind_node.meta["val"]))):
        return False

    return True


def _is_valid_select_cat_reshape(match):
    input = match.kwargs["input"]
    reshape_node = filter_nodes(match.nodes, aten.reshape)[0]
    if not statically_known_true(
        sym_eq(reshape_node.meta["val"].shape, input.meta["val"].shape)
    ):
        return False
    cat_dim = _get_cat_dim(filter_nodes(match.nodes, aten.cat)[0])
    if not _is_valid_select_cat(match, cat_dim):
        return False

    return True


unbind_src = CallFunctionVarArgs(aten.unbind.int, users=MULTIPLE)


@register_graph_pattern(
    CallFunction(
        aten.reshape,
        CallFunction(
            aten.cat,
            ListOf(
                CallFunction(
                    operator.getitem,
                    unbind_src,
                    Ignored(),
                )
            ),
            Ignored(),
        ),
        Ignored(),
    ),
    pass_dict=fold_passes["fold_stack"],
    extra_check=_is_valid_unbind_cat_reshape,
)
@register_graph_pattern(
    CallFunction(
        aten.reshape,
        CallFunction(
            aten.cat,
            ListOf(
                CallFunction(
                    operator.getitem,
                    unbind_src,
                    Ignored(),
                )
            ),
        ),
        Ignored(),
    ),
    pass_dict=fold_passes["fold_stack"],
    extra_check=_is_valid_unbind_cat_reshape,
)
@register_graph_pattern(
    CallFunction(
        aten.reshape,
        CallFunction(
            aten.cat,
            ListOf(
                CallFunction(
                    aten.select.int,
                    unbind_cat_tangent,
                    Ignored(),
                    Ignored(),
                )
            ),
            Ignored(),
        ),
        Ignored(),
    ),
    pass_dict=fold_passes["fold_stack"],
    extra_check=_is_valid_select_cat_reshape,
)
@register_graph_pattern(
    CallFunction(
        aten.reshape,
        CallFunction(
            aten.cat,
            ListOf(
                CallFunction(
                    aten.select.int,
                    unbind_cat_tangent,
                    Ignored(),
                    Ignored(),
                )
            ),
        ),
        Ignored(),
    ),
    pass_dict=fold_passes["fold_stack"],
    extra_check=_is_valid_select_cat_reshape,
)
def fold_unbind_cat_reshape(match, *args, **kwargs):
    def repl(input):
        return input.clone()

    unbind_nodes = filter_nodes(match.nodes, aten.unbind.int)
    if len(unbind_nodes) > 0:
        input = get_arg_value(unbind_nodes[0], 0, "self")
    else:
        input = kwargs["input"]

    # TODO(cdzhan): propagate stride to users of output?
    match.replace_by_example(repl, (input,))


# Ref XLA algebraic simplifier.
def _is_non_negative(node):
    if not isinstance(node, torch.fx.Node):
        return False
    if node.target == aten.mul.Tensor:
        if id(node.args[0]) != id(node.args[1]):
            return False
        # The multiply of signed integer may be negative.
        if is_signed_integer_tensor(node.meta["val"]):
            return False
        return True
    elif node.target == aten.abs.default:
        return True
    elif node.target == aten.maximum.default:
        return _is_non_negative(node.args[0]) or _is_non_negative(node.args[1])
    elif node.target == aten.where.self:
        return _is_non_negative(node.args[1]) and _is_non_negative(node.args[2])
    # Only need consider aten.pow.Tensor_Tensor and aten.pow.Scalar, other variants would be constant folded.
    elif node.target == aten.pow.Tensor_Tensor and not is_signed_integer_tensor(
        node.meta["val"]
    ):
        return _is_positive(node.args[0])
    elif (
        node.target == aten.pow.Scalar
        and not is_signed_integer_tensor(node.meta["val"])
        and isinstance(node.args[0], (int, float, torch.SymInt, torch.SymFloat))
    ):
        return statically_known_true(node.args[0] > 0)

    return _is_positive(node)


def _is_positive(node):
    if not isinstance(node, torch.fx.Node):
        return False
    if node.target == aten.full.default:
        full_res_tensor = node.meta["val"]
        # '>' not supported between instances of 'complex' and 'int'
        if full_res_tensor.is_complex():
            return False
        fill_val = node.args[1]
        if isinstance(fill_val, torch.fx.Node):
            fill_val = fill_val.meta.get("val", -1)
        return statically_known_true(fill_val > 0)
    return False


def _should_fold_abs(match):
    out = match.output_node()
    abs_arg = out.args[0]
    return (
        is_mlu_tensor_node(out)
        and out.meta["val"].dtype == abs_arg.meta["val"].dtype
        and _is_non_negative(abs_arg)
    )


def _should_fold_maximini(match):
    arg = match.args[0]
    if not is_mlu_tensor_node(arg):
        return False
    out = match.output_node()
    if arg.meta["val"].dtype != out.meta["val"].dtype or not statically_known_true(
        sym_eq(arg.meta["val"].shape, out.meta["val"].shape)
    ):
        return False
    is_maxi = out.target == aten.maximum.default
    dtype = out.meta["val"].dtype
    maxmin_val = None
    if dtype == torch.bool:
        maxmin_val = False if is_maxi else True
    elif is_integer_dtype(dtype):
        maxmin_val = torch.iinfo(dtype).min if is_maxi else torch.iinfo(dtype).max
    elif dtype.is_floating_point:
        maxmin_val = -torch.inf if is_maxi else torch.inf

    fill_val = match.kwargs["fill_val"]
    if isinstance(fill_val, torch.fx.Node):
        fill_val = fill_val.meta.get("val", None)
    return maxmin_val is not None and statically_known_true(maxmin_val == fill_val)


def _should_vanilla_fold(match):
    out = match.output_node()
    arg = match.args[0]
    return is_mlu_tensor_node(out) and arg.meta["val"].dtype == out.meta["val"].dtype


def _should_fold_log(match):
    if not _should_vanilla_fold(match):
        return False
    # log(exp(z)) != z for complex inputs because the imaginary part is wrapped to (-pi, pi]
    return not match.args[0].meta["val"].is_complex()


def _should_fold_logical_not(match):
    out = match.output_node()
    arg = match.args[0]
    return is_mlu_tensor_node(out) and arg.meta["val"].dtype == torch.bool


# abs(non_negative) => non_negative
@register_graph_pattern(
    CallFunction(aten.abs.default, Arg()),
    extra_check=_should_fold_abs,
    pass_dict=fold_passes["fold_abs"],
)
# maximum(x, type_min) => x, minimum(x, type_max) => x
@register_graph_pattern(
    CallFunction(
        [aten.maximum.default, aten.minimum.default],
        Arg(),
        CallFunction(
            aten.full.default, Ignored(), KeywordArg("fill_val"), _users=MULTIPLE
        ),
    ),
    extra_check=_should_fold_maximini,
    pass_dict=fold_passes["fold_maximini"],
)
@register_graph_pattern(
    CallFunction(
        [aten.maximum.default, aten.minimum.default],
        CallFunction(
            aten.full.default, Ignored(), KeywordArg("fill_val"), _users=MULTIPLE
        ),
        Arg(),
    ),
    extra_check=_should_fold_maximini,
    pass_dict=fold_passes["fold_maximini"],
)
# neg(neg(x)) => x
@register_graph_pattern(
    CallFunction(
        aten.neg.default,
        CallFunction(aten.neg.default, Arg(), _users=MULTIPLE),
    ),
    extra_check=_should_vanilla_fold,
    pass_dict=fold_passes["fold_neg"],
)
# not(not(x)) => x
@register_graph_pattern(
    CallFunction(
        aten.logical_not.default,
        CallFunction(aten.logical_not.default, Arg(), _users=MULTIPLE),
    ),
    extra_check=_should_fold_logical_not,
    pass_dict=fold_passes["fold_logical_not"],
)
# log(exp(x)) => x
@register_graph_pattern(
    CallFunction(
        aten.log.default,
        CallFunction(aten.exp.default, Arg(), _users=MULTIPLE),
    ),
    extra_check=_should_fold_log,
    pass_dict=fold_passes["fold_log"],
)
def vanilla_fold(match, inp, **kwargs):
    def repl(inp):
        return inp.clone()

    match.replace_by_example(repl, (inp,))


# Cover more scenes compared with native remove_noop_ops pass because we
# change some ops to aten.clone in the above pass.
noop_registry: dict[Any, Any] = {}


def register_noop_decomp(targets, nop_arg=0):
    def register_fun(cond):
        register_decomposition(targets, registry=noop_registry, unsafe=True)(
            (cond, nop_arg)  # type: ignore[arg-type]
        )
        return cond

    return register_fun


@register_noop_decomp(aten.clone)
def true_noop(*args, **kwargs):
    return True


def remove_noop_ops(graph: torch.fx.Graph):
    """
    Removes both operations that are essentially aten.clone and operations that are essentially aten.alias from the graph.
    """
    inputs = OrderedSet[torch.fx.Node]()
    input_storages = OrderedSet[Union[int, None]]()
    output_storages = OrderedSet[Union[int, None]]()

    for node in graph.find_nodes(op="placeholder"):
        inputs.add(node)
        input_storages.add(get_node_storage(node))

    output_node = next(iter(reversed(graph.nodes)))
    assert output_node.op == "output"
    outputs = output_node.args[0]
    if not isinstance(outputs, (list, tuple)):
        # nested subgraphs can have singleton outputs
        outputs = (outputs,)
    for out in outputs:
        if isinstance(out, torch.fx.Node):
            output_storages.add(get_node_storage(out))

    for node in graph.nodes:
        if node.target in noop_registry:
            cond, src_index = noop_registry[node.target]
            if isinstance(src_index, int):
                src = node.args[src_index]
            else:
                src = src_index(node.args)
            if not isinstance(src, torch.fx.Node):
                continue
            # Don't introduce new aliasing between inputs and outputs.
            # See fx_passes/README.md for a discussion of why this is
            # necessary.
            node_storage = get_node_storage(node)
            src_storage = get_node_storage(src)
            node_is_view = node_storage == src_storage
            if (
                not node_is_view
                and node_storage in output_storages
                and (src_storage in input_storages or src_storage in output_storages)
            ):
                continue

            # Even if input and outputs are expected to alias,
            # don't make "node is src" True
            if (
                node_is_view
                and node in output_node.args
                and (src in inputs or src in output_node.args)
            ):
                continue

            is_valid, args, kwargs = get_fake_args_kwargs(node)
            if not is_valid:
                continue
            if same_meta(node, src) and cond(*args, **kwargs):
                node.replace_all_uses_with(src)
                graph.erase_node(node)
