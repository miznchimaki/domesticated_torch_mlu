import operator

import torch
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Ignored,
    KeywordArg,
    Match,
    MULTIPLE,
    PatternMatcherPass,
    CallFunctionVarArgs,
    register_graph_pattern,
    register_lowering_pattern,
)
from torch._inductor.virtualized import ops
from torch._inductor.lowering import make_pointwise
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._inductor.utils import register_op_dtype_propagation_rules
from torch.fx.experimental.symbolic_shapes import (
    guard_size_oblivious,
    statically_known_true,
    sym_eq,
)
from torch._prims_common import (
    is_expandable_to,
    is_contiguous_or_false,
    _integer_dtypes,
)

from torch_mlu._inductor import config

from .utils import is_mlu_tensor_node

aten = torch.ops.aten
TRITONFUSION_KEEPALIVE_NAME = "tritonfusion_keepalive_node"

passes = {
    "cat_reshape": PatternMatcherPass("mlu_cat_reshape"),
    "aten_div_sqrt_replace": PatternMatcherPass("mlu_aten_div_sqrt_replace"),
    "div_exp_replace": PatternMatcherPass("mlu_div_exp_replace"),
}
silu_pass = PatternMatcherPass("silu_pass")
bmm_add_act_pass = PatternMatcherPass("mlu_fuse_tmo_bmm")
tmo_addmm_pass = PatternMatcherPass("mlu_fuse_tmo_addmm")
conv_relu_fusion_pass = PatternMatcherPass("conv_relu_fusion_pass")
tmo_layernorm_pass = PatternMatcherPass("mlu_fuse_tmo_layernorm")

BMM_TRITONFUSION_WEIGHT_LIMIT_BYTES = 128 * 128 * 4


def _is_mlu_pattern(match):
    out = match.output_node()
    return is_mlu_tensor_node(out)


def get_trans_input(node, permute_dims):
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target == aten.permute.default
        and (
            statically_known_true(sym_eq(list(node.args[1]), permute_dims))
            or statically_known_true(
                sym_eq(
                    list(node.args[1]),
                    list(map(lambda x: x - len(permute_dims) - 1, permute_dims)),
                )
            )
        )
        and is_contiguous_or_false(node.args[0].meta["val"])
    ):
        return True, node.args[0]
    return False, node


def normalize_dim_to_positive(dim: int, tensor: torch.Tensor) -> int:
    # Convert negative dimension index to positive.
    if dim < 0:
        dim += len(tensor.shape)
    return dim


def should_replace_cat_reshape_with_unsqueeze_cat(match):
    cat_node = match.nodes[0]
    reshape_node = match.nodes[1]
    cat_args = match.args[0]
    dim = match.kwargs.get("dim", 0)
    dim = normalize_dim_to_positive(dim, cat_node.meta["val"])
    size = match.kwargs["size"]
    cat_arg_shape = cat_args[0].meta["val"].shape
    # shape of args must be equal for aten.cat
    if any(
        [arg.meta["val"].shape != cat_arg_shape for arg in cat_args[1 : len(cat_args)]]
    ):
        return False

    args_count = len(cat_args)
    # reshape size must be equal
    if dim == 0:
        expect_reshape_size = [args_count, *list(cat_arg_shape)]
    else:
        expect_reshape_size = [
            *cat_arg_shape[0:dim],
            args_count,
            *cat_arg_shape[dim : len(cat_arg_shape)],
        ]
    for e, s in zip(expect_reshape_size, size):
        if isinstance(s, torch.fx.Node):
            s = s.meta["val"]
        if e != s:
            return False

    return True


def replace_cat_reshape_with_unsqueeze_cat(match: Match, inputs, dim, size):
    cat_node = match.nodes[0]
    dim = normalize_dim_to_positive(dim, cat_node.meta["val"])

    def repl(inputs, dim):
        return torch.cat([x.unsqueeze(dim) for x in inputs], dim=dim)

    match.replace_by_example(repl, (inputs, dim))


@register_graph_pattern(
    CallFunction(
        aten.reshape.default,
        CallFunction(
            aten.cat.default,
            Arg(),
            _users=1,
        ),
        KeywordArg("size"),
    ),
    extra_check=should_replace_cat_reshape_with_unsqueeze_cat,
    pass_dict=passes["cat_reshape"],
)
def cat_reshape_pattern_dim_is_zero(match: Match, inputs, size):
    replace_cat_reshape_with_unsqueeze_cat(match, inputs, 0, size)


@register_graph_pattern(
    CallFunction(
        aten.reshape.default,
        CallFunction(
            aten.cat.default,
            Arg(),
            KeywordArg("dim"),
            _users=1,
        ),
        KeywordArg("size"),
    ),
    extra_check=should_replace_cat_reshape_with_unsqueeze_cat,
    pass_dict=passes["cat_reshape"],
)
def cat_reshape_pattern_dim_not_zero(match: Match, inputs, dim, size):
    replace_cat_reshape_with_unsqueeze_cat(match, inputs, dim, size)


# x * sigmoid(x) = silu(x)
@register_lowering_pattern(
    CallFunction(
        aten.mul, KeywordArg("inputs"), CallFunction(aten.sigmoid, KeywordArg("inputs"))
    ),
    pass_dict=silu_pass,
)
# x / (exp(-x) + 1) = silu(x)
@register_lowering_pattern(
    CallFunction(
        aten.div,
        KeywordArg("inputs"),
        CallFunction(
            aten.add,
            CallFunction(
                aten.exp,
                CallFunction(aten.neg, KeywordArg("inputs")),
            ),
            1,
        ),
    ),
    pass_dict=silu_pass,
)
# x / (1 + exp(-x)) = silu(x)
@register_lowering_pattern(
    CallFunction(
        aten.div,
        KeywordArg("inputs"),
        CallFunction(
            aten.add,
            1,
            CallFunction(
                aten.exp,
                CallFunction(aten.neg, KeywordArg("inputs")),
            ),
        ),
    ),
    pass_dict=silu_pass,
)
def lower_silu(match, inputs):
    register_op_dtype_propagation_rules(
        "silu",
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        override_return_dtype=None,
    )

    def inner_fn(inputs):
        return ops.silu(inputs)

    return make_pointwise(inner_fn)(inputs)


def should_fuse_conv_relu(match):
    input = match.args[0].meta["val"]
    weight = match.args[1].meta["val"]
    bias = (
        match.args[2].meta["val"]
        if isinstance(match.args[2], torch.fx.Node)
        else match.args[2]
    )
    return (
        input.is_mlu
        and input.dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64)
        and weight.dtype == input.dtype
        and (bias is None or bias.dtype == input.dtype)
        and match.args[6] is False
        and match.args[8] == 1
    )


def should_fuse_bmm(match):
    input = match.args[0].meta["val"]
    if (
        not input.is_mlu
        or not torch.is_floating_point(input)
        or input.dtype == torch.double
    ):
        return False

    if not statically_known_true(match.output_node().meta["val"].numel() > 0):
        return False

    if len(match.args) > 1 and hasattr(match.args[1], "meta"):
        weight = match.args[1].meta.get("val", None)
        if isinstance(weight, torch.Tensor):
            weight_kn_bytes = (
                weight.shape[-2] * weight.shape[-1] * weight.element_size()
            )
            if config.enable_triton_fusion and guard_size_oblivious(
                weight_kn_bytes <= BMM_TRITONFUSION_WEIGHT_LIMIT_BYTES
            ):
                return False

    if match.kwargs.get("bias", None) is not None:
        if not hasattr(match.kwargs["bias"], "meta") or not isinstance(
            match.kwargs["bias"].meta["val"], torch.Tensor
        ):
            return False
        bias = match.kwargs["bias"].meta["val"]
        return bias.dtype == input.dtype and bias.dim() <= 3

    return True


def _get_tensor_meta(node):
    if isinstance(node, torch.fx.Node) and isinstance(
        node.meta.get("val", None), torch.Tensor
    ):
        return node.meta["val"]
    return None


def _shape_arg_value(value):
    if isinstance(value, torch.fx.Node):
        return value.meta.get("val", value)
    return value


def _shape_matches(a, b):
    a = _shape_arg_value(a)
    b = _shape_arg_value(b)
    try:
        return statically_known_true(sym_eq(a, b))
    except (AssertionError, TypeError):
        return False


def _shape_sequence_matches(a, b):
    if len(a) != len(b):
        return False
    return all(_shape_matches(x, y) for x, y in zip(a, b))


def _check_layernorm_affine_param(param, normalized_shape, input_dtype):
    if param is None:
        return False

    param_meta = _get_tensor_meta(param)
    if param_meta is None:
        return False

    # TMO fused_layer_norm accepts gamma/beta with the same dtype as input for
    # this residual/bias path. Keep mixed-dtype native_layer_norm on aten.
    if param_meta.dtype != input_dtype:
        return False

    # The residual/bias path supports regular last-dimension affine params.
    # TMO's 2D gamma special case does not support residual or input bias.
    if param_meta.dim() != 1 or len(normalized_shape) != 1:
        return False

    return _shape_matches(param_meta.shape[0], normalized_shape[0])


def _is_full_layernorm_input(tensor_meta, output_meta):
    return (
        tensor_meta is not None
        and tensor_meta.dtype == output_meta.dtype
        and _shape_sequence_matches(tensor_meta.shape, output_meta.shape)
    )


def _is_layernorm_bias_addend(tensor_meta, normalized_shape, output_dtype):
    return (
        tensor_meta is not None
        and tensor_meta.dtype == output_dtype
        and tensor_meta.dim() == 1
        and len(normalized_shape) == 1
        and _shape_matches(tensor_meta.shape[0], normalized_shape[0])
    )


def _is_full_shape_layernorm_addend(tensor_meta, input_shape, output_dtype):
    return (
        tensor_meta is not None
        and tensor_meta.dtype == output_dtype
        and _shape_sequence_matches(tensor_meta.shape, input_shape)
    )


def _is_expandable_layernorm_residual(tensor_meta, input_shape, output_dtype):
    if (
        tensor_meta is None
        or tensor_meta.dtype != output_dtype
        or tensor_meta.dim() != len(input_shape)
    ):
        return False
    try:
        return is_expandable_to(tensor_meta.shape, input_shape)
    except (AssertionError, TypeError):
        return False


def _normalize_permute_dims(dims, rank):
    if isinstance(dims, torch.fx.Node):
        dims = dims.meta.get("val", dims)
    if not isinstance(dims, (list, tuple)) or len(dims) != rank:
        return None
    normalized = []
    for dim in dims:
        if not isinstance(dim, int):
            return None
        if dim < 0:
            dim += rank
        normalized.append(dim)
    return normalized


def _is_supported_layernorm_transpose_node(node, normalized_shape, output_meta):
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target != aten.permute.default
    ):
        return False

    input_meta = _get_tensor_meta(node.args[0])
    permute_meta = _get_tensor_meta(node)
    if (
        input_meta is None
        or permute_meta is None
        or output_meta is None
        or input_meta.dtype != output_meta.dtype
        or permute_meta.dtype != output_meta.dtype
        or input_meta.dim() != 3
        or permute_meta.dim() != 3
        or not _shape_sequence_matches(permute_meta.shape, output_meta.shape)
    ):
        return False

    # TMO fused_layer_norm accepts non-contiguous inputs, but this pass only
    # targets the currently known profitable 3D pre-transpose form:
    #   [T, B, C] -> [B, T, C]
    # Keeping C as the last dimension preserves the layernorm normalized dim and
    # removes the permute.contiguous() clone seen before add+layernorm. This is
    # a pass-level optimization boundary, not the full TMO operator capability.
    dims = _normalize_permute_dims(node.args[1], input_meta.dim())
    return (
        dims == [1, 0, 2]
        and len(normalized_shape) == 1
        and _shape_matches(permute_meta.shape[-1], normalized_shape[0])
        and _shape_matches(input_meta.shape[-1], normalized_shape[0])
    )


def _classify_layernorm_addend(addend, input_shape, normalized_shape, output_dtype):
    addend_meta = _get_tensor_meta(addend)
    if _is_layernorm_bias_addend(addend_meta, normalized_shape, output_dtype):
        return "bias"
    if _is_full_shape_layernorm_addend(addend_meta, input_shape, output_dtype):
        return "residual"
    return None


def _check_common_tmo_layernorm(match):
    output = match.output_node()
    output_meta = _get_tensor_meta(output)
    if (
        output_meta is None
        or not output_meta.is_mlu
        or not torch.is_floating_point(output_meta)
        or output_meta.dtype == torch.double
        or not statically_known_true(output_meta.numel() > 0)
    ):
        return None

    native_layernorm = output.args[0]
    normalized_shape = native_layernorm.args[1]
    if len(normalized_shape) != 1:
        return None

    weight = native_layernorm.args[2]
    bias = native_layernorm.args[3]
    if not (
        _check_layernorm_affine_param(weight, normalized_shape, output_meta.dtype)
        and _check_layernorm_affine_param(bias, normalized_shape, output_meta.dtype)
    ):
        return None

    return output_meta, native_layernorm, normalized_shape


def _classify_add_layernorm_inputs(match):
    """Return the TMO input/addend layout for a matched add + layernorm.

    The add operand can map to TMO in two different ways:
      - full-shape tensor: residual argument
      - one-dimensional C tensor: bias argument
    """

    output = match.output_node()
    output_meta = _get_tensor_meta(output)
    if output_meta is None:
        return None

    native_layernorm = output.args[0]
    normalized_shape = native_layernorm.args[1]
    add = native_layernorm.args[0]
    lhs = add.args[0]
    rhs = add.args[1]
    lhs_meta = _get_tensor_meta(lhs)
    rhs_meta = _get_tensor_meta(rhs)

    lhs_is_full = _is_full_layernorm_input(lhs_meta, output_meta)
    rhs_is_full = _is_full_layernorm_input(rhs_meta, output_meta)
    lhs_is_bias = _is_layernorm_bias_addend(
        lhs_meta, normalized_shape, output_meta.dtype
    )
    rhs_is_bias = _is_layernorm_bias_addend(
        rhs_meta, normalized_shape, output_meta.dtype
    )

    if lhs_is_full and rhs_is_full:
        return lhs, rhs, False
    if lhs_is_full and rhs_is_bias:
        return lhs, rhs, True
    if rhs_is_full and lhs_is_bias:
        return rhs, lhs, True
    return None


def _classify_tmo_layernorm_inputs(match):
    """Return input/addend layout for TMO fused_layer_norm.

    Priority:
      1. permute + add + layernorm, so TMO can consume a supported 3D
         transpose view directly and avoid the permute.contiguous() clone.
      2. plain add + layernorm fallback, preserving the existing behavior.
    """

    common = _check_common_tmo_layernorm(match)
    if common is None:
        return None
    output_meta, native_layernorm, normalized_shape = common

    add = native_layernorm.args[0]
    lhs = add.args[0]
    rhs = add.args[1]
    lhs_is_transpose = _is_supported_layernorm_transpose_node(
        lhs, normalized_shape, output_meta
    )
    rhs_is_transpose = _is_supported_layernorm_transpose_node(
        rhs, normalized_shape, output_meta
    )

    if lhs_is_transpose:
        input, addend = lhs, rhs
    elif rhs_is_transpose:
        input, addend = rhs, lhs
    else:
        classified = _classify_add_layernorm_inputs(match)
        if classified is None:
            return None
        input, addend, addend_is_bias = classified
        input_meta = _get_tensor_meta(input)
        if input_meta is None or not _shape_matches(
            input_meta.shape[-1], normalized_shape[0]
        ):
            return None
        return (
            input,
            addend,
            "bias" if addend_is_bias else "residual",
            True,
        )

    addend_kind = _classify_layernorm_addend(
        addend, output_meta.shape, normalized_shape, output_meta.dtype
    )
    if addend_kind is None:
        return None
    return input, addend, addend_kind, False


def _classify_add_permute_layernorm_inputs(match):
    """Return input/addend layout for layer_norm(permute(add(x, residual))).

    This handles:
      layer_norm((x + residual).permute(1, 0, 2), weight, bias)
    by rewriting it as:
      fused_layer_norm(
          x.permute(1, 0, 2),
          residual.permute(1, 0, 2) or expanded residual,
          weight,
          bias,
          None,
          eps,
          False,
      )

    The explicit permute views carry shape/stride information into TMO; no extra
    transpose metadata is required by the fused_layer_norm interface.
    """

    common = _check_common_tmo_layernorm(match)
    if common is None:
        return None
    output_meta, native_layernorm, normalized_shape = common

    permute = native_layernorm.args[0]
    if not _is_supported_layernorm_transpose_node(
        permute, normalized_shape, output_meta
    ):
        return None

    add = permute.args[0]
    source_meta = _get_tensor_meta(add)
    if source_meta is None:
        return None

    lhs = add.args[0]
    rhs = add.args[1]
    lhs_meta = _get_tensor_meta(lhs)
    rhs_meta = _get_tensor_meta(rhs)
    lhs_is_source = _is_full_shape_layernorm_addend(
        lhs_meta, source_meta.shape, output_meta.dtype
    )
    rhs_is_source = _is_full_shape_layernorm_addend(
        rhs_meta, source_meta.shape, output_meta.dtype
    )

    def classify_addend(addend, addend_meta):
        if _is_layernorm_bias_addend(addend_meta, normalized_shape, output_meta.dtype):
            return "bias"
        if _is_full_shape_layernorm_addend(
            addend_meta, source_meta.shape, output_meta.dtype
        ):
            return "residual"
        if _is_expandable_layernorm_residual(
            addend_meta, source_meta.shape, output_meta.dtype
        ):
            return "expanded_residual"
        return None

    if lhs_is_source:
        input, addend, addend_meta = lhs, rhs, rhs_meta
    elif rhs_is_source:
        input, addend, addend_meta = rhs, lhs, lhs_meta
    else:
        return None

    addend_kind = classify_addend(addend, addend_meta)
    if addend_kind is None:
        return None
    return input, addend, permute.args[1], addend_kind


def _classify_permute_layernorm_input(match):
    """Return input layout for layer_norm(permute(x)).

    This is the fallback 3D pre-transpose fusion. It does not care what
    produces x: add, batch_matmul, or any other op can feed the permute. More
    specific residual-aware patterns, such as add+permute+LN, are registered
    before this one so they can still fuse the add into TMO's residual argument.
    """

    common = _check_common_tmo_layernorm(match)
    if common is None:
        return None
    output_meta, native_layernorm, normalized_shape = common

    permute = native_layernorm.args[0]
    if not _is_supported_layernorm_transpose_node(
        permute, normalized_shape, output_meta
    ):
        return None

    return permute.args[0], permute.args[1]


def should_fuse_tmo_layernorm(match):
    """Check whether add + native_layer_norm can use TMO fused_layer_norm.

    TMO's residual layernorm path computes layer_norm(x + residual, gamma, beta)
    and its bias path computes layer_norm(x + bias, gamma, beta). This pass only
    handles the normal layernorm result, float MLU tensors, alpha=1 tensor adds.
    When one add operand is a supported 3D transpose, that branch has priority
    over the plain add+layernorm fallback.
    """

    return _classify_tmo_layernorm_inputs(match) is not None


def should_fuse_add_permute_layernorm(match):
    return _classify_add_permute_layernorm_inputs(match) is not None


def should_fuse_permute_layernorm(match):
    return _classify_permute_layernorm_input(match) is not None


def _tmo_fused_layernorm_replace(
    match, input, addend, weight, bias, eps, addend_kind, *, make_input_contiguous
):
    def repl(input, addend, weight, bias, eps, addend_kind, make_input_contiguous):
        import torch_mlu_ops

        weight = weight.contiguous() if weight is not None else None
        bias = bias.contiguous() if bias is not None else None
        input = input.contiguous() if make_input_contiguous else input
        residual = None
        input_bias = None
        if addend_kind == "bias":
            input_bias = addend.contiguous()
        else:
            residual = addend.contiguous() if make_input_contiguous else addend
        return torch_mlu_ops.fused_layer_norm(
            input,
            residual,
            weight,
            bias,
            input_bias,
            eps,
            False,
        )

    match.replace_by_example(
        repl,
        (
            input,
            addend,
            weight,
            bias,
            eps,
            addend_kind,
            make_input_contiguous,
        ),
    )


def _tmo_fused_layernorm_after_permute_replace(
    match, input, addend, weight, bias, eps, dims, addend_kind
):
    def repl(input, addend, weight, bias, eps, dims, addend_kind):
        import torch_mlu_ops

        weight = weight.contiguous() if weight is not None else None
        bias = bias.contiguous() if bias is not None else None
        input = input.permute(dims)
        residual = None
        input_bias = None
        if addend_kind == "bias":
            input_bias = addend.contiguous()
        else:
            residual = addend.permute(dims)
            if addend_kind == "expanded_residual":
                residual = residual.expand_as(input)
        return torch_mlu_ops.fused_layer_norm(
            input,
            residual,
            weight,
            bias,
            input_bias,
            eps,
            False,
        )

    match.replace_by_example(
        repl, (input, addend, weight, bias, eps, dims, addend_kind)
    )


def _tmo_fused_layernorm_permute_replace(match, input, weight, bias, eps, dims):
    def repl(input, weight, bias, eps, dims):
        import torch_mlu_ops

        weight = weight.contiguous() if weight is not None else None
        bias = bias.contiguous() if bias is not None else None
        input = input.permute(dims)
        return torch_mlu_ops.fused_layer_norm(
            input,
            None,
            weight,
            bias,
            None,
            eps,
            False,
        )

    match.replace_by_example(repl, (input, weight, bias, eps, dims))


@register_graph_pattern(
    CallFunction(
        operator.getitem,
        CallFunction(
            aten.native_layer_norm.default,
            CallFunction(
                aten.permute.default,
                CallFunction(
                    aten.add.Tensor,
                    Arg(),
                    Arg(),
                    alpha=1,
                    _users=MULTIPLE,
                ),
                Ignored(),
                _users=MULTIPLE,
            ),
            KeywordArg("normalized_shape"),
            KeywordArg("weight"),
            KeywordArg("bias"),
            KeywordArg("eps"),
            _users=MULTIPLE,
        ),
        0,
    ),
    extra_check=should_fuse_add_permute_layernorm,
    pass_dict=tmo_layernorm_pass,
)
def add_permute_layernorm(
    match: Match, input, residual, normalized_shape, weight, bias, eps
):
    # Replace:
    #   layer_norm((x + residual).permute(1, 0, 2), weight, bias)
    # with:
    #   torch_mlu_ops.fused_layer_norm(
    #       x.permute(1, 0, 2),
    #       residual.permute(1, 0, 2) or expanded residual,
    #       weight,
    #       bias,
    #       None,
    #       eps,
    #       False,
    #   )
    # This uses the existing fused_layer_norm signature. The permute views carry
    # the transpose shape/stride information, and the residual argument lets TMO
    # perform the add internally before layernorm.
    classified = _classify_add_permute_layernorm_inputs(match)
    assert classified is not None
    input, addend, dims, addend_kind = classified
    _tmo_fused_layernorm_after_permute_replace(
        match,
        input,
        addend,
        weight,
        bias,
        eps,
        dims,
        addend_kind,
    )


@register_graph_pattern(
    CallFunction(
        operator.getitem,
        CallFunction(
            aten.native_layer_norm.default,
            CallFunction(
                aten.permute.default,
                Arg(),
                Ignored(),
                _users=MULTIPLE,
            ),
            KeywordArg("normalized_shape"),
            KeywordArg("weight"),
            KeywordArg("bias"),
            KeywordArg("eps"),
            _users=MULTIPLE,
        ),
        0,
    ),
    extra_check=should_fuse_permute_layernorm,
    pass_dict=tmo_layernorm_pass,
)
def permute_layernorm(match: Match, input, normalized_shape, weight, bias, eps):
    # Replace:
    #   layer_norm(x.permute(1, 0, 2), weight, bias)
    # with:
    #   torch_mlu_ops.fused_layer_norm(
    #       x.permute(1, 0, 2),
    #       None,
    #       weight,
    #       bias,
    #       None,
    #       eps,
    #       False,
    #   )
    # This fallback pass only cares about the supported 3D pre-transpose view.
    # It does not inspect the producer of x, so it also covers graphs where an
    # earlier pass has folded add into batch_matmul(..., bias=...). The
    # residual-aware add+permute+LN pass is registered first and therefore gets
    # the chance to fuse an explicit aten.add into TMO's residual path.
    classified = _classify_permute_layernorm_input(match)
    assert classified is not None
    input, dims = classified
    _tmo_fused_layernorm_permute_replace(match, input, weight, bias, eps, dims)


@register_graph_pattern(
    CallFunction(
        operator.getitem,
        CallFunction(
            aten.native_layer_norm.default,
            CallFunction(
                aten.add.Tensor,
                Arg(),
                Arg(),
                alpha=1,
                _users=MULTIPLE,
            ),
            KeywordArg("normalized_shape"),
            KeywordArg("weight"),
            KeywordArg("bias"),
            KeywordArg("eps"),
            _users=MULTIPLE,
        ),
        0,
    ),
    extra_check=should_fuse_tmo_layernorm,
    pass_dict=tmo_layernorm_pass,
)
def add_layernorm(match: Match, input, residual, normalized_shape, weight, bias, eps):
    # Replace:
    #   layer_norm(add(lhs, rhs), weight, bias)
    # with:
    #   torch_mlu_ops.fused_layer_norm(
    #       input, residual/input_bias, weight, bias, ..., eps
    #   )
    # The classifier decides priority. If either add operand is a supported 3D
    # permute, it keeps that operand non-contiguous and lets TMO consume the
    # transpose view directly. Otherwise it falls back to the original add+LN
    # fusion and makes the input/residual contiguous.
    classified = _classify_tmo_layernorm_inputs(match)
    assert classified is not None
    input, addend, addend_kind, make_input_contiguous = classified
    _tmo_fused_layernorm_replace(
        match,
        input,
        addend,
        weight,
        bias,
        eps,
        addend_kind,
        make_input_contiguous=make_input_contiguous,
    )


def _to_float_scalar(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_addmm_args(match):
    args = match.args
    kwargs = match.kwargs

    input = kwargs.get("input", args[0] if len(args) > 0 else None)
    mat1 = kwargs.get("mat1", args[1] if len(args) > 1 else None)
    mat2 = kwargs.get("mat2", args[2] if len(args) > 2 else None)
    beta = kwargs.get("beta", args[3] if len(args) > 3 else 1)
    alpha = kwargs.get("alpha", args[4] if len(args) > 4 else 1)

    if input is None or mat1 is None or mat2 is None:
        return None
    beta = _to_float_scalar(beta)
    alpha = _to_float_scalar(alpha)
    if beta is None or alpha is None:
        return None
    return input, mat1, mat2, beta, alpha


def should_fuse_tmo_addmm(match):
    # Skip tritionfusion fused nodes.
    if any([x.meta.get(TRITONFUSION_KEEPALIVE_NAME) for x in match.nodes]):
        return False

    addmm_args = _get_addmm_args(match)
    if addmm_args is None:
        return False
    input, mat1, mat2, beta, alpha = addmm_args

    if (
        not isinstance(input, torch.fx.Node)
        or not isinstance(mat1, torch.fx.Node)
        or not isinstance(mat2, torch.fx.Node)
    ):
        return False

    input = input.meta["val"]
    mat1 = mat1.meta["val"]
    mat2 = mat2.meta["val"]
    output = match.output_node().meta["val"]

    if (
        not input.is_mlu
        or not mat1.is_mlu
        or not mat2.is_mlu
        or not torch.is_floating_point(input)
        or input.dtype == torch.double
        or mat1.dtype != input.dtype
        or mat2.dtype != input.dtype
        or input.dim() != 2
        or mat1.dim() != 2
        or mat2.dim() != 2
    ):
        return False

    if not statically_known_true(output.numel() > 0):
        return False

    if not statically_known_true(sym_eq(mat1.shape[-1], mat2.shape[-2])):
        return False

    if not statically_known_true(
        sym_eq(list(input.shape), [mat1.shape[0], mat2.shape[1]])
    ):
        return False

    return statically_known_true(sym_eq(list(input.shape), list(output.shape)))


@register_graph_pattern(
    CallFunction(
        aten.relu.default,
        CallFunctionVarArgs(aten.addmm.default),
    ),
    extra_check=should_fuse_tmo_addmm,
    pass_dict=tmo_addmm_pass,
)
def replace_addmm_relu_with_tmo_matmul(match: Match, *args, **kwargs):
    # Optional TMO dependency is checked before the pass is applied.
    import torch_mlu_ops

    def repl(input, mat1, mat2, beta, alpha, trans_b):
        # v1.8 TMO matmul computes addmm semantics, then applies relu.
        return torch_mlu_ops.matmul(
            mat1.contiguous(),
            mat2.contiguous(),
            None,
            input.contiguous(),
            "relu",
            alpha,
            beta,
            True,
            True,
            input.dtype,
            None,
            None,
            False,
            trans_b,
        )

    input, mat1, mat2, beta, alpha = _get_addmm_args(match)
    trans_b, mat2 = get_trans_input(mat2, [1, 0])
    match.replace_by_example(repl, (input, mat1, mat2, beta, alpha, trans_b))


@register_graph_pattern(
    CallFunction(
        aten.gelu.default,
        CallFunction(
            aten.bmm.default,
            Arg(),
            Arg(),
        ),
        approximate="none",
    ),
    extra_check=should_fuse_bmm,
    pass_dict=bmm_add_act_pass,
)
@register_graph_pattern(
    CallFunction(
        aten.add.Tensor,
        CallFunction(
            aten.bmm.default,
            Arg(),
            Arg(),
        ),
        KeywordArg("bias"),
        alpha=1,
    ),
    extra_check=should_fuse_bmm,
    pass_dict=bmm_add_act_pass,
)
@register_graph_pattern(
    CallFunction(
        aten.add.Tensor,
        KeywordArg("bias"),
        CallFunction(
            aten.bmm.default,
            Arg(),
            Arg(),
        ),
        alpha=1,
    ),
    extra_check=should_fuse_bmm,
    pass_dict=bmm_add_act_pass,
)
@register_graph_pattern(
    CallFunction(
        aten.gelu.default,
        CallFunction(
            aten.add.Tensor,
            CallFunction(
                aten.bmm.default,
                Arg(),
                Arg(),
            ),
            KeywordArg("bias"),
            alpha=1,
        ),
        approximate="none",
    ),
    extra_check=should_fuse_bmm,
    pass_dict=bmm_add_act_pass,
)
@register_graph_pattern(
    CallFunction(
        aten.gelu.default,
        CallFunction(
            aten.add.Tensor,
            KeywordArg("bias"),
            CallFunction(
                aten.bmm.default,
                Arg(),
                Arg(),
            ),
            alpha=1,
        ),
        approximate="none",
    ),
    extra_check=should_fuse_bmm,
    pass_dict=bmm_add_act_pass,
)
@register_graph_pattern(
    CallFunction(
        aten.gelu.default,
        CallFunction(
            aten.baddbmm.default,
            KeywordArg("bias"),
            Arg(),
            Arg(),
            beta=1,
            alpha=1,
        ),
        approximate="none",
    ),
    extra_check=should_fuse_bmm,
    pass_dict=bmm_add_act_pass,
)
def bmm_add_act(match: Match, input, weight, *, bias=None):
    import torch_mlu_ops

    def repl(input, weight, bias, trans_weight, act):
        b, m, _ = input.shape
        n = weight.shape[-1] if not trans_weight else weight.shape[-2]
        if bias is not None:
            if bias.shape == torch.Size([b, m, n]):
                residual = bias.contiguous()
                bias = None
                beta = 1.0
            elif bias.shape == torch.Size([b, 1, n]):
                bias = bias.contiguous()
                residual = None
                beta = 0.0
            elif is_expandable_to(bias.shape, [b, 1, n]):
                bias = bias.expand([b, 1, n]).contiguous()
                residual = None
                beta = 0.0
            else:
                residual = bias.expand([b, m, n]).contiguous()
                bias = None
                beta = 1.0
        else:
            residual = None
            beta = 0.0

        return torch_mlu_ops.batch_matmul(
            input.contiguous(),
            weight.contiguous(),
            residual,
            1.0,
            beta,
            1.0,
            1.0,
            False,
            trans_weight,
            None,
            bias,
            act,
            input.dtype,
        )

    out = match.output_node()
    trans_weight, weight = get_trans_input(weight, [0, 2, 1])
    match.replace_by_example(
        repl,
        (
            input,
            weight,
            bias,
            trans_weight,
            "gelu" if out.target is aten.gelu.default else "none",
        ),
    )


@register_graph_pattern(
    CallFunction(
        [
            aten.div,
            aten.divide,
            aten.true_divide,
        ],
        Arg(),
        CallFunctionVarArgs(
            [aten.sqrt, aten.pow],
            users=MULTIPLE,
        ),
    ),
    extra_check=_is_mlu_pattern,
    pass_dict=passes["aten_div_sqrt_replace"],
)
def replace_div_sqrt(match: Match, *args):
    div = match.output_node()
    if div.kwargs.get("rounding_mode", None) is not None:
        return
    sqrt = div.args[1]
    input, other = args[:2]
    pow_fns = [aten.pow]
    pow_fns.extend(getattr(aten.pow, overload) for overload in aten.pow.overloads())
    if sqrt.target in pow_fns and (
        not isinstance(other, torch.fx.Node)
        or not isinstance(other.meta.get("val", None), torch.Tensor)
        or not isinstance(sqrt.args[1], float)
        or abs(sqrt.args[1] - 0.5) >= 1e-9
    ):
        return
    graph = match.graph
    with graph.inserting_before(div):
        rsqrt = graph.call_function(aten.rsqrt.default, (other,))
        rsqrt.meta.update(sqrt.meta)
        mul = graph.call_function(aten.mul.Tensor, (input, rsqrt))
        mul.meta.update(div.meta)
        div.replace_all_uses_with(mul)
    graph.erase_node(div)
    if len(sqrt.users) == 0:
        graph.erase_node(sqrt)


@register_graph_pattern(
    CallFunction(
        aten.relu.default,
        CallFunction(
            aten.convolution.default,
            Arg(),
            Arg(),
            Arg(),
            Arg(),
            Arg(),
            Arg(),
            Arg(),
            Arg(),
            Arg(),
        ),
    ),
    extra_check=should_fuse_conv_relu,
    pass_dict=conv_relu_fusion_pass,
)
def conv_relu_fusion_pattern(
    match: Match,
    x,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    def repl(
        x, weight, bias, stride, padding, dilation, transposed, output_padding, groups
    ):
        # cnnlFusedOps: conv -> relu
        return torch.ops.torch_mlu.fused_convolution(
            input=x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            transposed=transposed,
            output_padding=output_padding,
            groups=groups,
            mode="relu",
            slope=0.0,
        )

    match.replace_by_example(
        repl,
        (
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ),
    )


def _is_mlu_true_div(match):
    div = match.output_node()
    if isinstance(div, torch.fx.Node) and isinstance(
        div.meta.get("val", None), torch.Tensor
    ):
        return div.meta["val"].is_mlu and div.kwargs.get("rounding_mode", None) is None
    return False


# exp(A)/exp(B) => exp(A-B)
@register_graph_pattern(
    CallFunction(
        [
            aten.div,
            aten.divide,
            aten.true_divide,
        ],
        CallFunction(
            aten.exp.default,
            Arg(),
        ),
        CallFunction(
            aten.exp.default,
            Arg(),
            _users=MULTIPLE,
        ),
    ),
    extra_check=_is_mlu_true_div,
    pass_dict=passes["div_exp_replace"],
)
@register_graph_pattern(
    CallFunction(
        [
            aten.div,
            aten.divide,
            aten.true_divide,
        ],
        CallFunction(
            aten.exp.default,
            Arg(),
            _users=MULTIPLE,
        ),
        CallFunction(
            aten.exp.default,
            Arg(),
        ),
    ),
    extra_check=_is_mlu_true_div,
    pass_dict=passes["div_exp_replace"],
)
def replace_exp_div_exp(match: Match, a, b):
    def repl(a, b):
        return (a - b).exp()

    a_dtype = a.meta["val"].dtype
    b_dtype = b.meta["val"].dtype
    supported_dtypes = [torch.bfloat16, torch.half, torch.float, torch.double]
    supported_dtypes.extend(_integer_dtypes)
    if a_dtype not in supported_dtypes or b_dtype not in supported_dtypes:
        return
    # Avoid overflow of integer calculation.
    if a_dtype in _integer_dtypes and b_dtype in _integer_dtypes:
        return
    # Keep the dtype of res not change, because exp(integer) output fp32, but exp(half - integer) still half.
    if (a_dtype in _integer_dtypes and b_dtype in [torch.bfloat16, torch.half]) or (
        b_dtype in _integer_dtypes and a_dtype in [torch.bfloat16, torch.half]
    ):
        return
    match.replace_by_example(repl, (a, b))
