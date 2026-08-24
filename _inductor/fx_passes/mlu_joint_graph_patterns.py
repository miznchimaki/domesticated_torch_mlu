from typing import Optional
import functools

import torch
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Ignored,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    fwd_only,
    joint_fwd_bwd,
    init_once_fakemode,
    MultiOutputPattern,
    filter_nodes,
    get_arg_value,
)
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq

from .pattern_matcher import mlu_gen_register_replacement

aten = torch.ops.aten
prims = torch.ops.prims

repeat_gather_pass = PatternMatcherPass("mlu_repeat2expand")


def should_replace_repeat_expand(match):
    gather_node = match.output_node()
    repeat_node = gather_node.args[2]
    shape = repeat_node.args[0].meta["val"].shape
    desired = repeat_node.meta["val"].shape
    if len(shape) > len(desired):
        return False
    for i in range(len(shape)):
        if not statically_known_true(
            shape[-i - 1] == desired[-i - 1]
        ) and not statically_known_true(shape[-i - 1] == 1):
            return False

    return True


@register_graph_pattern(
    CallFunction(
        aten.gather.default,
        Arg(),
        Arg(),
        CallFunction(
            aten.repeat.default,
            Arg(),
            Ignored(),
            _users=1,
        ),
    ),
    extra_check=should_replace_repeat_expand,
    pass_dict=repeat_gather_pass,
)
def repeat_to_expand(match: Match, input, dim, repeat_input):
    def repl(input, dim, repeat_input, expand_sizes):
        return aten.gather.default(input, dim, repeat_input.expand(expand_sizes))

    gather_node = match.output_node()
    repeat_node = gather_node.args[2]
    match.replace_by_example(
        repl, (input, dim, repeat_input, repeat_node.meta["val"].shape)
    )


repeat_node = CallFunction(
    aten.repeat.default, KeywordArg("index"), Ignored(), _users=2
)
gather_dim = KeywordArg("dim")
output_node = MultiOutputPattern(
    [
        CallFunction(aten.gather.default, KeywordArg("input"), gather_dim, repeat_node),
        CallFunction(
            aten.scatter_add.default,
            KeywordArg("full"),
            gather_dim,
            repeat_node,
            KeywordArg("tangents_1"),
        ),
    ]
)


@register_graph_pattern(
    output_node,
    extra_check=should_replace_repeat_expand,
    pass_dict=repeat_gather_pass,
)
def repeat_to_expand_training(match: Match, input, dim, index, full, tangents_1):
    def repl(input, dim, index, full, tangents_1, expand_sizes):
        expand_node = index.expand(expand_sizes)
        gather_node = aten.gather.default(input, dim, expand_node)
        scatter_add_node = aten.scatter_add.default(full, dim, expand_node, tangents_1)
        return (gather_node, scatter_add_node)

    gather_node, _ = match.output_nodes()
    repeat_node = gather_node.args[2]
    match.replace_by_example(
        repl, (input, dim, index, full, tangents_1, repeat_node.meta["val"].shape)
    )


replace_layernorm_infer_pass = PatternMatcherPass("mlu_fuse_layernorm_infer")
replace_layernorm_training_pass = PatternMatcherPass("mlu_fuse_layernorm_training")


def naive_layernorm_target_with_cast(inputs, weight, bias, eps):
    return torch.nn.functional.layer_norm(
        inputs, inputs.shape[-1:], weight=weight, bias=bias, eps=eps
    )


def naive_layernorm_target(inputs, weight, bias, eps):
    output = torch.nn.functional.layer_norm(
        inputs, inputs.shape[-1:], weight=weight, bias=bias, eps=eps
    )
    # Currently only output of float dtype is possible when inputs have different dtypes.
    if inputs.dtype != weight.dtype and inputs.dtype != torch.float:
        return output.to(weight.dtype)
    return output


def naive_layernorm_target_1(inputs, weight, bias, eps):
    output = torch.nn.functional.layer_norm(
        inputs,
        (inputs.shape[-1],),
        weight=weight.squeeze(),
        bias=bias.squeeze(),
        eps=eps,
    )
    # Currently only output of float dtype is possible when inputs have different dtypes.
    if inputs.dtype != weight.dtype and inputs.dtype != torch.float:
        return output.to(weight.dtype)
    return output


def naive_layernorm_pattern_with_cast(inputs, weight, bias, eps):
    input_dtype = inputs.dtype
    inputs = inputs.float()
    weight = weight.float()
    bias = bias.float()
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)
    normalized = (inputs - mean) / ((variance + eps) ** (0.5))
    outputs = weight * normalized + bias
    outputs = outputs.to(input_dtype)
    return outputs


def naive_layernorm_pattern(inputs, weight, bias, eps):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)
    normalized = (inputs - mean) / ((variance + eps) ** (0.5))
    outputs = weight * normalized + bias
    return outputs


# rsqrt variants
def naive_layernorm_pattern_rsqrt(inputs, weight, bias, eps):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)
    normalized = (inputs - mean) * torch.rsqrt(variance + eps)
    outputs = weight * normalized + bias
    return outputs


def naive_layernorm_pattern_with_cast_rsqrt(inputs, weight, bias, eps):
    input_dtype = inputs.dtype
    inputs = inputs.float()
    weight = weight.float()
    bias = bias.float()
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)
    normalized = (inputs - mean) * torch.rsqrt(variance + eps)
    outputs = weight * normalized + bias
    outputs = outputs.to(input_dtype)
    return outputs


def naive_layernorm_pattern_1(inputs, weight, bias, eps):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)
    inv = torch.rsqrt(variance + eps)
    inv = inv * weight
    bias_term = bias - mean * inv
    out = inputs * inv + bias_term
    return out


def _layernorm_params_check(
    match, *, pow_based=False, rsqrt_based=False, with_cast=False
):
    """Unified check for both pow/rsqrt-based layernorm patterns"""
    inputs = match.kwargs["inputs"].meta["val"]
    if (
        not torch.is_floating_point(inputs)
        or not inputs.is_mlu
        or len(inputs.shape) < 1
    ):
        return False
    weight = match.kwargs["weight"]
    bias = match.kwargs["bias"]
    if not isinstance(weight, torch.fx.Node) or not isinstance(bias, torch.fx.Node):
        return False
    weight = weight.meta["val"]
    bias = bias.meta["val"]
    eps = match.kwargs["eps"]
    if (
        not isinstance(weight, torch.Tensor)
        or not isinstance(bias, torch.Tensor)
        or not (weight.dtype == bias.dtype)
        or not (inputs.device == weight.device == bias.device)
    ):
        return False
    # Currently do not support cast output to double, because of triton 64bit limitation.
    if inputs.dtype == torch.double or weight.dtype == torch.double:
        return False
    # Supported dtypes of cnnlLayerNormForward_v2
    if not (
        inputs.dtype == weight.dtype
        or (inputs.dtype == torch.half and weight.dtype == torch.float)
        or (inputs.dtype == torch.bfloat16 and weight.dtype == torch.float)
        or (inputs.dtype == torch.float and weight.dtype == torch.half)
        or (inputs.dtype == torch.float and weight.dtype == torch.bfloat16)
    ):
        return False
    if not isinstance(eps, (float, int)) and not isinstance(
        eps.meta.get("val", None), (torch.SymInt, torch.SymFloat)
    ):
        return False
    normalized_shape = inputs.shape[-1:]
    if not statically_known_true(
        sym_eq(weight.shape, normalized_shape)
    ) or not statically_known_true(sym_eq(bias.shape, normalized_shape)):
        return False
    add_nodes = filter_nodes(match.nodes, aten.add.Tensor)
    mul_nodes = filter_nodes(match.nodes, aten.mul.Scalar)
    div_nodes = filter_nodes(match.nodes, aten.div.Scalar)
    mean_nodes = filter_nodes(match.nodes, aten.mean.dim)
    var_nodes = filter_nodes(match.nodes, aten.var.correction)
    view_nodes = filter_nodes(match.nodes, aten.view.default)
    expand_nodes = filter_nodes(match.nodes, aten.expand.default)
    pow_nodes = filter_nodes(match.nodes, aten.pow.Tensor_Scalar)
    sum_nodes = filter_nodes(match.nodes, aten.sum.dim_IntList)

    for add_node in add_nodes:
        alpha = add_node.kwargs.get("alpha", None)
        if alpha is not None and alpha != 1:
            return False
    if len(mul_nodes) == 2:
        mul_other = (
            mul_nodes[1].args[1].meta.get("val", None)
            if isinstance(mul_nodes[1].args[1], torch.fx.Node)
            else mul_nodes[1].args[1]
        )
        if not statically_known_true(mul_other == 2.0 / weight.numel()):
            return False
        mul_other = (
            mul_nodes[0].args[1].meta.get("val", None)
            if isinstance(mul_nodes[0].args[1], torch.fx.Node)
            else mul_nodes[0].args[1]
        )
        if not statically_known_true(
            mul_other == (-0.5 if rsqrt_based else (0.5 if pow_based else 2))
        ):
            return False
    for div_node in div_nodes:
        div_other = (
            div_node.args[1].meta.get("val", None)
            if isinstance(div_node.args[1], torch.fx.Node)
            else div_node.args[1]
        )
        if not statically_known_true(div_other == inputs.shape[-1]):
            return False
    for mean_node in mean_nodes:
        dtype = mean_node.kwargs.get("dtype", None)
        if mean_node.args[1] != [-1] or dtype is not None:
            return False
    for view_node in view_nodes:
        if not statically_known_true(
            sym_eq(view_node.meta["val"].shape, normalized_shape)
        ):
            return False
    for expand_node in expand_nodes:
        if not statically_known_true(
            sym_eq(expand_node.meta["val"].shape, inputs.shape)
        ):
            return False
    if len(sum_nodes) == 4:
        try:
            if (
                list(sum_nodes[2].args[1]) != list(range(len(inputs.shape) - 1))
                or list(sum_nodes[3].args[1]) != list(range(len(inputs.shape) - 1))
                or list(sum_nodes[0].args[1]) != [len(inputs.shape) - 1]
                or list(sum_nodes[1].args[1]) != [len(inputs.shape) - 1]
            ):
                return False
        except Exception:
            return False

    if len(var_nodes) == 0:
        return False
    if var_nodes[0].args[1] != [-1] or var_nodes[0].kwargs["correction"] != 0:
        return False

    if pow_based and (
        len(pow_nodes) == 0
        or not statically_known_true(pow_nodes[0].args[1] == 0.5)
        or (
            len(pow_nodes) > 1
            and not statically_known_true(pow_nodes[1].args[1] == -0.5)
        )
    ):
        return False

    if rsqrt_based and (
        len(pow_nodes) == 1 and not statically_known_true(pow_nodes[0].args[1] == 3)
    ):
        return False

    # Check cast dtypes in naive_layernorm_pattern_with_cast
    if with_cast:
        cast_nodes = filter_nodes(match.nodes, prims.convert_element_type.default)
        # Check inputs cast
        inputs_users = []
        for key in ["inputs", "weight", "bias", "tangents_1"]:
            if key in match.kwargs:
                inputs_users.extend(list(match.kwargs[key].users.keys()))
        for user in inputs_users:
            if user in cast_nodes and user.args[1] != torch.float:
                return False
        # Check outputs cast back
        out_nodes = match.output_nodes()
        if out_nodes[0].args[1] != inputs.dtype:
            return False
        if len(out_nodes) == 5:
            if out_nodes[1].args[1] != inputs.dtype:
                return False
            if out_nodes[2].args[1] != weight.dtype:
                return False
            if out_nodes[3].args[1] != bias.dtype:
                return False
    return True


def _layernorm_params_check_1(match):
    inputs = match.kwargs["inputs"].meta["val"]
    if (
        not torch.is_floating_point(inputs)
        or not inputs.is_mlu
        or len(inputs.shape) < 1
    ):
        return False
    weight = match.kwargs["weight"]
    bias = match.kwargs["bias"]
    if not isinstance(weight, torch.fx.Node) or not isinstance(bias, torch.fx.Node):
        return False
    weight = weight.meta["val"]
    bias = bias.meta["val"]
    eps = match.kwargs["eps"]
    if (
        not isinstance(weight, torch.Tensor)
        or not isinstance(bias, torch.Tensor)
        or not (weight.dtype == bias.dtype)
        or not (inputs.device == weight.device == bias.device)
    ):
        return False
    # Currently do not support cast output to double, because of triton 64bit limitation.
    if inputs.dtype == torch.double or weight.dtype == torch.double:
        return False
    # Supported dtypes of cnnlLayerNormForward_v2
    if not (
        inputs.dtype == weight.dtype
        or (inputs.dtype == torch.half and weight.dtype == torch.float)
        or (inputs.dtype == torch.bfloat16 and weight.dtype == torch.float)
        or (inputs.dtype == torch.float and weight.dtype == torch.half)
        or (inputs.dtype == torch.float and weight.dtype == torch.bfloat16)
    ):
        return False
    if not isinstance(eps, (float, int)) and not isinstance(
        eps.meta.get("val", None), (torch.SymInt, torch.SymFloat)
    ):
        return False
    normalized_shape = inputs.shape[-1:]
    if not statically_known_true(
        sym_eq(weight.squeeze().shape, normalized_shape)
    ) or not statically_known_true(sym_eq(bias.squeeze().shape, normalized_shape)):
        return False
    add_nodes = filter_nodes(match.nodes, aten.add.Tensor)
    sub_nodes = filter_nodes(match.nodes, aten.sub.Tensor)
    mean_nodes = filter_nodes(match.nodes, aten.mean.dim)
    var_nodes = filter_nodes(match.nodes, aten.var.correction)

    for add_node in add_nodes:
        alpha = add_node.kwargs.get("alpha", None)
        if alpha is not None and alpha != 1:
            return False
    for sub_node in sub_nodes:
        alpha = sub_node.kwargs.get("alpha", None)
        if alpha is not None and alpha != 1:
            return False
    for mean_node in mean_nodes:
        dtype = mean_node.kwargs.get("dtype", None)
        if mean_node.args[1] != [-1] or dtype is not None:
            return False
    if var_nodes[0].args[1] != [-1] or var_nodes[0].kwargs["correction"] != 0:
        return False

    return True


@init_once_fakemode
def fuse_layernorm_pattern_init(input_device: Optional[torch.device] = None):
    # Register pow-based patterns
    gen_inputs = lambda dtype0, dtype1: [
        torch.empty(16, 128, dtype=dtype0, device="mlu", requires_grad=True),
        torch.empty(128, dtype=dtype1, device="mlu", requires_grad=True),
        torch.empty(128, dtype=dtype1, device="mlu", requires_grad=True),
    ]
    scalar_arg = {"eps": 0.0001}
    inputs = gen_inputs(torch.half, torch.half)
    mlu_gen_register_replacement(
        "naive_layernorm_pattern_with_cast_training",
        naive_layernorm_pattern_with_cast,
        naive_layernorm_target_with_cast,
        inputs,
        joint_fwd_bwd,
        [replace_layernorm_training_pass],
        functools.partial(_layernorm_params_check, pow_based=True, with_cast=True),
        scalar_arg,
    )

    mlu_gen_register_replacement(
        "naive_layernorm_pattern_with_cast_inference",
        naive_layernorm_pattern_with_cast,
        naive_layernorm_target_with_cast,
        inputs,
        fwd_only,
        [replace_layernorm_infer_pass],
        functools.partial(_layernorm_params_check, pow_based=True, with_cast=True),
        scalar_arg,
    )

    mlu_gen_register_replacement(
        "naive_layernorm_pattern_training_1",
        naive_layernorm_pattern,
        naive_layernorm_target,
        inputs,
        joint_fwd_bwd,
        [replace_layernorm_training_pass],
        functools.partial(_layernorm_params_check, pow_based=True),
        scalar_arg,
    )

    mlu_gen_register_replacement(
        "naive_layernorm_pattern_inference",
        naive_layernorm_pattern,
        naive_layernorm_target,
        inputs,
        fwd_only,
        [replace_layernorm_infer_pass],
        functools.partial(_layernorm_params_check, pow_based=True),
        scalar_arg,
    )

    inputs_mix_dtypes = gen_inputs(torch.half, torch.float)
    mlu_gen_register_replacement(
        "naive_layernorm_pattern_training_2",
        naive_layernorm_pattern,
        naive_layernorm_target,
        inputs_mix_dtypes,
        joint_fwd_bwd,
        [replace_layernorm_training_pass],
        functools.partial(_layernorm_params_check, pow_based=True),
        scalar_arg,
    )

    # Register rsqrt-based patterns
    mlu_gen_register_replacement(
        "naive_layernorm_pattern_rsqrt_training",
        naive_layernorm_pattern_rsqrt,
        naive_layernorm_target,
        inputs,
        joint_fwd_bwd,
        [replace_layernorm_training_pass],
        functools.partial(_layernorm_params_check, rsqrt_based=True),
        scalar_arg,
    )

    mlu_gen_register_replacement(
        "naive_layernorm_pattern_rsqrt_inference",
        naive_layernorm_pattern_rsqrt,
        naive_layernorm_target,
        inputs,
        fwd_only,
        [replace_layernorm_infer_pass],
        functools.partial(_layernorm_params_check, rsqrt_based=True),
        scalar_arg,
    )

    mlu_gen_register_replacement(
        "naive_layernorm_pattern_with_cast_rsqrt_training",
        naive_layernorm_pattern_with_cast_rsqrt,
        naive_layernorm_target_with_cast,
        inputs,
        joint_fwd_bwd,
        [replace_layernorm_training_pass],
        functools.partial(_layernorm_params_check, rsqrt_based=True, with_cast=True),
        scalar_arg,
    )

    mlu_gen_register_replacement(
        "naive_layernorm_pattern_with_cast_rsqrt_inference",
        naive_layernorm_pattern_with_cast_rsqrt,
        naive_layernorm_target_with_cast,
        inputs,
        fwd_only,
        [replace_layernorm_infer_pass],
        functools.partial(_layernorm_params_check, rsqrt_based=True, with_cast=True),
        scalar_arg,
    )

    mlu_gen_register_replacement(
        "naive_layernorm_pattern_training_rsqrt_2",
        naive_layernorm_pattern_rsqrt,
        naive_layernorm_target,
        inputs_mix_dtypes,
        joint_fwd_bwd,
        [replace_layernorm_training_pass],
        functools.partial(_layernorm_params_check, rsqrt_based=True),
        scalar_arg,
    )

    gen_inputs_1 = lambda dtype0, dtype1: [
        torch.empty(16, 128, 1536, dtype=dtype0, device="mlu"),
        torch.empty(1, 1, 1536, dtype=dtype1, device="mlu"),
        torch.empty(1, 1, 1536, dtype=dtype1, device="mlu"),
    ]
    inputs = gen_inputs_1(torch.float, torch.bfloat16)
    mlu_gen_register_replacement(
        "naive_layernorm_pattern_inference_2",
        naive_layernorm_pattern_1,
        naive_layernorm_target_1,
        inputs,
        fwd_only,
        [replace_layernorm_infer_pass],
        _layernorm_params_check_1,
        scalar_arg,
    )


fuse_layernorm_pattern_init()
