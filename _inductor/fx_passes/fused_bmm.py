import torch
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
    register_graph_pattern,
    PatternMatcherPass,
)

aten = torch.ops.aten

fused_bmm_pass = PatternMatcherPass("fused_bmm_pass")


ACTIVATION_SET = {
    torch.ops.aten.silu.default: "silu",
    torch.ops.aten.gelu.default: "gelu",
    torch.ops.aten.tanh.default: "tanh",
    torch.ops.aten.relu.default: "relu",
    torch.ops.aten.leaky_relu.default: "leakyrelu",
}

# When activation is in TRAINING_SKIP_ACTIVATIONS, we will skip fused_mm fusion in training to avoid extra recomputation overhead.
TRAINING_SKIP_ACTIVATIONS = {"tanh", "silu", "gelu", "leakyrelu"}


def should_skip_fused_bmm_in_training(activation, is_training):
    return is_training and activation in TRAINING_SKIP_ACTIVATIONS


def check_if_convert_dtype(inner):
    if not hasattr(inner, "target"):
        return False
    has_dtype_convert = inner.target == torch.ops.prims.convert_element_type.default
    has_bmm = (
        len(inner.args) > 0
        and hasattr(inner.args[0], "target")
        and inner.args[0].target
        in [aten.baddbmm.default, aten.bmm.default, aten.add.Tensor]
    )
    return has_dtype_convert and has_bmm


def get_unmatched_fused_bmm_nodes(inner):
    nodes = [inner]
    bmm_op_node = inner.args[0] if check_if_convert_dtype(inner) else inner
    if bmm_op_node is not inner:
        nodes.append(bmm_op_node)
    if bmm_op_node.target == aten.add.Tensor:
        nodes.extend(
            node
            for node in bmm_op_node.args[:2]
            if hasattr(node, "target") and node.target == aten.bmm.default
        )
    return nodes


def erase_unmatched_nodes(graph, nodes):
    for node in nodes:
        if not node._erased and not node.users:
            graph.erase_node(node)


def check_fused_bmm_bias_act(match):
    act_node = match.output_node()
    inner = act_node.args[0]
    bmm_op_node = inner.args[0] if check_if_convert_dtype(inner) else inner

    if not hasattr(bmm_op_node, "target"):
        return False

    if bmm_op_node.target == aten.baddbmm.default:
        return True
    elif bmm_op_node.target == aten.bmm.default:
        return True
    elif bmm_op_node.target == aten.add.Tensor:
        node1, node2 = bmm_op_node.args[0], bmm_op_node.args[1]
        return (hasattr(node1, "target") and node1.target == aten.bmm.default) or (
            hasattr(node2, "target") and node2.target == aten.bmm.default
        )
    return False


def _handle_fused_bmm_replacement(match, inner, activation, activation_param):
    """
    Parse the input parameters required by fused_bmm and return mat1, mat2, and bias.
    """

    if_convert_dtype = check_if_convert_dtype(inner)
    if if_convert_dtype:
        bmm_op_node = inner.args[0]
    else:
        bmm_op_node = inner

    if bmm_op_node.target == aten.baddbmm.default:
        bias = bmm_op_node.args[0]
        mat1 = bmm_op_node.args[1]
        mat2 = bmm_op_node.args[2]
        beta_val = (
            bmm_op_node.kwargs.get("beta")
            if isinstance(bmm_op_node.kwargs, dict)
            else None
        )
        alpha_val = (
            bmm_op_node.kwargs.get("alpha")
            if isinstance(bmm_op_node.kwargs, dict)
            else None
        )
        if beta_val is None and len(bmm_op_node.args) > 3:
            beta_val = bmm_op_node.args[3]
        if alpha_val is None and len(bmm_op_node.args) > 4:
            alpha_val = bmm_op_node.args[4]

        if (beta_val is None or beta_val == 1) and (
            alpha_val is None or alpha_val == 1
        ):
            return mat1, mat2, bias

    elif bmm_op_node.target == aten.bmm.default:
        return bmm_op_node.args[0], bmm_op_node.args[1], None

    elif bmm_op_node.target == aten.add.Tensor:
        node1, node2 = bmm_op_node.args[0], bmm_op_node.args[1]
        if node1.target == aten.bmm.default:
            return node1.args[0], node1.args[1], node2
        elif node2.target == aten.bmm.default:
            return node2.args[0], node2.args[1], node1

    return None


def bias_type_and_shape_check(bias, mat1, mat2):
    if bias is None:
        return True

    bias_val = bias.meta.get("val", None)
    if not isinstance(bias_val, torch.Tensor):
        return False

    def get_shape(tensor):
        if (
            not hasattr(tensor, "meta")
            or "val" not in tensor.meta
            or tensor.meta["val"] is None
        ):
            return None
        return tensor.meta["val"].size()

    mat1_shape = get_shape(mat1)
    mat2_shape = get_shape(mat2)
    bias_shape = get_shape(bias)

    if mat1_shape is None or mat2_shape is None or bias_shape is None:
        return False

    Bs = mat1_shape[0]
    N = mat2_shape[2]
    dim = len(bias_shape)

    if dim == 0:
        return True
    if dim == 1:
        return bias_shape[0] == N or bias_shape[0] == 1
    if dim == 2:
        return bias_shape[0] == 1 and (bias_shape[1] == N or bias_shape[1] == 1)
    if dim == 3:
        b0, b1, b2 = bias_shape
        if b1 != 1:
            return False
        if b0 != 1 and b0 != Bs:
            return False
        if b2 != N and b2 != 1:
            return False
        return True
    return False


def get_dtype(tensor):
    if (
        not hasattr(tensor, "meta")
        or "val" not in tensor.meta
        or tensor.meta["val"] is None
    ):
        return None
    return tensor.meta["val"].dtype


@register_graph_pattern(
    CallFunction(aten.relu.default, Arg()),
    pass_dict=fused_bmm_pass,
    extra_check=check_fused_bmm_bias_act,
)
@register_graph_pattern(
    CallFunction(aten.leaky_relu.default, Arg()),
    pass_dict=fused_bmm_pass,
    extra_check=check_fused_bmm_bias_act,
)
@register_graph_pattern(
    CallFunction(aten.leaky_relu.default, Arg(), Arg()),
    pass_dict=fused_bmm_pass,
    extra_check=check_fused_bmm_bias_act,
)
@register_graph_pattern(
    CallFunction(aten.tanh.default, Arg()),
    pass_dict=fused_bmm_pass,
    extra_check=check_fused_bmm_bias_act,
)
@register_graph_pattern(
    CallFunction(aten.silu.default, Arg()),
    pass_dict=fused_bmm_pass,
    extra_check=check_fused_bmm_bias_act,
)
@register_graph_pattern(
    CallFunction(aten.gelu.default, Arg()),
    pass_dict=fused_bmm_pass,
    extra_check=check_fused_bmm_bias_act,
)
@register_graph_pattern(
    CallFunction(aten.gelu.default, Arg(), dtype=KeywordArg("approximate")),
    pass_dict=fused_bmm_pass,
    extra_check=check_fused_bmm_bias_act,
)
def fold_fused_bmm_bias_activation(match: Match, *args):
    """
    activation(bmm(mat1, mat2) + bias) / activation(bias + bmm(mat2, mat2)) / activation(baddbmm(bias, mat1, mat2)
           ↓↓↓
    torch_mlu.fused_bmm(mat1, mat2, activation, bias, kwargs_act)

    activation(bmm)
        ↓↓↓
    torch_mlu.fused_bmm(mat1, mat2, activation, None, kwargs_act)
    """
    graph = match.graph
    gm = graph.owning_module
    is_inference = gm.meta.get("is_inference", False) if gm is not None else False
    is_training = not is_inference

    act_node = match.output_node()
    inner = act_node.args[0]
    activation = ACTIVATION_SET[act_node.target]
    activation_param = None
    if should_skip_fused_bmm_in_training(activation, is_training):
        return
    if activation == "gelu":
        approximate = act_node.kwargs.get("approximate", None)
        activation_param = 1 if approximate == "tanh" else 0
    elif activation == "leakyrelu":
        if len(args) == 1:
            # leaky_relu default activation_param=0.01
            activation_param = 0.01
        elif len(args) > 1 and args[1] is not None:
            activation_param = args[1]
        else:
            activation_param = 0.01
    result = _handle_fused_bmm_replacement(match, inner, activation, activation_param)
    if result:
        mat1, mat2, bias = result
        if bias is not None:
            if bias_type_and_shape_check(bias, mat1, mat2):

                def repl(
                    mat1,
                    mat2,
                    bias,
                    activation,
                    activation_param,
                    is_training,
                    target_dtype,
                ):
                    convert_bias = torch.ops.prims.convert_element_type.default(
                        bias, target_dtype
                    )
                    convert_mat1 = torch.ops.prims.convert_element_type.default(
                        mat1, target_dtype
                    )
                    convert_mat2 = torch.ops.prims.convert_element_type.default(
                        mat2, target_dtype
                    )
                    return torch.ops.torch_mlu.fused_bmm(
                        convert_mat1,
                        convert_mat2,
                        activation,
                        convert_bias,
                        activation_param,
                        is_training,
                    )

                target_dtype = torch.promote_types(get_dtype(mat1), get_dtype(bias))
                match.replace_by_example(
                    repl,
                    (
                        mat1,
                        mat2,
                        bias,
                        activation,
                        activation_param,
                        is_training,
                        target_dtype,
                    ),
                )
                erase_unmatched_nodes(graph, get_unmatched_fused_bmm_nodes(inner))
            return
        else:

            def repl(mat1, mat2, bias, activation, activation_param, is_training):
                return torch.ops.torch_mlu.fused_bmm(
                    mat1, mat2, activation, bias, activation_param, is_training
                )

            match.replace_by_example(
                repl, (mat1, mat2, bias, activation, activation_param, is_training)
            )
            erase_unmatched_nodes(graph, get_unmatched_fused_bmm_nodes(inner))
            return


def check_fused_bmm_bias_silu(match):
    mul_node = match.output_node()

    if len(mul_node.args) != 2:
        return False

    arg0, arg1 = mul_node.args

    if hasattr(arg0, "target") and arg0.target == aten.sigmoid.default:
        sigmoid_node = arg0
        other = arg1
    elif hasattr(arg1, "target") and arg1.target == aten.sigmoid.default:
        sigmoid_node = arg1
        other = arg0
    else:
        return False

    if sigmoid_node.args[0] is not other:
        return False

    inner = other

    bmm_op_node = inner.args[0] if check_if_convert_dtype(inner) else inner

    if hasattr(bmm_op_node, "target") and bmm_op_node.target == aten.baddbmm.default:
        return True
    else:
        if hasattr(bmm_op_node, "target") and bmm_op_node.target == aten.bmm.default:
            return True
        elif hasattr(bmm_op_node, "target") and bmm_op_node.target == aten.add.Tensor:
            node1 = bmm_op_node.args[0]
            node2 = bmm_op_node.args[1]
            if (hasattr(node1, "target") and node1.target == aten.bmm.default) or (
                hasattr(node2, "target") and node2.target == aten.bmm.default
            ):
                return True
    return False


@register_graph_pattern(
    CallFunction(aten.mul, Arg(), CallFunction(aten.sigmoid, Arg())),
    pass_dict=fused_bmm_pass,
    extra_check=check_fused_bmm_bias_silu,
)
@register_graph_pattern(
    CallFunction(aten.mul, CallFunction(aten.sigmoid, Arg()), Arg()),
    pass_dict=fused_bmm_pass,
    extra_check=check_fused_bmm_bias_silu,
)
def fold_fused_bmm_bias_silu(match: Match, *args):
    """
    step1:
         x * sigmoid(x)  -->  silu(x)

    step2:
        silu(bmm(mat1, mat2) + bias) / silu(bias + bmm(mat2, mat2)) / silu(baddbmm(bias, mat1, mat2)
               ↓↓↓
        torch_mlu.fused_bmm(mat1, mat2, "silu", bias, kwargs_act)

        silu(bmm)
            ↓↓↓
        torch_mlu.fused_bmm(mat1, mat2, "silu", None, kwargs_act)
    """
    graph = match.graph
    gm = graph.owning_module
    is_inference = gm.meta.get("is_inference", False) if gm is not None else False
    is_training = not is_inference
    if should_skip_fused_bmm_in_training("silu", is_training):
        return

    mul_node = match.output_node()
    inner = (
        mul_node.args[0]
        if mul_node.args[0].target != aten.sigmoid
        else mul_node.args[1]
    )

    result = _handle_fused_bmm_replacement(match, inner, "silu", None)
    if result:
        mat1, mat2, bias = result
        if bias is not None:
            if bias_type_and_shape_check(bias, mat1, mat2):

                def repl(
                    mat1,
                    mat2,
                    bias,
                    is_training,
                    target_dtype,
                ):
                    convert_bias = torch.ops.prims.convert_element_type.default(
                        bias, target_dtype
                    )
                    convert_mat1 = torch.ops.prims.convert_element_type.default(
                        mat1, target_dtype
                    )
                    convert_mat2 = torch.ops.prims.convert_element_type.default(
                        mat2, target_dtype
                    )
                    return torch.ops.torch_mlu.fused_bmm(
                        convert_mat1,
                        convert_mat2,
                        "silu",
                        convert_bias,
                        is_training=is_training,
                    )

                target_dtype = torch.promote_types(get_dtype(mat1), get_dtype(bias))
                match.replace_by_example(
                    repl,
                    (
                        mat1,
                        mat2,
                        bias,
                        is_training,
                        target_dtype,
                    ),
                )
                erase_unmatched_nodes(graph, get_unmatched_fused_bmm_nodes(inner))
            return
        else:

            def repl(mat1, mat2, bias, is_training):
                return torch.ops.torch_mlu.fused_bmm(
                    mat1, mat2, "silu", bias, is_training=is_training
                )

            match.replace_by_example(repl, (mat1, mat2, bias, is_training))
            erase_unmatched_nodes(graph, get_unmatched_fused_bmm_nodes(inner))
            return


def check_fused_bmm_bias_leak_relu(match):
    where_node = match.output_node()
    condition = where_node.args[0]
    inner = where_node.args[1]
    other = where_node.args[2]

    if len(condition.args) == 0 or len(other.args) == 0:
        return False
    if_convert_dtype = check_if_convert_dtype(inner)
    if if_convert_dtype:
        bmm_op_node = inner.args[0]
    else:
        bmm_op_node = inner
    if (
        (condition.args[0] == inner and other.args[0] == inner)
        and (condition.args[1] == 0)
        and (hasattr(condition, "target") and condition.target == aten.gt.Scalar)
        and (
            hasattr(bmm_op_node, "target")
            and bmm_op_node.target == aten.baddbmm.default
        )
        and (hasattr(other, "target") and other.target == aten.mul.Tensor)
    ):
        return True
    elif (
        (condition.args[0] == inner and other.args[0] == inner)
        and (condition.args[1] == 0)
        and (hasattr(condition, "target") and condition.target == aten.gt.Scalar)
        and (hasattr(other, "target") and other.target == aten.mul.Tensor)
    ):
        if hasattr(bmm_op_node, "target") and bmm_op_node.target == aten.bmm.default:
            return True
        elif hasattr(bmm_op_node, "target") and bmm_op_node.target == aten.add.Tensor:
            node1 = bmm_op_node.args[0]
            node2 = bmm_op_node.args[1]
            if (hasattr(node1, "target") and node1.target == aten.bmm.default) or (
                hasattr(node2, "target") and node2.target == aten.bmm.default
            ):
                return True
    return False


@register_graph_pattern(
    CallFunction(aten.where.self, Arg(), Arg(), Arg()),
    pass_dict=fused_bmm_pass,
    extra_check=check_fused_bmm_bias_leak_relu,
)
def fold_fused_bmm_bias_leaky_relu(match: Match, condition, input, other):
    """
    leaky_relu might be broken down into gt+mul+where, where this pattern is matched.

    (leaky_relu(bmm(mat1, mat2) + bias) / leaky_relu(bias + bmm(mat2, mat2)) / leaky_relu(baddbmm(bias, mat1, mat2)
           ↓↓↓
    torch_mlu.fused_bmm(mat1, mat2, activation, bias, kwargs_act)

    """
    graph = match.graph
    gm = graph.owning_module
    is_inference = gm.meta.get("is_inference", False) if gm is not None else False
    is_training = not is_inference

    where_node = match.output_node()
    condition = where_node.args[0]
    inner = where_node.args[1]
    other = where_node.args[2]
    activation_param = other.args[1]
    activation = "leakyrelu"
    if should_skip_fused_bmm_in_training(activation, is_training):
        return

    if_convert_dtype = check_if_convert_dtype(inner)
    if if_convert_dtype:
        bmm_op_node = inner.args[0]
    else:
        bmm_op_node = inner

    if bmm_op_node.target == aten.baddbmm.default:
        bias = bmm_op_node.args[0]
        mat1 = bmm_op_node.args[1]
        mat2 = bmm_op_node.args[2]
        beta_val = None
        alpha_val = None
        if isinstance(bmm_op_node.kwargs, dict):
            if "beta" in bmm_op_node.kwargs:
                beta_val = bmm_op_node.kwargs["beta"]
            if "alpha" in bmm_op_node.kwargs:
                alpha_val = bmm_op_node.kwargs["alpha"]
        if beta_val is None:
            if len(bmm_op_node.args) > 3:
                beta_val = bmm_op_node.args[3]
        if alpha_val is None:
            if len(bmm_op_node.args) > 4:
                alpha_val = bmm_op_node.args[4]
        ok_beta = False
        ok_alpha = False
        if beta_val is None or beta_val == 1:
            ok_beta = True
        if alpha_val is None or alpha_val == 1:
            ok_alpha = True
        if ok_beta and ok_alpha:
            if bias_type_and_shape_check(bias, mat1, mat2):

                def repl(
                    mat1,
                    mat2,
                    bias,
                    activation,
                    activation_param,
                    is_training,
                    target_dtype,
                ):
                    convert_bias = torch.ops.prims.convert_element_type.default(
                        bias, target_dtype
                    )
                    convert_mat1 = torch.ops.prims.convert_element_type.default(
                        mat1, target_dtype
                    )
                    convert_mat2 = torch.ops.prims.convert_element_type.default(
                        mat2, target_dtype
                    )
                    return torch.ops.torch_mlu.fused_bmm(
                        convert_mat1,
                        convert_mat2,
                        activation,
                        convert_bias,
                        activation_param,
                        is_training,
                    )

                target_dtype = torch.promote_types(get_dtype(mat1), get_dtype(bias))
                match.replace_by_example(
                    repl,
                    (
                        mat1,
                        mat2,
                        bias,
                        activation,
                        activation_param,
                        is_training,
                        target_dtype,
                    ),
                )
                erase_unmatched_nodes(
                    graph,
                    [condition, other, *get_unmatched_fused_bmm_nodes(inner)],
                )
            return

    elif hasattr(bmm_op_node, "target") and bmm_op_node.target == aten.bmm.default:
        mat1 = bmm_op_node.args[0]
        mat2 = bmm_op_node.args[1]
        bias = None

        def repl(mat1, mat2, bias, activation, activation_param, is_training):
            return torch.ops.torch_mlu.fused_bmm(
                mat1, mat2, activation, bias, activation_param, is_training
            )

        match.replace_by_example(
            repl, (mat1, mat2, bias, activation, activation_param, is_training)
        )
        erase_unmatched_nodes(
            graph,
            [condition, other, *get_unmatched_fused_bmm_nodes(inner)],
        )
        return
    elif hasattr(bmm_op_node, "target") and bmm_op_node.target == aten.add.Tensor:
        node1 = bmm_op_node.args[0]
        node2 = bmm_op_node.args[1]
        if hasattr(node1, "target") and node1.target == aten.bmm.default:
            mat1 = node1.args[0]
            mat2 = node1.args[1]
            bias = node2
        elif hasattr(node2, "target") and node2.target == aten.bmm.default:
            mat1 = node2.args[0]
            mat2 = node2.args[1]
            bias = node1
        else:
            return
        if bias_type_and_shape_check(bias, mat1, mat2):

            def repl(
                mat1,
                mat2,
                bias,
                activation,
                activation_param,
                is_training,
                target_dtype,
            ):
                convert_bias = torch.ops.prims.convert_element_type.default(
                    bias, target_dtype
                )
                convert_mat1 = torch.ops.prims.convert_element_type.default(
                    mat1, target_dtype
                )
                convert_mat2 = torch.ops.prims.convert_element_type.default(
                    mat2, target_dtype
                )
                return torch.ops.torch_mlu.fused_bmm(
                    convert_mat1,
                    convert_mat2,
                    activation,
                    convert_bias,
                    activation_param,
                    is_training,
                )

            target_dtype = torch.promote_types(get_dtype(mat1), get_dtype(bias))
            match.replace_by_example(
                repl,
                (
                    mat1,
                    mat2,
                    bias,
                    activation,
                    activation_param,
                    is_training,
                    target_dtype,
                ),
            )
            erase_unmatched_nodes(
                graph,
                [condition, other, *get_unmatched_fused_bmm_nodes(inner)],
            )
        return


def _match_exp_plus_one(denom, x, if_convert):
    if not hasattr(denom, "target") or denom.target != aten.add.Tensor:
        return None

    a, b = denom.args[0], denom.args[1]
    # case: add(exp(-x), 1)
    if (
        hasattr(a, "target")
        and a.target == aten.exp.default
        and hasattr(a.args[0], "target")
        and a.args[0].target == aten.neg.default
    ):
        if (
            if_convert
            and hasattr(a.args[0].args[0], "target")
            and a.args[0].args[0].target == torch.ops.prims.convert_element_type.default
        ):
            return a.args[0].args[0].args[0] is x
        else:
            return a.args[0].args[0] is x

    # case: add(1, exp(-x))
    if (
        hasattr(b, "target")
        and b.target == aten.exp.default
        and hasattr(b.args[0], "target")
        and b.args[0].target == aten.neg.default
    ):
        if (
            if_convert
            and hasattr(b.args[0].args[0], "target")
            and b.args[0].args[0].target == torch.ops.prims.convert_element_type.default
        ):
            return b.args[0].args[0].args[0] is x
        else:
            return b.args[0].args[0] is x

    return False


def check_fused_bmm_bias_silu_1(match):
    out = match.output_node()

    if not hasattr(out, "target") or out.target != aten.div.Tensor:
        return False

    inner = out.args[0]
    if_convert = check_if_convert_dtype(inner)
    x = inner.args[0] if if_convert else inner
    denom = out.args[1]

    if not _match_exp_plus_one(denom, x, if_convert):
        return False

    # x is bmm / baddbmm / add(bmm, bias)
    if hasattr(x, "target"):
        if x.target == aten.baddbmm.default:
            return True
        if x.target == aten.bmm.default:
            return True
        if x.target == aten.add.Tensor:
            a, b = x.args
            if (hasattr(a, "target") and a.target == aten.bmm.default) or (
                hasattr(b, "target") and b.target == aten.bmm.default
            ):
                return True

    return False


@register_graph_pattern(
    CallFunction(aten.div.Tensor, Arg(), Arg()),
    pass_dict=fused_bmm_pass,
    extra_check=check_fused_bmm_bias_silu_1,
)
def fold_fused_bmm_bias_silu_1(match: Match, input, denom):
    """
    silu might be broken down into div(x, add(exp(-x), 1)), where this pattern is matched.

    silu(bmm(mat1, mat2)) / silu(bmm(mat1, mat2) + bias) / silu(bias + bmm(mat2, mat2)) / silu(baddbmm(bias, mat1, mat2))
           ↓↓↓
    torch_mlu.fused_bmm(mat1, mat2, "silu", bias, kwargs_act)

    """
    graph = match.graph
    gm = graph.owning_module
    is_inference = gm.meta.get("is_inference", False) if gm is not None else False
    is_training = not is_inference

    activation = "silu"
    if should_skip_fused_bmm_in_training(activation, is_training):
        return
    activation_param = None

    x = input.args[0] if check_if_convert_dtype(input) else input

    # case 1: baddbmm(bias, mat1, mat2)
    if x.target == aten.baddbmm.default:
        bias = x.args[0]
        mat1 = x.args[1]
        mat2 = x.args[2]

        if bias_type_and_shape_check(bias, mat1, mat2):

            def repl(
                mat1,
                mat2,
                bias,
                activation,
                activation_param,
                is_training,
                target_dtype,
            ):
                convert_bias = torch.ops.prims.convert_element_type.default(
                    bias, target_dtype
                )
                convert_mat1 = torch.ops.prims.convert_element_type.default(
                    mat1, target_dtype
                )
                convert_mat2 = torch.ops.prims.convert_element_type.default(
                    mat2, target_dtype
                )
                return torch.ops.torch_mlu.fused_bmm(
                    convert_mat1,
                    convert_mat2,
                    activation,
                    convert_bias,
                    activation_param,
                    is_training,
                )

            target_dtype = torch.promote_types(get_dtype(mat1), get_dtype(bias))
            match.replace_by_example(
                repl,
                (
                    mat1,
                    mat2,
                    bias,
                    activation,
                    activation_param,
                    is_training,
                    target_dtype,
                ),
            )
        return

    # case 2: bmm(mat1, mat2)
    if x.target == aten.bmm.default:
        mat1 = x.args[0]
        mat2 = x.args[1]
        bias = None

        def repl(mat1, mat2, bias, activation, activation_param, is_training):
            return torch.ops.torch_mlu.fused_bmm(
                mat1, mat2, activation, bias, activation_param, is_training
            )

        match.replace_by_example(
            repl, (mat1, mat2, bias, activation, activation_param, is_training)
        )
        return

    # case 3: add(bmm, bias)
    # case 4: add(bias, bmm)
    if x.target == aten.add.Tensor:
        a, b = x.args
        if hasattr(a, "target") and a.target == aten.bmm.default:
            mat1 = a.args[0]
            mat2 = a.args[1]
            bias = b
        elif hasattr(b, "target") and b.target == aten.bmm.default:
            mat1 = b.args[0]
            mat2 = b.args[1]
            bias = a
        else:
            return

        if bias_type_and_shape_check(bias, mat1, mat2):

            def repl(
                mat1,
                mat2,
                bias,
                activation,
                activation_param,
                is_training,
                target_dtype,
            ):
                convert_bias = torch.ops.prims.convert_element_type.default(
                    bias, target_dtype
                )
                convert_mat1 = torch.ops.prims.convert_element_type.default(
                    mat1, target_dtype
                )
                convert_mat2 = torch.ops.prims.convert_element_type.default(
                    mat2, target_dtype
                )
                return torch.ops.torch_mlu.fused_bmm(
                    convert_mat1,
                    convert_mat2,
                    activation,
                    convert_bias,
                    activation_param,
                    is_training,
                )

            target_dtype = torch.promote_types(get_dtype(mat1), get_dtype(bias))
            match.replace_by_example(
                repl,
                (
                    mat1,
                    mat2,
                    bias,
                    activation,
                    activation_param,
                    is_training,
                    target_dtype,
                ),
            )
        return
