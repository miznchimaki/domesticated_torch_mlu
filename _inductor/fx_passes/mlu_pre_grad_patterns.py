import operator

import torch
from torch._inductor.pattern_matcher import (
    Arg,
    Match,
    CallMethod,
    CallFunction,
    CallFunctionVarArgs,
    CallMethodVarArgs,
    PatternMatcherPass,
    register_graph_pattern,
)

aten = torch.ops.aten


def _is_mlu_pattern(match):
    out = match.output_node()
    if isinstance(out, torch.fx.Node) and isinstance(
        out.meta.get("example_value", None), torch.Tensor
    ):
        return out.meta["example_value"].is_mlu
    return False


div_sqrt_pass = PatternMatcherPass("mlu_div_sqrt_replace")


@register_graph_pattern(
    CallFunction(
        [
            operator.truediv,
            torch.div,
            torch.divide,
            torch.true_divide,
            aten.div,
            aten.divide,
            aten.true_divide,
        ],
        Arg(),
        CallFunctionVarArgs(
            [operator.pow, torch.sqrt, torch.pow, aten.sqrt, aten.pow],
        ),
    ),
    extra_check=_is_mlu_pattern,
    pass_dict=div_sqrt_pass,
)
@register_graph_pattern(
    CallMethod(
        ["div", "true_divide"],
        Arg(),
        CallFunctionVarArgs(
            [operator.pow, torch.sqrt, torch.pow, aten.sqrt, aten.pow],
        ),
    ),
    extra_check=_is_mlu_pattern,
    pass_dict=div_sqrt_pass,
)
@register_graph_pattern(
    CallFunction(
        [
            operator.truediv,
            torch.div,
            torch.divide,
            torch.true_divide,
            aten.div,
            aten.divide,
            aten.true_divide,
        ],
        Arg(),
        CallMethodVarArgs(
            ["sqrt", "pow"],
        ),
    ),
    extra_check=_is_mlu_pattern,
    pass_dict=div_sqrt_pass,
)
@register_graph_pattern(
    CallMethod(
        ["div", "true_divide"],
        Arg(),
        CallMethodVarArgs(
            ["sqrt", "pow"],
        ),
    ),
    extra_check=_is_mlu_pattern,
    pass_dict=div_sqrt_pass,
)
def replace_div_sqrt(match: Match, *args):
    div = match.output_node()
    if (
        div.kwargs.get("rounding_mode", None) is not None
        or div.kwargs.get("out", None) is not None
    ):
        return
    sqrt = div.args[1]
    if sqrt.kwargs.get("out", None) is not None:
        return
    input, other = args[:2]
    pow_fns = [operator.pow, torch.pow, aten.pow]
    pow_fns.extend(getattr(aten.pow, overload) for overload in aten.pow.overloads())
    if (sqrt.target == "pow" or sqrt.target in pow_fns) and (
        not isinstance(other, torch.fx.Node)
        or not isinstance(other.meta.get("example_value", None), torch.Tensor)
        or not isinstance(sqrt.args[1], float)
        or abs(sqrt.args[1] - 0.5) >= 1e-9
    ):
        return
    graph = match.graph
    with graph.inserting_before(div):
        rsqrt = graph.call_function(torch.rsqrt, (other,))
        rsqrt.meta.update(sqrt.meta)
        mul = graph.call_function(torch.mul, (input, rsqrt))
        mul.meta.update(div.meta)
        div.replace_all_uses_with(mul)
    graph.erase_node(div)
    graph.erase_node(sqrt)
