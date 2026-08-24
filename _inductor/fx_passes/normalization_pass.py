import torch
from torch._inductor.pattern_matcher import (
    Arg,
    register_graph_pattern,
    PatternMatcherPass,
    CallFunction,
)

aten = torch.ops.aten

normalization_pass = PatternMatcherPass("mlu_normalization")


def _is_mlu_input(match):
    return match.args[0].meta["val"].is_mlu


@register_graph_pattern(
    CallFunction(
        aten.unbind.int,
        Arg(),
    ),
    extra_check=_is_mlu_input,
    pass_dict=normalization_pass,
)
def normalize_unbind(match, input):
    graph = match.graph
    node = match.output_node()
    with graph.inserting_before(node):
        new_node = graph.call_function(aten.unbind.int, (input, 0))
        new_node.meta.update(node.meta)
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
