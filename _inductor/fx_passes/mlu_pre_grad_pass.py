import functools

import torch

from torch_mlu._inductor import config


def _mlu_pre_grad_pass(prev_custom_pass, graph: torch.fx.graph.Graph):
    if prev_custom_pass is not None:
        prev_custom_pass(graph)

    if not torch.mlu.is_available():
        return

    from . import mlu_pre_grad_patterns

    if "div_sqrt_replace" not in config.skipped_fx_passes:
        mlu_pre_grad_patterns.div_sqrt_pass.apply(graph)


mlu_pre_grad_pass = functools.partial(
    _mlu_pre_grad_pass, torch._inductor.config.pre_grad_custom_pass
)
