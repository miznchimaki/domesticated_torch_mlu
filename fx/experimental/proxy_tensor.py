import functools
import importlib

import torch
from torch_mlu.utils import gorilla
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch_mlu.utils import gorilla


def load_op_from_str(op_str):
    op_name_fragmts = op_str.split(".")
    mod_name = op_name_fragmts[0]
    try:
        mod = importlib.import_module(mod_name)
        for attr_name in op_name_fragmts[1:]:
            if not hasattr(mod, attr_name):
                importlib.import_module(f".{attr_name}", mod.__name__)
            mod = getattr(mod, attr_name)
    except:
        raise ValueError(f"{op_str} cannot be found")
    return mod


@functools.cache
def get_emulate_precision_casts_ops_name():
    from torch_mlu.fx.experimental.proxy_tensor import load_op_from_str
    import torch_mlu._inductor.config as mlu_config

    ops_name = []
    if not mlu_config.emulate_precision_casts_ops:
        return ops_name

    for op in mlu_config.emulate_precision_casts_ops:
        op = load_op_from_str(op)
        if isinstance(op, torch._ops.OpOverloadPacket):
            for op_overload in op.overloads():
                op_overload = getattr(op, op_overload)
                ops_name.append(op_overload.name())
        elif isinstance(op, torch._ops.OpOverload):
            ops_name.append(op.name())

    return set(ops_name)


@gorilla.patch(torch.fx.experimental.proxy_tensor)
def _maybe_record_pointwise_barrier(
    func: object, proxy_mode: ProxyTorchDispatchMode
) -> None:
    """
    Records operators whose tensor outputs or inputs are fp16/bf16 so downstream pointwise code can
    emulate eager's rounding behavior when emulate_precision_casts is enabled.
    """
    if proxy_mode.decomp_layers or not proxy_mode.emulate_precision_casts:
        return

    if not isinstance(func, torch._ops.OpOverload):
        return

    last_node = next(iter(reversed(proxy_mode.tracer.graph.nodes)))
    # Modified by Cambricon
    from torch_mlu.fx.experimental.proxy_tensor import (
        get_emulate_precision_casts_ops_name,
    )

    emulate_precision_casts_ops = get_emulate_precision_casts_ops_name()
    if not isinstance(last_node.target, torch._ops.OpOverload):
        return

    if (
        emulate_precision_casts_ops
        and last_node.target.name() not in emulate_precision_casts_ops
    ):
        return

    # end Modified by Cambricon
    t = last_node.meta.get("val")
    low_pr_fp = (torch.bfloat16, torch.float16)

    output_low_precision = isinstance(t, torch.Tensor) and t.dtype in low_pr_fp

    if not output_low_precision:
        for input_node in last_node.all_input_nodes:
            val = input_node.meta.get("val") if hasattr(input_node, "meta") else None
            if isinstance(val, torch.Tensor) and val.dtype in low_pr_fp:
                output_low_precision = True
                break

    if not output_low_precision:
        return

    last_node.meta["low_precision_pointwise_barrier"] = True
