import sys
import logging
from importlib import import_module
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

import torch

from torch.distributed.tensor.experimental._context_parallel._attention import (
    _templated_ring_attention,
    _templated_ring_attention_backward,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

aten = torch.ops.aten
logger = logging.getLogger(__name__)


def _scaled_dot_product_ring_fused_attention_overrideable(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    if attn_bias is not None:
        raise NotImplementedError("attn_bias is not supported yet")
    if return_debug_mask:
        raise NotImplementedError("return_debug_mask is not supported yet")

    seq_dim = 2
    group = mesh.get_group()
    return _templated_ring_attention(
        group,
        seq_dim,
        aten._scaled_dot_product_fused_attention_overrideable,
        query=query,
        key=key,
        value=value,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def _scaled_dot_product_ring_fused_attention_overrideable_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    grad_input_mask: Tuple[bool, ...],
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    seq_dim = 2
    group = mesh.get_group()
    return _templated_ring_attention_backward(
        group,
        seq_dim,
        aten._scaled_dot_product_fused_attention_overrideable_backward.default,
        grad_out=grad_out,
        grad_out_name="grad_out",
        query=query,
        key=key,
        value=value,
        attn_bias=bias,
        grad_input_mask=grad_input_mask,
        out=out,
        logsumexp=logsumexp,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
        dropout_p=dropout_p,
        is_causal=is_causal,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        scale=scale,
    )


def _sdpa_overrideable_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    logger.debug("Dispatching op_call: %s", op_info.schema)

    # sharding propagation
    # TODO: remove the context parallel strategy from the default propagation
    # rule. Either figure out how to dynamically enable it or just don't call
    # propagate.
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    assert not output_sharding.needs_redistribute, "inputs need to be redistributed"

    local_results = _scaled_dot_product_ring_fused_attention_overrideable(
        op_info.compute_mesh,
        *op_info.local_args,  # type: ignore[arg-type]
        **op_info.local_kwargs,  # type: ignore[arg-type]
    )
    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


def _sdpa_overrideable_backward_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # Redistribute grad_output tensor to the same placement as output tensor
    args = list(args)
    args = tuple(args)

    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    logger.debug("Dispatching op_call: %s", op_info.schema)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    assert not output_sharding.needs_redistribute, "inputs need to be redistributed"

    local_results = _scaled_dot_product_ring_fused_attention_overrideable_backward(
        op_info.compute_mesh,
        *op_info.local_args,  # type: ignore[arg-type]
        **op_info.local_kwargs,  # type: ignore[arg-type]
    )

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


def apply_context_parallel_patch():
    full_module_name = (
        f"torch.distributed.tensor.experimental._context_parallel._attention"
    )
    sys.modules[
        f"torch.distributed._tensor.experimental._context_parallel._attention"
    ] = import_module(full_module_name)
    from torch.distributed._tensor.experimental._context_parallel._attention import (
        custom_ops,
    )

    custom_ops[
        aten._scaled_dot_product_fused_attention_overrideable.default
    ] = _sdpa_overrideable_handler
    custom_ops[
        aten._scaled_dot_product_fused_attention_overrideable_backward.default
    ] = _sdpa_overrideable_backward_handler
