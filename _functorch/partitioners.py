from typing import Callable

import torch
from torch.utils._ordered_set import OrderedSet
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch._functorch.partitioners import OpTypes, pointwise_ops

aten = torch.ops.aten

from ..utils import gorilla


@gorilla.patch(torch._functorch.partitioners)
def get_default_op_list() -> OpTypes:
    default_recomputable_ops: list[Callable[..., Any]] = [
        aten.add,
        aten.sub,
        aten.div,
        aten.atan2,
        aten.mul,
        aten.max,
        aten.min,
        aten.pow,
        aten.remainder,
        aten.fmod,
        aten.__and__,
        aten.__or__,
        aten.__xor__,
        aten.__lshift__,
        aten.__rshift__,
        aten.eq,
        aten.ne,
        aten.ge,
        aten.gt,
        aten.le,
        aten.lt,
        aten.abs,
        aten.bitwise_not,
        aten.ceil,
        aten.floor,
        aten.frac,
        aten.neg,
        aten.relu,
        aten.round,
        aten.silu,
        aten.trunc,
        aten.log,
        aten.log10,
        aten.log1p,
        aten.log2,
        aten.lgamma,
        aten.exp,
        aten.expm1,
        aten.erf,
        aten.erfc,
        aten.cos,
        aten.acos,
        aten.cosh,
        aten.sin,
        aten.asin,
        aten.sinh,
        aten.tan,
        aten.atan,
        aten.tanh,
        aten.atanh,
        aten.sqrt,
        aten.rsqrt,
        aten.reciprocal,
        aten.sigmoid,
        aten.softplus,
        aten.threshold,
        aten.threshold_backward,
        aten.clamp,
        aten.where,
        aten.lerp,
        aten.addcmul,
        aten.gelu,
        aten.gelu_backward,
        aten.sum,
        aten.mean,
        aten._grad_sum_to_size,
        aten.sum_to_size,
        aten.amax,
        aten.to,
        aten.type_as,
        operator.getitem,
        aten.squeeze,
        aten.unsqueeze,
        aten.rsub,
        aten._to_copy,
    ]  # noqa: E501,B950
    recomputable_view_ops = [aten.squeeze, aten.unsqueeze, aten.alias]
    recomputable_view_ops += [
        aten.view,
        aten.slice,
        aten.t,
        prims.broadcast_in_dim,
        aten.expand,
        aten.as_strided,
        aten.permute,
        aten.select,
        aten.split,
    ]
    view_ops = recomputable_view_ops
    default_recomputable_ops += [
        prims.div,
        prims.convert_element_type,
        aten.clone,
        aten._to_copy,
        aten.full_like,
        prims.var,
        prims.sum,
        aten.var,
        aten.std,
        prims.broadcast_in_dim,
        aten.select,
        aten._unsafe_view,
        aten.view,
        aten.expand,
        aten.slice,
        aten.reshape,
        aten.broadcast_tensors,
        aten.scalar_tensor,
        aten.ones,
        aten.new_zeros,
        aten.lift_fresh_copy,
        aten.arange,
        aten.triu,
        aten.var_mean,
        aten.isinf,
        aten.any,
        aten.full,
        aten.as_strided,
        aten.zeros,
        aten.empty,
        aten.empty_like,
        aten.argmax,
        aten.maximum,
        prims.iota,
        prims._low_memory_max_pool_offsets_to_indices,
    ]  # noqa: E501,B950
    # Natalia said that we should allow recomputing indexing :)
    default_recomputable_ops += [aten.index, aten.gather]
    default_recomputable_ops += view_ops

    default_recomputable_ops += pointwise_ops()

    default_recomputable_ops += [
        aten.zeros_like,
    ]

    default_recomputable_ops += [method_to_operator(m) for m in magic_methods]
    recomputable_ops = OrderedSet(default_recomputable_ops)

    random_ops = OrderedSet[Callable[..., Any]](
        [aten.native_dropout, aten.rand_like, aten.randn_like]
    )
    compute_intensive_ops = [
        aten.mm,
        aten.convolution,
        aten.convolution_backward,
        aten.bmm,
        aten.addmm,
        aten._scaled_dot_product_flash_attention,
        aten._scaled_dot_product_efficient_attention,
        aten._flash_attention_forward,
        aten._efficient_attention_forward,
        aten.upsample_bilinear2d,
        aten._scaled_mm,
    ]  # noqa: E501,B950

    fusible_ops = recomputable_ops | random_ops
    # Added by cambricon: exclude some ops from recomputable_ops list
    import os

    exclude_ops = [
        aten.tanh,
    ]

    use_mlu_recompute_ops = (
        os.environ.get("TORCH_MLU_USE_MLU_RECOMPUTE_OPS", "0") == "1"
    )
    if use_mlu_recompute_ops:
        for op in exclude_ops:
            recomputable_ops.remove(op)
    # end Added by cambricon

    return OpTypes(
        fusible_ops,
        OrderedSet(compute_intensive_ops),
        random_ops,
        OrderedSet(view_ops),
        recomputable_ops,
    )
