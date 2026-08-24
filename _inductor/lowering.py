import logging
import sympy

import torch
from torch._inductor.lowering import (
    FALLBACK_ALLOW_LIST,
    register_lowering,
    require_contiguous,
    make_fallback,
    make_pointwise,
    make_reduction,
    fallback_handler,
    floor,
    trunc,
    div_mode,
    _validate_reduction_axis,
    prims,
    empty_like,
    to_dtype,
)
from torch._inductor.ir import Pointwise, TensorBox
from torch._inductor.virtualized import ops
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    is_boolean_dtype,
    is_integer_dtype,
)
from torch._inductor import ir
from torch._inductor.utils import register_op_dtype_propagation_rules
from torch.utils._ordered_set import OrderedSet
from . import lowering_utils
from . import config as inductor_config
from ..utils import gorilla


log = logging.getLogger(__name__)
# fix cudagraph tests
FALLBACK_ALLOW_LIST.add("aten::clone")
make_fallback(torch.ops.torch_mlu.grouped_gemm, require_contiguous)


def mlu_register_lowering(
    aten_fn,
    broadcast=False,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
):
    lowering_utils.delete_lowerings(aten_fn)
    return register_lowering(
        aten_fn, broadcast, type_promotion_kind, convert_input_to_bool
    )


lowering_utils.remove_register_lowering(lowering_utils.remove_list)


# register for libdevice
# Currently, libdevice gelu only supports approximate='tanh'
@mlu_register_lowering(torch.ops.aten.gelu)
def gelu(x, approximate: str = "none"):
    mlu_device = x.get_device().type == "mlu"
    if (
        (not mlu_device)
        or (not inductor_config.use_ultra_math)
        or (not inductor_config.use_ultra_gelu)
    ):
        return fallback_handler(torch.ops.aten.gelu.default, add_to_fallback_set=False)(
            x, approximate=approximate
        )

    def _gelu(x):
        return ops.gelu(x)

    return make_pointwise(_gelu)(x)


@make_pointwise
def div(a, b):
    return ops.div(a, b)


@mlu_register_lowering(torch.ops.aten.div, broadcast=True)
def mlu_div_mode(a, b, *, rounding_mode=None):
    mlu_device = lowering_utils.is_mlu_device(a) or lowering_utils.is_mlu_device(b)
    mlu_support_float = lowering_utils.is_mlu_float_type(
        a
    ) and lowering_utils.is_mlu_float_type(b)
    if not mlu_device or (not mlu_support_float) or (not inductor_config.use_fast_div):
        return div_mode(a, b, rounding_mode=rounding_mode)

    register_op_dtype_propagation_rules(
        "div",
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        override_return_dtype=None,
    )

    if rounding_mode == "floor":
        return floor(div(a, b))
    if rounding_mode == "trunc":
        return trunc(div(a, b))

    return div(a, b)


@register_lowering([torch.ops.aten.sum, torch.ops.prims.sum])
def sum_(x, axis=None, keepdims=False, *, dtype=None):
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64

    size = x.get_size()
    axis_ = OrderedSet[int](_validate_reduction_axis(x, axis))
    reduced_size = 1
    has_symbol = False
    for i in range(len(size)):
        if i in axis_:
            if isinstance(size[i], sympy.Integer):
                reduced_size *= size[i]
            else:
                has_symbol = True
                break
    # fallback in the following cases:
    # - Multi-axis reduction
    # - Dynamic shape
    # - Large reduce dim_size
    # - Middle axis reduction
    if (
        len(axis_) > 1
        or has_symbol
        or reduced_size >= 8192
        or not (axis_ == OrderedSet([0]) or axis_ == OrderedSet([len(size) - 1]))
    ):
        return fallback_handler(
            torch.ops.aten.sum.dim_IntList, add_to_fallback_set=False
        )(x, axis, keepdim=keepdims, dtype=dtype)
    fn = make_reduction("sum", override_return_dtype=dtype)
    return fn(x, axis, keepdims, dtype=dtype)


patch = gorilla.Patch(torch._inductor.lowering, "sum_", sum_)
gorilla.apply(patch)


@register_lowering(torch.ops.aten.clone)
def clone(x, *, memory_format=None):
    # Modify by Cambricon: fallback clone(memory_format=torch.contiguous)
    if memory_format is not None:
        # TODO(jansel): memory format
        if isinstance(x, TensorBox) and isinstance(x.data, PermuteView):
            return x
        elif (
            isinstance(x, TensorBox)
            and isinstance(x.data, ir.StorageBox)
            and isinstance(x.data.data, Pointwise)
        ):
            return Pointwise.create(
                device=x.get_device(),
                dtype=x.get_dtype(),
                inner_fn=x.make_loader(),
                ranges=list(x.get_size()),
            )
        else:
            return fallback_handler(
                torch.ops.aten.clone.default, add_to_fallback_set=False
            )(x, memory_format=memory_format)
    # end Modify by Cambricon
    else:
        return Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=x.make_loader(),
            ranges=list(x.get_size()),
        )


patch = gorilla.Patch(torch._inductor.lowering, "clone", clone)
gorilla.apply(patch)


@register_lowering(prims.convert_element_type, type_promotion_kind=None)
def _convert_element_type(x: TensorBox, dtype: torch.dtype):
    # Add by CAMBRICON
    # when dtype of input is float64, fallback copy to prims.convert_element_type.default.
    if isinstance(x, TensorBox) and (
        dtype == torch.float64 or x.get_dtype() == torch.float64
    ):
        return fallback_handler(
            prims.convert_element_type.default, add_to_fallback_set=False
        )(x, dtype)
    # end Add by CAMBRICON
    if dtype.is_complex or x.get_dtype().is_complex:
        if x.get_size():
            # Decompose since aa aten fallback is more friendly for c++ codegen.
            # This decomposition doesn't work for empty tensor, which needs more investigation.
            dst = empty_like(x, dtype=dtype)
            ir.InplaceCopyFallback.create(dst, x)
            return dst
        else:
            return fallback_handler(
                prims.convert_element_type.default, add_to_fallback_set=False
            )(x, dtype)
    return to_dtype(x, dtype, copy=True)


patch = gorilla.Patch(
    torch._inductor.lowering, "_convert_element_type", _convert_element_type
)
gorilla.apply(patch)
