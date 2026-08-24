import torch
import torch._inductor.inductor_prims
from torch._inductor.ir import TensorBox
from torch._inductor.lowering import (
    lowerings,
    fallbacks,
    make_fallback,
)

aten = torch.ops.aten
prims = torch.ops.prims

# Exclude below lowerings to get a better performance for now.
# The following blacklist will be gradually reduced to incorporate more operators.
remove_list = [
    # slice related.
    aten.glu,
    aten.slice_scatter,
    # pool related.
    aten._adaptive_avg_pool2d,
    aten.adaptive_max_pool2d,
    aten.avg_pool2d,
    aten.avg_pool2d_backward,
    aten.avg_pool3d,
    aten.avg_pool3d_backward,
    aten.fractional_max_pool2d,
    aten.max_pool2d_with_indices,
    aten.max_pool2d_with_indices_backward,
    prims._low_memory_max_pool_offsets_to_indices,
    prims._low_memory_max_pool_with_offsets,
    # conv related.
    aten._convolution,
    aten.convolution,
    aten.convolution_backward,
    torch.ops.torch_mlu.fused_convolution,
    # index put related.
    aten._unsafe_masked_index_put_accumulate,
    # reduction related.
    aten.any,
    aten.mean,
    aten.var,
    aten.var_mean,
    aten.sum,
    prims.sum,
    prims.var,
    aten._softmax,
    aten.cumsum,
    aten.logcumsumexp,
    aten.cumprod,
    # upsample related.
    aten._upsample_bicubic2d_aa,
    aten.upsample_bicubic2d,
    aten.upsample_bilinear2d,
    aten.upsample_nearest1d,
    aten.upsample_nearest2d,
    aten.upsample_nearest3d,
    aten.upsample_nearest1d_backward,
    aten.upsample_nearest2d_backward,
    aten.upsample_nearest3d_backward,
    # scatter related.
    aten.scatter,
    aten.scatter_,
    aten.scatter_add,
    aten.scatter_add_,
    aten.scatter_reduce,
    aten.scatter_reduce_,
    aten.select_scatter,
    # Uncategorized
    # aten.clone,
    aten.diagonal,
    aten.diagonal_scatter,
    aten.embedding,
    aten.gather,
    aten.reshape,
    aten.sort,
    aten.split,
    aten.unfold,
    aten.view,
    aten.index,
    aten.index_put,
    aten.index_put_,
    aten._unsafe_index,
    aten.expand_as,
    aten.constant_pad_nd,
    aten._unsafe_masked_index,
    aten.repeat,
    prims.iota,
    aten.amax,
    aten.amin,
    aten.argmax,
    aten.argmin,
    aten.any.dim,
    aten.unsqueeze,
    aten.max,
    aten.min,
]
extra_lowering_deny_list = []
extra_lowering_allow_list = []

orig_lowerings = lowerings.copy()


def get_all_overloads(aten_fn):
    if not isinstance(aten_fn, (list, tuple)):
        aten_fn = [aten_fn]
    else:
        aten_fn = list(aten_fn)

    for fn in list(aten_fn):
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                aten_fn.append(other_fn)
    return aten_fn


def delete_lowerings(aten_fn):
    if not isinstance(aten_fn, (list, tuple)):
        aten_fn = [aten_fn]
    else:
        aten_fn = list(aten_fn)

    for fn in list(aten_fn):
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                if other_fn in lowerings:
                    lowerings.pop(other_fn)
        elif isinstance(fn, torch._ops.OpOverload):
            if fn in lowerings:
                lowerings.pop(fn, None)

    # deleta all function related to aten_fn(like aten.reshape, aten.reshape.default)
    # above code only delete overloads func of aten_fn(like aten.reshape.default)
    # fn_overloads = get_all_overloads(aten_fn)
    # for fn in list(fn_overloads):
    #     if fn in lowerings:
    #         lowerings.pop(fn, None)


def remove_register_lowering(op_list=[]):
    for op in op_list:
        delete_lowerings(op)
        make_fallback(op, None, False)


def allow_aten_fn_lowering():
    if not extra_lowering_allow_list:
        return
    fn_overloads = get_all_overloads(extra_lowering_allow_list)
    for op in fn_overloads:
        if handler := orig_lowerings.get(op, None):
            lowerings[op] = handler
        if op in fallbacks:
            fallbacks.remove(op)
    extra_lowering_allow_list.clear()


def deny_aten_fn_lowering():
    if not extra_lowering_deny_list:
        return
    remove_register_lowering(extra_lowering_deny_list)
    extra_lowering_deny_list.clear()


from .config import add_lowering_list, remove_lowering_list

for fn in add_lowering_list:
    if fn.startswith("aten."):
        op = getattr(aten, fn[5:], None)
    elif fn.startswith("prims."):
        op = getattr(prims, fn[6:], None)
    else:
        continue

    if op is not None and op not in remove_list:
        remove_list.append(op)
        extra_lowering_deny_list.append(op)

for fn in remove_lowering_list:
    if fn.startswith("aten."):
        op = getattr(aten, fn[5:], None)
    elif fn.startswith("prims."):
        op = getattr(prims, fn[6:], None)
    else:
        continue

    if op is not None and op in remove_list:
        remove_list.remove(op)
        extra_lowering_allow_list.append(op)


def _get_aten_fn_name(fn):
    if isinstance(fn, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        return fn.__str__()
    return f"{fn.__module__}.{fn.__name__}"


def add_to_lowering_denylist(aten_fns=[]):
    for fn in aten_fns:
        if fn not in remove_list:
            remove_list.append(fn)
            extra_lowering_deny_list.append(fn)
    allow_aten_fn_lowering()
    deny_aten_fn_lowering()


def remove_from_lowering_denylist(aten_fns=[]):
    for fn in aten_fns:
        if fn in remove_list:
            remove_list.remove(fn)
            extra_lowering_allow_list.append(fn)
    allow_aten_fn_lowering()
    deny_aten_fn_lowering()


def get_lowering_denylist():
    return [_get_aten_fn_name(fn) for fn in remove_list]


def is_mlu_device(x):
    if isinstance(x, TensorBox):
        return x.data.get_device().type == "mlu"
    if isinstance(x, torch.Tensor):
        return x.get_device().type == "mlu"
    return False  # scalar or others


_ALLOWED_FLOAT_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


def is_mlu_float_type(x):
    if isinstance(x, TensorBox):
        try:
            dtype = x.get_dtype()
        except AttributeError:
            x = x.data
            if isinstance(x, torch.Tensor):
                dtype = x.dtype
            else:
                return False
        return dtype in _ALLOWED_FLOAT_DTYPES

    if isinstance(x, torch.Tensor):
        return x.dtype in _ALLOWED_FLOAT_DTYPES

    if isinstance(x, float):
        return True

    return False
