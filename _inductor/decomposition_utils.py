import torch
from torch._decomp import remove_decompositions
from torch._inductor.lowering import lowerings, make_fallback
from torch._inductor.decomposition import decompositions
from .lowering_utils import get_all_overloads, orig_lowerings

aten = torch.ops.aten
mlu_decomps_to_exclude = [
    aten._log_softmax,
    aten._log_softmax_backward_data,
    aten.cat,
    aten._reshape_alias,
    aten._softmax,
    aten._safe_softmax,
    aten._softmax_backward_data,
    aten._upsample_bicubic2d_aa,
    aten.convolution_backward,
    aten.embedding,
    aten.embedding_dense_backward,
    aten.flip,
    aten.gelu_backward,
    aten.max_pool2d_with_indices,
    aten.masked_scatter,
    aten.native_batch_norm,
    aten.native_dropout,
    aten.native_dropout_backward,
    aten.native_group_norm,
    aten.native_layer_norm,
    aten.native_layer_norm_backward,
    aten.nll_loss_backward,
    aten.rand_like,
    aten.randn_like,
    aten.randint_like,
    aten.resize_as,
    aten.upsample_bicubic2d,
    aten.upsample_bilinear2d,
    aten.upsample_nearest2d,
    aten.reflection_pad2d,
    aten.reflection_pad2d_backward,
    aten._scaled_dot_product_fused_attention_overrideable,
    aten.native_batch_norm_backward,
    aten.hardtanh,
    aten.silu_backward,
    aten.mish_backward,
    aten.rms_norm,
    aten._fused_rms_norm,
    aten._fused_rms_norm_backward,
    # https://github.com/pytorch/pytorch/pull/167667
    # The community uses a fallback approach for `_weight_norm_interface_backward`
    # but the forward op is decomposed normally. However, the norms stored in the CNNL interface differ
    # from those in the community. We must use cnnl forward/backward at the same time,
    # so we cannot decompose the forward _weight_norm_interface either.
    aten._weight_norm_interface,
]
mlu_libdevice_to_exclude = [
    aten.gelu,
]

extra_decomp_deny_list = list()
extra_decomp_allow_list = list()

orig_decompositions = decompositions.copy()


def make_fallbacks(op_list=[]):
    for op in op_list:
        make_fallback(op, None, False)


def deny_aten_fn_decomposition():
    if not extra_decomp_deny_list:
        return
    remove_decompositions(decompositions, extra_decomp_deny_list)
    make_fallbacks(extra_decomp_deny_list)
    extra_decomp_deny_list.clear()


def allow_aten_fn_decomposition():
    if not extra_decomp_allow_list:
        return
    fn_overloads = get_all_overloads(extra_decomp_allow_list)
    for fn in fn_overloads:
        if func := orig_decompositions.get(fn, None):
            decompositions[fn] = func
        if handler := orig_lowerings.get(fn, None):
            lowerings[fn] = handler
    extra_decomp_allow_list.clear()


from .config import add_decomp_list, remove_decomp_list

for fn in add_decomp_list:
    if fn.startswith("aten."):
        op = getattr(aten, fn[5:], None)
    else:
        continue

    if op is not None and op not in mlu_decomps_to_exclude:
        mlu_decomps_to_exclude.append(op)
        extra_decomp_deny_list.append(op)

for fn in remove_decomp_list:
    if fn.startswith("aten."):
        op = getattr(aten, fn[5:], None)
    else:
        continue

    if op is not None and op in mlu_decomps_to_exclude:
        mlu_decomps_to_exclude.remove(op)
        extra_decomp_allow_list.append(op)


def _get_aten_fn_name(fn):
    if isinstance(fn, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        return fn.__str__()
    return f"{fn.__module__}.{fn.__name__}"


def get_decompositions_denylist():
    return [_get_aten_fn_name(fn) for fn in mlu_decomps_to_exclude]


def add_to_decompositions_denylist(aten_fns=[]):
    for fn in aten_fns:
        if fn not in mlu_decomps_to_exclude:
            extra_decomp_deny_list.append(fn)
            mlu_decomps_to_exclude.append(fn)
    allow_aten_fn_decomposition()
    deny_aten_fn_decomposition()


def remove_from_decompositions_denylist(aten_fns=[]):
    for fn in aten_fns:
        if fn in mlu_decomps_to_exclude:
            extra_decomp_allow_list.append(fn)
            mlu_decomps_to_exclude.remove(fn)
    allow_aten_fn_decomposition()
    deny_aten_fn_decomposition()
