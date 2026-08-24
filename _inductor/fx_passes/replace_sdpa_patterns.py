from typing import Optional

import torch
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    fwd_only,
    init_once_fakemode,
)

from .pattern_matcher import mlu_gen_register_replacement
import math

aten = torch.ops.aten
replace_sdpa_pass = PatternMatcherPass("mlu_use_tmo_fa")


def sdpa_replace_check(match):
    try:
        import torch_mlu_ops
    except ImportError:
        return False

    q = match.kwargs["query"].meta["val"]
    k = match.kwargs["key"].meta["val"]
    v = match.kwargs["value"].meta["val"]
    a = match.kwargs["attn_mask"].meta["val"]

    if not isinstance(q, torch.Tensor) or q.dim() != 4:
        return False
    if not isinstance(k, torch.Tensor) or k.dim() != 4:
        return False
    if not isinstance(v, torch.Tensor) or v.dim() != 4:
        return False
    if not isinstance(a, torch.Tensor) or a.dim() != 4:
        return False

    sca = match.kwargs["sca"]
    sca_value = 1 / math.sqrt(math.sqrt(q.size(-1)))
    if not math.isclose(sca, sca_value, abs_tol=1e-12):
        return False

    return True


def sdpa_pattern_mlu_1(query, key, value, attn_mask, sca):
    # when use PYTORCH_MLU_GEN_PATTERNS=1 to gen serialized_patterns, should rewrite view to reshape.
    # The default generator produces torch.ops.aten.mul.Tensor, which does not match.
    query = torch.ops.aten.mul.Scalar(query, sca)
    key = key.transpose(-2, -1)
    key = torch.ops.aten.mul.Scalar(key, sca)
    attn_weight = query @ key
    # Do not use the += inplace operator.
    attn_weight = attn_weight + attn_mask
    attn_weight = torch.ops.aten._safe_softmax.default(attn_weight, dim=-1)
    return attn_weight @ value


def sdpa_pattern_target(query, key, value, attn_mask, sca):
    import torch_mlu_ops as tmo

    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    B, S_q, H, D = q.shape
    _, S_k, _, _ = k.shape
    a = None
    if attn_mask is not None:
        attn_shape = (B, H, S_q, S_k)
        a = attn_mask.expand(attn_shape).contiguous().to(query.dtype)
    result = tmo.flash_attention(
        q,
        k,
        v,
        None,
        None,
        None,
        None,
        a,
        q.size(1),
        k.size(1),
        1.0 / math.sqrt(D),
        False,
        out_dtype=query.dtype,
    )
    result = result.permute([0, 2, 1, 3])
    return result


def sdpa_pattern_mlu_2(query, key, value, attn_mask, sca):
    # when use PYTORCH_MLU_GEN_PATTERNS=1 to gen serialized_patterns, should rewrite view to reshape.
    convert_element_type = torch.ops.prims.convert_element_type.default(
        query, torch.float32
    )
    mul = torch.ops.aten.mul.Scalar(convert_element_type, sca)
    view = torch.ops.aten.reshape.default(mul, [4800, 121, 128])
    convert_element_type_1 = torch.ops.prims.convert_element_type.default(
        key, torch.float32
    )
    permute = torch.ops.aten.permute.default(convert_element_type_1, [0, 1, 3, 2])
    mul_1 = torch.ops.aten.mul.Scalar(permute, sca)
    view_1 = torch.ops.aten.reshape.default(mul_1, [4800, 128, 121])
    bmm = torch.ops.aten.bmm.default(view, view_1)
    view_2 = torch.ops.aten.reshape.default(bmm, [400, 12, 121, 121])
    full_default_1 = torch.ops.aten.full.default(
        [],
        0.0,
        dtype=torch.bfloat16,
        layout=torch.strided,
        device=torch.device(type="mlu", index=0),
        pin_memory=False,
    )
    full_default = torch.ops.aten.full.default(
        [],
        -torch.inf,
        dtype=torch.bfloat16,
        layout=torch.strided,
        device=torch.device(type="mlu", index=0),
        pin_memory=False,
    )
    where = torch.ops.aten.where.self(attn_mask, full_default_1, full_default)
    add = torch.ops.aten.add.Tensor(view_2, where)
    _safe_softmax = torch.ops.aten._safe_softmax.default(add, -1)
    view_3 = torch.ops.aten.reshape.default(_safe_softmax, [4800, 121, 121])
    convert_element_type_2 = torch.ops.prims.convert_element_type.default(
        value, torch.float32
    )
    view_4 = torch.ops.aten.reshape.default(convert_element_type_2, [4800, 121, 128])
    bmm_1 = torch.ops.aten.bmm.default(view_3, view_4)
    view_5 = torch.ops.aten.reshape.default(bmm_1, [400, 12, 121, 128])
    convert_element_type_4 = torch.ops.prims.convert_element_type.default(
        view_5, torch.bfloat16
    )
    return convert_element_type_4


def sdpa_pattern_mlu_3(query, key, value, attn_mask, sca):
    # when use PYTORCH_MLU_GEN_PATTERNS=1 to gen serialized_patterns, should rewrite view to reshape.
    mul = torch.ops.aten.mul.Scalar(query, sca)
    view = torch.ops.aten.reshape.default(mul, [4800, 121, 128])
    permute = torch.ops.aten.permute.default(key, [0, 1, 3, 2])
    mul_1 = torch.ops.aten.mul.Scalar(permute, sca)
    view_1 = torch.ops.aten.reshape.default(mul_1, [4800, 128, 121])
    bmm = torch.ops.aten.bmm.default(view, view_1)
    view_2 = torch.ops.aten.reshape.default(bmm, [400, 12, 121, 121])
    full_default_1 = torch.ops.aten.full.default(
        [],
        0.0,
        dtype=torch.bfloat16,
        layout=torch.strided,
        device=torch.device(type="mlu", index=0),
        pin_memory=False,
    )
    full_default = torch.ops.aten.full.default(
        [],
        -torch.inf,
        dtype=torch.bfloat16,
        layout=torch.strided,
        device=torch.device(type="mlu", index=0),
        pin_memory=False,
    )
    where = torch.ops.aten.where.self(attn_mask, full_default_1, full_default)
    add = torch.ops.aten.add.Tensor(view_2, where)
    _safe_softmax = torch.ops.aten._safe_softmax.default(add, -1)
    view_3 = torch.ops.aten.reshape.default(_safe_softmax, [4800, 121, 121])
    view_4 = torch.ops.aten.reshape.default(value, [4800, 121, 128])
    bmm_1 = torch.ops.aten.bmm.default(view_3, view_4)
    view_5 = torch.ops.aten.reshape.default(bmm_1, [400, 12, 121, 128])
    return view_5


def sdpa_pattern_target_2(query, key, value, attn_mask, sca):
    import torch_mlu_ops as tmo

    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    B, S_q, H, D = q.shape
    _, S_k, _, _ = k.shape
    a = None
    if attn_mask is not None:
        attn_shape = (B, H, S_q, S_k)
        a = attn_mask.expand(attn_shape)
        a = a.logical_not()
        a = a.to(q.dtype)
        a = a * torch.finfo(q.dtype).min
        a = a.contiguous()
    result = tmo.flash_attention(
        q,
        k,
        v,
        None,
        None,
        None,
        None,
        a,
        q.size(1),
        k.size(1),
        1.0 / math.sqrt(D),
        False,
        out_dtype=query.dtype,
    )
    result = result.permute([0, 2, 1, 3])
    return result


@init_once_fakemode
def replace_sdpa_pattern_init(input_device: Optional[torch.device] = None):
    gen_inputs_1 = lambda dtype_1, dtype_2: [
        torch.empty(128, 4, 1024, 64, dtype=dtype_1, device="mlu", requires_grad=True),
        torch.empty(128, 4, 1024, 64, dtype=dtype_1, device="mlu", requires_grad=True),
        torch.empty(128, 4, 1024, 64, dtype=dtype_1, device="mlu", requires_grad=True),
        torch.empty(128, 4, 1, 1024, dtype=dtype_2, device="mlu", requires_grad=True),
    ]
    inputs = gen_inputs_1(torch.float, torch.float)
    mlu_gen_register_replacement(
        "sdpa_pattern_mlu_inference_1",
        sdpa_pattern_mlu_1,
        sdpa_pattern_target,
        inputs,
        fwd_only,
        [replace_sdpa_pass],
        sdpa_replace_check,
        {"sca": 1 / math.sqrt(math.sqrt(64))},
    )

    gen_inputs_2 = lambda dtype_1, dtype_2: [
        torch.empty(400, 12, 121, 128, dtype=dtype_1, device="mlu", requires_grad=True),
        torch.empty(400, 12, 121, 128, dtype=dtype_1, device="mlu", requires_grad=True),
        torch.empty(400, 12, 121, 128, dtype=dtype_1, device="mlu", requires_grad=True),
        torch.empty(400, 1, 121, 121, dtype=dtype_2, device="mlu"),
    ]
    inputs = gen_inputs_2(torch.bfloat16, torch.bool)
    mlu_gen_register_replacement(
        "sdpa_pattern_mlu_inference_2",
        sdpa_pattern_mlu_2,
        sdpa_pattern_target_2,
        inputs,
        fwd_only,
        [replace_sdpa_pass],
        sdpa_replace_check,
        {"sca": 1 / math.sqrt(math.sqrt(128))},
    )

    inputs = gen_inputs_2(torch.float, torch.bool)
    mlu_gen_register_replacement(
        "sdpa_pattern_mlu_inference_3",
        sdpa_pattern_mlu_3,
        sdpa_pattern_target_2,
        inputs,
        fwd_only,
        [replace_sdpa_pass],
        sdpa_replace_check,
        {"sca": 1 / math.sqrt(math.sqrt(128))},
    )


replace_sdpa_pattern_init()
