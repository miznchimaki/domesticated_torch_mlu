import math
from typing import Tuple

import torch
import torch_mlu

from .utils import get_total_core_num


def _transform_bias_rescale_qkv_mlu(
    qkv: torch.Tensor,
    qkv_bias: torch.Tensor,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert num_heads != 0, f"num_heads must not be 0"
    assert qkv.dim() == 3
    B = qkv.size(0)
    T = qkv.size(1)
    _3D = qkv.size(2)
    D = int(_3D / 3)
    dim_per_head = int(D / num_heads)
    assert D % num_heads == 0
    assert _3D % 3 == 0
    if not qkv.is_contiguous():
        qkv = qkv.clone()
    if not qkv_bias.is_contiguous():
        qkv_bias = qkv_bias.clone()
    orig_dtype = qkv.dtype
    if orig_dtype == torch.float64:
        qkv = qkv.to(torch.float32)
        qkv_bias = qkv_bias.to(torch.float32)
    dtype = qkv.dtype
    out_q = torch.empty((B, num_heads, T, dim_per_head), device=qkv.device, dtype=dtype)
    out_k = torch.empty((B, num_heads, T, dim_per_head), device=qkv.device, dtype=dtype)
    out_v = torch.empty((B, num_heads, T, dim_per_head), device=qkv.device, dtype=dtype)
    # 500 series:
    #     _3D must be less than or equal to 18432 when dtype is fp32
    #     _3D must be less than or equal to 32760 when dtype is fp16
    # 300 series:
    #     _3D must be less than or equal to 27960 when dtype is fp32
    #     _3D must be less than or equal to 32760 when dtype is fp16
    if (dtype == torch.float32 and _3D > 18432) or (
        dtype == torch.float16 and _3D > 32760
    ):
        raise RuntimeError(
            "_transform_bias_rescale_qkv: MLU does not support qkv.size(2) exceed {} for {}.".format(
                _3D, dtype
            )
        )

    import triton
    from ._transform_bias_rescale_qkv_triton_kernel import (
        transform_bias_rescale_qkv_no_split_lowest_dim_impl,
    )

    if qkv.numel() > 0:
        inv_sqrt_dim_per_head = 1.0 / math.sqrt(dim_per_head)
        grid = lambda META: (
            min(
                triton.cdiv(B, META["YBLOCK"]) * triton.cdiv(T, META["XBLOCK"]),
                get_total_core_num(),
            ),
        )
        transform_bias_rescale_qkv_no_split_lowest_dim_impl[grid](
            qkv,
            qkv_bias,
            out_q,
            out_k,
            out_v,
            inv_sqrt_dim_per_head,
            B,
            T,
            D,
            num_heads,
            dim_per_head,
        )

    if dtype != orig_dtype:
        out_q = out_q.to(orig_dtype)
        out_k = out_k.to(orig_dtype)
        out_v = out_v.to(orig_dtype)
    return out_q, out_k, out_v
