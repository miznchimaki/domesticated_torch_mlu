# mypy: allow-untyped-defs
import torch
from torch._inductor.kernel import mm_plus_mm
from torch._inductor.kernel.mm_common import mm_grid
from torch._inductor.select_algorithm import TritonTemplate
from ...utils import gorilla


_ = TritonTemplate.all_templates.pop("mm_plus_mm")
mm_plus_mm_template = TritonTemplate(
    name="mm_plus_mm",
    grid=mm_grid,
    debug=False,
    source=r"""
{{def_kernel("A", "B", "C", "D")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K1 = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    # K2 = {{size("C", 1)}}
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}
    stride_cm = {{stride("C", 0)}}
    stride_ck = {{stride("C", 1)}}
    stride_dk = {{stride("D", 0)}}
    stride_dn = {{stride("D", 1)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rk = tl.arange(0, BLOCK_K)
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    C = C + (rm[:, None] * stride_cm + rk[None, :] * stride_ck)
    D = D + (rk[:, None] * stride_dk + rn[None, :] * stride_dn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k1 in range(K1, 0, -BLOCK_K):
        # First matmul with A @ B
        if EVEN_K:
            a = tl.load(A, mask=(rm < M)[:, None], other=0.0)
            b = tl.load(B, mask=(rn < N)[None, :], other=0.0)
        else:
            a = tl.load(A, mask=(rk[None, :] < k1) & (rm < M)[:, None], other=0.)
            b = tl.load(B, mask=(rk[:, None] < k1) & (rn < N)[None, :], other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    for k2 in range(K1, 0, -BLOCK_K):

        # Second matmul with C @ D
        if EVEN_K:
            c = tl.load(C, mask=(rm < M)[:, None], other=0.0)
            d = tl.load(D, mask=(rn < N)[None, :], other=0.0)
        else:
            c = tl.load(C, mask=(rk[None, :] < k2) & (rm < M)[:, None], other=0.)
            d = tl.load(D, mask=(rk[:, None] < k2) & (rn < N)[None, :], other=0.)
        acc += tl.dot(c, d, allow_tf32=ALLOW_TF32)
        C += BLOCK_K * stride_ck
        D += BLOCK_K * stride_dk


    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)
patch = gorilla.Patch(mm_plus_mm, "mm_plus_mm_template", mm_plus_mm_template)
gorilla.apply(patch)
