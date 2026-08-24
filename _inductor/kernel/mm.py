# mypy: allow-untyped-defs
import torch
from torch._inductor.lowering import register_lowering
from torch._inductor.kernel import mm
from torch._inductor.virtualized import V
from torch._inductor.kernel.mm_common import mm_grid
from torch._inductor.select_algorithm import (
    TritonTemplate,
    ExternKernelChoice,
    extern_kernels,
)
from ...utils import gorilla

aten = torch.ops.aten

aten = torch.ops.aten
if hasattr(extern_kernels, "bias_addmm"):
    del extern_kernels.bias_addmm

aten_bias_addmm = ExternKernelChoice(
    mm.bias_addmm, "at::addmm_out", op_overload=aten.addmm.out
)
patch = gorilla.Patch(mm, "aten_bias_addmm", aten_bias_addmm)
gorilla.apply(patch)

# _ = TritonTemplate.all_templates.pop("mm")
# mm_template = TritonTemplate(
#     name="mm",
#     grid=mm_grid,
#     source=r"""
# {{def_kernel("A", "B")}}
#     M = {{size("A", 0)}}
#     N = {{size("B", 1)}}
#     K = {{size("A", 1)}}
#     if M * N == 0:
#         # early exit due to zero-size input(s)
#         return
#     stride_am = {{stride("A", 0)}}
#     stride_ak = {{stride("A", 1)}}
#     stride_bk = {{stride("B", 0)}}
#     stride_bn = {{stride("B", 1)}}
#
#     # based on triton.ops.matmul
#     pid = tl.program_id(0)
#     grid_m = (M + BLOCK_M - 1) // BLOCK_M
#     grid_n = (N + BLOCK_N - 1) // BLOCK_N
#
#     # re-order program ID for better L2 performance
#     width = GROUP_M * grid_n
#     group_id = pid // width
#     group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
#     pid_m = group_id * GROUP_M + (pid % group_size)
#     pid_n = (pid % width) // (group_size)
#
#     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#
#     rk = tl.arange(0, BLOCK_K)
#     A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
#     B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
#
#     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
#     for k in range(K, 0, -BLOCK_K):
#         if EVEN_K:
#             a = tl.load(A, mask=(rm < M)[:, None], other=0.0)
#             b = tl.load(B, mask=(rn < N)[None, :], other=0.0)
#         else:
#             a = tl.load(A, mask=(rk[None, :] < k) & (rm < M)[:, None], other=0.)
#             b = tl.load(B, mask=(rk[:, None] < k) & (rn < N)[None, :], other=0.)
#         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
#         A += BLOCK_K * stride_ak
#         B += BLOCK_K * stride_bk
#
#     # rematerialize rm and rn to save registers
#     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#     idx_m = rm[:, None]
#     idx_n = rn[None, :]
#     mask = (idx_m < M) & (idx_n < N)
#
#     # inductor generates a suffix
#     {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
# """,
# )
# patch = gorilla.Patch(mm, "mm_template", mm_template)
# gorilla.apply(patch)


@register_lowering(aten.addmm, type_promotion_kind=None)
def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    """
    Lowering for autotuning aten.addmm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    if use_native_matmul(mat1, mat2):
        if beta == 0:
            arg1 = 0
        else:
            arg1 = lowerings[aten.mul](beta, inp)

        if alpha == 0:
            arg2 = 0
        else:
            arg2 = lowerings[aten.mul](alpha, lowerings[aten.mm](mat1, mat2))

        return lowerings[aten.add](arg1, arg2)

    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(mat1, mat2, inp, layout=layout)
    static_shape, is_nonzero = _is_static_problem(layout)
    name = "addmm"

    # Create MMKernelInputs for AddMM at the top
    kernel_inputs = MMKernelInputs(
        [inp_expanded, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
    )
    choices: list[ChoiceCaller] = []

    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten.addmm_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten.addmm: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )
    if (not is_nonzero) or (
        not (inductor_config.max_autotune or inductor_config.max_autotune_gemm)
    ):
        # TODO(coconutruben): combine this with the main flow of addmm through
        # a subgraph or something as inp vs inp_expanded causes some slight numeric
        # differences
        kernel_inputs = MMKernelInputs(
            [inp, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
        )
        choices.extend(
            V.choices.get_template_configs(
                kernel_inputs,
                [aten_addmm],
                name,
            )
        )
        return autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)

    # Modify by Cambricon: Don't expand bias for addmm in max-autotune mode, ref https://github.com/pytorch/pytorch/pull/179808.
    kernel_inputs_aten = MMKernelInputs(
        [inp, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
    )
    # end Modify by Cambricon

    # Collect all templates for unified call
    templates_to_use: list[Union[ExternKernelChoice, KernelTemplate]] = []
    if use_aten_gemm_kernels():
        # Modify by Cambricon: Don't expand bias for addmm in max-autotune mode, ref https://github.com/pytorch/pytorch/pull/179808.
        # templates_to_use.extend([aten_bias_addmm, aten_addmm])
        aten_templates: list[ExternKernelChoice | KernelTemplate] = [aten_addmm]
        if (
            inp.get_stride()[0] == 0
            and len(inp.get_size()) == 2
            and inductor_config.triton.autotune_cublasLt
            and not V.graph.cpp_wrapper  # bias_addmm only has a Python implementation
        ):
            aten_templates.append(aten_bias_addmm)
        choices.extend(
            V.choices.get_template_configs(kernel_inputs_aten, aten_templates, name)
        )
        # end Modify by Cambricon

    if is_nonzero and use_triton_template(layout, check_max_autotune=False):
        templates_to_use.append(mm_template)

        if use_triton_blackwell_tma_template(mat1, mat2, output_layout=layout):
            templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)
        elif use_triton_tma_template(mat1, mat2, output_layout=layout):
            templates_to_use.append(persistent_tma_mm_template)

        templates_to_use.append(addmm_contiguous_subgraph_template)

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(kernel_inputs, templates_to_use, name)
    )

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op(name)
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices,
            layout,
            # reorder here because CUTLASS expects (x, w, bias) but torch
            # is bias, x, w
            kernel_inputs.nodes(reorder=[1, 2, 0]),
            alpha=alpha,
            beta=beta,
        )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(
            choices,
            layout,
            # reorder here because CK expects (x, w, bias) but torch
            # is bias, x, w
            kernel_inputs.nodes(reorder=[1, 2, 0]),
            alpha=alpha,
            beta=beta,
            input_reorder=[2, 0, 1],
        )

    if use_cpp_gemm_template(layout, mat1, mat2):
        CppGemmTemplate.add_choices(
            choices,
            layout,
            kernel_inputs.nodes(),
            alpha=alpha,
            beta=beta,
            has_bias=True,
        )

    return autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)


patch = gorilla.Patch(mm, "tuned_addmm", tuned_addmm)
gorilla.apply(patch)
