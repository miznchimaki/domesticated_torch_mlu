from typing import (
    no_type_check,
)
from collections.abc import Sequence
import sympy
import torch
from torch.utils._sympy.symbol import (
    prefix_str,
    SymT,
)
from torch._inductor import config, scheduler
from torch._inductor.scheduler import WhyNoFuse
from torch._inductor.ir import TritonTemplateBuffer
from torch._inductor.virtualized import V
from torch._inductor.utils import (
    cache_on_self,
)
from torch._inductor.codegen import simd
from torch._inductor.codegen.simd import (
    SIMDKernel,
    log,
)

from ...utils import gorilla


def codegen_combo_kernel(self, combo_kernel_node):
    subkernel_nodes = combo_kernel_node.get_subkernel_nodes()
    custom_part_algorithm = combo_kernel_node.use_custom_partition_algo
    enable_autotune = combo_kernel_node.enable_autotune
    mixed_sizes = config.combo_kernel_allow_mixed_sizes > 1 or (
        config.combo_kernel_allow_mixed_sizes == 1 and custom_part_algorithm
    )

    kernel_code_list = self.generate_combo_kernel_code(
        subkernel_nodes, custom_part_algorithm, enable_autotune, mixed_sizes
    )

    # Modify by CAMBRICON: kernel_name have fuse op
    # for src_code, kernel, _ in kernel_code_list:
    #     kernel_name = self.define_kernel(src_code, [combo_kernel_node], kernel)
    #     self.codegen_comment([combo_kernel_node])
    for src_code, kernel, node_group in kernel_code_list:
        kernel_name = self.define_kernel(src_code, node_group, kernel)
        if config.trace.enabled:
            set_kernel_post_grad_provenance_tracing(
                combo_kernel_node.snodes, kernel_name
            )
        self.codegen_comment(node_group)
        log.debug("ComboKernels: generated kernel %s.", kernel_name)
        kernel.call_kernel(V.graph.wrapper_code, kernel_name)

    self.free_buffers_in_scheduler()


patch = gorilla.Patch(
    torch._inductor.codegen.simd.SIMDScheduling,
    "codegen_combo_kernel",
    codegen_combo_kernel,
)
gorilla.apply(patch)


@classmethod
def create_partial_tiling(
    cls,
    tiling: Sequence[sympy.Expr],
    is_pointwise: bool,
) -> dict[str, sympy.Expr]:
    return cls.create_tiling(
        # Modified by Cambricon
        tiling if is_pointwise else tiling[:-1],
        # tiling if is_pointwise else [],
        tiling[-1:] if not is_pointwise else [],
        # tiling if not is_pointwise else [],
        # end Modify by Cambricon
    )


patch = gorilla.Patch(
    torch._inductor.codegen.simd.SIMDScheduling,
    "create_partial_tiling",
    create_partial_tiling,
)
gorilla.apply(patch)


def can_fuse(self, node1, node2):
    """
    Hook called by Scheduler to determine if the Triton backend
    can fuse node1 and node2.  These nodes might already be
    FusedSchedulerNodes.
    """
    if isinstance(node1, scheduler.ForeachKernelSchedulerNode) or isinstance(
        node2, scheduler.ForeachKernelSchedulerNode
    ):
        return scheduler.ForeachKernelSchedulerNode.can_fuse(node1, node2)

    _, (numel1, rnumel1) = node1.group
    _, (numel2, rnumel2) = node2.group
    why = WhyNoFuse(node1, node2)

    if node1.is_split_scan() and not node2.is_split_scan():
        if node2.is_reduction():
            why("Split scan cannot fuse with reductions")
    elif node2.is_split_scan() and not node1.is_split_scan():
        if node1.is_reduction():
            why("Split scan cannot fuse with reductions")

    if node1.is_reduction() and node2.is_reduction():
        reduction_can_fuse = numel1 == numel2 and rnumel1 == rnumel2
        if not reduction_can_fuse:
            why(
                "numel/rnumel mismatch (reduce) (%s, %s), (%s, %s)",
                numel1,
                numel2,
                rnumel1,
                rnumel2,
            )
        return reduction_can_fuse

    if not node1.is_reduction() and not node2.is_reduction():
        if not (numel1 == numel2 and rnumel1 == rnumel2):
            if not node2.is_template():
                why(
                    "numel/rnumel mismatch (non-reduce) (%s, %s), (%s, %s)",
                    numel1,
                    numel2,
                    rnumel1,
                    rnumel2,
                )
                return False
            else:
                # prologue fusion input sizes differ from output group
                # fuse so long as this node matches the group of existing prologue nodes
                for node in node2.get_nodes():
                    # dont need to check epilogue nodes for prologue fusion, break after template
                    if node.is_template():
                        break
                    # we would have already restricted prologue from fusing if it had multiple
                    # uses, so it must be fusing into this node
                    if not node.used_buffer_names() & node1.get_buffer_names():
                        continue
                    _, (pro_numel, pro_rnumel) = node.group
                    if not (numel1 == pro_numel and rnumel1 == pro_rnumel):
                        why(
                            "numel/rnumel mismatch prologue mismatch (%s, %s), (%s, %s)",
                            numel1,
                            pro_numel,
                            rnumel1,
                            pro_rnumel,
                        )
                        return False

        for n, node_name in zip((node1, node2), ("node1", "node2")):
            if n.is_template():
                # Only allow fusion for TritonTemplates for now.
                # Fusion for CUDATemplates are not supported.
                is_triton_template = isinstance(
                    n.get_template_node(), TritonTemplateBuffer
                )
                if not is_triton_template:
                    why(f"{node_name} is not TritonTemplateBuffer")
                return is_triton_template

        # check for a bad combined tiling
        tiling1 = self.select_tiling(node1.get_nodes(), numel1, rnumel1)
        tiling2 = self.select_tiling(node2.get_nodes(), numel1, rnumel1)
        tiling3 = self.select_tiling(
            node1.get_nodes() + node2.get_nodes(), numel1, rnumel1
        )
        if config.triton.tiling_prevents_pointwise_fusion:
            cond = True
            if len(tiling1) > 2:
                if len(tiling2) > 2:
                    cond = tiling1 == tiling2 == tiling3
                else:
                    cond = tiling1 == tiling3
            elif len(tiling2) > 2:
                cond = tiling2 == tiling3
            if not cond:
                why(
                    "tiling mismatch (%s, %s, %s)",
                    tiling1,
                    tiling2,
                    tiling3,
                )
                return False

        return True

    if not node1.is_reduction() and node2.is_reduction():
        assert rnumel1 == 1 and rnumel2 != 1
        if numel1 == numel2 * rnumel2:
            if not all(
                SIMDKernel.is_compatible((numel2, rnumel2), n.get_ranges())
                for n in node1.get_nodes()
            ):
                why("nodes numel/rnumel incompatibility")
                return False
            if (
                config.triton.tiling_prevents_reduction_fusion
                and not node1.is_template()
            ):
                # Modified by Cambricon
                # is_reduction_tiling_valid = tuple(
                #     self.select_tiling(node1.get_nodes(), numel1).values()
                # ) in (
                #     (numel1, 1),
                #     (numel2, rnumel2, 1),
                # )
                node1_tilings = tuple(
                    self.select_tiling(node1.get_nodes(), numel1).values()
                )
                is_reduction_tiling_valid = node1_tilings in (
                    (numel1, 1),
                    (numel2, rnumel2, 1),
                )
                if (
                    len(node1_tilings) == 4
                    and node1_tilings[-1] == 1
                    and node1_tilings[-2] == rnumel2
                ):
                    is_reduction_tiling_valid = True
                # end Modify by Cambricon

                if not is_reduction_tiling_valid:
                    why("invalid tiling for reduction")
                return is_reduction_tiling_valid
            return True

        if numel1 != numel2:
            why("nodes numel incompatibility")
        return numel1 == numel2

    assert node1.is_reduction() and not node2.is_reduction()
    # swap args to hit the case above
    return self.can_fuse_horizontal(node2, node1)


patch = gorilla.Patch(
    torch._inductor.codegen.simd.SIMDScheduling,
    "can_fuse",
    can_fuse,
)
gorilla.apply(patch)
