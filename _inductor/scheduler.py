import logging
import sympy

import torch
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    ExternKernelSchedulerNode,
    NopKernelSchedulerNode,
    ForeachKernelSchedulerNode,
)
from torch._inductor import dependencies
from torch.utils._ordered_set import OrderedSet
from ..utils import gorilla


log = logging.getLogger(__name__)


@classmethod
def combinable_nodes(cls, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    extern = [x for x in nodes if isinstance(x, ExternKernelSchedulerNode)]
    if extern:
        log.debug(
            "ComboKernels: %d external nodes are filtered %s",
            len(extern),
            [node.node.get_origins() for node in extern if node.node is not None],
        )
    filtered_nodes = [
        x
        for x in nodes
        if not isinstance(x, (NopKernelSchedulerNode, ExternKernelSchedulerNode))
    ]
    foreach_nodes = [
        x for x in filtered_nodes if isinstance(x, ForeachKernelSchedulerNode)
    ]
    if foreach_nodes:
        log.debug("ComboKernels: %d foreach nodes are filtered", len(foreach_nodes))
    filtered_nodes = [
        x for x in filtered_nodes if not isinstance(x, ForeachKernelSchedulerNode)
    ]
    template_nodes = [x for x in filtered_nodes if x.is_template()]
    if template_nodes:
        log.debug(
            "ComboKernels: %d template nodes are filtered: %s",
            len(template_nodes),
            template_nodes,
        )
    filtered_nodes = [x for x in filtered_nodes if x not in template_nodes]
    # Modify by CAMBRICON: we do not want fuse reduction node into horitanal fusion
    filtered_nodes = [x for x in filtered_nodes if not x.is_reduction()]
    filtered_nodes = [x for x in filtered_nodes if not x.is_cpu()]
    return filtered_nodes


patch = gorilla.Patch(
    torch._inductor.scheduler.ForeachKernelSchedulerNode,
    "combinable_nodes",
    combinable_nodes,
)
gorilla.apply(patch)


def pointwise_or_reduction_read_writes(
    self, pointwise: bool = True
) -> dependencies.ReadWrites:
    """
    Get the memory dependencies in either the pointwise or the reduction axes.
    """
    # Modify by Cambricon
    # keep_sizes, ignore_sizes = self._sizes if pointwise else reversed(self._sizes)
    keep_sizes, ignore_sizes = self._sizes
    # end Modify by Cambricon
    return dependencies.extract_read_writes(
        self._body, keep_sizes, hidden_args=[[sympy.S.Zero] * len(ignore_sizes)]
    )


patch = gorilla.Patch(
    torch._inductor.scheduler.SchedulerNode,
    "pointwise_or_reduction_read_writes",
    pointwise_or_reduction_read_writes,
)
gorilla.apply(patch)
