import torch
from torch._dynamo.backends.common import AotAutograd

from ...utils import gorilla


@gorilla.patch(
    torch._dynamo.backends.common.AotAutograd,
    settings=gorilla.Settings(use_replace_references=True),
)
def __init__(self, **kwargs) -> None:
    self.__name__ = "compiler_fn"
    self.kwargs = kwargs
    # Add by CAMBRICON
    import os

    force_default_partitoner = (
        os.environ.get("TORCHINDUCTOR_MLU_FORCE_DEFAULT_PARTITIONER", "0") == "1"
    )
    from collections.abc import Sequence
    from torch._functorch.partitioners import default_partition
    from torch.fx import GraphModule
    from torch._inductor.compile_fx import (
        get_cuda_device_context,
        _recursive_joint_graph_passes,
    )

    def _partition_fn(
        gm: GraphModule,
        joint_inputs: Sequence[object],
        **kwargs: object,
    ) -> tuple[GraphModule, GraphModule]:
        cuda_context = get_cuda_device_context(gm)
        with cuda_context:
            _recursive_joint_graph_passes(gm)
        return default_partition(gm, joint_inputs, **kwargs)

    if force_default_partitoner:
        self.kwargs["partition_fn"] = _partition_fn
    # end Add by CAMBRICON
