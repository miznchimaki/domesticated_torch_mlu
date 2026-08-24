import os
import torch
from torch._dynamo.repro.after_aot import save_graph_repro
from ..utils import gorilla


@gorilla.patch(torch._inductor.debug.DebugFormatter)
def fx_graph(
    self,
    gm: torch.fx.GraphModule,
    inputs: list[torch.Tensor],
) -> None:
    with self.fopen("fx_graph_runnable.py") as fd:
        save_dir = None
        if torch._inductor.config.trace.save_real_tensors:
            inputs = torch._subclasses.fake_utils.try_convert_fake_to_real(inputs)
            save_dir = os.path.dirname(fd.name)

        # dont try to use stable hash torchinductor compilation if saving real tensors
        # and avoid recursively trying to save real tensors inside of the inductor compilation
        # regardless
        stable_hash = torch._inductor.config.trace.save_real_tensors
        with torch._inductor.config.patch(
            {"trace.enabled": False, "trace.save_real_tensors": False}
        ):
            save_graph_repro(
                fd,
                gm,
                inputs,
                "inductor",
                save_dir=save_dir,
                stable_hash=stable_hash,
            )

    with self.fopen("fx_graph_readable.py") as fd:
        # Modify by CAMBRICON
        # fd.write(gm.print_readable(print_output=False))
        fd.write(
            gm.print_readable(
                print_output=False, include_stride=True, include_device=True
            )
        )
        # end Modify by CAMBRICON
