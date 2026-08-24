import torch
from torch._inductor.kernel.flex import flex_attention
from torch._inductor.select_algorithm import SymbolicGridFn, TritonTemplate

from .common import load_flex_template
from ....utils import gorilla


@SymbolicGridFn
def flex_attention_grid(batch_size, q_heads, num_queries, d_model, meta, *, cdiv):
    """How is this kernel parallelized?
    We create a grid of (ceil_div(n_queries, query_block_size), batch_size, num_heads)
    Each block is responsible for iterating over blocks of keys and values calculating
    the final attention output.
    """
    # Modify by CAMBRICON
    # return (cdiv(num_queries, meta["BLOCK_M"]), batch_size, q_heads)
    processor_count = torch.mlu.get_device_properties(
        torch.mlu.current_device()
    ).multi_processor_count
    return (processor_count, 1, 1)
    # end Modify by CAMBRICON


_ = TritonTemplate.all_templates.pop("flex_attention")
flex_attention_template = TritonTemplate(
    name="flex_attention",
    grid=flex_attention_grid,
    source=load_flex_template("flex_attention")
    + load_flex_template("utilities")
    + load_flex_template("common"),
)


patch = gorilla.Patch(
    flex_attention, "flex_attention_template", flex_attention_template
)
gorilla.apply(patch)
