from typing import Optional

import torch
from torch._inductor.cudagraph_utils import (
    format_default_skip_message,
    _get_use_stack_trace,
)
from torch._inductor.utils import is_using_cudagraph_partition

from ..utils import gorilla


@gorilla.patch(torch._inductor.cudagraph_utils)
def check_multiple_devices_or_any_cpu_nodes(
    device_node_mapping: dict[torch.device, torch.fx.Node],
) -> Optional[str]:
    # meta tensors are supported since there is no compute
    device_node_mapping.pop(torch.device("meta"), None)

    # dynamo cudagraph does not support graph partition
    if is_using_cudagraph_partition():
        # graph partition supports splitting on cpu op. So we can ignore cpu nodes.
        device_node_mapping.pop(torch.device("cpu"), None)

    if cpu_node := device_node_mapping.get(torch.device("cpu")):
        msg = f"cpu device ({cpu_node.name})"
        if stack_trace := _get_use_stack_trace(cpu_node):
            return format_default_skip_message(f"{msg}. Found from : \n {stack_trace}")

        return format_default_skip_message(msg)

    # Modify by CAMBRICON
    # if (
    #     len(device_node_mapping) == 1
    #     and next(iter(device_node_mapping.keys())).type == "cuda"
    # ):
    if (
        len(device_node_mapping) == 1
        and next(iter(device_node_mapping.keys())).type == "mlu"
    ):
        # end Modify by CAMBRICON
        return None

    keys_repr = (repr(key) for key in device_node_mapping)
    return format_default_skip_message(f"multiple devices: {', '.join(keys_repr)}")
