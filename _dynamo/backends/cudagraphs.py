from collections.abc import Sequence
from typing import Any, Callable
import torch
from torch._dynamo.backends.cudagraphs import (
    get_device_node_mapping,
)

from ...utils import gorilla


@gorilla.patch(torch._dynamo.backends.cudagraphs)
def get_device_index(gm) -> int:
    device = next(iter(get_device_node_mapping(gm)))
    # Modify by CAMBRICON
    # assert device.type == "cuda"
    assert device.type == "mlu"
    # end Modify by CAMBRICON
    return device.index


@gorilla.patch(torch._dynamo.backends.cudagraphs)
def cudagraphs_inner(
    model: Callable[..., Any],
    inputs: Sequence[Any],
    copy_outputs: bool = True,
    copy_inputs: bool = True,
) -> Callable[..., Sequence[Any]]:
    """This isn't registered as a backend, but is used in some benchmarks"""
    assert isinstance(inputs, (list, tuple))
    if copy_inputs:
        # pyrefly: ignore [bad-argument-type]
        static_inputs = [torch.zeros_like(x) for x in inputs]
    else:
        static_inputs = list(inputs)

    # warmup
    # Modify by Cambricon
    # torch.cuda.synchronize()
    # stream = torch.cuda.Stream()
    # stream.wait_stream(torch.cuda.current_stream())
    # with torch.cuda.stream(stream):
    #     model(*inputs)
    # stream.synchronize()
    # torch.cuda.current_stream().wait_stream(stream)
    # torch.cuda.synchronize()

    # # record
    # graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(graph, stream=stream):
    #     static_outputs = model(*static_inputs)
    torch.mlu.synchronize()
    stream = torch.mlu.Stream()
    stream.wait_stream(torch.mlu.current_stream())
    with torch.mlu.stream(stream):
        model(*inputs)
    stream.synchronize()
    torch.mlu.current_stream().wait_stream(stream)
    torch.mlu.synchronize()

    # record
    graph = torch.mlu.MLUGraph()
    with torch.mlu.graph(graph, stream=stream):
        static_outputs = model(*static_inputs)
    # end Modify by Cambricon
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    def run(*new_inputs: Any) -> Sequence[Any]:
        assert len(static_inputs) == len(new_inputs)
        if copy_inputs:
            for dst, src in zip(static_inputs, new_inputs):
                dst.copy_(src)
        graph.replay()
        if copy_outputs:
            return [x.clone() for x in static_outputs]
        else:
            return static_outputs

    return run
