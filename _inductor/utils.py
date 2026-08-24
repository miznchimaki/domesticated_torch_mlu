import functools
import logging
import time
import operator
from typing import List, Callable, Any, Union, Optional, Literal, Sequence

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor.utils import (
    aggregate_origins,
    synchronize,
    GPU_TYPES,
)
from torch.autograd import DeviceType
from torch.autograd.profiler_util import EventList
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.ir import ExternKernel
from torch.utils._ordered_set import OrderedSet
from torch.fx.node import Node
from ..utils import gorilla

log = logging.getLogger(__name__)
GPU_TYPES = ["mlu"]


@gorilla.patch(torch._inductor.utils)
def aggregate_origins(
    node_schedule: Union[Sequence[BaseSchedulerNode], ExternKernel],
) -> OrderedSet[Node]:
    # Modify by CAMBRICON
    # from . import ir
    from torch._inductor import ir

    if isinstance(node_schedule, list):
        # return functools.reduce(
        #     operator.or_,
        #     [
        #         # pyrefly: ignore [missing-attribute]
        #         node.node.origins
        #         for node in node_schedule
        #         if hasattr(node, "node") and node.node
        #     ],
        #     OrderedSet(),
        # )
        origins_list = []
        for node in node_schedule:
            if hasattr(node, "node") and node.node and hasattr(node.node, "origins"):
                origins_list.append(node.node.origins)
            elif hasattr(node, "snodes") and node.snodes:
                origins_list.extend(
                    sub_node.node.origins
                    for sub_node in node.snodes
                    if sub_node.node is not None and hasattr(sub_node.node, "origins")
                )
        # end Modify by CAMBRICON
        return functools.reduce(operator.or_, origins_list, OrderedSet())
    elif isinstance(node_schedule, ir.ExternKernel):
        return node_schedule.origins
    else:
        return OrderedSet()


@gorilla.patch(torch._inductor.utils)
def get_fused_kernel_name(
    node_schedule: Sequence[BaseSchedulerNode],
    descriptive_names: Literal[True, "torch", "original_aten", "inductor_node"],
) -> str:
    all_origins = aggregate_origins(node_schedule)
    if descriptive_names == "original_aten":

        def get_origin_meta_str(origin):
            original_aten = origin.meta["original_aten"]
            key = ""
            if isinstance(original_aten, torch._ops.OpOverload):
                key = original_aten._overloadpacket.__name__
            elif isinstance(original_aten, torch._ops.HigherOrderOperator):
                key = str(original_aten.name())
            return key

        # Bases the kernel name off of the top-level aten operator (i.e. pre-decompositions)
        # Modify by CAMBRICON: replace meta["original_aten"] with target
        """
        sources = [
            get_origin_meta_str(origin)
            for origin in all_origins
            if origin.op == "call_function"
            and "original_aten" in origin.meta
            and origin.meta["original_aten"] is not None
        ]
        """
        sources = []
        for origin in all_origins:
            # remove the patch when we do not use xpu_graph
            if origin.op == "call_function":
                if hasattr(origin, "target") and hasattr(
                    origin.target, "_overloadpacket"
                ):
                    sources.append(origin.target._overloadpacket.__name__)
                elif hasattr(origin, "target") and isinstance(
                    origin.target, torch._ops.HigherOrderOperator
                ):
                    sources.append(str(origin.target.name()))

        # end Modify by CAMBRICON
        sources = sorted(OrderedSet(sources))
    elif descriptive_names == "torch":
        # Bases the kernel name off of the top-level "torch" operator (i.e. post-dynamo graph)
        sources = []
        for origin in all_origins:
            if origin.op == "call_function":
                source_fn = None
                suffix = ""
                if "source_fn_stack" in origin.meta:
                    source_fn = origin.meta["source_fn_stack"][-1]
                elif "fwd_source_fn_stack" in origin.meta:
                    # backward nodes have "fwd_source_fn_stack" instead
                    source_fn = origin.meta["fwd_source_fn_stack"][-1]
                    suffix = "backward"
                if not source_fn:
                    continue
                if isinstance(source_fn[1], str):
                    sources.append(source_fn[1] + suffix)
                else:
                    sources.append(source_fn[1].__name__ + suffix)

        sources = sorted(OrderedSet(sources))
    elif descriptive_names == "inductor_node":
        sources = [
            origin.name for origin in all_origins if origin.op == "call_function"
        ]
    else:
        raise NotImplementedError
    return "_".join(["fused"] + sources)


@gorilla.patch(
    torch._inductor.utils, settings=gorilla.Settings(use_replace_references=True)
)
def timed(
    model: Callable[..., Any],
    example_inputs: Sequence[Any],
    times: int = 1,
    device: str = "mlu",
) -> float:
    synchronize(device)
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize(device)
    t1 = time.perf_counter()
    # GC the result after timing
    assert result is not None  # type: ignore[possibly-undefined]
    return t1 - t0


@gorilla.patch(
    torch._inductor.utils, settings=gorilla.Settings(use_replace_references=True)
)
def print_performance(
    model: Callable[..., Any],
    example_inputs: Sequence[Any] = (),
    times: int = 10,
    repeat: int = 10,
    baseline: float = 1.0,
    # Modify by CAMBRICON
    # device: str = "cuda",
    device: str = "mlu",
    # end Modify by CAMBRICON
) -> float:
    timings = torch.tensor(
        [timed(model, example_inputs, times, device) for _ in range(repeat)]
    )
    took = torch.median(timings) / times
    print(f"{took / baseline:.6f}")
    return took.item()


@gorilla.patch(torch._inductor.utils)
@functools.cache
def is_big_gpu(index_or_device: Union[int, torch.device] = 0) -> bool:
    # Modify by CAMBRICON
    # if isinstance(index_or_device, torch.device):
    #     device = index_or_device
    # else:
    #     device = torch.device(get_gpu_type(), index_or_device)

    # prop = DeviceProperties.create(device)

    # # SM logic is not relevant to ROCm gpus
    # # Arbitrarily skipping the older models
    # if torch.version.hip:
    #     assert prop.major is not None
    #     if prop.major < 9 or prop.major == 10:
    #         log.warning("GPU arch does not support max_autotune_gemm mode usage")
    #         return False
    #     return True

    # min_sms = 16 if device.type == "xpu" else 68  # 3080
    # avail_sms = prop.multi_processor_count
    # if avail_sms < min_sms:
    #     log.warning(
    #         "Not enough SMs to use max_autotune_gemm mode",
    #         extra={"min_sms": min_sms, "avail_sms": avail_sms},
    #     )
    #     return False
    # return True
    return True
    # end Modify by CAMBRICON


@gorilla.patch(torch._inductor.utils)
def is_gpu(device: Optional[str]) -> bool:
    # Modify by CAMBRICON
    # return device in GPU_TYPES
    return device in ["cuda", "xpu", "mlu"]
    # end Modify by CAMBRICON


@gorilla.patch(torch._inductor.utils)
def decode_device(device: Union[Optional[torch.device], str]) -> torch.device:
    if device is None:
        return torch.tensor(0.0).device  # default device
    if isinstance(device, str):
        # Modify by CAMBRICON
        device = device.replace("cuda", "mlu")
        # end Modify by CAMBRICON
        device = torch.device(device)
    # Modify by CAMBRICON
    if device.type not in ("cpu", "meta") and device.index is None:
        # device_interface = get_interface_for_device(device.type)
        # return torch.device(device.type, index=device_interface.Worker.current_device())
        device_interface = get_interface_for_device("mlu")
        device = torch.device("mlu", index=device_interface.Worker.current_device())
        # end Modify by CAMBRICON
    return device


@gorilla.patch(torch._inductor.utils)
def _do_bench_using_profiling(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    is_vetted_benchmarking: bool = False,
) -> float:
    """
    Returns benchmark results by examining torch profiler events.
    This could be more accurate as it doesn't count CPU side overhead.
    However, this also requires manually excluding irrelevant event, e.g.
    vectorized_elementwise_kernel which is used to fill L2 cache,
    various CUDA events, etc, so could also be fragile.
    """

    if not is_vetted_benchmarking:
        from torch._inductor.runtime.benchmarking import may_ban_benchmarking

        may_ban_benchmarking()

    # Modify by CAMBRICON
    # device_type = get_gpu_type()
    device_type = "mlu"
    device_type_upper = device_type.upper()
    # end Modify by CAMBRICON
    device_interface = get_interface_for_device(device_type)
    fn()
    device_interface.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device_type)

    # Estimate the runtime of the function
    start_event = device_interface.Event(enable_timing=True)
    end_event = device_interface.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    device_interface.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # Warm-up
    for _ in range(n_warmup):
        fn()

    device_interface.synchronize()
    with torch.profiler.profile(
        activities=[
            getattr(torch.profiler.ProfilerActivity, device_type_upper),
        ]
    ) as p:
        # Benchmark
        for _ in range(n_repeat):
            # we clear the L2 cache before each run
            cache.zero_()
            # record time of `fn`
            fn()
        # Record clocks
        device_interface.synchronize()

    log.debug("raw events")
    log.debug(p.key_averages().table(sort_by="self_device_time_total", row_limit=-1))

    filtered_events = EventList(
        [
            event
            for event in p.events()
            # Modify by CAMBRICON
            # if event.device_type == getattr(DeviceType, device_type_upper)
            if event.device_type == DeviceType.PrivateUse1
            and event.name != "Context Sync"
            and not event.is_user_annotation
            # end Modify by CAMBRICON
        ]
    )
    # Filter out cache.zero_() events by name pattern instead of relying on
    # positional grouping. cache.zero_() may not always generate an event
    # making positional assumptions unreliable.
    # The kernel name contains "FillFunctor" when generated by cache.zero_().
    # Modify by CAMBRICON
    # actual_events = EventList(
    #     [event for event in filtered_events if "FillFunctor" not in event.name]
    # )
    if len(filtered_events) % n_repeat != 0:
        raise RuntimeError(
            "Failed to divide all profiling events into #repeat groups. "
            "#%s events: %d, #repeats: %s",
            device_type,
            len(filtered_events),
            n_repeat,
        )
    num_event_per_group = len(filtered_events) / n_repeat
    actual_events = EventList(
        [
            event
            for i, event in enumerate(filtered_events)
            if i % num_event_per_group != 0
        ]
    )
    # end Modify by CAMBRICON
    if len(actual_events) == 0:
        raise RuntimeError(
            f"Failed to capture any events after filtering cache clearing events. "
            f"{device_type} events: {len(filtered_events)}, repeats: {n_repeat}"
        )
    actual_events._build_tree()
    actual_events = actual_events.key_averages()

    log.debug("profiling time breakdown")
    log.debug(actual_events.table(row_limit=-1))

    res = sum(event.device_time_total for event in actual_events) / 1000.0 / n_repeat
    log.debug("profiling results: %s ms", res)
    return res


@gorilla.patch(torch._inductor.utils)
def synchronize(device: str = "mlu") -> None:
    if device == "cpu":
        return
    device_interface = get_interface_for_device(device)
    if device_interface.is_available():
        device_interface.synchronize()


@gorilla.patch(torch._inductor.utils)
def get_max_numwarps() -> int:
    # Modify by CAMBRICON
    # if torch.cuda.is_available():
    if torch.mlu.is_available():
        # Defaults
        warp_size = 32
        max_threads_per_block = 1024
    elif torch.cuda.is_available():
        # end Modify by CAMBRICON
        warp_size = torch.cuda.get_device_properties().warp_size
        # pyrefly: ignore [missing-attribute]
        max_threads_per_block = torch.cuda.get_device_properties().max_threads_per_block
    else:
        # Defaults
        warp_size = 32
        max_threads_per_block = 1024
    return max_threads_per_block // warp_size


@gorilla.patch(torch._inductor.utils)
def get_kernel_metadata(
    # Modify by Cambricon:
    # node_schedule: Union[Sequence[BaseSchedulerNode], ExternKernel],
    # wrapper: PythonWrapperCodegen,
    node_schedule,
    wrapper,
    # end Modify by Cambricon
) -> tuple[str, str]:
    """
    Retrieves metadata information for a kernel.
    Args:
        node_schedule (Union[Sequence[BaseSchedulerNode], ExternKernel]):
            Either a sequence of BaseSchedulerNode objects or an ExternKernel instance.
        wrapper (PythonWrapperCodegen):
            An instance of PythonWrapperCodegen, used to define the code comment format.
    Returns:
        tuple[str, str]:
            A tuple containing two strings:
                - The first string represents the kernel's metadata.
                - The second string represent the kernel's detailed metadata.
    """

    all_origins = aggregate_origins(node_schedule)
    inductor_nodes = [origin for origin in all_origins if origin.op == "call_function"]

    from_node_dict = collections.defaultdict(list)
    original_aten_dict = collections.defaultdict(list)

    # Attempt to sort `inductor_nodes` topologically. Note that the case
    # where `inductor_nodes` contains nodes from multiple graph instances
    # is not supported. An example of this is conditional statements.
    single_graph = None
    # Modify by Cambricon
    from collections.abc import Iterable

    match_triton_fusion = False
    if len(inductor_nodes) == 1 and "keep_transform_triton" in inductor_nodes[0].meta:
        inductor_nodes = list(
            inductor_nodes[0].meta["keep_transform_triton"].targetnodes
        )
        match_triton_fusion = True
        single_graph = inductor_nodes[0].graph

    # if inductor_nodes:
    if len(inductor_nodes) and not match_triton_fusion:
        # end Modify by Cambricon
        unique_graphs = OrderedSet(n.graph for n in inductor_nodes)
        if len(unique_graphs) == 1:
            single_graph = inductor_nodes[0].graph
            # create a map of idx -> node and cache it
            if not hasattr(single_graph, "_inductor_kernel_metadata_node_to_idx_map"):
                node_to_idx_map = {n: idx for idx, n in enumerate(single_graph.nodes)}
                single_graph._inductor_kernel_metadata_node_to_idx_map = node_to_idx_map  # type: ignore[attr-defined]
            inductor_nodes.sort(
                key=lambda n: single_graph._inductor_kernel_metadata_node_to_idx_map[n]  # type: ignore[attr-defined]
            )

    for node in inductor_nodes:
        if "original_aten" in node.meta and node.meta["original_aten"] is not None:
            original_aten = node.meta["original_aten"]
            key = None
            if isinstance(original_aten, torch._ops.OpOverload):
                key = str(original_aten._overloadpacket)
            elif isinstance(original_aten, torch._ops.HigherOrderOperator):
                key = str(original_aten.name())
            if key:
                original_aten_dict[key].append(node.name)
        if "from_node" in node.meta:
            key = node.meta["from_node"][0].name
            from_node_dict[key].append(node.name)
        elif node.meta.get("partitioner_tag") == "is_backward":
            # backward nodes currently don't have a "from node"
            from_node_dict[node.name].append(node.name)
    sort_str = "Topologically Sorted" if single_graph is not None else "Unsorted"
    metadata = (
        f"{wrapper.comment} {sort_str} Source Nodes: [{', '.join(from_node_dict.keys())}], "
        f"Original ATen: [{', '.join(original_aten_dict.keys())}]"
    )

    # trace back to original node here
    detailed_metadata = [f"{wrapper.comment} Source node to ATen node mapping:"]
    for original_node, nodes in sorted(from_node_dict.items()):
        detailed_metadata.append(
            f"{wrapper.comment}   {original_node} => {', '.join(sorted(nodes))}"
        )

    # print the aot_autograd graph fragment
    if single_graph is not None:
        from . import ir

        detailed_metadata.append(f"{wrapper.comment} Graph fragment:")
        all_reads: OrderedSet[str] = OrderedSet()
        all_writes: list[str] = []
        if not isinstance(node_schedule, ir.ExternKernel):
            from .virtualized import V

            def get_buffer_info(
                buffer: Union[ir.TensorBox, ir.Buffer, ir.TorchBindObject], rw_name: str
            ) -> tuple[str, ir.Layout | None]:
                if isinstance(buffer, ir.TensorBox) and isinstance(
                    buffer.data, ir.StorageBox
                ):
                    origin_node = buffer.data.data.origin_node
                else:
                    origin_node = buffer.origin_node
                if origin_node is None:
                    # use the read/write name if no origin node is found
                    name = rw_name
                else:
                    name = origin_node.name
                try:
                    layout = buffer.get_layout()
                except NotImplementedError:
                    layout = None
                return name, layout

            def stringify_shape(shape: Iterable[int]) -> str:
                return f"[{', '.join([str(x) for x in shape])}]"

            def stringfy_layout(layout: ir.Layout | None) -> str:
                if layout is None:
                    return ""
                shape_annotation = f"{stringify_shape(layout.size)}"
                stride_annotation = f"{stringify_shape(layout.stride)}"
                device_annotation = f"{layout.device}"

                return (
                    f'"{dtype_abbrs[layout.dtype]}{shape_annotation}'
                    f'{stride_annotation}{device_annotation}"'
                )

            for n in node_schedule:
                if not hasattr(n, "read_writes") or n.read_writes is None:
                    continue
                if hasattr(n.read_writes, "reads") and n.read_writes.reads is not None:
                    for r in n.read_writes.reads:
                        # Remove the dupricated inputs
                        if r.name in all_reads:
                            continue
                        all_reads.add(r.name)
                        buffer = V.graph.try_get_buffer(r.name)
                        if buffer is None:
                            continue
                        input_name, layout = get_buffer_info(buffer, r.name)
                        detailed_metadata.append(
                            f"{wrapper.comment}   %{input_name} : Tensor "
                            f"{stringfy_layout(layout)} = PlaceHolder[target={input_name}]"
                        )

                if (
                    hasattr(n.read_writes, "writes")
                    and n.read_writes.writes is not None
                ):
                    for w in n.read_writes.writes:
                        buffer = V.graph.try_get_buffer(w.name)
                        if buffer is None:
                            continue
                        output_name, _ = get_buffer_info(buffer, w.name)

                        all_writes.append("%" + output_name)

        for node in inductor_nodes:
            detailed_metadata.append(
                f"{wrapper.comment}   {node.format_node(include_tensor_metadata=True)}"
            )

        detailed_metadata.append(f"{wrapper.comment}   return {','.join(all_writes)}")
        # Add by Cambricon
        if single_graph is not None:
            detailed_metadata.append(
                f"{wrapper.comment} Mapping Between Source Code and Graph Fragments:"
            )
            for n in inductor_nodes:
                # TODO(future): maybe refactor torch/fx/graph.py to make it easy to
                # generate python code for graph fragments
                detailed_metadata.append(f"{wrapper.comment}   {n.format_node()}")
                if n.stack_trace:
                    stack_trace = n.stack_trace.replace("\n", " ")
                    detailed_metadata.append(
                        f"{wrapper.comment}           -> Source Code: {stack_trace}"
                    )

        source_codes = set()
        for n in inductor_nodes:
            if n.stack_trace:
                stack_trace = n.stack_trace.replace("\n", " ")
                source_codes.add(stack_trace)
        if source_codes:
            detailed_metadata.append(f"{wrapper.comment} Source code:")
            for stack_trace in sorted(source_codes):
                detailed_metadata.append(f"{wrapper.comment}   {stack_trace}")
        # end Add by Cambricon

    return metadata, "\n".join(detailed_metadata)
