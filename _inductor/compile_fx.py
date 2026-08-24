from typing import (
    Any,
    Callable,
    Dict,
    Sequence,
)

import torch
from torch.fx import GraphModule
from torch._dynamo.utils import dynamo_timed
from torch._inductor import config
from torch._inductor.compile_fx import (
    align_inputs_from_check_idxs,
    copy_misaligned_inputs,
    get_expanded_dims,
    index_expanded_dims_and_copy_,
    remove_unaligned_input_idxs,
    static_input,
    _get_subgraph_names,
)
from torch._inductor.fx_passes.post_grad import post_grad_passes
from torch._inductor.output_code import index_expanded_dims
from torch._inductor.utils import (
    InputType,
)
from torch._dynamo.utils import dynamo_timed
from torch.utils._ordered_set import OrderedSet
from ..utils import gorilla
from .graph import codegen_with_cpp_wrapper


def get_input_idxs_to_check(
    inputs: Sequence[InputType],
    static_input_idxs: Sequence[int],
) -> Sequence[int]:
    """
    This function runs at compile time, and generates a list of indices for which we
    might need to do a copy to preserve alignment requirements.
    """
    # Modify by Cambricon
    # ids_to_check = []

    # for i, input in enumerate(inputs):
    #     if not isinstance(input, torch.Tensor):
    #         # non-tensors don't need alignment
    #         continue
    #     if not is_gpu(input.device.type):
    #         # right now we only care for gpu tensors
    #         continue
    #     with maybe_get_suppress_shape_guards_ctx():
    #         # suppress guards so that tensor_is_aligned and should_assume_input_aligned
    #         # do not add guards on input's storage offset
    #         if i in static_input_idxs and tensor_is_aligned(input):
    #             continue
    #         if not should_assume_input_aligned(input):
    #             continue

    #     # if we get here, then
    #     # (a) our triton code assumes that the input is aligned
    #     # (b) we can't be sure ahead of time that the input will actually be aligned.
    #     # therefore, at runtime, we'll need to check that the input is aligned
    #     # (and if not, clone it to make it aligned.)
    #     ids_to_check.append(i)

    # return ids_to_check
    # end Modify by Cambricon

    # Since we have enforced all inputs to Triton to be contiguous in make_contiguous_clone,
    # all tensors passed to Triton meet the alignment requirements. Therefore,
    # there is no need to check the tensors for MLU.
    # Detailed docs can be seen in wiki:pageId=453340020.
    return []


patch = gorilla.Patch(
    torch._inductor.compile_fx, "get_input_idxs_to_check", get_input_idxs_to_check
)
gorilla.apply(patch)


def cudagraphify_impl(
    model: Callable[..., Any],
    inputs: list[torch.Tensor],
    static_input_idxs: Sequence[int] = (),
) -> Callable[[list[InputType]], Any]:
    """
    Assumes inputs[static_input_idxs[i]] are always the same memory address
    """
    check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)  # type: ignore[arg-type]
    # pyrefly: ignore [annotation-mismatch, redefinition]
    static_input_idxs: OrderedSet[int] = OrderedSet(
        remove_unaligned_input_idxs(inputs, static_input_idxs)  # type: ignore[arg-type]
    )
    copy_misaligned_inputs(inputs, check_input_idxs)  # type: ignore[arg-type]

    assert isinstance(inputs, list)

    inps_expanded_dims = [
        get_expanded_dims(x) if idx not in static_input_idxs else []
        for idx, x in enumerate(inputs)
    ]

    # allocate static tensor inputs
    static_inputs = [
        (
            x
            if not isinstance(x, torch.Tensor)
            else static_input(x)
            if idx not in static_input_idxs
            else x.detach()
        )
        for idx, x in enumerate(inputs)
    ]

    # copy over input values for fresh allocations
    for idx, (x, expanded_dims) in enumerate(zip(inputs, inps_expanded_dims)):
        if isinstance(x, torch.Tensor) and idx not in static_input_idxs:
            index_expanded_dims_and_copy_(static_inputs[idx], x, expanded_dims)

    # warmup
    # Modify by CAMBRICON
    from typing import Callable

    """
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # copy static_inputs because it will be cleared in model
    with torch.cuda.stream(stream):
        model(list(static_inputs))
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
    """
    torch.mlu.synchronize()
    stream = torch.mlu.Stream()
    stream.wait_stream(torch.mlu.current_stream())
    # copy static_inputs because it will be cleared in model
    with torch.mlu.stream(stream):
        model(list(static_inputs))
    stream.synchronize()
    torch.mlu.current_stream().wait_stream(stream)
    torch.mlu.synchronize()

    # record
    graph = torch.mlu.MLUGraph()
    with torch.mlu.graph(graph, stream=stream, capture_error_mode="thread_local"):
        # end Modify by CAMBRICON
        static_outputs = model(list(static_inputs))
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    if config.size_asserts:

        def run(new_inputs: list[InputType]) -> Callable[[list[InputType]], Any]:
            assert len(static_inputs) == len(new_inputs)
            for idx, (dst, src, expanded_dims) in enumerate(
                zip(static_inputs, new_inputs, inps_expanded_dims)
            ):
                if not isinstance(dst, torch.Tensor):
                    continue
                assert isinstance(src, torch.Tensor)
                if idx in static_input_idxs:
                    assert dst.data_ptr() == src.data_ptr()
                else:
                    # TODO - could make one single op of multiple slices
                    # and avoid dispatch.
                    # Could also pre-index the `dst` tensors
                    index_expanded_dims_and_copy_(dst, src, expanded_dims)
            new_inputs.clear()
            graph.replay()
            # pyrefly: ignore [bad-return]
            return static_outputs

    else:
        copy_indices = [
            idx for idx in range(len(static_inputs)) if idx not in static_input_idxs
        ]

        def run(new_inputs: list[InputType]) -> Callable[[list[InputType]], Any]:
            for idx in copy_indices:
                expanded_dims = inps_expanded_dims[idx]
                src = new_inputs[idx]
                assert isinstance(src, torch.Tensor)
                index_expanded_dims_and_copy_(static_inputs[idx], src, expanded_dims)
            new_inputs.clear()
            graph.replay()
            # pyrefly: ignore [bad-return]
            return static_outputs

    return align_inputs_from_check_idxs(run, check_input_idxs, OrderedSet())


patch = gorilla.Patch(
    torch._inductor.compile_fx, "cudagraphify_impl", cudagraphify_impl
)
gorilla.apply(patch)


def _recursive_post_grad_passes(gm: GraphModule, is_inference: bool = False) -> None:
    # Modify by CAMBRICON
    from torch_mlu._inductor import config as inductor_config

    # end Modify by CAMBRICON

    with dynamo_timed(
        "_recursive_post_grad_passes",
        log_pt2_compile_event=True,
        dynamo_compile_column_us="post_grad_pass_time_us",
    ):
        if not config.use_post_grad_passes:
            return

        # Modify by CAMBRICON
        if inductor_config.enable_post_grad:
            for subgraph_name in _get_subgraph_names(gm):
                subgraph = getattr(gm, subgraph_name)
                _recursive_post_grad_passes(subgraph, is_inference)
            post_grad_passes(gm, is_inference)
        # end Modify by CAMBRICON


patch = gorilla.Patch(
    torch._inductor.compile_fx,
    "_recursive_post_grad_passes",
    _recursive_post_grad_passes,
)
gorilla.apply(patch)
