from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import is_in_torch_dispatch_mode
from torch._dynamo.utils import counters
from torch._higher_order_ops.wrap import inductor_compiled_code
from torch._inductor.cudagraph_utils import (
    BoxedDeviceIndex,
    CudagraphCachedInfo,
    get_placeholder_info,
    log_cudagraph_skip_and_bump_counter,
)
from torch._inductor.freezing_utils import has_frozen_params, is_frozen_param
from torch._inductor.utils import (
    BoxedBool,
    InputType,
    output_node,
    set_tracing_context_output_strides,
)
from torch._inductor import config
from torch._inductor import metrics
from torch._inductor.graph import GraphLowering
from torch._inductor.compile_fx import _CompileFxKwargs

log = logging.getLogger(__name__)

from ..utils import gorilla


@gorilla.patch(torch._inductor.output_code)
def cudagraph_post_compile(
    example_inputs: Sequence[InputType],
    compiled_graph: CompiledFxGraph,
    cudagraphs: BoxedBool,
    constants: dict[str, Union[torch.Tensor, type]],
    boxed_forward_device_index: Optional[BoxedDeviceIndex],
) -> None:
    """
    Checks for any reasons not to run cudagraphs and then
    runs it on compiled_graph.
    Mutates the `compiled_graph.current_callable` and `cudagraphs`
    """
    from torch._inductor.compiler_bisector import CompilerBisector

    assert compiled_graph.current_callable is not None
    assert compiled_graph.cudagraph_info is not None
    cached_info = compiled_graph.cudagraph_info
    cudagraph_fail_reasons = cached_info.cudagraph_fail_reasons
    is_inference = compiled_graph.fx_kwargs["is_inference"]
    is_backward = compiled_graph.fx_kwargs["is_backward"]

    # Check if bisector wants to disable cudagraphs for this graph
    if CompilerBisector.disable_subsystem("inductor", "cudagraphs"):
        BoxedBool.disable(cudagraphs)
        maybe_handle_backward_generation(compiled_graph, boxed_forward_device_index)
        log_cudagraph_skip_and_bump_counter("skipping cudagraphs due to bisector")
        return

    if not cudagraph_fail_reasons:
        fx_kwargs = compiled_graph.fx_kwargs
        static_input_idxs = fx_kwargs["static_input_idxs"]

        placeholders = cached_info.placeholders
        stack_traces = cached_info.stack_traces

        prepare_cudagraph_post_compile(
            compiled_graph, example_inputs, boxed_forward_device_index
        )

        from .compile_fx import cudagraphify

        current_callable = compiled_graph.current_callable
        assert current_callable is not None
        # Filter to only tensor constants (exclude opaque value type classes)
        tensor_constants = {
            k: v for k, v in constants.items() if isinstance(v, torch.Tensor)
        }
        compiled_graph.current_callable = cudagraphify(
            current_callable,
            static_input_idxs=static_input_idxs or (),
            device_index=next(iter(compiled_graph.device_idxs)),
            stack_traces=stack_traces,
            is_backward=is_backward,
            is_inference=is_inference,
            constants=tuple(tensor_constants.values()),
            placeholders=placeholders,
            mutated_input_idxs=tuple(compiled_graph.mutated_input_idxs),
        )

    else:
        BoxedBool.disable(cudagraphs)
        maybe_handle_backward_generation(compiled_graph, boxed_forward_device_index)

        # Modify by CAMBRICON
        # if "cuda" in compiled_graph.device_types:
        if "mlu" in compiled_graph.device_types:
            # end Modify by CAMBRICON
            # prefer better disable_cudagraphs_reason bc stack trace
            # TODO: migrate all disable reasons to stack trace, refactor
            if compiled_graph.disabled_cudagraphs_reason:
                log_cudagraph_skip_and_bump_counter(
                    compiled_graph.disabled_cudagraphs_reason
                )
            else:
                log_cudagraph_skip_and_bump_counter(
                    f"skipping cudagraphs due to {cudagraph_fail_reasons}"
                )


@gorilla.patch(torch._inductor.output_code.CompiledFxGraph)
def post_compile(
    self,
    example_inputs: Sequence[InputType],
    constants: CompiledFxGraphConstants,
    graph_kwargs: _CompileFxKwargs,
) -> None:
    """
    Run a set of post processing steps after loading from the cache. These involve:
     - Setting the tracing context output strides
     - Running cudagraphs if enabled
     - Realigning inputs

    This runs whether or not we have a cache hit, and always runs directly after we get a CompiledFxGraph.
    The results of this function are *not* saved in the cache itself.
    """
    if config.graph_partition and _unstable_customized_partition_wrapper.wrapper:
        # Mechanically apply user-specified cudagraph wrappers without modification
        assert self.recursively_apply_fns is not None
        assert self.compiled_fn_runner is not None
        num_partitions = len(self.compiled_fn_runner.partitions)
        wrapper_metadatas = [
            CUDAGraphWrapperMetadata(num_partitions, i) for i in range(num_partitions)
        ]
        customized_wrapper = _unstable_customized_partition_wrapper.wrapper
        customized_wrappers_with_metadata = [
            lambda f, m=metadata: customized_wrapper(f, m)
            for metadata in wrapper_metadatas
        ]
        self.recursively_apply_fns(customized_wrappers_with_metadata)
        return

    set_tracing_context_output_strides(example_inputs, self)
    assert graph_kwargs["cudagraphs"] is not None
    assert graph_kwargs["is_backward"] is not None
    is_backward = graph_kwargs["is_backward"]
    cudagraphs: BoxedBool = graph_kwargs["cudagraphs"]
    if cudagraphs:
        # It's possible that cudagraphs is enabled, but was disabled
        # during a previous compilation we're loading from the cache.
        # If so, we need to disable it on this new process too.
        if self.disabled_cudagraphs_reason:
            # Modify by CAMBRICON
            # if "cuda" in self.device_types:
            if "mlu" in self.device_types:
                # end Modify by CAMBRICON
                log_cudagraph_skip_and_bump_counter(
                    f"skipping cudagraphs due to {self.disabled_cudagraphs_reason}"
                )
            else:
                counters["inductor"]["cudagraph_skips"] += 1
            BoxedBool.disable(cudagraphs)
        else:
            if is_backward:
                assert "boxed_forward_device_index" in graph_kwargs
                boxed_forward_device_index = graph_kwargs["boxed_forward_device_index"]
            else:
                # On the forward we don't know whether or not
                # boxed_forward_device_index is set yet
                boxed_forward_device_index = graph_kwargs.get(
                    "boxed_forward_device_index", None
                )
            if config.graph_partition:
                # with graph_partition=True, we skip some cudagraph checks if it's supported
                # with partition. So we have to use cudagraph_partition_post_compile.
                cudagraph_partition_post_compile(
                    example_inputs,
                    self,
                    cudagraphs,
                    constants.unwrap(self),
                    boxed_forward_device_index,
                )
            else:
                cudagraph_post_compile(
                    example_inputs,
                    self,
                    cudagraphs,
                    constants.unwrap(self),
                    boxed_forward_device_index,
                )
    inputs_to_check = self.inputs_to_check
    # cudagraphs could have been disabled from the earlier conditions
    # so we still need to realign inputs if that happens
    maybe_realign_inputs(
        cudagraphs,
        self,
        inputs_to_check,
        self.mutated_input_idxs,
    )

    # Apply inductor_compiled_code HOP wrapper if configured
    # This is done in post_compile to ensure it works with cached artifacts
    if self._wrap_compiled_regions and self.current_callable is not None:
        original_callable = self.current_callable

        def wrapped_callable(inputs):
            if is_in_torch_dispatch_mode():
                return inductor_compiled_code(original_callable, inputs)
            else:
                return original_callable(inputs)

        self.current_callable = wrapped_callable


@gorilla.patch(torch._inductor.output_code.CompiledFxGraph)
def __init__(
    self,
    current_callable: Optional[Callable[..., Any]],
    graph: GraphLowering,
    gm: torch.fx.GraphModule,
    output_strides: list[Optional[tuple[_StrideExprStr, ...]]],
    disabled_cudagraphs_reason: Optional[str],
    metrics_deltas: metrics.CachedMetricsDeltas,
    counter_deltas: Counter[str],
    cudagraphs: BoxedBool,
    example_inputs: Sequence[InputType],
    static_input_idxs: Sequence[int],
    fx_kwargs: _CompileFxKwargs,
    inputs_to_check: Sequence[int],
    runnable_graph_str: str,
    inductor_post_grad_graph_str: str,
    compiled_fn_runner: Optional[Any] = None,
    inductor_provenance_mapping_str: Optional[str] = None,
    inductor_provenance_stack_traces_str: Optional[str] = None,
) -> None:
    self.current_callable = current_callable
    self.compiled_fn_runner = compiled_fn_runner
    self.recursively_apply_fns = (
        compiled_fn_runner.recursively_apply_fns
        if compiled_fn_runner is not None
        else None
    )
    self.cache_key = graph.cache_key
    if graph.cache_path:
        with open(graph.cache_path) as f:
            self.source_code = f.read()
    self.runnable_graph_str = runnable_graph_str
    self.inductor_post_grad_graph_str = inductor_post_grad_graph_str
    self.inductor_provenance_mapping_str = inductor_provenance_mapping_str
    self.inductor_provenance_stack_traces_str = inductor_provenance_stack_traces_str
    self.cache_linemap = graph.cache_linemap
    # TODO - ordered set
    self.device_types = OrderedSet(graph.device_types)
    self.device_idxs = OrderedSet(graph.device_idxs)
    self.mutated_inputs = OrderedSet(graph.mutated_inputs)
    self.mutated_input_idxs = OrderedSet(graph.mutated_input_idxs)
    # We store the constant attributes in the cache entry and re-attach them
    # to the module created in PyCodeCache.load_by_key_path. In the case that
    # the graph has frozen parameters, we save the mapping from the attribute
    # names in the GraphLowering to the original name of the attribute in the
    # GraphModule. When we create the module from the cache entry, we then
    # look up the constants from the current GraphModule. This scheme allows
    # us to support caching with freezing.
    if not has_frozen_params(gm):
        self.constants = graph.constants
        self.frozen_param_names = {}
    else:
        self.constants = {}
        self.frozen_param_names = {}
        for k, v in graph.constants.items():
            if is_frozen_param(v):
                self.frozen_param_names[k] = graph.allocated_constant_name[k]
            else:
                self.constants[k] = v
    self.torchbind_constants = graph.torchbind_constants
    self.opaque_value_type_classes = graph.opaque_value_type_classes
    self.output_strides = output_strides
    self.disabled_cudagraphs_reason = disabled_cudagraphs_reason
    self.metrics_deltas = metrics_deltas
    self.counter_deltas = counter_deltas
    self.guards_expr = None
    self.cudagraph_info = None
    self.partition_maps = graph.partition_maps
    self.fx_kwargs = {}
    self.inputs_to_check = ()
    cudagraph_info = None
    if cudagraphs:
        # check cudagraph disabling reasons from inductor lowering
        if self.disabled_cudagraphs_reason:
            # Modify by CAMBRICON
            # if "cuda" in self.device_types:
            if "mlu" in self.device_types:
                # end Modify by CAMBRICON
                log_cudagraph_skip_and_bump_counter(
                    f"skipping cudagraphs due to {self.disabled_cudagraphs_reason}"
                )
            else:
                counters["inductor"]["cudagraph_skips"] += 1
            BoxedBool.disable(cudagraphs)
        else:
            complex_memory_overlap_inputs = any(
                complex_memory_overlap(t)
                for t in example_inputs
                if isinstance(t, torch.Tensor)
            )
            if not config.triton.cudagraph_support_input_mutation:
                # Skip supports for cudagraph-managed tensors
                from torch._inductor.cudagraph_utils import (
                    check_for_mutation_ignore_cuda_graph_managed_tensor,
                )

                has_mutation_str = check_for_mutation_ignore_cuda_graph_managed_tensor(
                    gm,
                    self.mutated_inputs,
                    self.mutated_input_idxs,
                    static_input_idxs,
                )
                has_mutation = has_mutation_str is not None
                if has_mutation:
                    self.disabled_cudagraphs_reason = has_mutation_str
            else:
                # Check mutation later to support cudagraph-managed tensors
                has_mutation = None
            cudagraph_tests = [
                (not has_mutation, "mutated inputs"),
                (not complex_memory_overlap_inputs, "complex memory overlap"),
                (
                    all(
                        isinstance(t, (torch.Tensor, torch.SymInt, torch.Generator))
                        for t in example_inputs
                    ),
                    "non-Tensor inputs",
                ),
            ]
            output = output_node(gm)
            # output args are tuple of first argument
            assert len(output.args) == 1
            stack_traces = [
                (arg.stack_trace if isinstance(arg, torch.fx.node.Node) else None)
                for arg in output.args[0]  # type: ignore[union-attr]
            ]
            cudagraph_fail_reasons = [s for b, s in cudagraph_tests if not b]
            placeholders = tuple(get_placeholder_info(gm.graph))
            cudagraph_info = CudagraphCachedInfo(
                placeholders, stack_traces, cudagraph_fail_reasons
            )
    self.cudagraph_info = cudagraph_info
    self.inputs_to_check = inputs_to_check
    self.fx_kwargs = fx_kwargs
    # aot autograd needs to know to pass in inputs as a list
    self._boxed_call = True

    # Store whether to wrap compiled regions in inductor_compiled_code HOP
    # This is set at compile time to avoid runtime overhead
    self._wrap_compiled_regions = config.wrap_inductor_compiled_regions
