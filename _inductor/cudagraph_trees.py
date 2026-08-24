import contextlib
import functools
import itertools
import threading
import warnings
import weakref
from collections import defaultdict
from typing import (
    Any,
    Generator,
    Optional,
    Sequence,
    Set,
    Union,
    NewType,
)
import torch
import torch_mlu
from torch import Tensor

from torch._inductor.utils import InputType
from torch.types import _bool

from torch._dynamo.utils import counters, preserve_rng_state
from torch._inductor.cudagraph_trees import (
    AliasesPriorGraphOutput,
    AliasesNewOutput,
    CompilationMode,
    CUDAGraphNode,
    CUDAGraphTreeManager,
    CUDAWarmupNode,
    GraphID,
    ExecutionState,
    FunctionID,
    OutputAliasInfo,
    PathOutputIndex,
    StackTraces,
    StorageDataPtr,
    StorageWeakRefWrapper,
    TreeManagerContainer,
    UnaliasedStorage,
    WrappedFunction,
    check_memory_pool,
    get_container,
    get_history_recording,
    is_live,
    map_to_ref,
    InputList,
    OutputList,
    LevelList,
    format_inputs_log,
)
from torch._inductor.compile_fx import (
    get_expanded_dims,
    static_input,
)
from torch._inductor.cudagraph_utils import (
    ModelType,
    OutputType,
    PlaceholderInfo,
)
from torch.utils.weak import TensorWeakRef
from torch.utils._ordered_set import OrderedSet
from torch.storage import UntypedStorage
from torch._inductor import config
from torch._guards import CompileId

_POOL_HANDLE = NewType("_POOL_HANDLE", tuple[int, int])

from ..utils import gorilla


if torch.backends.mlu.is_built():
    from torch._C import (
        _set_cached_tensors_enabled as _set_cached_tensors_enabled,
    )
    from torch_mlu._MLUC import (
        _mlu_MLUAllocator_AllocatorState as AllocatorState,
    )
else:

    class AllocatorState:  # type: ignore[no-redef]
        pass

    def _set_cached_tensors_enabled(enabled: _bool) -> None:
        pass


log = torch._logging.getArtifactLogger(__name__, "cudagraphs")


@contextlib.contextmanager
def enable_history_recording() -> Generator[None, None, None]:
    # Modify by CAMBRICON
    """
    "Turns on history recording in the CUDA Caching Allocator"
    enabled = torch._C._cuda_isHistoryEnabled()
    try:
        if not enabled:
            torch.cuda.memory._record_memory_history()
        yield
    finally:
        if not enabled:
            torch.cuda.memory._record_memory_history(None)
    """
    "Turns on history recording in the MLU Caching Allocator"
    import torch_mlu

    enabled = torch_mlu._MLUC._mlu_isHistoryEnabled()
    try:
        if not enabled:
            torch.mlu.memory._record_memory_history()
        yield
    finally:
        if not enabled:
            torch.mlu.memory._record_memory_history(None)
    # end Modify by CAMBRICON


patch = gorilla.Patch(
    torch._inductor.cudagraph_trees,
    "enable_history_recording",
    enable_history_recording,
)
gorilla.apply(patch)


def __init__(self, device_index: int) -> None:
    # This class keeps a strong reference to tree_manager,
    # but upon all other strong references to the tree_manager will reset it to None.
    # We need a strong reference so that we can still access its attributes upon cleanup.
    self.tree_manager: Optional[CUDAGraphTreeManager] = None

    # Number of outstanding references to the current tree manager
    self.live_cudagraphify_fns = 0

    self.device_index = device_index

    # Following two objects are only set in the case that Tensor outputs outlive
    # the cudagraphify_fns. Reference to the Graph is needed to keep the private pool from
    # deallocation.
    self.live_storages_count = 0
    # Modify by CAMBRICON
    # self.graph: Optional[torch.cuda.CUDAGraph] = None
    self.graph: Optional[torch.mlu.MLUGraph] = None
    # end Modify by CAMBRICON

    self.lock = threading.Lock()


patch = gorilla.Patch(
    torch._inductor.cudagraph_trees.TreeManagerContainer, "__init__", __init__
)
gorilla.apply(patch)


@contextlib.contextmanager
def _use_cuda_memory_pool_manager(
    device: int, mem_pool: tuple[int, int], stream: torch.mlu.Stream
) -> Generator[None, None, None]:
    """
    Context manager to use cuda graph pool for new allocations. If you use this manager
    all cudagraph tensors in use should be reflected in the allocator or they will be overwritten.
    existing_graph should already have been used in a capture, and the mem_pool must already exist,
    because this manager will not preserve a reference to the pool which keeps it alive.
    """
    # Modify by CAMBRICON: cuda -> mlu
    # torch.cuda.synchronize()
    # stream.wait_stream(torch.cuda.current_stream())

    # with torch.cuda.stream(stream), torch.device(device):
    #     # Begin allocate to mem pool for all memory allocation on the current thread.
    #     # This is thread safe since a thread can only warmup or record 1 cudagraph
    #     # at the same time.
    #     torch._C._cuda_beginAllocateCurrentThreadToPool(device, mem_pool)
    #     try:
    #         yield
    #     finally:
    #         torch._C._cuda_endAllocateToPool(device, mem_pool)
    #         torch._C._cuda_releasePool(device, mem_pool)

    # torch.cuda.current_stream().wait_stream(stream)
    import torch_mlu

    torch.mlu.synchronize()
    stream.wait_stream(torch.mlu.current_stream())

    with torch.mlu.stream(stream), torch.device(device):
        torch_mlu._MLUC._mlu_beginAllocateCurrentStreamToPool(device, mem_pool)
        try:
            yield
        finally:
            torch_mlu._MLUC._mlu_endAllocateCurrentStreamToPool(device, mem_pool)
            torch_mlu._MLUC._mlu_releasePool(device, mem_pool)

    torch.mlu.current_stream().wait_stream(stream)
    # end Modify by CAMBRICON


patch = gorilla.Patch(
    torch._inductor.cudagraph_trees,
    "_use_cuda_memory_pool_manager",
    _use_cuda_memory_pool_manager,
)
gorilla.apply(patch)


def __init__(
    self,
    wrapped_function: WrappedFunction,
    parent: Optional[Union[CUDAGraphNode, CUDAWarmupNode]],
    cuda_graphs_pool: tuple[int, int],
    # Modify by CAMBRICON
    # existing_cuda_graph: Optional[torch.cuda.CUDAGraph],
    existing_cuda_graph: Optional[torch.mlu.MLUGraph],
    device_index: int,
    stack_traces: Optional[StackTraces],
    # stream: torch.cuda.Stream,
    stream: torch.mlu.Stream,
    # end Modify by CAMBRICON
    already_warm: bool,
    id: GraphID,
) -> None:
    self.wrapped_function = wrapped_function
    self.parent: Optional[Union[CUDAGraphNode, CUDAWarmupNode]] = parent
    self.cuda_graphs_pool = cuda_graphs_pool
    self.outputs_weakrefs: list[Optional[StorageWeakRefWrapper]] = []
    self.tensor_weakrefs: list[Optional[TensorWeakRef]] = []
    self.existing_cuda_graph = existing_cuda_graph
    self.has_run = False
    self.device_index = device_index
    self.stack_traces = stack_traces
    self.stream = stream
    self.already_warm = already_warm
    self.id = id


patch = gorilla.Patch(
    torch._inductor.cudagraph_trees.CUDAWarmupNode, "__init__", __init__
)
gorilla.apply(patch)


def run(self, new_inputs: Any) -> OutputType:
    assert not self.has_run, "Wrapped function should never be run twice"

    # See: output_is_alias_of_persistent_static_inputs below. We should only be returning freshly created
    # storages in path_live_weakrefs.
    existing_path_data_ptrs = OrderedSet(
        [t.data_ptr() for t in self.path_live_weakrefs() if t()]
    )

    def get_non_cudagraph_inps() -> list[weakref.ReferenceType[UntypedStorage]]:
        non_cudagraph_inps = [
            weakref.ref(t.untyped_storage())
            for t in itertools.chain(new_inputs, self.wrapped_function.constants)
            if isinstance(t, torch.Tensor)
            and t.untyped_storage().data_ptr() not in existing_path_data_ptrs
        ]
        return non_cudagraph_inps

    non_cudagraph_inps_storages = get_non_cudagraph_inps()

    if config.triton.slow_path_cudagraph_asserts and not self.already_warm:
        refs = list(self.path_live_weakrefs())
        check_memory_pool(self.device_index, self.cuda_graphs_pool, refs)

    # Modify by CAMBRICON
    # with (
    #     torch.cuda.device(self.device_index),
    #     disable_conv_cache_emptying(),
    #     clear_cublas_manager(),
    #     _use_cuda_memory_pool_manager(
    #         self.device_index, self.cuda_graphs_pool, self.stream
    #     ),
    #     get_history_recording(),
    # ):
    with (
        torch.mlu.device(self.device_index),
        _use_cuda_memory_pool_manager(
            self.device_index, self.cuda_graphs_pool, self.stream
        ),
        get_history_recording(),
    ):
        out = self.wrapped_function.model(new_inputs)
    # end Modify by CAMBRICON

    # We need to know which outputs are allocated within the cudagraph pool
    # so that we can deallocate them at the beginning of the next cudagraph step,
    # and set their access to error.
    # We use a weakref to the inputs storage, in case a block which was previously
    # allocated to the general caching allocator pool gets reallocated to a private pool.

    non_cudagraph_inps_storage_ptrs = OrderedSet[Any]()
    for storage in non_cudagraph_inps_storages:
        s = storage()
        if s is not None:
            non_cudagraph_inps_storage_ptrs.add(s._cdata)

    assert len(new_inputs) == 0

    # sdpa returns cpu tensors when not recording cuda graph
    def add_ref(o: Any) -> bool:
        # Modify by CAMBRICON
        # return (
        #     isinstance(o, torch.Tensor)
        #     and o.is_cuda
        #     and o.untyped_storage()._cdata not in non_cudagraph_inps_storage_ptrs
        #     and o.untyped_storage().data_ptr() != 0
        # )
        return (
            isinstance(o, torch.Tensor)
            and o.is_mlu
            and o.untyped_storage()._cdata not in non_cudagraph_inps_storage_ptrs
            and o.untyped_storage().data_ptr() != 0
        )

    self.outputs_weakrefs.extend([map_to_ref(o) if add_ref(o) else None for o in out])
    self.tensor_weakrefs.extend([TensorWeakRef(o) if add_ref(o) else None for o in out])
    # end Modify by CAMBRICON

    if config.triton.slow_path_cudagraph_asserts and not self.already_warm:
        out_refs = list(self.path_live_weakrefs())
        check_memory_pool(self.device_index, self.cuda_graphs_pool, out_refs)

    return out


patch = gorilla.Patch(torch._inductor.cudagraph_trees.CUDAWarmupNode, "run", run)
gorilla.apply(patch)


def __init__(
    self,
    wrapped_function: WrappedFunction,
    id: GraphID,
    parent: Optional[CUDAGraphNode],
    inputs: list[InputType],
    cuda_graphs_pool: _POOL_HANDLE,
    device_index: int,
    stack_traces: Optional[StackTraces],
    stream: torch.mlu.Stream,
    mode: Optional[CompilationMode],
    compile_id: Optional[CompileId],
) -> None:
    assert isinstance(inputs, (list, tuple))

    self.wrapped_function = wrapped_function
    self.id = id
    self.device = device_index
    self.stack_traces = stack_traces
    self.stream = stream

    # Enable re-record a cudagraph when static tensor address changed.
    # if not we should error when it changed.
    self.rerecord_if_static_inputs_change = (
        torch._dynamo.config.inline_inbuilt_nn_modules
        or torch._inductor.config.triton.cudagraph_support_input_mutation
    )

    # if this is a root parent will be None. use weakref to prevent reference cycle
    self._parent = weakref.ref(parent) if parent is not None else None
    # reference to the shared memory pool for the entire cuda graphs tree
    self.cuda_graphs_pool = cuda_graphs_pool

    # A single wrapped function may be recorded multiple times if memory patterns or
    # invariants change from one execution to the next
    self.children: dict[FunctionID, list[CUDAGraphNode]] = defaultdict(list)

    # StorageWeakRef maintains whether the Storage C++ object remains allocated,
    # not whether the corresponding memory has been deallocated. In order
    # to use them to track memory deallocations we must maintain a single StorageWeakRef
    # for all Storages that reference that memory (even if we are constructing Storages
    # that do not have a deallocator function). We maintain one single storage_cache
    # as we execute any tree path. When we retrieve a storage from the cache we
    # check that it is still alive, and we hash based on observed recording data ptr
    # and storage cdata.

    # we preserve a single reference to executed outputs that is then referenced
    # in children to avoid children having to chase parent pointers in the hot path
    # DO NOT reassign output_weakrefs, only call `clear()`
    # Path is a series of nodes from root to the current node
    self.outputs_weakrefs: OutputList[Optional[StorageWeakRefWrapper]] = []
    self.path_weakrefs: LevelList[OutputList[Optional[StorageWeakRefWrapper]]] = [
        node.outputs_weakrefs for node in self._path_from_root
    ]
    self.path_stacktraces: LevelList[Optional[StackTraces]] = [
        node.stack_traces for node in self._path_from_root
    ]
    self.tensor_weakrefs: OutputList[Optional[TensorWeakRef]] = []

    # tensors which are outputs of previous graphs in the tree
    self.cudagraph_managed_idxs: list[int] = [
        idx
        for idx, t in enumerate(inputs)
        if isinstance(t, torch.Tensor) and self._is_cuda_graph_recorded_tensor(t)
    ]

    # (depth, offset) of live tensors which are alias of previous graph outputs
    self.live_cudagraph_managed_path_refs: InputList[Optional[PathOutputIndex]] = [
        (
            self._is_alias_of_live_recorded_tensor(t)
            if isinstance(t, torch.Tensor)
            else None
        )
        for t in inputs
    ]

    # when replay, preserve the liveness of an input if it AliasesPriorGraphOutput
    # and also aliases an output of the current CUDAGraphNode
    self.preserved_aliased_inputs: InputList[bool] = [False] * len(inputs)

    self.static_input_idxs: list[int] = list(
        OrderedSet(wrapped_function.static_input_idxs)
        | OrderedSet(self.cudagraph_managed_idxs)
    )

    self.non_static_input_idx: LevelList[int] = [
        i for i in range(len(inputs)) if i not in self.static_input_idxs
    ]

    counters["inductor"]["cudagraph_recorded_non_static_inputs"] += len(
        self.non_static_input_idx
    )

    self.non_managed_static_input_idxs: LevelList[int] = [
        i
        for i in wrapped_function.static_input_idxs
        if i not in self.cudagraph_managed_idxs
    ]

    # Modify by CAMBRICON
    from torch._inductor.utils import InputType

    # end Modify by CAMBRICON

    def maybe_get_static_data_ptr(
        idx: int,
        inputs: list[InputType],
        static_input_idxs: list[int],
    ) -> Optional[int]:
        inp = inputs[idx]
        if isinstance(inp, torch.Tensor) and idx in static_input_idxs:
            return inp.data_ptr()
        return None

    self.static_input_data_ptrs: InputList[Optional[int]] = [
        # pyrefly: ignore [bad-argument-type]
        maybe_get_static_data_ptr(i, inputs, self.static_input_idxs)
        for i in range(len(inputs))
    ]

    # When we checkpoint, and free generations, we will be manually freeing the outputs
    # of CUDAGraphNodes. We should not be freeing parameters, not do we need to account for
    # their liveness (they are static), so we need to compute which outputs are aliases of
    # parameters. Some static inputs are saved tensors from the forward that die in the backward.
    # Their locations are static but lifetimes are not. We only include the persistent static
    # data ptrs below because the non persistent data ptrs may be outputs of this record and
    # fresh allocations.

    # precompute expanded dims to avoid computing in the hot path
    self.expanded_dims: list[list[int]] = [
        get_expanded_dims(x)
        if isinstance(x, torch.Tensor) and idx not in self.static_input_idxs
        else []
        for idx, x in enumerate(inputs)
    ]

    # For each node in path, which outputs were observed to be live
    # before invoking graph recording, and after graph recording
    self.recorded_liveness_before_graph: LevelList[OutputList[bool]] = []
    self.recorded_liveness_after_graph: LevelList[OutputList[bool]] = []

    # List of tuples of (depth, output_index) that index into node at depth
    # number of nodes from root and output_index of outputs. Will index into
    # path_weakrefs.
    self.expected_dead_indices_before_graph: list[PathOutputIndex] = []
    self.expected_dead_indices_after_graph: list[PathOutputIndex] = []

    # all live indices after graph recording
    self.live_indices_after_graph: list[PathOutputIndex] = []

    if self.parent is not None:
        previous_liveness = self.parent.recorded_liveness_after_graph
        curr_liveness = self._get_liveness(self.path_weakrefs)

        different_indices = self._get_different_indices(
            previous_liveness, curr_liveness
        )

        self.recorded_liveness_before_graph = curr_liveness
        self.expected_dead_indices_before_graph = different_indices

    rng_states = [inp for inp in inputs if isinstance(inp, torch.Generator)]
    # pyrefly: ignore [bad-argument-type]
    recording_inputs = self._allocate_and_copy_recording_inputs(inputs)
    # recording inputs will copy over memory, so we can free non recording inputs
    # pyrefly: ignore [missing-attribute]
    inputs.clear()
    del inputs

    # graph used for recording model invocation
    # Modify by CAMBRICON
    # self.graph: Optional[torch.cuda.CUDAGraph] = torch.cuda.CUDAGraph()
    self.graph: Optional[torch.mlu.MLUGraph] = torch.mlu.MLUGraph()
    # end Modify by CAMBRICON

    # TODO: register_generator_state should potentially take explicit device
    # Modify by CAMBRICON
    # with torch.cuda.device(self.device):
    with torch.mlu.device(self.device):
        # end Modify by CAMBRICON
        for rng_state in rng_states:
            self.graph.register_generator_state(rng_state)

    # we allocate non-static inputs within the same memory pool as the CUDAGraph
    # which we will record the model with. For memory efficiency, it is important
    # to reclaim the input memory when the inputs are no longer live. To accomplish this,
    # we reconstruct tensors at the correct data pointers of our inputs which are
    # non owning and do not prevent deallocation. On subsequent executions, input values
    # will be copied over to these tensors.
    self.reconstructed_inputs: list[InputType] = [
        self._reconstruct_from_tensor_metadata(self._tensor_metadata(x))
        if isinstance(x, torch.Tensor)
        else x
        for x in recording_inputs
    ]

    # DO THE RECORDING!!!
    # We record the CUDA graph in the constructor of CUDAGraphNode, which
    # gives you what the CPU side compute of the function would do.  We
    # don't throw the recording outputs away: their memory is
    # correctly accounted for in the CUDAGraphs caching allocator.  This
    # means on the very FIRST run of the CUDA graph node, we can directly
    # do more recording, because we have a valid caching allocator state.
    # NB: This relies on run() being called immediately after the
    # constructor, otherwise this optimization would not be valid.

    # initialized below in _record

    self.checkpointed_caching_state: Optional[AllocatorState] = None

    # Output Storage Alias information, can be:
    # - A new, unaliased storage, or the output is None
    # - An alias of an output of a prior graph
    # - An alias of an output already created in the reconstructed outputs
    # This is None if the output in question is an int
    self.output_storage_alias: OutputList[Optional[OutputAliasInfo]] = []

    # is the output Storage unaliased in subsequent outputs, of all subsequent paths
    # if it is, we cached the output tensor and adjust storage liveness tracking to also
    # check if the output tensor does not have an additional python reference.
    # If a descendent node discovers it has an alias of a prior output, then the output
    # will no longer be cached in the ancestor.
    # The large majority of tensors are unaliased, and preserving aliased output tensors would add
    # significant additional complexity with marginal gains
    # The cached tensor outputs are added on the first execution, and cleared whenever we need
    # to do subsequent recording
    self.unaliased_in_all_paths: OutputList[bool] = []
    self.cached_tensor_outputs: OutputList[Optional[Tensor]] = []

    # if an output aliases a static, persistent input then the corresponding Tensor will
    # be set here. These are different than cached tensors, because they are tensors that
    # are aliases of parameters that are always live.
    self.static_output_tensors: OutputList[Optional[Tensor]] = []

    # Cleared after recording
    with dynamo_timed_cudagraph("CUDAGraphNode.record", compile_id, mode):
        self.recording_outputs: Optional[OutputType] = self._record(
            wrapped_function.model, recording_inputs
        )
    self.outputs_metadata: OutputList[Union[dict[str, Any], int, None]] = []

    # As with inputs, we do not want to keep the outputs permanently alive because that would prevent
    # their memory being reclaimed in subsequent cuda graph recordings. We record the tensor metadata
    # needed to reconstruct instead.
    assert self.recording_outputs is not None
    for out in self.recording_outputs:
        if isinstance(out, torch.Tensor):
            self.outputs_metadata.append(
                self._tensor_metadata(out, ignore_storage_offset=False)
            )
        else:
            assert isinstance(out, (int, type(None))), type(out)
            self.outputs_metadata.append(out)

    self.graph.replay()


patch = gorilla.Patch(
    torch._inductor.cudagraph_trees.CUDAGraphNode, "__init__", __init__
)
gorilla.apply(patch)


def _copy_inputs_and_remove_from_src(
    self, dsts: list[InputType], srcs: list[InputType]
) -> None:
    dst_tensors = []
    src_tensors = []

    for idx in self.non_static_input_idx:
        if not isinstance(srcs[idx], torch.Tensor):
            continue
        expanded_dims = self.expanded_dims[idx]
        # Modify by CAMBRICON
        # dst_tensors.append(index_expanded_dims(dsts[idx], expanded_dims))  # type: ignore[arg-type]
        # src_tensors.append(index_expanded_dims(srcs[idx], expanded_dims))  # type: ignore[arg-type]
        if expanded_dims == []:
            dst_tensors.append(dsts[idx])
            src_tensors.append(srcs[idx])
        else:
            dst_tensors.append(index_expanded_dims(dsts[idx], expanded_dims))  # type: ignore[arg-type]
            src_tensors.append(index_expanded_dims(srcs[idx], expanded_dims))  # type: ignore[arg-type]
        # end Modify by CAMBRICON
        srcs[idx] = None  # type: ignore[call-overload]

    # Fails on empty lists
    if dst_tensors:
        torch._foreach_copy_(dst_tensors, src_tensors)


patch = gorilla.Patch(
    torch._inductor.cudagraph_trees.CUDAGraphNode,
    "_copy_inputs_and_remove_from_src",
    _copy_inputs_and_remove_from_src,
)
gorilla.apply(patch)


def run(self, new_inputs: list[InputType]) -> OutputType:
    self.check_static_inputs_are_stable(new_inputs)

    self._copy_inputs_and_remove_from_src(self.reconstructed_inputs, new_inputs)

    self.run_graph()

    outputs = self.reconstruct_outputs()
    new_inputs.clear()

    if config.triton.fast_path_cudagraph_asserts:
        self.debug_check_invariants_after_invocation()

    # Modify by CAMBRICON
    if config.triton.force_cudagraph_sync:
        # torch.cuda.synchronize()
        torch.mlu.synchronize()
    # end Modify by CAMBRICON

    # Reset this to run the check in the future
    self.static_inputs_stable = False

    return outputs


patch = gorilla.Patch(torch._inductor.cudagraph_trees.CUDAGraphNode, "run", run)
gorilla.apply(patch)


def _record(self, model: ModelType, inputs: list[InputType]) -> OutputType:
    "Record the model"
    assert self.graph is not None
    # Modify by CAMBRICON
    from collections.abc import Generator

    # end Modify by CAMBRICON

    def static_input_iter() -> Generator[torch.Tensor, None, None]:
        for i in self.wrapped_function.static_input_idxs:
            _inp = inputs[i]
            if isinstance(
                _inp, torch.Tensor
            ) and not self._is_cuda_graph_recorded_tensor(_inp):
                yield _inp

    # see: output_is_alias_of_persistent_static_inputs above
    static_input_persistent_storage_ptrs: dict[int, StorageWeakRefWrapper] = {
        inp.untyped_storage().data_ptr(): StorageWeakRefWrapper(inp)
        for inp in itertools.chain(static_input_iter(), self.wrapped_function.constants)
    }

    if config.triton.slow_path_cudagraph_asserts:
        # need to use parent live weakrefs because live_indices isn't set yet
        memory = [] if self.parent is None else list(self.parent.path_live_weakrefs())
        memory += [
            StorageWeakRefWrapper(elem)
            for i, elem in enumerate(inputs)
            if isinstance(elem, torch.Tensor)
            and i not in self.wrapped_function.static_input_idxs
            and elem.untyped_storage().data_ptr() != 0
        ]
        check_memory_pool(self.device, self.cuda_graphs_pool, memory)

    # Modify by CAMBRICON
    # with (
    #    preserve_rng_state(),
    #    torch.cuda.device(self.device),
    #    clear_cublas_manager(),
    #    torch.cuda.graph(
    #        self.graph,
    #        stream=self.stream,
    #        pool=self.cuda_graphs_pool,
    #        capture_error_mode="thread_local",
    #    ),
    #    get_history_recording(),
    # ):
    with (
        preserve_rng_state(),
        torch.mlu.device(self.device),
        torch.mlu.graph(
            self.graph,
            stream=self.stream,
            pool=self.cuda_graphs_pool,
            capture_error_mode="thread_local",
        ),
        get_history_recording(),
    ):
        # end Modify by CAMBRICON
        static_outputs = model(inputs)

    # running model should reclaim memory
    assert len(inputs) == 0

    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    # pyrefly: ignore [bad-argument-type]
    self._add_first_outputs(static_outputs, static_input_persistent_storage_ptrs)

    # pyrefly: ignore [bad-return]
    return static_outputs


patch = gorilla.Patch(torch._inductor.cudagraph_trees.CUDAGraphNode, "_record", _record)
gorilla.apply(patch)


def _add_first_outputs(
    self,
    outputs: OutputType,
    static_input_persistent_storage_ptrs: dict[int, StorageWeakRefWrapper],
) -> None:
    "Add the outputs from the first invocation of the node and set up metadata"

    # getting liveness before we have added the outputs to path, so the length
    # of the two lists is equal
    prev_liveness = self.recorded_liveness_before_graph
    curr_liveness = self._get_liveness(self.path_weakrefs)

    delta = self._get_different_indices(prev_liveness, curr_liveness)
    self.expected_dead_indices_after_graph = delta

    assert len(self.outputs_weakrefs) == 0
    # index from data pointer to index in outputs
    output_new_storages_index: dict[StorageDataPtr, int] = {}

    self.unaliased_in_all_paths = [False for _ in range(len(outputs))]
    self.static_output_tensors = [None for _ in range(len(outputs))]

    for i, o in enumerate(outputs):
        if o is None or not isinstance(o, torch.Tensor):
            self.output_storage_alias.append(UnaliasedStorage)
            continue

        # Modify by CAMBRICON
        """
        torch._check(
            o.is_cuda or o.untyped_storage().data_ptr() == 0,

            lambda: (
                "Expected all cuda outputs in cuda graph recording. Non cuda output "
                f"from {self.stack_traces[i] if self.stack_traces else '(unknown)'}"
            ),
        )
        """
        torch._check(
            o.is_mlu or o.untyped_storage().data_ptr() == 0,
            lambda: (
                "Expected all cuda outputs in cuda graph recording. Non cuda output "
                f"from {self.stack_traces[i] if self.stack_traces else '(unknown)'}"
            ),
        )
        # end Modify by CAMBRICON

        ref = static_input_persistent_storage_ptrs.get(o.untyped_storage().data_ptr())
        # also treat empty storages as static outputs because we do not need to manage their lifetime
        # and they should not participate in checkpointing
        is_empty_storage = o.untyped_storage().data_ptr() == 0
        if (ref and ref() is not None) or is_empty_storage:
            self.output_storage_alias.append(None)
            self.static_output_tensors[i] = o
            continue

        path_ref = self._is_alias_of_live_recorded_tensor(o)
        if path_ref is not None:
            self._mark_prior_graph_output_as_aliased(path_ref)

            for idx, inp_path_ref in enumerate(self.live_cudagraph_managed_path_refs):
                if path_ref == inp_path_ref:
                    self.preserved_aliased_inputs[idx] = True
            self.output_storage_alias.append(AliasesPriorGraphOutput(path_ref))
            continue

        if o.untyped_storage().data_ptr() in output_new_storages_index:
            index = output_new_storages_index[o.untyped_storage().data_ptr()]
            self.unaliased_in_all_paths[index] = False
            self.output_storage_alias.append(AliasesNewOutput(index))
            continue

        output_new_storages_index[o.untyped_storage().data_ptr()] = i
        self.output_storage_alias.append(UnaliasedStorage)
        self.unaliased_in_all_paths[i] = True

    if self.stack_traces is None:
        self.stack_traces = [None for _ in range(len(outputs))]
    else:
        assert len(self.stack_traces) == len(
            outputs
        ), "Wrong number of stack traces passed in"

    assert not self.outputs_weakrefs
    for out, static_output_tensor in zip(outputs, self.static_output_tensors):
        if not isinstance(out, torch.Tensor) or static_output_tensor is not None:
            self.outputs_weakrefs.append(None)
            self.tensor_weakrefs.append(None)
        else:
            self.outputs_weakrefs.append(StorageWeakRefWrapper(out))
            self.tensor_weakrefs.append(TensorWeakRef(out))

    self.recorded_liveness_after_graph = self._get_liveness(self.path_weakrefs)
    self.checkpointed_caching_state = torch._C._cuda_getCheckpointState(
        self.device, self.cuda_graphs_pool
    )

    # now, get liveness with outputs added
    for depth in range(len(self.path_weakrefs)):
        for output_index in range(len(self.path_weakrefs[depth])):
            if is_live(self.path_weakrefs[depth][output_index]):
                self.live_indices_after_graph.append((depth, output_index))

    self.debug_check_invariants_after_invocation()
    if config.triton.slow_path_cudagraph_asserts:
        check_memory_pool(
            self.device, self.cuda_graphs_pool, list(self.path_live_weakrefs())
        )


patch = gorilla.Patch(
    torch._inductor.cudagraph_trees.CUDAGraphNode,
    "_add_first_outputs",
    _add_first_outputs,
)
gorilla.apply(patch)


def _allocate_and_copy_recording_inputs(
    self, inputs: list[InputType]
) -> list[InputType]:
    """
    Allocate inputs for non static, non cudagraph managed tensors in the memory pool
    and copy over the tensor values.
    """

    # Modify by CAMBRICON
    # torch.cuda.synchronize()
    # self.stream.wait_stream(torch.cuda.current_stream())
    torch.mlu.synchronize()
    self.stream.wait_stream(torch.mlu.current_stream())
    # end Modify by CAMBRICON
    recording_inputs: list[InputType] = []

    with (
        warnings.catch_warnings(record=True),
        # Modify by CAMBRICON
        # torch.cuda.device(self.device),
        torch.mlu.device(self.device),
        # end Modify by CAMBRICON
        _use_cuda_memory_pool_manager(
            self.device,
            mem_pool=self.cuda_graphs_pool,
            stream=self.stream,
        ),
    ):
        for i, inp in enumerate(inputs):
            if not isinstance(inp, torch.Tensor):
                assert isinstance(inp, (int, torch.Generator))
                # pyrefly: ignore [bad-argument-type]
                recording_inputs.append(inp)
            elif i not in self.static_input_idxs:
                # static_input does an allocation!
                recording_inputs.append(static_input(inp))
            else:
                recording_inputs.append(inp)

        self._copy_inputs_and_remove_from_src(recording_inputs, inputs)

    return recording_inputs


patch = gorilla.Patch(
    torch._inductor.cudagraph_trees.CUDAGraphNode,
    "_allocate_and_copy_recording_inputs",
    _allocate_and_copy_recording_inputs,
)
gorilla.apply(patch)


def get_cudagraph_segments(pool_id: tuple[int, int]) -> Any:
    # Modify by CAMBRICON
    # segments = torch.cuda.memory_snapshot()
    segments = torch.mlu.memory_snapshot()
    # end Modify by CAMBRICON
    return [segment for segment in segments if segment["segment_pool_id"] == pool_id]


patch = gorilla.Patch(
    torch._inductor.cudagraph_trees, "get_cudagraph_segments", get_cudagraph_segments
)
gorilla.apply(patch)


def __init__(self, device_index: int) -> None:
    # roots are functions which have no dependencies on an other node. I.e.,
    # when they are first invoked, none of their inputs are outputs are outputs
    # of another node, nor are there any live outputs of another node whose
    # liveness would create a dependency.
    self.roots: dict[FunctionID, list[CUDAGraphNode]] = defaultdict(list)

    # mapping from function id to wrapped function
    self.ids_to_funcs: dict[FunctionID, WrappedFunction] = {}

    self.ids_to_stack_traces: dict[FunctionID, Optional[StackTraces]] = {}

    self.warmed_up_functions: OrderedSet[FunctionID] = OrderedSet()
    # if we fail to increment generation, and are stuck warming up,
    # only warn on each function once
    self.warned_functions: OrderedSet[FunctionID] = OrderedSet()
    torch._C._set_cached_tensors_enabled(True)

    # warn only once if a function mutates inputs
    self.warned_mutation: OrderedSet[FunctionID] = OrderedSet()

    # NB: cuda caching allocator will remember the stream a segment is allocated to
    # and only allocate that segment to the same stream. we need to use a single stream
    # for all allocations to the memory pool, otherwise the allocations to separate streams
    # will not be reused; separate recordings would have use the same memory pool, but not
    # the same memory.

    # Modify by CAMBRICON
    """
    with torch.cuda.device(device_index):
        torch.cuda.synchronize()
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())

        # Keeps Memory Pool Alive
        self.graph: Optional[torch.cuda.CUDAGraph] = torch.cuda.CUDAGraph()
        self.cuda_graphs_thread_pool = torch.cuda.graph_pool_handle()

        with (
            warnings.catch_warnings(record=True),
            torch.cuda.graph(
                self.graph,
                pool=self.cuda_graphs_thread_pool,
                stream=self.stream,
                capture_error_mode="thread_local",
            ),
        ):
            pass
    """
    with torch.mlu.device(device_index):
        torch.mlu.synchronize()
        self.stream = torch.mlu.Stream()
        self.stream.wait_stream(torch.mlu.current_stream())

        # Keeps Memory Pool Alive
        self.graph: Optional[torch.mlu.MLUGraph] = torch.mlu.MLUGraph()
        self.cuda_graphs_thread_pool = torch.mlu.graph_pool_handle()

        with (
            warnings.catch_warnings(record=True),
            torch.mlu.graph(
                self.graph,
                pool=self.cuda_graphs_thread_pool,
                stream=self.stream,
                capture_error_mode="thread_local",
            ),
        ):
            pass
        # end Modify by CAMBRICON

    self.graph_counter = itertools.count(0)
    self.func_counter = itertools.count(0)

    # mapping from graph_id to (function id to mutation type hint) since we are
    # specializing on a particular combination of Parent Node -> Function ID.
    self.non_cudagraph_managed_mutation_hint: dict[
        Optional[GraphID], dict[FunctionID, bool]
    ] = defaultdict(dict)
    self.warmup_node_counter = itertools.count(start=-1, step=-1)

    # mapping from graph_id to (function id to re-record count). We fall back to
    # eager function if a function is re-recorded frequently on a node.
    self.num_rerecord: dict[Optional[GraphID], dict[FunctionID, int]] = defaultdict(
        lambda: defaultdict(lambda: 0)
    )

    # whether we the current node is in a state of warmup, recording, execution. If
    # there is no current node the state will be ExecutionState.None.
    self.path_state = ExecutionState.NONE
    self.device_index = device_index

    # the most recently invoked cudagraph wrapping of a function. Will be None
    # when there is no output from a previous recording or execution whose memory
    # we need to respect in the cuda caching allocation. If you incremented generation,
    # this will also be none, as ignore those allocations.
    self.current_node: Optional[Union[CUDAGraphNode, CUDAWarmupNode]] = None

    # current generation of cudagraph invocations. when torch.compile is run
    # we increment the current generation. are willing to ignore live outputs
    # of a previous generation in checking liveness.
    self.current_gen: int = -1

    # number of instances we are in execution and failed to match to an
    # existing child
    self.debug_fail_counter = 0
    # number of instances we had to checkpoint the function
    self.debug_checkpointing_counter = 0

    self.id_to_mode: dict[FunctionID, CompilationMode] = {}
    self.id_to_compile_id: dict[FunctionID, Optional[CompileId]] = {}

    # Note: [Backward Generation Handling]
    # We generally perform a sequence of forward executions followed by backward executions.
    # If multiple torch.compile wrapped forwards are executed with their backwards pending,
    # we should not disregard the outputs from a prior torch.compile since the entire training
    # loop hasn't completed.  Occasionally, a backward pass corresponding to a forward pass may
    # not be executed, so we cannot wait for all pending forward pass backward completions, so
    # we cannot wait for all backwards to have been invoked. Instead we wait for a single backward
    # invocation. Triggering a backward pass typically doesn't lead to another torch.compile
    # invocation, making it less likely for the generation to increase between multiple
    # backward calls. The following use case is covered by this approach:
    # mod1 = torch.compile(...)
    # mod2 = torch.compile(...)
    # mod2(mod1(x)).sum().backward()

    self.running_forwards_with_pending_backwards = False
    self.mode: Optional[CompilationMode] = None

    self.disable_invalidate_aliases = (
        False
        if not torch._environment.is_fbcode()
        else torch._utils_internal.justknobs_check(
            "pytorch/inductor:disable_cudagraph_alias_invalidation"
        )
    )


patch = gorilla.Patch(
    torch._inductor.cudagraph_trees.CUDAGraphTreeManager, "__init__", __init__
)
gorilla.apply(patch)


def record_function(
    self, new_inputs: list[InputType], function_id: FunctionID
) -> OutputType:
    assert not isinstance(self.current_node, CUDAWarmupNode)
    with torch._dynamo.callback_handler.install_callbacks(
        CallbackTrigger.CUDAGRAPH_RECORDING, str(self.compile_id)
    ):
        graph_id = self.new_graph_id()
        log.debug(
            "[%s] Recording function=%s, cuda_graph_id=%d, inputs: %s",
            self.compile_id,
            self.get_func_name(function_id),
            graph_id.id,
            format_inputs_log(new_inputs),
        )
        # Modify by CAMBRICON
        # torch.cuda.synchronize()
        torch.mlu.synchronize()
        # end Modify by CAMBRICON
        node = CUDAGraphNode(
            self.ids_to_funcs[function_id],
            graph_id,
            self.current_node,
            new_inputs,
            self.cuda_graphs_thread_pool,
            self.device_index,
            self.ids_to_stack_traces[function_id],
            self.stream,
            self.mode,
            self.compile_id,
        )
        if self.current_node is None:
            self.roots[function_id].append(node)
        else:
            self.current_node.add_child(function_id, node)
        self.current_node = node
        self.path_state = ExecutionState.RECORDING
        self.update_generation()
        # Modify by CAMBRICON
        # torch.cuda.synchronize()
        torch.mlu.synchronize()
        # end Modify by CAMBRICON
        return node.run_first_inputs(new_inputs)


patch = gorilla.Patch(
    torch._inductor.cudagraph_trees.CUDAGraphTreeManager,
    "record_function",
    record_function,
)
gorilla.apply(patch)


def add_function(
    self,
    model: ModelType,
    inputs: list[InputType],
    static_input_idxs: Sequence[int],
    stack_traces: Optional[StackTraces],
    mode: CompilationMode,
    constants: tuple[torch.Tensor, ...],
    placeholders: tuple[PlaceholderInfo, ...],
    mutated_input_idxs: tuple[int, ...],
    compile_id: Optional[CompileId],
) -> tuple[ModelType, OutputType,]:
    id = self.new_func_id()
    self.ids_to_stack_traces[id] = stack_traces
    # Modify by CAMBRICON
    """
    self.ids_to_funcs[id] = WrappedFunction(
        model,
        list(static_input_idxs),
        id,
        tuple(t for t in constants if isinstance(t, torch.Tensor) and t.is_cuda),
        placeholders,
        mutated_input_idxs,
    )
    """
    self.ids_to_funcs[id] = WrappedFunction(
        model,
        list(static_input_idxs),
        id,
        tuple(t for t in constants if isinstance(t, torch.Tensor) and t.is_mlu),
        placeholders,
        mutated_input_idxs,
    )
    # end Modify by CAMBRICON
    self.id_to_mode[id] = mode
    self.id_to_compile_id[id] = compile_id
    fn = functools.partial(self.run, function_id=id)

    # container needs to set clean up when fn dies
    get_container(self.device_index).add_strong_reference(fn)
    return fn, fn(inputs)


patch = gorilla.Patch(
    torch._inductor.cudagraph_trees.CUDAGraphTreeManager, "add_function", add_function
)
gorilla.apply(patch)


@contextlib.contextmanager
def disable_conv_cache_emptying() -> Generator[None, None, None]:
    # Modify by CAMBRICON
    """
    prev = torch._C._cuda_get_conv_benchmark_empty_cache()
    torch._C._cudnn_set_conv_benchmark_empty_cache(False)
    try:
        yield
    finally:
        torch._C._cudnn_set_conv_benchmark_empty_cache(prev)
    """
    try:
        yield
    finally:
        pass
    # end Modify by CAMBRICON


patch = gorilla.Patch(
    torch._inductor.cudagraph_trees,
    "disable_conv_cache_emptying",
    disable_conv_cache_emptying,
)
gorilla.apply(patch)
