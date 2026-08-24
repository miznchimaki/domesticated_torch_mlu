import functools
import re
from typing import Set, Union, Optional, Any

import torch
from torch._dynamo.resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX
from torch._dynamo.trace_rules import (
    _as_posix_path,
    _module_dir,
    get_legacy_mod_inlinelist,
    get_torch_obj_rule_map,
    is_aten_op_or_tensor_method,
    is_fbcode,
    is_torch_inline_allowed,
    load_object,
    SkipResult,
    FBCODE_INLINE_FILES_IN_SKIPPED_DIRS_RE,
    FBCODE_SKIP_DIRS,
    FBCODE_SKIP_DIRS_RE,
    FBCODE_SKIP_TORCHREC_DIRS,
    FBCODE_SKIP_TORCHREC_DIRS_RE,
    FORCE_SKIP_FILES,
    MOD_INLINELIST,
    SKIP_DIRS,
)
from torch._dynamo.utils import getfile, hashable, is_lru_cache_wrapped_function
from torch._dynamo.variables import (
    SkipFunctionVariable,
    TorchInGraphFunctionVariable,
    UserFunctionVariable,
)
from torch._dynamo import config
from torch._dynamo.variables.base import VariableTracker
from ..utils import gorilla


@gorilla.patch(torch._dynamo.trace_rules)
@functools.cache
def get_mod_inlinelist() -> set[str]:
    torch_dir = _module_dir(torch)
    if torch_dir is None:
        return set()
    inlinelist = {
        _as_posix_path(torch_dir + m[len("torch.") :].replace(".", "/"))
        for m in MOD_INLINELIST
    }

    # Add by CAMBRICON
    import torch_mlu

    MLU_MOD_INLINELIST = [
        "torch_mlu._dynamo.compiled_autograd",
        "torch_mlu._prims",
        "torch_mlu.mlu.amp.autocast_mode",
        "torch_mlu.fx.experimental.proxy_tensor",
        "torch_mlu.nn",
        "torch_mlu.backends.mlu",
    ]

    mlu_inlinelist = {
        _as_posix_path(
            _module_dir(torch_mlu) + m[len("torch_mlu.") :].replace(".", "/")
        )
        for m in MLU_MOD_INLINELIST
    }

    inlinelist.update(mlu_inlinelist)
    # end Add by CAMBRICON
    return inlinelist


@gorilla.patch(torch._dynamo.trace_rules)
@functools.lru_cache(None)
def get_mod_skiplist():
    torch_dir = _module_dir(torch)
    if torch_dir is None:
        return set()
    skiplist = {
        _as_posix_path(torch_dir + m[len("torch.") :].replace(".", "/"))
        for m in MOD_SKIPLIST
    }
    # Add by CAMBRICOM
    import torch_mlu

    MLU_MOD_SKIPLIST = [
        "torch_mlu.utils.gpu_migration",
        "torch_mlu.__init__",
        "torch_mlu._dynamo",
        "torch_mlu._functorch",
        "torch_mlu._inductor",
        "torch_mlu._meta_registrations",
        "torch_mlu._prims",
        "torch_mlu._tensor_str",
        "torch_mlu.backends",
        "torch_mlu.distributed",
        "torch_mlu.fx",
        "torch_mlu.nn",
        "torch_mlu.profiler",
        "torch_mlu.utils",
        "torch_mlu.mlu",
    ]

    mlu_skiplist = {
        _as_posix_path(
            _module_dir(torch_mlu) + m[len("torch_mlu.") :].replace(".", "/")
        )
        for m in MLU_MOD_SKIPLIST
    }
    skiplist.update(mlu_skiplist)
    # end Add by CAMBRICOM
    return skiplist


# We should consider whether all MLU APIs corresponding to the CUDA APIs in the native torch_non_c_binding_in_graph_functions(torch._dynamo.trace_rules) should be added to this list.
torch_mlu_non_c_binding_in_graph_functions = dict.fromkeys(
    [
        "torch.mlu.current_stream",
        "torch.mlu.stream",
        "torch.mlu.is_available",
        "torch.mlu._lazy_init",
        "torch.mlu.amp.autocast_mode._cast",
        "torch.mlu.amp.autocast_mode.custom_bwd",
        "torch.mlu.amp.autocast_mode.custom_fwd",
        "torch.mlu.default_stream",
        "torch.mlu.device_count",
        "torch.mlu.is_initialized",
        "torch.backends.mlu.can_use_efficient_attention",
        "torch.backends.mlu.can_use_flash_attention",
        "torch.backends.mlu.can_use_cudnn_attention",
        "torch.backends.mlu.enable_flash_sdp",
        "torch.backends.mlu.enable_math_sdp",
        "torch.backends.mlu.allow_fp16_bf16_reduction_math_sdp",
        "torch.backends.mlu.enable_mem_efficient_sdp",
        "torch.backends.mlu.flash_sdp_enabled",
        "torch.backends.mlu.is_built",
        "torch.backends.mlu.is_flash_attention_available",
        "torch.backends.mlu.math_sdp_enabled",
        "torch.backends.mlu.fp16_bf16_reduction_math_sdp_allowed",
        "torch.backends.mlu.mem_efficient_sdp_enabled",
        "torch.backends.mlu.cudnn_sdp_enabled",
        "torch.backends.mlu.enable_cudnn_sdp",
        "torch.backends.mlu.sdp_kernel",
        "torch.backends.cnnl.flags",
        "torch.backends.cnnl.set_flags",
        "torch.mlu._device_count_cndev",
        "torch.mlu._get_cndev_handler",
        "torch.mlu._get_cndev_device_index",
        "torch.mlu._get_device",
        "torch.mlu._get_generator",
        "torch.mlu._get_rng_state_offset",
        "torch.mlu._is_compiled",
        "torch.mlu._lazy_call",
        "torch.mlu._lazy_init",
        "torch.mlu._cndev_based_avail",
        "torch.mlu._parse_visible_devices",
        "torch.mlu._raw_device_count_cndev",
        "torch.mlu._raw_device_uuid_cndev",
        "torch.mlu._set_rng_state_offset",
        "torch.mlu._set_stream_by_id",
        "torch.mlu._sleep",
        "torch.mlu._transform_uuid_to_ordinals",
        "torch.mlu._utils._get_device_index",
        "torch.mlu.amp.common.amp_definitely_not_available",
        "torch.mlu.can_device_access_peer",
        "torch.mlu.check_error",
        "torch.mlu.clock_rate",
        "torch.mlu.cnrt",
        "torch.mlu.get_device_capability",
        "torch.mlu.get_device_name",
        "torch.mlu.get_device_properties",
        "torch.mlu.graphs.graph_pool_handle",
        "torch.mlu.graphs.is_current_stream_capturing",
        "torch.mlu.graphs.make_graphed_callables",
        "torch.mlu.init",
        "torch.mlu.ipc_collect",
        "torch.mlu.is_bf16_supported",
        "torch.mlu.memory._dump_snapshot",
        "torch.mlu.memory._get_current_allocator",
        "torch.mlu.memory._record_memory_history_impl",
        "torch.mlu.memory._record_memory_history_legacy",
        "torch.mlu.memory._record_memory_history",
        "torch.mlu.memory._save_memory_usage",
        "torch.mlu.memory._save_segment_usage",
        "torch.mlu.memory._set_allocator_settings",
        "torch.mlu.memory.caching_allocator_alloc",
        "torch.mlu.memory.caching_allocator_delete",
        "torch.mlu.memory.caching_allocator_enable",
        "torch.mlu.memory.change_current_allocator",
        "torch.mlu.memory.empty_cache",
        "torch.mlu.memory.get_allocator_backend",
        "torch.mlu.memory.get_per_process_memory_fraction",
        "torch.mlu.memory.host_memory_stats_as_nested_dict",
        "torch.mlu.memory.host_memory_stats",
        "torch.mlu.memory.max_memory_allocated",
        "torch.mlu.memory.max_memory_cached",
        "torch.mlu.memory.max_memory_reserved",
        "torch.mlu.memory.mem_get_info",
        "torch.mlu.memory.memory_allocated",
        "torch.mlu.memory.memory_cached",
        "torch.mlu.memory.memory_reserved",
        "torch.mlu.memory.memory_snapshot",
        "torch.mlu.memory.memory_stats_as_nested_dict",
        "torch.mlu.memory.memory_stats",
        "torch.mlu.memory.memory_summary",
        "torch.mlu.memory.reset_accumulated_host_memory_stats",
        "torch.mlu.memory.reset_accumulated_memory_stats",
        "torch.mlu.memory.reset_max_memory_allocated",
        "torch.mlu.memory.reset_max_memory_cached",
        "torch.mlu.memory.reset_peak_host_memory_stats",
        "torch.mlu.memory.reset_peak_memory_stats",
        "torch.mlu.memory.set_per_process_memory_fraction",
        "torch.mlu.cncl.is_cncl_available",
        "torch.mlu.cncl.version",
        "torch.mlu.cnpx.mark",
        "torch.mlu.cnpx.range_end",
        "torch.mlu.cnpx.range_pop",
        "torch.mlu.cnpx.range_push",
        "torch.mlu.cnpx.range_start",
        "torch.mlu.cnpx.range",
        "torch.mlu.power_draw",
        "torch.mlu.profiler.init",
        "torch.mlu.profiler.profile",
        "torch.mlu.profiler.start",
        "torch.mlu.profiler.stop",
        "torch.mlu.random.get_rng_state_all",
        "torch.mlu.random.initial_seed",
        "torch.mlu.random.manual_seed_all",
        "torch.mlu.random.manual_seed",
        "torch.mlu.random.seed_all",
        "torch.mlu.random.seed",
        "torch.mlu.random.set_rng_state_all",
        "torch.mlu.set_stream",
        "torch.mlu.stream",
        "torch.mlu.temperature",
        "torch.mlu.utilization",
        # extra add by CAMBRICON
        "torch._utils._maybe_view_chunk_cat",
        # end add by CAMBRICON
    ],
    TorchInGraphFunctionVariable,
)


torch_mlu_c_binding_in_graph_functions = dict.fromkeys(
    [
        "torch._C._get_fp32_precision_getter",
        "torch_mlu._MLUC._get_cnmatmul_allow_tf32",
        "torch_mlu._MLUC._mlu_synchronize",
        "torch_mlu._MLUC._can_use_flash_attention",
        "torch_mlu._MLUC._can_use_mem_efficient_attention",
        # "torch._C._get_cublas_allow_bf16_reduced_precision_reduction",
        # "torch._C._get_cublas_allow_fp16_reduced_precision_reduction",
        # "torch._C._set_cublas_allow_bf16_reduced_precision_reduction",
        # "torch._C._set_cublas_allow_fp16_reduced_precision_reduction",
        # "torch._C._set_cublas_allow_tf32",
        "torch_mlu._MLUC._get_cnnl_allow_tf32",
        "torch_mlu._MLUC._get_cnnl_benchmark",
        "torch_mlu._MLUC._get_cnnl_deterministic",
        # "torch._C._get_cudnn_enabled",
        # "torch._C._get_cudnn_sdp_enabled",
        # "torch._C._set_sdp_use_cudnn",
        "torch_mlu._MLUC._mlu_attach_out_of_memory_observer",
        "torch_mlu._MLUC._mlu_beginAllocateCurrentStreamToPool",
        "torch_mlu._MLUC._mlu_canDeviceAccessPeer",
        "torch_mlu._MLUC._mlu_changeCurrentAllocator",
        "torch_mlu._MLUC._mlu_checkPoolLiveAllocations",
        "torch_mlu._MLUC._mlu_mluCachingAllocator_raw_alloc",
        "torch_mlu._MLUC._mlu_mluCachingAllocator_raw_delete",
        "torch_mlu._MLUC._mlu_mluCachingAllocator_set_allocator_settings",
        # "torch._C._cuda_cudaHostAllocator",
        "torch_mlu._MLUC._mlu_customAllocator",
        "torch_mlu._MLUC._mlu_emptyCache",
        "torch_mlu._MLUC._mlu_endAllocateCurrentStreamToPool",
        "torch_mlu._MLUC._mlu_exchangeDevice",
        "torch_mlu._MLUC._mlu_getAllocator",
        "torch_mlu._MLUC._mlu_getAllocatorBackend",
        "torch_mlu._MLUC._mlu_getCheckpointState",
        "torch_mlu._MLUC._mlu_getCurrentRawStream",
        # "torch._C._cuda_getCurrentStream",
        "torch_mlu._MLUC._mlu_getDefaultStream",
        "torch_mlu._MLUC._mlu_getDevice",
        "torch_mlu._MLUC._mlu_getDeviceCount",
        "torch_mlu._MLUC._mlu_hasPrimaryContext",
        "torch_mlu._MLUC._mlu_hostMemoryStats",
        "torch_mlu._MLUC._mlu_init",
        "torch_mlu._MLUC._mlu_ipc_collect",
        "torch_mlu._MLUC._mlu_isCurrentStreamCapturing",
        "torch_mlu._MLUC._mlu_isHistoryEnabled",
        "torch_mlu._MLUC._mlu_isInBadFork",
        "torch_mlu._MLUC._mlu_maybeExchangeDevice",
        "torch_mlu._MLUC._mlu_memorySnapshot",
        "torch_mlu._MLUC._mlu_memoryStats",
        "torch_mlu._MLUC._mlu_record_memory_history_legacy",
        "torch_mlu._MLUC._mlu_record_memory_history",
        "torch_mlu._MLUC._mlu_releasePool",
        "torch_mlu._MLUC._mlu_resetAccumulatedHostMemoryStats",
        "torch_mlu._MLUC._mlu_resetAccumulatedMemoryStats",
        "torch_mlu._MLUC._mlu_resetPeakHostMemoryStats",
        "torch_mlu._MLUC._mlu_resetPeakMemoryStats",
        "torch_mlu._MLUC._mlu_setCheckpointPoolState",
        "torch_mlu._MLUC._mlu_setDevice",
        "torch_mlu._MLUC._mlu_setMemoryFraction",
        "torch_mlu._MLUC._mlu_setStream",
        "torch_mlu._MLUC._mlu_sleep",
        "torch_mlu._MLUC._mlu_synchronize",
    ],
    TorchInGraphFunctionVariable,
)


@gorilla.patch(
    torch._dynamo.trace_rules, settings=gorilla.Settings(use_replace_references=True)
)
@functools.cache
def get_torch_obj_rule_map() -> dict[Any, type["VariableTracker"]]:
    # Add by CAMBRICON
    from torch._dynamo.trace_rules import torch_name_rule_map
    from torch_mlu._dynamo.trace_rules import (
        torch_mlu_non_c_binding_in_graph_functions,
        torch_mlu_c_binding_in_graph_functions,
    )

    torch_name_rule_map.append(torch_mlu_non_c_binding_in_graph_functions)
    torch_name_rule_map.append(torch_mlu_c_binding_in_graph_functions)
    # end Add by CAMBRICON

    d: dict[Any, type[VariableTracker]] = {}
    for m in torch_name_rule_map:
        for k, v in m.items():  # type: ignore[attr-defined]
            if ".py#" not in k:
                obj = load_object(k)
            else:
                torch_dir = _module_dir(torch)
                if torch_dir is None:
                    continue
                obj = torch_dir + k[len("torch/") :]
            if obj is not None:
                # Modify by CAMBRICON
                # if is_lru_cache_wrapped_function(obj):
                if (
                    hasattr(obj, "is_mlu_gpu_migration")
                    and obj.is_mlu_gpu_migration is True
                    and hasattr(obj, "__wrapped__")
                ) or is_lru_cache_wrapped_function(obj):
                    obj = obj.__wrapped__
                # end Modify by CAMBRICON
                if obj in d and d[obj] != v:
                    raise AssertionError(
                        f"Duplicate torch object {obj} with different rules: {v}, {d[obj]}"
                    )
                else:
                    d[obj] = v
    return d


@functools.lru_cache(None)
def get_corrected_path():
    import torch_mlu
    from torch_mlu.utils.gpu_migration.migration import original_device_constructors

    return {
        # torch_mlu.utils.gpu_migration.migration.original_device_constructors corresponding torch.utils._device._device_constructors
        original_device_constructors.__wrapped__: _as_posix_path(
            _module_dir(torch) + "utils/_device"
        ),
    }


@functools.lru_cache(None)
def is_gpu_migration_enabled():
    func = torch.ones
    if (
        hasattr(func, "is_mlu_gpu_migration")
        and func.is_mlu_gpu_migration is True
        and hasattr(func, "__wrapped__")
    ):
        return True
    return False


# workaround for PYTORCH-12898
@gorilla.patch(torch._dynamo.trace_rules)
def _lookup_inner(
    obj: Any,
    name: Optional[str] = None,
    filename: Optional[str] = None,
    is_direct_call: bool = True,
    reasons: Optional[set[str]] = None,
) -> Optional[type[VariableTracker]]:
    # Step 1: lookup obj's tracing rule in `torch_name_rule_map`.
    # The rules defined in `torch_name_rule_map` mainly includes two parts:
    # - Manually defined rules for any functions.
    # - The list of torch in graph functions.
    try:
        can_hash = hashable(obj)
    except Exception:
        can_hash = False
    if not can_hash:
        if reasons is not None:
            reasons.add("obj is not hashable")
        return None
    if obj is not None:
        # Add by CAMBRICON
        # Enabling gpu_migration changes the function type from <built-in method **>
        # to <_VariableFunctionsClass.**> and wrapping <built-in method **> or <method of TensorBase>
        if (
            hasattr(obj, "is_mlu_gpu_migration")
            and obj.is_mlu_gpu_migration is True
            and hasattr(obj, "__wrapped__")
        ):
            obj = obj.__wrapped__
            if filename is not None:
                filename = getfile(obj)
        # end Add by CAMBRICON

        if is_aten_op_or_tensor_method(obj):
            return TorchInGraphFunctionVariable
        rule = get_torch_obj_rule_map().get(obj, None)
        if rule is not None:
            if reasons is not None:
                reasons.add("get_torch_obj_rule_map")
            return rule
    elif name is not None and filename is not None and not is_direct_call:
        if name.startswith(TORCH_DYNAMO_RESUME_IN_PREFIX):
            rule = get_torch_obj_rule_map().get(
                filename + "#" + TORCH_DYNAMO_RESUME_IN_PREFIX, None
            )
        else:
            rule = get_torch_obj_rule_map().get(filename + "#" + name, None)
        if rule is not None:
            if reasons is not None:
                reasons.add("get_torch_obj_rule_map")
            return rule
    elif name == "<listcomp>":
        if reasons is not None:
            reasons.add("inlining frame from list comprehension")
        return UserFunctionVariable

    # Step 2: lookup obj's tracing rule by function name.
    if is_direct_call:
        if name == "patched_init":
            if reasons is not None:
                reasons.add("func name is patched_init")
            return SkipFunctionVariable
        elif name == "__torch_function__" or (
            obj and getattr(obj, "__name__", None) == "__torch_function__"
        ):
            if reasons is not None:
                reasons.add("func name is __torch_function__")
            return UserFunctionVariable

    if not is_direct_call:
        if name == "__getattr__":
            # is_direct_call = False indicates that this is the top-level frame
            # being traced (i.e., it is not inlined and not called from
            # InliningInstructionTranslator).  Tracing __getattr__ at the top
            # level is unlikely because we inline it for
            # UserDefinedObjectVariable. This scenario occurs only for
            # UnspecializedNNModuleVariable, where Dynamo directly calls
            # __getattr__ during trace time, generating LOAD_ATTR bytecode
            # without going through the underlying __getattr__ data structures.
            # When this optimized bytecode is executed, Dynamo is triggered
            # again on the __getattr__ call. Therefore, we skip Dynamo tracing
            # in this case.
            if reasons is not None:
                reasons.add(
                    "Tracing __getattr__ as the top level frame, unsuitable for tracing."
                )
            return SkipFunctionVariable

    # Step 3: lookup obj's tracing rule by filename.
    if filename is None:
        filename = getfile(obj)

    # Modify by CAMBRICON
    from torch_mlu._dynamo.trace_rules import is_gpu_migration_enabled

    if is_gpu_migration_enabled():
        from torch_mlu._dynamo.trace_rules import get_corrected_path

        filename = get_corrected_path().get(obj, filename)
    # end Modify by CAMBRICON

    skip_result = check_file(filename, is_direct_call)
    if reasons is not None and skip_result.reason is not None:
        reasons.add(skip_result.reason)
    if skip_result.skipped:
        return SkipFunctionVariable
    else:
        return UserFunctionVariable
