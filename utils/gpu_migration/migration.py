from functools import wraps
from typing import Callable, Dict, List, Tuple, Union
import warnings
import functools
import sys
import os
import importlib
import types
import inspect

import torch
import torch.autograd.profiler as prof
import torch.distributed.distributed_c10d as c10d
import torch.distributed.fsdp

# Fix the "torch.Stream | None" type annotation fail because migration modify the type of torch.Stream
from torch.distributed.checkpoint.filesystem import _OverlappingCpuLoader
from torch._prims import context
from torch._prims.context import torch_to_refs_map as native_torch_to_refs_map
import torch.utils._device

import torch_mlu
from torch.backends.cudnn import CudnnModule
from torch.backends import ContextProp


# class can accept device arg
class_list = [
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    torch.nn.LazyConv1d,
    torch.nn.LazyConv2d,
    torch.nn.LazyConv3d,
    torch.nn.LazyConvTranspose1d,
    torch.nn.LazyConvTranspose2d,
    torch.nn.LazyConvTranspose3d,
    torch.nn.MultiheadAttention,
    torch.nn.PReLU,
    torch.nn.AdaptiveLogSoftmaxWithLoss,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.LazyBatchNorm1d,
    torch.nn.LazyBatchNorm2d,
    torch.nn.LazyBatchNorm3d,
    torch.nn.GroupNorm,
    torch.nn.SyncBatchNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LazyInstanceNorm1d,
    torch.nn.LazyInstanceNorm2d,
    torch.nn.LazyInstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.RNNBase,
    torch.nn.RNN,
    torch.nn.LSTM,
    torch.nn.GRU,
    torch.nn.RNNCell,
    torch.nn.LSTMCell,
    torch.nn.GRUCell,
    torch.nn.Transformer,
    torch.nn.TransformerEncoderLayer,
    torch.nn.TransformerDecoderLayer,
    torch.nn.Linear,
    torch.nn.Bilinear,
    torch.nn.LazyLinear,
    torch.nn.Embedding,
    torch.nn.EmbeddingBag,
    # DDP
    torch.nn.parallel.DistributedDataParallel,
    # amp
    torch.amp.autocast,
    torch.mlu.device,
    torch.storage.UntypedStorage,
    torch.UntypedStorage,
    torch.TypedStorage,
    # fsdp
    torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel,
    torch.distributed.fsdp._fully_shard._fsdp_param.FSDPParam,
    # DCP
    torch.distributed._remote_device,
    torch.utils._device.DeviceContext,
    # TP
    torch.distributed.DeviceMesh,
    torch.utils.data.DataLoader,
    torch_mlu._MLUC._aoti.AOTIModelContainerRunnerMlu,
]


# torch.* that needs to replace device, selected from torch/_torch_docs.py
torch_fn_list = [
    "set_default_device",
    "as_tensor",
    "asarray",
    "autocast",
    "eye",
    "linspace",
    "logspace",
    "ones",
    "ones_like",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
    "tensor",
    "range",
    "arange",
    "sparse_compressed_tensor",
    "sparse_csr_tensor",
    "sparse_csc_tensor",
    "sparse_bsr_tensor",
    "sparse_bsc_tensor",
    "sparse_coo_tensor",
    "tril_indices",
    "triu_indices",
    "zeros",
    "zeros_like",
    "empty",
    "empty_like",
    "empty_strided",
    "empty_permuted",
    "full",
    "full_like",
    "hann_window",
    "hamming_window",
    "bartlett_window",
    "blackman_window",
    "kaiser_window",
    # the following torch functions are not listed in _torch_docs.py,
    # but can still use torch.* (selected from native_functions.yaml).
    "_cudnn_init_dropout_state",
    "_empty_affine_quantized",
    "_empty_per_channel_affine_quantized",
    "empty_quantized",
    "from_file",
    "_pin_memory",
    "scalar_tensor",
    "_efficientzerotensor",
    "_sparse_compressed_tensor_unsafe",
    "_sparse_csr_tensor_unsafe",
    "_sparse_csc_tensor_unsafe",
    "_sparse_bsr_tensor_unsafe",
    "_sparse_bsc_tensor_unsafe",
    "_sparse_coo_tensor_unsafe",
    "normal",
    "_nested_tensor_from_tensor_list",
    "get_autocast_dtype",
    "set_autocast_dtype",
    # they are not available in the torch module
    # 'fft_fftfreq',
    # 'fft_rfftfreq',
    # '_sparse_coo_tensor_with_dims'
    # '_sparse_coo_tensor_with_dims_and_tensors',
]

# torch.ops.* that needs to replace device
torch_ops_fn_list = [
    #
    # aten namespace
    #
    "aten._assert_tensor_metadata",
    "aten.empty",
    "aten.empty_strided",
    "aten.zeros",
    "aten._sparse_coo_tensor_with_dims_and_tensors",
    "aten.empty_like",
    "aten.full_like",
    "aten.ones_like",
    "aten.rand_like",
    "aten.randn_like",
    "aten.randint_like",
    "aten.zeros_like",
    "aten.new_empty",
    "aten.new_empty_strided",
    "aten.new_full",
    "aten.new_zeros",
    "aten.new_ones",
    "aten._resize_output_",
    "aten._nested_tensor_from_tensor_list",
    "aten.pin_memory",
    "aten.to",
    "aten.is_pinned",
    "aten._pin_memory",
    "aten._resize_output",
    "aten.eq",
    "aten.ne",
    "aten.empty_permuted",
    "aten.full",
    "aten.scalar_tensor",
    "aten.normal",
    "aten._to_copy",
    "aten.arange",
    "aten.ones",
    "aten.linspace",
    "aten.logspace",
    "aten.eye",
    "aten.randn",
    "aten.tril_indices",
    "aten.triu_indices",
    "aten.randperm",
    "aten.randint",
    "aten.rand",
    "aten._make_dep_token",
    "aten.sparse_bsc_tensor",
    "aten.sparse_csr_tensor",
    "aten.sparse_bsr_tensor",
    "aten._sparse_coo_tensor_unsafe",
    "aten.sparse_csc_tensor",
    "aten._sparse_bsr_tensor_unsafe",
    "aten._sparse_compressed_tensor_unsafe",
    "aten._sparse_bsc_tensor_unsafe",
    "aten._cufft_clear_plan_cache",
    "aten.sparse_coo_tensor",
    "aten._cufft_set_plan_cache_max_size",
    "aten._sparse_csr_tensor_unsafe",
    "aten._cufft_get_plan_cache_size",
    "aten._sparse_csc_tensor_unsafe",
    "aten._cufft_get_plan_cache_max_size",
    "aten._efficientzerotensor",
    #
    # prims namespace
    #
    "prims.device_put",
    "prims.iota",
    "prims.empty",
    "prims.empty_strided",
    "prims.empty_permuted",
    "prims.full",
    "prims.full_like",
    "prims.scalar_tensor",
    "prims.normal",
    "prims.uniform",
    "prims.inductor_seed",
    "prims.inductor_seeds",
    #
    # rngprims namespace
    #
    "rngprims.philox_rand",
    "streams.fork",
    "streams.join",
]

torch_cuda_fn_list = [
    "set_device",
    "get_device_name",
    "get_device_capability",
    "get_device_properties",
    "can_device_access_peer",
    "synchronize",
    "current_stream",
    "default_stream",
    # skip nvml fn
    # '_get_nvml_device_index',
    # '_get_pynvml_handler',
    # 'memory_usage',
    # 'utilization',
    # 'temperature',
    # 'power_draw',
    # 'clock_rate',
    "list_gpu_processes",
    "mem_get_info",
    "get_per_process_memory_fraction",
    "memory_stats",
    "memory_summary",
    "memory_allocated",
    "max_memory_allocated",
    "reset_max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "set_per_process_memory_fraction",
    "memory_cached",
    "max_memory_cached",
    "reset_max_memory_cached",
    "reset_peak_memory_stats",
]


# torch.Tensor.* that needs to replace device, selected from torch/_tensor_docs.py
tensor_fn_list = [
    "new_tensor",
    "new_full",
    "new_empty",
    "new_empty_strided",
    "new_ones",
    "new_zeros",
    "to",
    "is_pinned",
    "pin_memory",
    "type",
    "mlu",
    "new",
]

distributed_fn_list = [
    "broadcast_object_list",
    "init_device_mesh",
]

library_fn_list = [
    "__init__",
    "impl",
    "_impl_with_aoti_compile",
    "fallback",
]

module_fn_list = [
    "to",
    "to_empty",
]

# this list is for functions whose modules are not in torch, torch.distributed and torch.module
# TODO: Merge all fn lists into one list
other_fn_list = [
    {
        "name": "torch.random.fork_rng",
        "mod": torch.random,
        "fn_list": [
            "fork_rng",
        ],
    },
    {
        "name": "torch.utils.cpp_extension.include_paths",
        "mod": torch.utils.cpp_extension,
        "fn_list": [
            "include_paths",
        ],
    },
    {
        "name": "torch.distributed.Backend.register_backend",
        "mod": torch.distributed.Backend,
        "fn_list": [
            "register_backend",
        ],
    },
    {
        "name": "torch.distributed.tensor.parallel.parallelize_module",
        "mod": torch.distributed.tensor.parallel,
        "fn_list": [
            "parallelize_module",
        ],
    },
    {
        "name": "torch.testing.make_tensor",
        "mod": torch.testing,
        "fn_list": [
            "make_tensor",
        ],
    },
    {
        "name": "torch.distributed.device_mesh.DeviceMesh.from_group",
        "mod": torch.distributed.device_mesh.DeviceMesh,
        "fn_list": [
            "from_group",
        ],
    },
    {
        "name": "torch.distributed.tensor.distribute_module",
        "mod": torch.distributed.tensor,
        "fn_list": [
            "distribute_module",
        ],
    },
    {
        "name": "torch.distributed.tensor.distribute_tensor",
        "mod": torch.distributed.tensor,
        "fn_list": [
            "distribute_tensor",
        ],
    },
    {
        "name": "torch.distributed.tensor.DTensor.from_local",
        "mod": torch.distributed.tensor.DTensor,
        "fn_list": [
            "from_local",
        ],
    },
    {
        "name": "torch.distributed.tensor.DTensor.redistribute",
        "mod": torch.distributed.tensor.DTensor,
        "fn_list": [
            "redistribute",
        ],
    },
    {
        "name": "torch._export.aot_load",
        "mod": torch._export,
        "fn_list": [
            "aot_load",
        ],
    },
]

torch_mlu_memory_fn_list = ["mem_get_info", "get_per_process_memory_fraction"]

# default value of input args is cuda related and needs to be processed separately.
default_cuda_args_list = [
    "torch.amp.GradScaler.__init__",
    "torch.random.fork_rng",
]

# add new tuples here to adapt new substitution
substitution_list = {
    "torch.cuda": [
        ("nccl", "cncl"),
        ("nvtx", "cnpx"),
        ("CUDAGraph", "MLUGraph"),
        ("CUDAPluggableAllocator", "MLUPluggableAllocator"),
    ],
    "torch.cuda.graphs": [("CUDAGraph", "MLUGraph")],
    "torch.cuda.nccl": [("is_available", "is_cncl_available")],
    "torch.cuda.memory": [("CUDAPluggableAllocator", "MLUPluggableAllocator")],
}


def exec_only_once(func):
    """A decorator that limits the decorated function to be executed only once."""
    # use variable obj as a flag because of shallow copy in closure
    flag = [False]

    def inner(*args, **kwargs):
        if flag[0]:
            pass
        else:
            flag[0] = True
            func(*args, **kwargs)

    return inner


class MetaStream(type):
    """
    Metaclass to make isinstance(orig_stream(), Stream) and
    issubclass(orig_stream, Stream) pass.
    """

    def __instancecheck__(cls, instance):
        type_obj = type(instance)
        # orig_stream is the torch.Stream before it gets replaced
        # We use cls.base_class which is set at class definition time
        return (
            type_obj is cls
            or type_obj is torch.mlu.Stream
            or type_obj is cls.base_class
            or issubclass(type_obj, cls)
        )

    def __subclasscheck__(cls, subclass):
        if subclass is cls or subclass is cls.base_class:
            return True
        return type.__subclasscheck__(cls, subclass)


class MetaEvent(type):
    """
    Metaclass to make isinstance(orig_event(), Event) and
    issubclass(orig_event, Event) pass.
    """

    def __instancecheck__(cls, instance):
        type_obj = type(instance)
        # orig_event is the torch.Event before it gets replaced
        # We use cls.base_class which is set at class definition time
        return (
            type_obj is cls
            or type_obj is torch.mlu.Event
            or type_obj is cls.base_class
            or issubclass(type_obj, cls)
        )

    def __subclasscheck__(cls, subclass):
        if subclass is cls or subclass is cls.base_class:
            return True
        return type.__subclasscheck__(cls, subclass)


class Generator(torch.Generator):
    def __new__(cls, device: Union[torch.device, str, None] = None):
        return super().__new__(cls, device)


class Stream(torch.Stream, metaclass=MetaStream):
    """Subclass of torch.Stream that handles device replacement."""

    # Save reference to the original torch.Stream class for isinstance checks
    base_class = torch.Stream

    def __new__(cls, *args, **kwargs):
        # Replace cuda/CUDA in args/kwargs with mlu/MLU before delegating to torch.Stream
        if args:
            args_list = list(args)
            for i, arg in enumerate(args_list):
                args_list[i] = replace_cuda_with_mlu(arg)
            args = tuple(args_list)

        if kwargs:
            for key in ["device", "_device"]:
                if key in kwargs:
                    kwargs[key] = replace_cuda_with_mlu(kwargs[key])

        # Ensure we create an MLU stream by default if no device specified
        if len(args) == 0 and "device" not in kwargs:
            device_str = replace_cuda_with_mlu("cuda")
            kwargs["device"] = torch.device(device_str)

        return super().__new__(cls, *args, **kwargs)


class Event(torch.Event, metaclass=MetaEvent):
    """Subclass of torch.Event that handles device replacement."""

    # Save reference to the original torch.Event class for isinstance checks
    base_class = torch.Event

    def __new__(cls, *args, **kwargs):
        enable_timing = bool(kwargs.get("enable_timing", False))
        # Replace cuda/CUDA in args/kwargs with mlu/MLU before delegating to torch.Stream
        if args:
            args_list = list(args)
            for i, arg in enumerate(args_list):
                args_list[i] = replace_cuda_with_mlu(arg)
            args = tuple(args_list)

        if kwargs:
            for key in ["device", "_device"]:
                if key in kwargs:
                    kwargs[key] = replace_cuda_with_mlu(kwargs[key])

        # Ensure we create an MLU event by default if no device specified
        if len(args) == 0 and "device" not in kwargs:
            device_str = replace_cuda_with_mlu("cuda")
            kwargs["device"] = torch.device(device_str)

        event = super().__new__(cls, *args, **kwargs)
        # Preserve the Python-level flag so the migration wrapper can match
        # torch.Event error semantics before calling into CNRT directly.
        event._migration_enable_timing = enable_timing
        return event

    def elapsed_time(self, end_event: "Event") -> float:
        r"""Return the time elapsed in milliseconds.

        This override handles the type checking issue where PyTorch's C-level
        elapsed_time checks ob_type directly against THPEventClass, which fails
        for Python subclasses like this migration Event.
        """
        # Type validation: end_event must be the same type as self
        # (migration.Event should only accept migration.Event, not torch.mlu.Event)
        if type(end_event) is not type(self):
            raise RuntimeError("expected other to be a torch.Event object")

        # For MLU events, use cnrtNotifierElapsedTime directly
        if self.device.type == "mlu":
            # Validate that end_event is also an MLU event
            if end_event.device.type != "mlu":
                raise RuntimeError(
                    f"Event device type mlu does not match other's device type {end_event.device.type}."
                )

            if not getattr(self, "_migration_enable_timing", False) or not getattr(
                end_event, "_migration_enable_timing", False
            ):
                raise ValueError(
                    "Both events must be created with argument 'enable_timing=True'."
                )

            if self.event_id == 0 or end_event.event_id == 0:
                raise ValueError(
                    "Both events must be recorded before calculating elapsed time."
                )

            if not self.query() or not end_event.query():
                raise RuntimeError(
                    "Both events must be completed before calculating elapsed time."
                )

            import ctypes

            cnrt = ctypes.CDLL("libcnrt.so")
            cnrtNotifierElapsedTime = cnrt.cnrtNotifierElapsedTime
            cnrtNotifierElapsedTime.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
            ]
            cnrtNotifierElapsedTime.restype = ctypes.c_int

            time_ms = ctypes.c_float()
            result = cnrtNotifierElapsedTime(
                self.event_id, end_event.event_id, ctypes.byref(time_ms)
            )
            if result != 0:
                raise RuntimeError(
                    f"cnrtNotifierElapsedTime failed with error code {result}"
                )
            return float(time_ms.value)
        # For other devices, use the base class method
        return self.base_class.elapsed_time(self, end_event)


def replace_device_args(fn, ignore_self_arg=False, skip_qual_name=False):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        if args:
            args = list(args)
            for i, arg in enumerate(args):
                # do not replace_device for self argument of <class>.__init__ function
                if ignore_self_arg and i == 0:
                    continue
                if skip_qual_name and isinstance(arg, str) and "::" in arg:
                    continue
                args[i] = replace_cuda_with_mlu(arg)

        if kwargs:
            # device=* , device_type=*, output_device=*
            # type can be str, torch.device or int
            for key in [
                "device",
                "device_type",
                "output_device",
                "device_id",
                "map_location",
                "dispatch_key",
                "from_device",
                "to_device",
            ]:
                device = kwargs.get(key, None)
                if device:
                    kwargs[key] = replace_cuda_with_mlu(device)

            # list of int or torch.device
            device_ids = kwargs.get("device_ids", None)
            if isinstance(device_ids, list):
                for i, arg in enumerate(device_ids):
                    device_ids[i] = replace_cuda_with_mlu(arg)
                kwargs["device_ids"] = device_ids
        return fn(*args, **kwargs)

    wrapper_fn.is_mlu_gpu_migration = True
    return wrapper_fn


# For torch.load: skip the first argument (file path) to avoid replacing 'cuda' in paths
def replace_load_args(fn):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        # args[0] is file path, should not be replaced
        # args[1] is map_location (optional), should be replaced
        if len(args) >= 2:
            args = list(args)
            args[1] = replace_cuda_with_mlu(args[1])
            args = tuple(args)

        # Also handle kwargs form
        if kwargs:
            map_location = kwargs.get("map_location", None)
            if map_location:
                kwargs["map_location"] = replace_cuda_with_mlu(map_location)

        return fn(*args, **kwargs)

    wrapper_fn.is_mlu_gpu_migration = True
    return wrapper_fn


def replace_cuda_with_mlu(arg):
    # 'cuda*' -> 'mlu*'
    if isinstance(arg, str) and "cuda" in arg:
        arg = arg.replace("cuda", "mlu")
    elif isinstance(arg, str) and "CUDA" in arg:
        arg = arg.replace("CUDA", "MLU")
    # torch.device('cuda*') -> torch.device('mlu*')
    if isinstance(arg, torch.device) and "cuda" in arg.type:
        device = f"mlu:{arg.index}" if arg.index is not None else "mlu"
        arg = torch.device(device)
    # int ids are no need to handle, because relative
    # modifications have already patched to Pytorch
    return arg


def replace_device(module, fn_list):
    for fn_name in fn_list:
        fn = getattr(module, fn_name, None)
        if fn:
            if fn_name == "__init__":
                setattr(module, fn_name, replace_device_args(fn, ignore_self_arg=True))
            elif fn_name == "impl":
                setattr(module, fn_name, replace_device_args(fn, skip_qual_name=True))
            else:
                setattr(module, fn_name, replace_device_args(fn))


class OpOverloadPacketWrapper(torch._ops.OpOverloadPacket):
    __call__ = replace_device_args(torch._ops.OpOverloadPacket.__call__)


class OpOverloadWrapper(torch._ops.OpOverload):
    __call__ = replace_device_args(torch._ops.OpOverload.__call__)


# Only add wrapper for ops with 'device' argument rather than all torch.ops.*, because
# wrapper for all ops would reduce performance especially for torch.compile scenarios.
# Based on above reason, do not wrapper for __call__ of torch._ops.OpOverloadPacket and
# torch._ops.OpOverload directly. Just wrapper for __call__ of instances in whitelist.
def replace_torch_ops_args(fn_list):
    fn = torch.ops
    # wrapper for torch.ops.my_namespace.my_op_name
    for fn_name in fn_list:
        attrs = fn_name.split(".")
        for attr in attrs:
            fn = getattr(fn, attr, None)
        if isinstance(fn, torch._ops.OpOverloadPacket):
            # patch fn.__class__ since we can't patch __call__ directly for an instance
            fn.__class__ = OpOverloadPacketWrapper
            # wrapper for all overloads: torch.ops.my_namespace.my_op_name.overload_name
            for overload_name in fn.overloads():
                fn_overload = getattr(fn, overload_name, None)
                if isinstance(fn_overload, torch._ops.OpOverload):
                    fn_overload.__class__ = OpOverloadWrapper
                else:
                    warnings.warn(
                        f"When wrapping for torch.ops.{fn_name}.{overload_name}, expected it as an instance of OpOverload, "
                        f"but got a {type(fn_overload)} instance: {fn_overload}, will skip wrapping for it."
                    )
        else:
            warnings.warn(
                f"When wrapping for torch.ops.{fn_name}, expected it as an instance of OpOverloadPacket, "
                f"but got a {type(fn)} instance: {fn}, will skip wrapping for it."
            )
        fn = torch.ops


# init_process_group
# 1. 'nccl' -> 'cncl'
# 2. backend='nccl' -> backend='cncl'
# 3. backend='cuda:nccl' -> backend='mlu:cncl'
# 4. device_id='cuda:0'  -> device_id='mlu:0'
def replace_nccl_with_cncl(fn):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        if args:
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, str) and "nccl" in arg:
                    args[i] = arg.replace("nccl", "cncl")
                    if "cuda" in args[i]:
                        args[i] = args[i].replace("cuda", "mlu")
                if isinstance(arg, str) and "NCCL" in arg:
                    args[i] = arg.replace("NCCL", "CNCL")
        if kwargs:
            backend = kwargs.get("backend", None)
            if isinstance(backend, str) and "nccl" in backend:
                backend = backend.replace("nccl", "cncl")
                if "cuda" in backend:
                    backend = backend.replace("cuda", "mlu")
                kwargs["backend"] = backend
            device_id = kwargs.get("device_id", None)
            if device_id:
                kwargs["device_id"] = replace_cuda_with_mlu(device_id)
            device = kwargs.get("device", None)
            if device:
                kwargs["device"] = replace_cuda_with_mlu(device)
        return fn(*args, **kwargs)

    return wrapper_fn


# profiler
# sort_by='*cuda*' -> sort_by='*mlu*'
def replace_profiler_args(fn):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        if args:
            args = list(args)
            for i, arg in enumerate(args):
                args[i] = replace_cuda_with_mlu(arg)

        if kwargs:
            sort_by = kwargs.get("sort_by", None)
            if sort_by:
                kwargs["sort_by"] = replace_cuda_with_mlu(sort_by)
            metric = kwargs.get("metric", None)
            if metric:
                kwargs["metric"] = replace_cuda_with_mlu(metric)
            use_cuda = kwargs.get("use_cuda", None)
            use_device = kwargs.get("use_device", None)
            if (use_device and use_device == "cuda") or use_cuda:
                kwargs["use_cuda"] = False
                kwargs["use_device"] = "mlu"
                # mlu profiler only available with kineto
                kwargs["use_kineto"] = True

        return fn(*args, **kwargs)

    return wrapper_fn


def replace_profiler(module, fn_name):
    fn = getattr(module, fn_name, None)
    if fn:
        setattr(module, fn_name, replace_profiler_args(fn))


# dataloader
# pin_memory_device=''->pin_memory_device='mlu'
# pin_memory_device='cuda'->pin_memory_device='mlu'
def replace_dataloader(fn):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        if kwargs:
            pin_memory = kwargs.get("pin_memory", False)
            pin_memory_device = kwargs.get("pin_memory_device", None)
            if pin_memory and not pin_memory_device:
                kwargs["pin_memory_device"] = "mlu"
            if (
                pin_memory
                and isinstance(pin_memory_device, str)
                and "cuda" in pin_memory_device
            ):
                kwargs["pin_memory_device"] = pin_memory_device.replace("cuda", "mlu")
        return fn(*args, **kwargs)

    return wrapper_fn


# For autocast API.
# set_autocast_enabled(device_type: str, enabled: _bool)
# set_autocast_enabled(enabled: _bool)
# This API is default for cuda, and will be deprecated in pytorch 2.5;
# is_autocast_enabled()
# is_autocast_enabled(device_type: str)
# replace_autocast function:
# replace torch.is_autocast_enabled() to torch.is_autocast_enabled('mlu')
# torch.is_autocast_enabled('cuda') to torch.is_autocast_enabled('mlu')
# torch.set_autocast_enabled(bool) to torch.set_autocast_enabled('mlu', bool)
# torch.set_autocast_enabled('cuda', bool) to torch.set_autocast_enabled('mlu', bool)
def replace_autocast(fn):
    @wraps(fn)
    def warp_fn(*args, **kwargs):
        parameters_len = len(args) + len(kwargs)
        if parameters_len == 0:
            # is_autocast_enabled()
            args = [torch._C._get_privateuse1_backend_name()]
            return fn(*args, **kwargs)
        elif parameters_len == 1:
            # for set_autocast_enabled(enabled: _bool)
            if (
                len(args) == 1
                and isinstance(args[0], bool)
                or len(kwargs) == 1
                and "enabled" in kwargs.keys()
            ):
                return fn(torch._C._get_privateuse1_backend_name(), *args, **kwargs)
            else:
                # for is_autocast_enabled(device_type: str)
                return replace_device_args(fn)(*args, **kwargs)
        else:
            # set_autocast_enabled(device_type: str, enabled: _bool)
            return replace_device_args(fn)(*args, **kwargs)

    warp_fn.is_mlu_gpu_migration = True
    return warp_fn


def replace_default_cuda_args_with_mlu(fn):
    @wraps(fn)
    def warp_fn(*args, **kwargs):
        sig = inspect.signature(fn)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        device = bound_args.arguments.get("device", None)
        if device == "cuda":
            bound_args.arguments["device"] = "mlu"

        device_type = bound_args.arguments.get("device_type", None)
        if device_type == "cuda":
            bound_args.arguments["device_type"] = "mlu"

        return fn(*bound_args.args, **bound_args.kwargs)

    return warp_fn


def replace_default_cuda_args(func_name):
    components = func_name.split(".")
    module = torch
    fn_name = components[-1]
    for comp in components[1:-1]:
        module = getattr(module, comp, None)
    if module is None:
        raise AttributeError(f"module '{func_name}' does not exist")
    fn = getattr(module, fn_name, None)
    if fn is not None:
        setattr(module, fn_name, replace_default_cuda_args_with_mlu(fn))


def script(
    obj,
    optimize=None,
    _frames_up=0,
    _rcb=None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
):
    warnings.warn(
        f"The model-transfer tool does not support torch.jit.script and use eager mode instead."
    )
    return obj


@functools.lru_cache(None)
def torch_to_refs_map():
    """
    Mapping of torch API functions to torch._refs functions.
    E.g. torch_to_refs_map()[torch.add] == torch._refs.add
    """
    r = native_torch_to_refs_map()

    # Recover mapping relation of native API hijacked by model transfer.
    for fn in torch_fn_list:
        if fn in torch._refs.__all__:
            r[getattr(torch._C._VariableFunctions, fn)] = torch._refs.__dict__.get(fn)

    for fn in tensor_fn_list:
        if fn in torch._refs.__all__:
            r[getattr(torch._C.TensorBase, fn)] = torch._refs.__dict__.get(fn)

    return r


# This function is used to check whether an object needs to be patched (called by `update_dict``)
def needs_skip(obj, module):
    # filter out functions which are not defined in current module
    if isinstance(obj, (types.FunctionType, type)) and (
        obj.__module__.startswith(module.__name__) or obj.__module__.startswith("torch")
    ):
        return False
    # filter out modules like os, List, Union
    elif isinstance(obj, types.ModuleType) and (
        ".".join(obj.__name__.split(".")[:-1]).startswith(module.__name__)
        or ".".join(obj.__name__.split(".")[:-1]).startswith("torch")
    ):
        return False
    return True


# Here we only need to ensure that the two input obj are both a class/func rather than
# both of the two inputs have exactly the same type
def rough_type_comparison(obj1, obj2):
    if isinstance(obj1, types.ModuleType) and isinstance(obj2, types.ModuleType):
        return True
    elif inspect.isclass(obj2) and inspect.isclass(obj2):
        return True
    elif (inspect.isfunction(obj1) or isinstance(obj1, types.BuiltinFunctionType)) and (
        inspect.isfunction(obj2) or isinstance(obj2, types.BuiltinFunctionType)
    ):
        return True
    elif inspect.ismethod(obj1) and inspect.ismethod(obj2):
        return True
    return False


# Check whether all of the modules in __all__ have been wrapped/ patched. If not, a warning will be provided
def check_patch_succeeded(source_module, dest_module):
    if hasattr(source_module, "__all__"):
        substitutions = substitution_list.get(dest_module.__name__, [])
        for attr in source_module.__all__:
            # process substitution list
            attr_dst = attr
            for dst_sub, src_sub in substitutions:
                if attr == src_sub:
                    attr_dst = dst_sub
            try:
                if not isinstance(dest_module.__dict__[attr], types.ModuleType):
                    if dest_module.__dict__[attr_dst] != source_module.__dict__[attr]:
                        warnings.warn(
                            f"{source_module.__dict__[attr].__name__} is supported in {source_module.__name__} and not patched"
                            f" in {dest_module.__name__}. Please check whether {source_module.__dict__[attr].__name__} needs to be migrated."
                        )
            except:
                pass


# for unsupported funcs/classes/modules, we add wrappers.
def warning_wrapper(obj, obj_name):
    warning_message = "gpu_migration: " + obj_name + " is not yet suppoorted on MLU"

    def decorator(obj):
        # decorate func
        if isinstance(obj, types.FunctionType):

            @wraps(obj)
            def wrapped_function(*args, **kwargs):
                warnings.warn(warning_message, UserWarning)
                return obj(*args, **kwargs)

            return wrapped_function

        # decorate classes
        elif isinstance(obj, type):

            class WrappedClass(obj):
                def __init__(self, *args, **kwargs):
                    warnings.warn(warning_message, UserWarning)
                    super().__init__(*args, **kwargs)

                # for callable classes
                def __call__(self, *args, **kwargs):
                    warnings.warn(warning_message, UserWarning)
                    return super().__call__(*args, **kwargs)

            return WrappedClass

        # decorate callable obj (e.g. an instance of a callable class)
        elif callable(obj):

            class WrappedCallable:
                def __init__(self, callable_obj):
                    self._callable_obj = callable_obj

                def __call__(self, *args, **kwargs):
                    warnings.warn(warning_message, UserWarning)
                    return self._callable_obj(*args, **kwargs)

                def __getattr__(self, name):
                    return getattr(self._callable_obj, name)

            return WrappedCallable(obj)

        # return original module
        else:
            return obj

    return decorator(obj)


# This func is used to apply monkey patches and update sys.modules and module.__dict__
# All attrs that are in dest_module.__dict__ but not in source_module.__dict__ will be
# wrapped(warning added) and added to source_module.__dict__. In addition, sys.modules[dest_module]
# will be set to source_module.
def update_dict(source_module, dest_module, cache: set | None):
    # avoid RecursionError
    if id(source_module) == id(dest_module):
        return
    if cache is not None:
        if (id(source_module), id(dest_module)) in cache:
            return
        cache.add((id(source_module), id(dest_module)))
    source_dict = source_module.__dict__
    dest_dict = dest_module.__dict__
    substitutions = substitution_list.get(dest_module.__name__, [])
    # torch.cuda
    updates = {}
    for dest_key in dest_dict.copy():
        patched = False
        # filter out all attrs that not begins with "__" and not defined in current module
        if dest_key.startswith("__") or needs_skip(dest_dict[dest_key], dest_module):
            continue
        # torch_mlu.mlu
        for source_key in source_dict:
            if source_key == dest_key or (dest_key, source_key) in substitutions:
                # Report a warning and skip the current substitution when two eponymous have different types
                if not rough_type_comparison(
                    source_dict[source_key], dest_dict[dest_key]
                ):
                    warnings.warn(
                        f"Warning: The object {source_dict[source_key]} in source_module {source_module.__name__} is a {type(source_dict[source_key])}, "
                        f"but in dest_module {dest_module.__name__}, it is a {type(dest_dict[dest_key])}. Please align the types."
                        f" This substitution will be skipped"
                    )
                    # Add a warning msg and process the next obj
                    break
                else:
                    # process modules
                    if isinstance(source_dict[source_key], types.ModuleType):
                        update_dict(source_dict[source_key], dest_dict[dest_key], cache)
                        updates[dest_key] = source_dict[source_key]
                        patched = True
                        break
                    # process classes and funcs
                    else:
                        # Do nothing when dest_key equals source_key
                        updates[dest_key] = source_dict[source_key]
                        patched = True
                        break

        if not patched and os.getenv("TORCH_MLU_MIGRATION_WITH_WARNING", "1") == "1":
            updates[dest_key] = warning_wrapper(dest_dict[dest_key], dest_key)
    # update __dict__
    source_dict.update(updates)

    # get relative module name and parent module
    parent_module_name = ".".join(dest_module.__name__.split(".")[:-1])

    # for 'root' modules, will not be run
    if not parent_module_name:
        sys.modules[dest_module.__name__] = source_module
        return

    parent_module = sys.modules[parent_module_name]
    dest_module_name = dest_module.__name__.removeprefix(parent_module.__name__ + ".")

    # monkey patches for modules
    setattr(parent_module, dest_module_name.split(".")[-1], source_module)
    parent_module.__dict__[dest_module_name.split(".")[-1]] = source_module
    sys.modules[dest_module.__name__] = source_module


def _get_available_device_type():
    if torch.mlu.is_available():
        return "mlu"
    return None


old_device_constructors_ = torch.utils._device._device_constructors()


@functools.lru_cache(1)
def original_device_constructors():
    global old_device_constructors_
    return old_device_constructors_


def _new_privateuse1_deserialize(obj, location):
    if location.startswith("cuda"):
        location = location.replace("cuda", "mlu")
    origin_privateuse1_deserialize = functools.partial(
        torch.serialization._deserialize, "privateuse1"
    )
    return origin_privateuse1_deserialize(obj, location)


torch.serialization.register_package(
    11,
    functools.partial(torch.serialization._backend_tag, "privateuse1"),
    _new_privateuse1_deserialize,
)


def replace_flags():
    # torch.backends.cudnn.allow_tf32 -> torch.backends.cnnl.allow_tf32
    torch._C._get_cudnn_allow_tf32 = torch_mlu._MLUC._get_cnnl_allow_tf32
    torch._C._set_cudnn_allow_tf32 = torch_mlu._MLUC._set_cnnl_allow_tf32
    CudnnModule.allow_tf32 = ContextProp(
        torch_mlu._MLUC._get_cnnl_allow_tf32, torch_mlu._MLUC._set_cnnl_allow_tf32
    )

    # torch.backends.cudnn.deterministic -> torch.backends.cnnl.deterministic
    torch._C._get_cudnn_deterministic = torch_mlu._MLUC._get_cnnl_deterministic
    torch._C._set_cudnn_deterministic = torch_mlu._MLUC._set_cnnl_deterministic
    CudnnModule.deterministic = ContextProp(
        torch_mlu._MLUC._get_cnnl_deterministic, torch_mlu._MLUC._set_cnnl_deterministic
    )


def _apply_monkey_patches():
    # replace order MATTERS, this functions need to be replaced before torch_fn_list
    torch.utils._device._device_constructors = original_device_constructors
    # torch.*
    replace_device(torch, torch_fn_list)
    # torch.load: use replace_load_args to skip file path replacement
    torch.load = replace_load_args(torch.load)
    # Replace torch.Stream/Event with Stream/Event subclass to handle device replacement
    torch.Stream = Stream
    torch.Event = Event

    torch.get_autocast_gpu_dtype = functools.partial(
        torch.get_autocast_dtype, torch._C._get_privateuse1_backend_name()
    )
    torch.set_autocast_gpu_dtype = functools.partial(
        torch.set_autocast_dtype, torch._C._get_privateuse1_backend_name()
    )
    torch.is_autocast_enabled = replace_autocast(torch.is_autocast_enabled)
    torch.set_autocast_enabled = replace_autocast(torch.set_autocast_enabled)
    torch.amp.custom_fwd = replace_autocast(torch.amp.custom_fwd)
    torch.amp.custom_bwd = replace_autocast(torch.amp.custom_bwd)
    torch.amp.autocast_mode.custom_bwd = replace_autocast(
        torch.amp.autocast_mode.custom_bwd
    )
    torch.amp.autocast_mode.custom_fwd = replace_autocast(
        torch.amp.autocast_mode.custom_fwd
    )
    torch._C._cuda_getCurrentRawStream = torch_mlu._MLUC._mlu_getCurrentRawStream
    torch._C._cuda_setCheckpointPoolState = torch_mlu._MLUC._mlu_setCheckpointPoolState
    torch._C._cuda_getCheckpointState = torch_mlu._MLUC._mlu_getCheckpointState
    torch._C._cuda_cudaCachingAllocator_raw_delete = (
        torch_mlu._MLUC._mlu_mluCachingAllocator_raw_delete
    )
    torch._C._cuda_checkPoolLiveAllocations = (
        torch_mlu._MLUC._mlu_checkPoolLiveAllocations
    )
    torch._C._cuda_beginAllocateCurrentStreamToPool = (
        torch_mlu._MLUC._mlu_beginAllocateCurrentStreamToPool
    )
    torch._C._cuda_endAllocateCurrentStreamToPool = (
        torch_mlu._MLUC._mlu_endAllocateCurrentStreamToPool
    )
    torch._C._cuda_releasePool = torch_mlu._MLUC._mlu_releasePool
    torch._C._cuda_getAllocatorBackend = torch_mlu._MLUC._mlu_getAllocatorBackend
    torch._C._cuda_attach_out_of_memory_observer = (
        torch_mlu._MLUC._mlu_attach_out_of_memory_observer
    )
    torch._C._host_emptyCache = torch_mlu._MLUC._host_emptyCache

    # torch.ops.*
    replace_torch_ops_args(torch_ops_fn_list)

    # torch.cuda.*
    # cache_tmp = set()
    cache_tmp = None
    update_dict(torch.mlu, torch.cuda, cache_tmp)
    check_patch_succeeded(torch_mlu.mlu, torch.cuda)
    replace_device(torch.cuda, torch_cuda_fn_list)
    torch.cuda.Stream.__cuda_stream__ = torch_mlu.mlu.Stream.__mlu_stream__

    # torch.Tensor.*
    replace_device(torch.Tensor, tensor_fn_list)
    torch.Tensor.cuda = torch.Tensor.mlu
    torch.Tensor.is_cuda = torch.Tensor.is_mlu

    # torch.library.Library.*
    replace_device(torch.library.Library, library_fn_list)

    # some functions like torch.Tensor.to is first recorded in _allowed_methods
    # and then wrapped here, so the method in _allowed_methods needs to be updated.
    for i, method in enumerate(
        torch.nn.parameter.UninitializedTensorMixin._allowed_methods
    ):
        if method.__name__ in tensor_fn_list and method.__name__ != "mlu":
            torch.nn.parameter.UninitializedTensorMixin._allowed_methods[i] = getattr(
                torch.Tensor, method.__name__
            )

    # torch.nn.Module.*
    replace_device(torch.nn.Module, module_fn_list)
    torch.nn.Module.cuda = torch.nn.Module.mlu

    replace_device(torch.mlu.memory, torch_mlu_memory_fn_list)
    replace_device(torch_mlu.mlu.storage, ["_typed_storage_init"])

    # torch.distributed.*
    replace_device(torch.distributed, distributed_fn_list)

    # Other funcs
    for item in other_fn_list:
        mod = item["mod"]
        fn_list = item["fn_list"]
        replace_device(mod, fn_list)

    for mod in class_list:
        replace_device(mod, ["__init__"])

    # torch.distributed.tensor.distribute_module and distribute_tensor
    torch.distributed.tensor._api.distribute_module = (
        torch.distributed.tensor.distribute_module
    )
    torch.distributed.tensor.parallel.style.distribute_module = (
        torch.distributed.tensor.distribute_module
    )
    torch.distributed.tensor._api.distribute_tensor = (
        torch.distributed.tensor.distribute_tensor
    )
    torch.distributed.tensor.parallel.style.distribute_tensor = (
        torch.distributed.tensor.distribute_tensor
    )
    torch.distributed._state_dict_utils.distribute_tensor = (
        torch.distributed.tensor.distribute_tensor
    )
    torch.distributed._state_dict_utils.pin_memory_utils = (
        torch_mlu.mlu._pin_memory_utils
    )

    # torch.Generator
    # can't set attributes of extension type 'torch._C.Generator',
    # so we use a subclass of Generator
    torch.Generator = Generator
    replace_device(torch.Generator, ["__new__"])
    replace_device(torch.UntypedStorage, ["__new__"])
    replace_device(torch.TypedStorage, ["__new__"])

    # torch.distributed.init_process_group
    import torch.distributed._dist2 as dist2

    c10d.init_process_group = replace_nccl_with_cncl(c10d.init_process_group)
    dist2.new_group = replace_nccl_with_cncl(dist2.new_group)
    torch.distributed.init_process_group = replace_nccl_with_cncl(
        torch.distributed.init_process_group
    )
    torch.distributed.BackendConfig.__init__ = replace_nccl_with_cncl(
        torch.distributed.BackendConfig.__init__
    )
    c10d.new_group = replace_nccl_with_cncl(c10d.new_group)
    torch.distributed.new_group = replace_nccl_with_cncl(torch.distributed.new_group)
    c10d.is_nccl_available = torch.distributed.is_cncl_available
    torch.distributed.is_nccl_available = torch.distributed.is_cncl_available
    torch.distributed.ProcessGroup._get_backend = replace_device_args(
        torch.distributed.ProcessGroup._get_backend
    )
    c10d.ProcessGroupNCCL = torch_mlu._MLUC.ProcessGroupCNCL
    torch.distributed.ProcessGroupNCCL = torch_mlu._MLUC.ProcessGroupCNCL
    torch._C._distributed_c10d._dump_nccl_trace = (
        torch._C._distributed_c10d._dump_cncl_trace
    )

    # TODO(PYTORCH-11776): A workround to skip ProcessGroupCudaP2P,
    # remove following code when MLU support ProcessGroupCudaP2P.
    c10d.ProcessGroupCudaP2P = torch_mlu._MLUC.ProcessGroupCNCL
    torch.distributed.ProcessGroupCudaP2P = torch_mlu._MLUC.ProcessGroupCNCL

    torch._C._cuda_hasPrimaryContext = torch_mlu._MLUC._mlu_hasPrimaryContext
    torch._C._cudart = torch_mlu._MLUC._cnrt
    torch._C._cudart.cudaError = torch_mlu._MLUC._cnrt.mluError
    torch._C._cudart.cudaGetErrorString = torch_mlu._MLUC._cnrt.mluGetErrorStr
    torch._C._cudart.cudaProfilerStart = torch_mlu._MLUC._cnrt.mluProfilerStart
    torch._C._cudart.cudaProfilerStop = torch_mlu._MLUC._cnrt.mluProfilerStop
    torch._C._cudart.cudaStreamCreate = torch_mlu._MLUC._cnrt.mluStreamCreate
    torch._C._cudart.cudaStreamDestroy = torch_mlu._MLUC._cnrt.mluStreamDestroy
    torch._C._cudart.cudaHostRegister = torch_mlu._MLUC._cnrt.mluHostRegister
    torch._C._cudart.cudaHostUnregister = torch_mlu._MLUC._cnrt.mluHostUnregister
    torch.cuda.cudart = torch.mlu.cnrt

    # torch.utils.data.DataLoader
    torch.utils.data.DataLoader.__init__ = replace_dataloader(
        torch.utils.data.DataLoader.__init__
    )

    # torch.profiler
    torch.profiler.ProfilerActivity.CUDA = torch.profiler.ProfilerActivity.PrivateUse1
    torch.autograd.ProfilerState.CUDA = torch.autograd.ProfilerState.PRIVATEUSE1
    replace_profiler(torch.profiler._KinetoProfile, "__init__")
    replace_profiler(torch.autograd.profiler.profile, "__init__")
    replace_profiler(torch.autograd.profiler_util, "_build_table")
    replace_profiler(torch.autograd.profiler_util.EventList, "export_stacks")
    torch.profiler._pattern_matcher.ExtraCUDACopyPattern = (
        torch_mlu.profiler._pattern_matcher.ExtraMLUCopyPattern
    )

    # storage
    torch.TypedStorage.cuda = torch.TypedStorage.mlu
    torch.TypedStorage.is_cuda = torch.TypedStorage.is_mlu

    torch.jit.script = script

    torch.autograd.profiler.emit_nvtx = torch.autograd.profiler.emit_cnpx

    torch.backends.cuda = torch.backends.mlu
    torch.backends.__setattr__("cuda", torch.backends.mlu)

    torch._prims.context.torch_to_refs_map = torch_to_refs_map

    torch._utils._get_available_device_type = _get_available_device_type

    # cuda default
    torch.UntypedStorage._release_ipc_counter_cuda = (
        torch.UntypedStorage._release_ipc_counter_mlu
    )
    torch.UntypedStorage._share_cuda_ = torch.UntypedStorage._share_mlu_
    torch.UntypedStorage._new_shared_cuda = torch.UntypedStorage._new_shared_mlu
    # Regardless of whether the device parameter is cuda or other,
    # torch.TypedStorage._release_ipc_counter will call cuda's API.
    # Here it is replaced with torch.TypedStorage._release_ipc_counter_mlu,
    # and the mlu API will be called anyway.
    torch.TypedStorage._release_ipc_counter = (
        torch.TypedStorage._release_ipc_counter_mlu
    )
    torch.TypedStorage._share_cuda_ = torch.TypedStorage._share_mlu_
    torch.TypedStorage._new_shared_cuda = torch.TypedStorage._new_shared_mlu

    for func_name in default_cuda_args_list:
        replace_default_cuda_args(func_name)

    # detected by monkey patch leak check

    torch.serialization.load = torch.load
    torch.nn.parallel.data_parallel._get_available_device_type = (
        _get_available_device_type
    )
    torch.mtia.Event = torch.Event
    torch.utils.data.dataset.Generator = Generator
    torch.mtia.Stream = torch.Stream
    torch.cuda._get_device_index = torch.mlu._utils._get_device_index

    parallel_apply_module = importlib.import_module("torch.nn.parallel.parallel_apply")
    parallel_apply_module._get_device_index = torch.mlu._utils._get_device_index

    data_parallel_module = importlib.import_module("torch.nn.parallel.data_parallel")
    data_parallel_module._get_available_device_type = _get_available_device_type

    sys.modules["torch.backends.cuda"] = torch.backends.mlu
    torch.nn.attention.can_use_efficient_attention = (
        torch_mlu.backends.mlu.can_use_efficient_attention
    )
    torch.nn.attention.can_use_flash_attention = (
        torch_mlu.backends.mlu.can_use_flash_attention
    )
    torch.nn.attention.enable_flash_sdp = torch_mlu.backends.mlu.enable_flash_sdp
    torch.nn.attention.enable_math_sdp = torch_mlu.backends.mlu.enable_math_sdp
    torch.nn.attention.enable_mem_efficient_sdp = (
        torch_mlu.backends.mlu.enable_mem_efficient_sdp
    )
    torch.nn.attention.flash_sdp_enabled = torch_mlu.backends.mlu.flash_sdp_enabled
    torch.nn.attention.math_sdp_enabled = torch_mlu.backends.mlu.math_sdp_enabled
    torch.nn.attention.mem_efficient_sdp_enabled = (
        torch_mlu.backends.mlu.mem_efficient_sdp_enabled
    )

    torch.mlu.graphs.MLUGraph.raw_cuda_graph = torch.mlu.graphs.MLUGraph.raw_mlu_graph
    torch.cuda.graphs.CUDAGraph = torch.mlu.graphs.MLUGraph
    torch.cuda.memory.is_initialized = torch.mlu.is_initialized
    torch.nn.parallel.comm.nccl = torch.mlu.cncl
    torch.distributed.device_mesh.init_process_group = (
        torch.distributed.distributed_c10d.init_process_group
    )
    torch.distributed.device_mesh.new_group = (
        torch.distributed.distributed_c10d.new_group
    )
    torch.utils.data.dataset.randperm = torch.randperm
    torch.jit._script.script = script
    torch.jit._trace.script = script

    torch.accelerator._utils._device_t = torch_mlu.mlu.device
    torch.accelerator._device_t = torch_mlu.mlu.device
    torch.utils.dlpack._Device = torch_mlu.mlu.device

    # _inductor monkey patch
    torch._inductor.utils.DeviceProperties = (
        torch_mlu._inductor.runtime.hints.DeviceProperties
    )
    torch._C._aoti.AOTIModelContainerRunnerCuda = (
        torch_mlu._MLUC._aoti.AOTIModelContainerRunnerMlu
    )

    # _dynamo monkey patch
    replace_device(torch._dynamo.device_interface, ["get_interface_for_device"])
    torch._inductor.utils.get_interface_for_device = (
        torch._dynamo.device_interface.get_interface_for_device
    )
    torch._dynamo.output_graph.get_interface_for_device = (
        torch._dynamo.device_interface.get_interface_for_device
    )

    # migration backends.cudnn.*
    replace_flags()


def apply_monkey_patches():
    # Limit warning control range.
    with warnings.catch_warnings():
        warnings.filterwarnings(action="once")
        _apply_monkey_patches()
