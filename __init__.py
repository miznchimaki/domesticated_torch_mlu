from typing import Tuple
import re
import sys
import os
import warnings
# Disable auto load torch_mlu when load torch_mlu to
# avoid recursive init when only import torch_mlu without
# import torch
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch
os.environ.pop('TORCH_DEVICE_BACKEND_AUTOLOAD')
from torch.torch_version import Version
from torch_mlu import version

# PyTorch Environment Compatibility Check for torch_mlu
#
# Autoload will be skipped if any of the following conditions are met:
#   1. PyTorch is not built for CPU.
#   2. The PyTorch major or minor version differs from the required version.
#   3. The C++11 ABI setting does not match the expected configuration.
#   4. The compiler type differs (e.g., GCC vs. Clang).
acc = torch.accelerator.current_accelerator() or torch.device("cpu")
try:
    torch_version = Version(torch.__version__).release  # (2, 7, 1)
except InvalidVersion:
    torch_version = None
    warnings.warn(
        f"[torch_mlu] Failed to parse PyTorch version: '{torch.__version__}'. "
        f"Please manually check whether it matches the version required by torch_mlu."
    )
torch_git_version = torch.version.git_version
torch_cxx11_abi = torch.compiled_with_cxx11_abi()
torch_built_with_clang = "clang" in torch.__config__.show().lower()

def extract_torch_version(mlu_version: str) -> Tuple[int, ...]:
    match = re.search(r"\+torch(\d+\.\d+\.\d+)", mlu_version)
    return tuple(int(x) for x in match.group(1).split(".")) if match else None

expected_version = extract_torch_version(version.mlu_version)  # (2, 7, 1)
expected_git_version = version.pytorch_git_version
expected_cxx11_abi = version.compiled_with_cxx11_abi
expected_built_with_clang = version.built_with_clang

if torch_git_version != expected_git_version:
    warnings.warn(
        f"[torch_mlu] PyTorch Git version mismatch: "
        f"expected '{expected_git_version[:7]}', but found '{torch_git_version[:7]}'. "
        f"Please manually check whether this version satisfies the requirements of torch_mlu."
    )

# Check whether torch_cpu was built with OpenMP.
torch_build_config = torch.__config__.show()
if not re.search(r"OpenMP\s+\d+", torch_build_config):
    warnings.warn(
        "[torch_mlu] PyTorch CPU was built without OpenMP support. "
    )

version_mismatch = (
    torch_version is not None and expected_version is not None and
    (torch_version[0] != expected_version[0] or torch_version[1] != expected_version[1])
)
incompatible = (
    acc.type != "cpu" or version_mismatch
    or torch_cxx11_abi != expected_cxx11_abi
    or torch_built_with_clang != expected_built_with_clang
)

if incompatible:
    warnings.warn(
        "[torch_mlu] Autoload skipped due to environment mismatch:\n"
        f"  • Expected:\n"
        f"      PyTorch {expected_version}\n"
        f"      platform       : cpu\n"
        f"      git            : {expected_git_version[:7]}\n"
        f"      cxx11_abi      : {expected_cxx11_abi}\n"
        f"      compiler_clang : {expected_built_with_clang}\n"
        f"  • Found:\n"
        f"      PyTorch {torch_version}\n"
        f"      platform       : {acc.type}\n"
        f"      git            : {torch_git_version[:7]}\n"
        f"      cxx11_abi      : {torch_cxx11_abi}\n"
        f"      compiler_clang : {torch_built_with_clang}"
    )
else:
    from torch.utils.checkpoint import DefaultDeviceType
    from torch.overrides import (
        handle_torch_function,
        has_torch_function_unary,
    )
    from torch_mlu import _MLUC
    import torch_mlu.mlu
    import torch_mlu.mlu.amp
    import torch_mlu.mlu.cnpx
    import torch_mlu.backends
    from torch_mlu.backends.cnnl import CnnlModule
    from torch_mlu.backends.mlufusion import MlufusionModule
    from torch_mlu.profiler import apply__pattern_matcher_patch
    from torch_mlu.utils import gorilla
    from torch_mlu.utils import (
        apply_utils_patch,
        apply_torch_overrides_patch,
        apply_ddp_patch,
        apply_benchmark_utils_patch,
        apply_as_tensor_patch,
    )
    from torch_mlu.nn import apply_functional_patch
    from torch_mlu.distributed import apply_distributed_patch, _local_tensor, device_mesh
    from torch_mlu.mlu import apply_storage_patch, apply_reductions_patch
    from torch_mlu.mlu import _pin_memory_utils

    from torch_mlu.optimizers import optimizer

    def get_version():
        return _MLUC._get_version()

    __version__ = get_version()

    def get_driver_version():
        return _MLUC._get_driver_version()

    def get_git_version() -> str:
        try:
            from torch_mlu.version import git_version
            return git_version
        except Exception:
            pass
        return "unknown"

    def apply_patches():
        apply_utils_patch()
        apply_functional_patch()
        apply_torch_overrides_patch()
        apply_distributed_patch()
        apply_ddp_patch()
        apply_benchmark_utils_patch()
        apply__pattern_matcher_patch()
        apply_as_tensor_patch()


    def _check_register_once(module, attr):
        if hasattr(module, attr):
            raise RuntimeError(f"The custom device module of {module} has already been registered with {attr}")

    torch.utils.rename_privateuse1_backend("mlu")
    torch._register_device_module('mlu', torch_mlu.mlu)

    unsupported_dtype = [
        torch.quint8, torch.quint4x2,
        torch.quint2x4, torch.qint32, torch.qint8
    ]

    # tensor and storage have custom implementation, we generate methods in other way
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True, for_module=True, for_packed_sequence = True, for_storage=True,
        unsupported_dtype=unsupported_dtype)

    apply_patches()

    ### torch.backends
    torch._C._set_cublas_allow_tf32 = torch_mlu.backends.mlu.fake_set_cublas_allow_tf32
    torch._C._set_cublas_allow_fp16_reduced_precision_reduction = torch_mlu.backends.mlu.fake_set_cublas_allow_fp16_reduced_precision_reduction
    torch._C._set_cublas_allow_bf16_reduced_precision_reduction = torch_mlu.backends.mlu.fake_set_cublas_allow_bf16_reduced_precision_reduction
    torch._C._conv_determine_backend_memory_format = torch_mlu._MLUC._conv_determine_backend_memory_format
    torch._C._cuda_getCurrentRawStream = torch_mlu._MLUC._mlu_getCurrentRawStream
    torch._C._cuda_CUDAAllocator_AllocatorState = torch_mlu._MLUC._mlu_MLUAllocator_AllocatorState
    torch._C._cuda_checkPoolLiveAllocations = torch_mlu._MLUC._mlu_checkPoolLiveAllocations
    torch._C._cuda_getCheckpointState = torch_mlu._MLUC._mlu_getCheckpointState
    torch._C._storage_Use_Count = torch_mlu._MLUC._storage_Use_Count
    torch._C._set_storage_access_error_msg = torch_mlu._MLUC._set_storage_access_error_msg
    torch._C._set_storage_data_ptr_access_error_msg = torch_mlu._MLUC._set_storage_data_ptr_access_error_msg
    torch._C._free_And_Remove_DeleterFn = torch_mlu._MLUC._free_And_Remove_DeleterFn
    torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata = torch_mlu._MLUC._construct_MLU_Tensor_From_Storage_And_Metadata
    torch._C._tensors_data_ptrs_at_indices_equal = torch_mlu._MLUC._tensors_data_ptrs_at_indices_equal
    torch._C._cuda_setCheckpointPoolState = torch_mlu._MLUC._mlu_setCheckpointPoolState
    torch._C._cuda_cudaCachingAllocator_raw_delete = torch_mlu._MLUC._mlu_mluCachingAllocator_raw_delete
    torch._C._has_Standard_Deleter = torch_mlu._MLUC._has_Standard_Deleter
    torch._C._tensors_data_ptrs_at_indices_equal = torch_mlu._MLUC._tensors_data_ptrs_at_indices_equal
    torch.backends.__setattr__("mlu", torch_mlu.backends.mlu)
    sys.modules["torch.backends.mlu"] = torch_mlu.backends.mlu
    torch.backends.__setattr__("cnnl", CnnlModule(torch_mlu.backends.cnnl, "torch.backends.cnnl"))
    torch.backends.__setattr__("mlufusion", MlufusionModule(torch_mlu.backends.mlufusion, "torch.backends.mlufusion"))
    torch.version.mlu = version.cntoolkit_version

    _MLUC._initExtension()

    ### tf32 cnmatmul control
    setattr(torch._C, "_get_cnmatmul_allow_tf32", torch_mlu._MLUC._get_cnmatmul_allow_tf32)
    setattr(torch._C, "_set_cnmatmul_allow_tf32", torch_mlu._MLUC._set_cnmatmul_allow_tf32)
    # set default device type mlu for checkpointing
    DefaultDeviceType.set_device_type("mlu")

    apply_storage_patch()
    apply_reductions_patch()

    # Register CNSHMEM symmetric memory backend if available
    try:
        from torch_mlu.mlu.cnshmem import _register_cnshmem_backend
        _register_cnshmem_backend()
    except (ImportError, RuntimeError):
        pass

    # add torch.Tensor.mlu in _allowed_methods to align torch.Tensor.cuda
    torch.nn.parameter.UninitializedTensorMixin._allowed_methods.append(
        torch.Tensor.mlu
    )

    from . import _dynamo
    from . import _inductor
    from ._inductor import (
        runtime as inductor_runtime,
        utils as inductor_utils,
        codecache as inductor_codecache,
        cudagraph_utils as inductor_cudagraph_utils,
        debug as inductor_debug,
        sizevars as inductor_sizevars,
        choices as inductor_choices,
        lowering as inductor_lowering,
        decomposition as inductor_decomposition,
        select_algorithm as inductor_select_algorithm,
        kernel as inductor_kernel,
        compile_fx as inductor_compile_fx,
        cudagraph_trees as inductor_cudagraph_trees,
        output_code as inductor_output_code,
        ir as inductor_ir,
        scheduler as inductor_scheduler,
    )
    from ._inductor.codegen import (
        triton,
        triton_combo_kernel,
        wrapper,
    )
    from ._inductor.fx_passes import (
        post_grad,
    )
    from ._inductor.runtime import (
        autotune_cache,
    )
    from ._inductor.decomposition_utils import (
        allow_aten_fn_decomposition,
        deny_aten_fn_decomposition,
    )
    from ._inductor.lowering_utils import (
        allow_aten_fn_lowering,
        deny_aten_fn_lowering,
    )
    from ._dynamo import (
        compiled_autograd as dynamo_compiled_autograd,
        device_interface as dynamo_device_interface,
        trace_rules as dynamo_trace_rules,
        guards as dynamo_guards,
    )
    from ._dynamo.backends import cudagraphs as dynamo_cudagraphs
    from ._dynamo.backends import common as dynamo_common
    from ._dynamo.variables import (
        builder as dynamo_builder,
        torch as dynamo_torch,
        user_defined as dynamo_user_defined,
        functions as dynamo_functions,
    )
    from ._dynamo.repro import (
        after_dynamo as dynamo_after_dynamo,
        after_aot as dynamo_after_aot,
    )
    from . import _export
    from .utils import _sympy, _triton
    from .utils._sympy import interp
    from .fx.experimental import validator, proxy_tensor
    from .fx.passes import _tensorify_python_scalars, graph_transform_observer
    from ._functorch import autograd_cache, partitioners
    from ._subclasses import fake_impls
    from .nn import flex_attention
    from . import _tensor_str

    gorilla_patches = gorilla.find_patches([
        _tensorify_python_scalars,
        _tensor_str,
        _triton,
        dynamo_after_aot,
        dynamo_after_dynamo,
        dynamo_builder,
        dynamo_common,
        dynamo_compiled_autograd,
        dynamo_cudagraphs,
        dynamo_device_interface,
        dynamo_functions,
        dynamo_guards,
        dynamo_torch,
        dynamo_trace_rules,
        dynamo_user_defined,
        fake_impls,
        flex_attention,
        graph_transform_observer,
        autograd_cache,
        inductor_runtime,
        inductor_utils,
        inductor_codecache,
        inductor_cudagraph_utils,
        inductor_debug,
        inductor_ir,
        inductor_sizevars,
        inductor_output_code,
        interp,
        optimizer,
        partitioners,
        proxy_tensor,
        validator,
        wrapper,
        _local_tensor,
        _pin_memory_utils,
        device_mesh,
    ])
    for patch in gorilla_patches:
        gorilla.apply(patch)

    from . import fx
    from . import triton_kernel
    from . import _functorch
    from . import _prims
    from . import _higher_order_ops

    from torch_mlu import _meta_registrations

    # Support environment variable to auto-enable gpu_migration.
    # This ensures migration patches are applied before any third-party code
    # can access torch.cuda interfaces, which is critical in complex environments
    # (ray, megatron, etc.) where import order cannot be guaranteed.
    if os.getenv("TORCH_MLU_GPU_MIGRATION", "0") == "1":
        print("[torch_mlu] GPU migration auto-enabled via TORCH_MLU_GPU_MIGRATION=1")
        import torch_mlu.utils.gpu_migration

def init():
    return
