import os
import functools
import hashlib
import json
from typing import (
    Any,
    Union,
)
import torch
import importlib
from torch._inductor.codecache import (
    code_hash,
    FxGraphHashDetails,
    OrderedSetHolder,
    build_code_hash,
    torch_key_cache,
)
from torch._inductor.codegen.common import (
    custom_backend_codegen_configs,
    custom_backend_passes,
    init_backend_registration,
)
from torch._inductor.compile_fx import _CompileFxKwargs
from torch._dynamo.utils import dynamo_timed
from torch._inductor import config
from ..utils import gorilla
from ctypes import c_void_p

from collections.abc import Sequence
from torch._inductor.utils import InputType
from torch._inductor.utils import XPU_KERNEL_FORMAT


@gorilla.patch(torch._inductor.codecache)
def _get_cpp_wrapper_header(device: str, aot_mode: bool = False) -> str:
    """Given a device type (and optionally whether we're in AOT Inductor mode), returns
    the path to the cpp_wrapper header file to be precompiled."""
    base_device = device.split(":", maxsplit=1)[0]
    is_array_ref = config.aot_inductor.allow_stack_allocation and base_device == "cpu"
    # Modify by cambricon
    base_path = "torch/csrc/inductor/"
    if device == "mlu":
        base_path = "framework/inductor/"
    return (
        # "torch/csrc/inductor/"
        f"{base_path}"
        f"{'aoti_include' if aot_mode else 'cpp_wrapper'}/"
        f"{'array_ref' if is_array_ref else base_device}.h"
    )
    # end Modify by cambricon


@gorilla.patch(torch._inductor.codecache.CacheBase)
@staticmethod
@functools.cache
def get_system() -> dict[str, Any]:
    from torch._inductor.runtime.triton_compat import HAS_TRITON, triton_key

    if HAS_TRITON:
        # Use triton_key instead of triton.__version__ as the version
        # is not updated with each code change
        triton_version = triton_key()
    else:
        triton_version = None

    try:
        system: dict[str, Any] = {
            "device": {"name": None},
            "version": {
                "triton": triton_version,
            },
        }
        # Modify by CAMBRICON: add a control block to handle mlu.
        # device_properties = torch.cuda.get_device_properties(
        #     torch.cuda.current_device()
        # )
        if torch.version.mlu is not None:
            device_properties = torch.mlu.get_device_properties(
                torch.mlu.current_device()
            )
        else:
            device_properties = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            )
        # end Modify by CAMBRICON
        if torch.version.cuda is not None:
            system["device"]["name"] = device_properties.name
            system["version"]["cuda"] = torch.version.cuda
        # Modify by CAMBRICON: add codes for mlu.
        elif torch.version.mlu is not None:
            system["device"]["name"] = device_properties.name
            # Note, this version is the version of cntoolkit. When the y or z
            # of version is changed, the cache doesn't need to be recompiled.
            system["version"]["mlu"] = torch.version.mlu[0]
        # end Modify by CAMBRICON
        else:
            system["device"]["name"] = device_properties.gcnArchName
            system["version"]["hip"] = torch.version.hip
    except (AssertionError, RuntimeError):
        # If cuda is not installed, none of the above config is relevant.
        system = {}

    system["hash"] = hashlib.sha256(
        json.dumps(system, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return system


@gorilla.patch(torch._inductor.codecache)
def get_hash(
    content: Union[str, bytes], extra: str = "", hash_type: str = "code"
) -> str:
    if hash_type in {"amdgcn", "code", "ptx", "spv"}:
        return code_hash(content, extra)
    # Modify by CAMBRICON
    # if hash_type in {"cubin", "hsaco", XPU_KERNEL_FORMAT}:
    if hash_type in ["cubin", "hsaco", XPU_KERNEL_FORMAT, "cnbin"]:
        # end Modify by CAMBRICON
        return code_hash(repr(content))
    raise AssertionError(f"Unknown hash type {hash_type}")


@gorilla.patch(torch._inductor.codecache)
def custom_op_wrapper(op: str, *args: Any) -> Union[list[c_void_p], c_void_p, None]:
    # This function will be called from generated cpp wrapper code in the JIT mode.
    # Because tensors will be passed in as AtenTensorHandle, we need to explicitly convert them.
    def convert_arg(arg: Any) -> Any:
        if str(type(arg)) == "<class 'PyCapsule'>":
            # No easy way to do isinstance check on PyCapsule
            return torch._C._aoti.alloc_tensor_by_stealing_from_void_ptr(arg)
        elif isinstance(arg, (list, tuple)):
            return type(arg)(convert_arg(a) for a in arg)
        else:
            return arg

    converted_args = [convert_arg(arg) for arg in args]

    assert op.startswith("torch.ops."), (
        op + " can not be called through custom_op_wrapper"
    )
    func = None
    for i, s in enumerate(op.split(".")):
        if i == 0:
            func = importlib.import_module(s)
        func = getattr(func, s)

    assert callable(func), op + " can not be loaded through custom_op_wrapper"

    # convert any kwarg-only arguments to kwargs
    kwargs = dict()
    # pyrefly: ignore [missing-attribute]
    for func_arg, conv_arg in zip(func._schema.arguments, converted_args):
        if func_arg.kwarg_only:
            kwargs[func_arg.name] = conv_arg
    if kwargs:
        del converted_args[-len(kwargs) :]

    result = func(*converted_args, **kwargs)
    if result is None:
        return None

    if isinstance(result, (list, tuple)):
        # unsafe_alloc_void_ptrs_from_tensors expects result contains tensor only
        result = [torch.tensor([]) if r is None else r for r in result]
        for r in result:
            assert isinstance(r, torch.Tensor), op + " returns a list of non-tensors"

        # modified by Cambricon: fix a corner case.
        # case1: output of custom op is list[Tensor], and length of output is 1.
        # case2: output of custom op is Tensor.
        # generate_fallback_kernel_with_runtime_lookup_jit can't distinguish between case1 and case2.
        if len(result) == 1:
            return torch._C._aoti.unsafe_alloc_void_ptr_from_tensor(result[0])
        return torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(result)  # type: ignore[arg-type]

    assert isinstance(result, torch.Tensor), op + " returns a non-tensor"
    return torch._C._aoti.unsafe_alloc_void_ptr_from_tensor(result)


@torch_key_cache
def torch_mlu_key() -> bytes:
    """
    Compute a key that contains relevant information about torch source files
    """
    with dynamo_timed("inductor_codecache_torch_key", log_pt2_compile_event=False):
        if not config.is_fbcode():

            def get_code_hash(root: str) -> bytes:
                # This function isn't meant to be used outside of torch_key, just a
                # helper for clarity. Instead, use torch_key() directly when you need
                # a hash representing the state of the source code.
                extra_files = (
                    "codegen/aoti_runtime/interface.cpp",
                    "script.ld",
                )
                inductor_root = os.path.dirname(__file__)
                extra_files = [os.path.join(inductor_root, x) for x in extra_files]
                hasher = hashlib.sha256()
                import torch_mlu

                hasher.update(torch_mlu.__version__.encode("utf-8"))
                build_code_hash([root], "", hasher)
                for path in extra_files:
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            hasher.update(f.read())
                return hasher.digest()

            _HERE = os.path.abspath(__file__)
            _TORCH_MLU_PATH = os.path.dirname(os.path.dirname(_HERE))
            return get_code_hash(_TORCH_MLU_PATH)

        from libfb.py import parutil

        return parutil.get_file_contents("torch/src_hash.txt").rstrip().encode("ascii")


@gorilla.patch(torch._inductor.codecache.FxGraphHashDetails)
def __init__(
    self,
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[InputType],
    fx_kwargs: _CompileFxKwargs,
    inputs_to_check: Sequence[int],
) -> None:
    self.gm = gm
    self.example_inputs = example_inputs
    self.cache_key_tag = cconfig.cache_key_tag

    # Order kwargs so hashing is stable to changes in kwarg order. Although
    # it's technically a _CompileFxKwargs we don't actually need it typed as
    # such since we're just using it to generate a hash.
    self.fx_kwargs: dict[str, object] = {}
    for k, v in sorted(fx_kwargs.items()):
        if k not in self.EXCLUDED_KWARGS:
            if type(v) in (set, OrderedSet):  # noqa: set_linter
                # Special case to handle set params. Python sets can't be
                # ordered, so sort the elements and store them in a proxy.
                self.fx_kwargs[k] = OrderedSetHolder(sorted(v))  # type: ignore[call-overload]
            else:
                self.fx_kwargs[k] = v

    from torch._higher_order_ops.triton_kernel_wrap import (
        kernel_side_table,
        triton_kernel_wrapper_functional,
        triton_kernel_wrapper_mutation,
    )
    from torch._inductor.codegen.wrapper import (
        user_defined_triton_kernel_transitive_closure_source_code,
    )

    # Node meta will not be part of gm's reduce function, so lets remember
    # the kernel source code separately
    self.user_defined_triton_source: list[Any] = []
    if gm is not None:
        for module in gm.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in itertools.chain(
                module.graph.find_nodes(
                    op="call_function", target=triton_kernel_wrapper_functional
                ),
                module.graph.find_nodes(
                    op="call_function", target=triton_kernel_wrapper_mutation
                ),
            ):
                from triton.runtime.autotuner import Autotuner

                kernel = kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
                configs = None
                if isinstance(kernel, Autotuner):
                    if kernel.configs:
                        configs = str(
                            sorted(
                                sorted(str(kv) for kv in c.all_kwargs().items())
                                for c in kernel.configs
                            )
                        )
                    kernel = kernel.fn

                kernel_source = (
                    user_defined_triton_kernel_transitive_closure_source_code(kernel)
                )
                constant_args = kernel_side_table.get_constant_args(
                    node.kwargs["constant_args_idx"]
                )
                self.user_defined_triton_source.append(
                    (kernel_source, constant_args, configs)
                )

    # Alignment checks
    self.inputs_to_check = inputs_to_check

    no_tensor_inputs = not any(isinstance(x, torch.Tensor) for x in example_inputs)
    # This device index is usually already encoded by the device of the inputs
    # but fx graphs don't necessarily have tensor inputs. If there aren't any,
    # we need to guard on the device index in case we allocate cuda tensors
    if no_tensor_inputs and torch.accelerator.is_available():
        self.default_cuda_device_index = torch.accelerator.current_device_index()

    # 'Deterministic algorithms' can affect codegen via lowering to cuda kernels.
    self.deterministic_algorithms_settings = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.utils.deterministic.fill_uninitialized_memory,  # type: ignore[attr-defined]
    )

    # Global settings affecting matmul codegen.
    # Modify by CAMBRICON
    self.cuda_matmul_settings = (
        # torch.backends.cuda.matmul.fp32_precision,
        # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        # torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.mlu.matmul.fp32_precision,
        torch.backends.mlu.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.mlu.matmul.allow_bf16_reduced_precision_reduction,
    )
    # end Modify by CAMBRICON

    # Also hash on various system info (including the triton compiler version).
    self.torch_version = torch_key()
    self.system_info = CacheBase.get_system()
    self.inductor_config = config.save_config_portable(ignore_private_configs=False)
    # Custom post grad passes should provide an ID to hash.
    self.post_grad_custom_pre_pass = self._get_custom_pass_detail(
        config.post_grad_custom_pre_pass
    )
    # TODO: change to more holistic config rather than bundled_autograd_cache
    self.precompile_enabled = torch._functorch.config.bundled_autograd_cache
    self.post_grad_custom_post_pass = self._get_custom_pass_detail(
        config.post_grad_custom_post_pass
    )
    self.joint_custom_pre_pass = self._get_custom_pass_detail(
        config.joint_custom_pre_pass
    )
    self.joint_custom_post_pass = self._get_custom_pass_detail(
        config.joint_custom_post_pass
    )
    self._pre_fusion_custom_pass = self._get_custom_pass_detail_unsafe(
        config._pre_fusion_custom_pass
    )
    self._fuse_ddp_communication_passes = self._get_custom_pass_detail_unsafe(
        config._fuse_ddp_communication_passes
    )

    # Register indcutor backends and custom passes and get their UUIDs.
    init_backend_registration()
    self.custom_backend_passes = tuple(
        map(self._get_custom_pass_detail, custom_backend_passes.values())
    )

    # Save custom inductor codegen configs
    self.custom_backend_codegen_configs = {
        device: custom_config.save_config_portable(ignore_private_configs=False)
        for device, custom_config in custom_backend_codegen_configs.items()
        if custom_config is not None
    }

    # Register the custom partitioner function
    self._custom_partitioner_fn = self._get_custom_partitioner_fn_detail(
        config.custom_partitioner_fn
    )

    # Include hint overrides in the cache key because _reduce_symint
    # only hashes symbol names, not hint values.
    self.var_to_hint_override: dict[str, int] = {}
    shape_env = FxGraphCache._get_shape_env()
    if shape_env is not None and shape_env.var_to_hint_override:
        self.var_to_hint_override = {
            str(sym): val
            for sym, val in sorted(
                shape_env.var_to_hint_override.items(), key=lambda x: str(x[0])
            )
        }

    # Add by CAMBRICON: add torch_mlu properties to cache key calculation
    import torch_mlu
    from torch_mlu._inductor.codecache import torch_mlu_key
    from torch_mlu._inductor.lowering_utils import remove_list
    from torch_mlu._inductor.decomposition_utils import mlu_decomps_to_exclude

    self.torch_mlu_version = torch_mlu_key()
    self.mlu_inductor_config = torch_mlu._inductor.config.save_config_portable()
    self.remove_list = tuple(str(fn) for fn in remove_list)
    self.mlu_decomps_to_exclude = tuple(str(fn) for fn in mlu_decomps_to_exclude)
    # end Add by CAMBRICON
