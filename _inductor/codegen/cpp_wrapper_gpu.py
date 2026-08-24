from ...utils import gorilla
from torch._inductor.codegen.cpp_wrapper_gpu import (
    DeferredTritonCallWrapper,
    CppWrapperGpu,
    UnwrapUnspecArg,
)
from torch._inductor.codecache import CudaKernelParamCache
from torch._inductor.codegen.wrapper import SymbolicCallArg
from torch._inductor.virtualized import V
from torch._inductor.codegen.aoti_hipify_utils import maybe_hipify_code_wrapper
from torch._inductor import config


def generate(self, wrapper: CppWrapperGpu):
    """
    Generate the GPU kernel definition, as well as load and launch code.
    """
    prefix = wrapper.prefix
    if self.kernel_name.startswith("multi_kernel_"):
        # MultiKernel will select one kernel after running the autotune block
        self.kernel_name = MultiKernelCall.lookup_choice(self.kernel_name)
    params = CudaKernelParamCache.get(self.kernel_name)
    assert params, f"CudaKernelParamCache not populated for {self.kernel_name}"
    def_args = params["def_args"]
    arg_types = self.arg_types
    inductor_meta = params["inductor_meta"]

    if "extra_launcher_args" in inductor_meta and len(def_args) > len(arg_types):
        # extra_launcher_args should already be in def_args
        assert len(def_args) == len(arg_types) - len(
            inductor_meta["extra_launcher_args"]
        )
        arg_types = arg_types + [SymbolicCallArg] * len(
            inductor_meta["extra_launcher_args"]
        )

    if not V.graph.aot_mode:
        prefix.writeline(
            maybe_hipify_code_wrapper(
                f"static {wrapper.device_codegen.cpp_kernel_type()} {self.kernel_name} = nullptr;"
            )
        )
        kernel_var_name = self.kernel_name
    else:
        kernel_var_name = f"kernels_.{self.kernel_name}"

    # tensors can be RAIIAtenTensorHandle or ConstantHandle, so make them template types
    template_types = [
        f"typename {name}_type_"
        for name, arg_type in zip(def_args, arg_types)
        if isinstance(arg_type, (torch_dtype, UnwrapUnspecArg))
    ]
    if V.graph.aot_mode:
        template_types.append("typename kernels_type_")
    if template_types:
        prefix.writeline(f"template <{', '.join(template_types)}>")
    prefix.writeline(f"static inline void {self.wrapper_name}(")
    # Add by CAMBRICON
    # Add more cases for other types as needed
    signature2dtype = {
        "i32": "int64_t",
        "i64": "int64_t",
        "fp32": "float",
    }
    signature = params["triton_meta"]["signature"]
    # end Add by CAMBRICON
    with prefix.indent():
        assert len(def_args) == len(arg_types), (def_args, arg_types)
        for name, arg_type in zip(def_args, arg_types):
            if isinstance(arg_type, (torch_dtype, UnwrapUnspecArg)):
                prefix.writeline(f"const {name}_type_& {name},")
            elif issubclass(arg_type, (SymbolicCallArg, sympy.Expr, int)):
                # Modify by CAMBRICON
                # prefix.writeline(f"int64_t {name},")
                if issubclass(arg_type, (sympy.Expr)) and name in signature:
                    try:
                        dtype = signature2dtype[signature[name]]
                        prefix.writeline(f"{dtype} {name},")
                    except KeyError:
                        prefix.writeline(f"int64_t {name},")
                else:
                    prefix.writeline(f"int64_t {name},")
                # end Modify by CAMBRICON
            elif arg_type is float:
                prefix.writeline(f"float {name},")
            elif arg_type is bool:
                prefix.writeline(f"bool {name},")
            else:
                raise ValueError(f"Unexpected arg type {arg_type}")
        prefix.writeline("int32_t device_idx_,")
        prefix.writeline(
            maybe_hipify_code_wrapper(
                f"{wrapper.device_codegen.cpp_stream_type()} stream_,"
            )
        )
        if V.graph.aot_mode:
            prefix.writeline("kernels_type_& kernels_,")
        prefix.writeline("const std::optional<std::string>& cubin_dir_ = std::nullopt")
    prefix.writeline("){")
    with prefix.indent():
        if V.graph.aot_mode:
            # Emit the original Triton kernel for debugging purposes
            prefix.writeline("/*")
            prefix.splice(self.kernel_name_to_body[self.kernel_name])
            prefix.writeline("*/")
        self.generate_grid(prefix, inductor_meta, params)
        self.generate_load_kernel(prefix, kernel_var_name, params)
        self.generate_launch_kernel(prefix, wrapper, kernel_var_name, params)
    prefix.writeline("}")

    if not config.aot_inductor.embed_kernel_binary:
        # Ensure the cubin file is included in the package
        V.graph.wrapper_code.additional_files.append(
            params[get_cpp_wrapper_cubin_path_name()]
        )


patch = gorilla.Patch(DeferredTritonCallWrapper, "generate", generate)
gorilla.apply(patch)
