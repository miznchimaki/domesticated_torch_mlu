import torch
from ...utils import gorilla
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from typing import Callable, Optional
from collections.abc import Sequence
from torch._inductor import ir
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor import ir


def generate_py_arg(self, py_args_var, idx, raw_arg, arg_type):
    def generate_py_arg_inner(lines, raw_arg, arg_type):
        def add_py_newref():
            if sys.version_info < (3, 10):
                # Py_NewRef is only available since Python 3.10
                self.include_extra_header("torch/csrc/utils/pythoncapi_compat.h")

        def handle_scalar(scalar):
            if isinstance(scalar, int):
                return f"PyLong_FromLongLong({scalar})"
            if isinstance(scalar, float):
                return f"PyFloat_FromDouble({self.generate_float_value(scalar)})"
            if isinstance(scalar, bool):
                return f"PyBool_FromLong({1 if scalar else 0})"
            if isinstance(scalar, complex):
                real = self.generate_float_value(scalar.real)
                imag = self.generate_float_value(scalar.imag)
                return f"PyComplex_FromDoubles({real}, {imag})"
            if isinstance(scalar, SymTypes):
                scalar_var = cexpr(scalar.node.expr)
                if isinstance(scalar, torch.SymBool):
                    return f"PyBool_FromLong({scalar_var})"
                if isinstance(scalar, torch.SymFloat):
                    return f"PyFloat_FromDouble({scalar_var})"
                return f"PyLong_FromLongLong({scalar_var})"
            raise NotImplementedError(
                f"scalar {scalar}, {type(scalar)} cannot be handled by handle_scalar"
            )

        if raw_arg is None:
            # Py_None is a singleton, so we have to explicitly incref it here
            lines.append("Py_INCREF(Py_None);\n")
            return "Py_None"
        elif isinstance(arg_type, torch.TensorType):
            # In some cases, scalar arguments may be passed in place of tensors.
            if not hasattr(raw_arg, "codegen_reference"):
                return handle_scalar(raw_arg)

            # Store AtenTensorHandle as void*.  All Python args are constructed in a
            # nested scope, so this handle will self-destruct after the function
            # call.
            base_handle = self.create_tmp_raii_handle_var_if_needed(
                raw_arg.codegen_reference(), lines
            )
            return f"PyCapsule_New(reinterpret_cast<void*>({base_handle}.get()), NULL, NULL)"
        elif isinstance(arg_type, torch.OptionalType):
            return generate_py_arg_inner(lines, raw_arg, arg_type.getElementType())
        elif isinstance(arg_type, torch.IntType):
            # int
            return f"PyLong_FromLongLong({raw_arg})"
        elif isinstance(arg_type, torch.SymIntType):
            # SymInt
            expr = raw_arg.node.expr if isinstance(raw_arg, torch.SymInt) else raw_arg
            return f"PyLong_FromLongLong({cexpr(expr)})"
        elif isinstance(arg_type, torch.FloatType):
            return f"PyFloat_FromDouble({self.generate_float_value(raw_arg)})"
        elif isinstance(arg_type, torch.BoolType):
            return f"PyBool_FromLong({1 if raw_arg else 0})"
        elif isinstance(arg_type, torch.StringType):
            return f'PyUnicode_FromString("{raw_arg}")'
        elif isinstance(arg_type, torch.NumberType):
            # Union[bool, int, float, complex]
            # torch/_prims_common/__init__.py
            return handle_scalar(raw_arg)
        elif isinstance(raw_arg, torch.device):
            # device
            self.include_extra_header("torch/csrc/Device.h")
            device_str, device_index = self.codegen_device(raw_arg).split(", ")
            return f"THPDevice_New(c10::Device(static_cast<c10::DeviceType>({device_str}), {device_index}))"
        elif isinstance(raw_arg, torch.dtype):
            # dtype
            add_py_newref()
            self.include_extra_header("torch/csrc/DynamicTypes.h")
            return f"Py_NewRef(torch::getTHPDtype(static_cast<c10::ScalarType>({self.codegen_dtype(raw_arg)})))"
        elif isinstance(raw_arg, torch.layout):
            # memory layout
            add_py_newref()
            self.include_extra_header("torch/csrc/DynamicTypes.h")
            return f"Py_NewRef(torch::getTHPLayout(static_cast<c10::Layout>({self.codegen_layout(raw_arg)})))"
        elif isinstance(raw_arg, torch.memory_format):
            # memory_format
            add_py_newref()
            self.include_extra_header("torch/csrc/utils/tensor_memoryformats.h")
            return (
                "Py_NewRef(torch::utils::getTHPMemoryFormat(static_cast<c10::MemoryFormat>("
                f"{self.codegen_memory_format(raw_arg)})))"
            )
        elif isinstance(arg_type, torch.DictType):
            return f"PyDict_New()"
        else:
            raise NotImplementedError(
                f"arg type {arg_type} is not yet supported by custom_op_wrapper"
            )

    lines = []

    # Modified by Cambricon.
    if isinstance(arg_type, torch.OptionalType):
        arg_type = arg_type.getElementType()

    if isinstance(arg_type, torch.ListType) and raw_arg is not None:
        assert isinstance(raw_arg, (list, tuple)), str(raw_arg) + " is not a list"
        lines.append(f"PyObject* {py_args_var}_{idx} = PyList_New({len(raw_arg)});\n")
        for i, elem in enumerate(raw_arg):
            lines.append(
                f"PyList_SetItem({py_args_var}_{idx}, {i}, {generate_py_arg_inner(lines, elem, arg_type.getElementType())});\n"
            )
        lines.append(f"PyTuple_SetItem({py_args_var}, {idx}, {py_args_var}_{idx});\n")
    else:
        lines.append(
            f"PyTuple_SetItem({py_args_var}, {idx}, {generate_py_arg_inner(lines, raw_arg, arg_type)});\n"
        )
    return "".join(lines)


patch = gorilla.Patch(CppWrapperCpu, "generate_py_arg", generate_py_arg)
gorilla.apply(patch)


def generate_fallback_kernel_with_runtime_lookup_nopython(
    self,
    get_args: Callable[[], Sequence[str]],
    op_overload: torch._ops.OpOverload,
    output_args: Sequence[Optional[str]],
    raw_outputs: Sequence[ir.Buffer],
) -> None:
    """Generate fallback kernel calls with runtime (non-AOT) dispatch.  This can
    only be called in cpp_wrapper mode, and assumes that the input is a non-None
    OpOverload.
    In the future, we may switch over to directly calling c10::Dispatcher if we need
    to support more datatypes."""
    if raw_outputs:
        declarations_before_scope = [
            f"RAIIAtenTensorHandle {output_arg};"
            for output_arg, raw_output_arg in zip(output_args, raw_outputs)  # type: ignore[arg-type]
            if output_arg is not None
            and not isinstance(raw_output_arg, ir.MutationOutput)
        ]
    else:
        declarations_before_scope = [
            f"RAIIAtenTensorHandle {output_arg};"
            for output_arg in output_args  # type: ignore[arg-type]
            if output_arg is not None
        ]
    dispatch_lines = IndentedBuffer()
    dispatch_lines.writelines(declarations_before_scope)
    dispatch_lines.writeline("{")
    with dispatch_lines.indent():
        tmp_var_number = count()

        def parse_arg(arg_type, codegen_arg: str) -> str:
            # Strip off any temporary references; we're in an indented context, so
            # any saved-off variables will be auto-destroyed.
            new_codegen_arg = codegen_arg.removeprefix("&temporary_reference(")
            if new_codegen_arg != codegen_arg:
                # If we removed temporary_reference, there's a good chance the
                # variable ends with get() (which would retrieve an ATenTensorHandle
                # from a temporary RAII handle).  Strip that off too, since we're
                # going to save this in a temporary RAII handle.
                if codegen_arg.endswith(".get())"):
                    codegen_arg = new_codegen_arg.removesuffix(".get())")
                else:
                    codegen_arg = new_codegen_arg.removesuffix(")")
            if isinstance(arg_type, torch.OptionalType):
                # If we have a pointer to a variable, strip it off and let
                # from<std::optional> handle any internal pointers.
                codegen_arg = codegen_arg.removeprefix("&")
                if codegen_arg == "nullptr":
                    return "from(std::nullopt)"
                var_name = f"tmp_var_{next(tmp_var_number)}"
                dispatch_lines.writeline(
                    f"std::optional {var_name}{{{parse_arg(arg_type.getElementType(), codegen_arg)}}};"
                )
                return f"from({var_name})"
            raii_var = self.create_tmp_raii_handle_var_if_needed(
                codegen_arg, dispatch_lines
            )
            temp_handle = raii_var != codegen_arg
            if isinstance(arg_type, torch.TensorType):
                if not temp_handle:
                    # If the RAII tensor being referenced _isn't_ a temporary,
                    # scoped to this fallback call, then create a new handle
                    # referencing it which from<AtenTensorHandle> can steal.
                    var_name = f"tmp_var_{next(tmp_var_number)}"
                    dispatch_lines.writeline(f"AtenTensorHandle {var_name};")
                    dispatch_lines.writeline(
                        f"aoti_torch_new_tensor_handle({raii_var}, &{var_name});"
                    )
                    return f"from({var_name})"
                # If the RAII tensor _is_ a temporary scoped to this fallback call,
                # simply release and steal the handle.
                return f"from({raii_var}.release())"
            if isinstance(arg_type, torch.DeviceObjType):
                return f"from(torch_mlu::aot_inductor::WrapDevice({codegen_arg}))"
            return f"from({codegen_arg})"

        codegen_args = get_args()
        ivalue_args = (
            parse_arg(a.type, c)
            for a, c in zip(op_overload._schema.arguments, codegen_args)
        )
        array_len = max(len(codegen_args), len(output_args))
        dispatch_lines.writeline(
            f"std::array<StableIValue, {array_len}> dispatch_vars{{{', '.join(ivalue_args)}}};"
        )
        dispatch_lines.writeline("AOTI_TORCH_ERROR_CODE_CHECK(")
        with dispatch_lines.indent():
            dispatch_lines.writeline(
                f'aoti_torch_call_dispatcher("{op_overload._schema.name}", "{op_overload._schema.overload_name}", dispatch_vars.data())'  # noqa: B950
            )
        dispatch_lines.writeline(");")
        if len(output_args) == 1 and (output := output_args[0]) is not None:
            # result is a single tensor
            dispatch_lines.writeline(
                f"{output} = to<AtenTensorHandle>(dispatch_vars[0]);"
            )
        else:
            # result is a tuple of tensors
            for idx, output_arg in enumerate(output_args):
                if output_arg is None:
                    continue
                dispatch_lines.writeline(
                    f"{output_arg} = to<AtenTensorHandle>(dispatch_vars[{idx}]);"
                )
    dispatch_lines.writeline("}")
    self.writelines(dispatch_lines.getvalue().splitlines())


patch = gorilla.Patch(
    CppWrapperCpu,
    "generate_fallback_kernel_with_runtime_lookup_nopython",
    generate_fallback_kernel_with_runtime_lookup_nopython,
)
gorilla.apply(patch)


def generate_c_shim_fallback_kernel(
    self, fallback_kernel: ir.FallbackKernel, args: list[str]
) -> None:
    output_args = []
    output_raii_handles = []
    output_name_base = fallback_kernel.get_name()

    # Modified by Cambricon
    from torchgen.model import FunctionSchema, SchemaKind

    func_schema = FunctionSchema.parse(str(fallback_kernel.op_overload._schema))
    # end Modified by Cambricon

    # Modified by Cambricon
    # 1. if returns of schema is empty, output_args will be empty list.
    # 2. fallback inplace op, output_args will be empty list.
    if (
        len(fallback_kernel.op_overload._schema.returns) == 0
        or func_schema.kind() == SchemaKind.inplace
    ):
        pass
    # Modified by Cambricon: cpp wrapper support return list[Tensor].
    elif (
        len(fallback_kernel.op_overload._schema.returns) == 1
        and isinstance(
            fallback_kernel.op_overload._schema.returns[0].type, torch.ListType
        )
        and isinstance(
            fallback_kernel.op_overload._schema.returns[0].type.getElementType(),
            torch.TensorType,
        )
    ):
        output_num = len(fallback_kernel.outputs)
        out_args = ", ".join(
            f"&{output.get_name()}_handle" for output in fallback_kernel.outputs
        )
        out_args = f"std::vector<AtenTensorHandle*>{{{out_args}}}.data()"
        for idx, output in enumerate(fallback_kernel.outputs):
            name = f"{output.get_name()}"
            output_handle_name = f"{name}_handle"
            self.writeline(f"AtenTensorHandle {output_handle_name};")
            output_raii_handles.append(
                f"RAIIAtenTensorHandle {name}({output_handle_name});"
            )
        output_args.append(out_args)
    else:
        for idx, output in enumerate(fallback_kernel.outputs):
            if isinstance(output, ir.MultiOutput):
                # TODO: handle integer output (e.g., as in attention)
                name = f"{output.get_name()}"
                output_handle_name = f"{name}_handle"
                if output.indices:
                    assert (
                        output.indices[0][1] == idx
                    ), f"expected {output.indices[0][1]=} == {idx=} for {output_name_base=}"
                self.writeline(f"AtenTensorHandle {output_handle_name};")
                output_args.append(f"&{output_handle_name}")
                output_raii_handles.append(
                    f"RAIIAtenTensorHandle {name}({output_handle_name});"
                )
            elif isinstance(output, int):
                output_name = f"{output_name_base}_{idx}"
                self.writeline(f"int64_t {output_name} = {output};")
                output_args.append(f"&{output_name}")
            elif isinstance(output, sympy.Expr):
                output_name = f"{output_name_base}_{idx}"
                self.writeline(f"auto {output_name} = {cexpr(output)};")
                output_args.append(f"&{output_name}")
            elif output is None:
                output_args.append("nullptr")
            else:
                raise NotImplementedError(f"unsupported type of {output=}")
    args = args + output_args
    device = d.type if (d := fallback_kernel.get_device()) else self.device
    self.generate_c_shim_extern_kernel_call(
        fallback_kernel.cpp_kernel_name,  # type: ignore[arg-type]
        args,
        device,
    )
    for raii_handle in output_raii_handles:
        self.writeline(raii_handle)


patch = gorilla.Patch(
    CppWrapperCpu, "generate_c_shim_fallback_kernel", generate_c_shim_fallback_kernel
)
gorilla.apply(patch)


@staticmethod
def get_c_shim_func_name(kernel: str, device: str) -> str:
    if kernel.startswith("aoti_torch_"):
        return kernel

    assert "::" in kernel, "Cpp kernel name: " + kernel + " does not contain '::'"
    kernel_tokens = kernel.split("::")
    kernel_suffix = kernel_tokens[-1]
    namespace = kernel_tokens[0]
    if kernel_suffix == "call":
        kernel_suffix = kernel_tokens[-2]
    if namespace == "torch_mlu_ops" and device == "mlu":
        shim_fn = f"aoti_{namespace}_{kernel_suffix}"
    else:
        shim_fn = f"aoti_torch_{device}_{kernel_suffix}"
    return shim_fn


patch = gorilla.Patch(CppWrapperCpu, "get_c_shim_func_name", get_c_shim_func_name)
gorilla.apply(patch)


def write_prefix(self):
    if V.graph.is_const_graph:
        # We do not write prefix for constant graph, it will be written by main module.
        return

    # Modify by Cambricon
    import torch_mlu

    if len(torch_mlu._inductor.config.aot_inductor.custom_ops_to_c_shims) > 0:
        torch_mlu._inductor.config._warn_custom_op_config()

    custom_ops_to_c_shims = {}
    custom_ops_to_c_shims.update(
        torch_mlu._inductor.config.aot_inductor.custom_ops_to_c_shims
    )
    custom_ops_to_c_shims.update(
        torch._inductor.config.aot_inductor.custom_ops_to_c_shims
    )

    # if config.aot_inductor.custom_ops_to_c_shims:
    if custom_ops_to_c_shims:
        # custom_ops_to_c_shims contains declaration of custom ops with C shim.
        # TODO: this could be auto-generated from a passed-in custom op schema
        # custom_c_shims = list(
        #     chain(*config.aot_inductor.custom_ops_to_c_shims.values())
        # )
        custom_c_shims = list(chain(*custom_ops_to_c_shims.values()))
        # end Modify by Cambricon
        declarations = "\n".join(
            [f"extern {textwrap.dedent(shim)};" for shim in custom_c_shims]
        )
        self.prefix.splice(
            f"""
           extern "C" {{
               {declarations}
           }}
           """
        )
    if V.graph.aot_mode:
        self.prefix.writeline("namespace torch::aot_inductor {")


patch = gorilla.Patch(CppWrapperCpu, "write_prefix", write_prefix)
gorilla.apply(patch)


def write_header(self):
    if V.graph.is_const_graph:
        # We do not write header for constant graph, it will be written by main module.
        return

    if not V.graph.aot_mode:
        # Modify by Cambricon
        import torch_mlu

        custom_op_libs = []
        if torch_mlu._inductor.config.aot_inductor.custom_op_libs:
            torch_mlu._inductor.config._warn_custom_op_config()
            custom_op_libs += torch_mlu._inductor.config.aot_inductor.custom_op_libs
        if torch._inductor.config.aot_inductor.custom_op_libs:
            custom_op_libs += torch._inductor.config.aot_inductor.custom_op_libs
        if custom_op_libs is not None:
            for lib_name in custom_op_libs:
                self.header.splice(
                    f"""
                    try:
                        import {lib_name}
                    except ImportError:
                        pass
                    """
                )
        # end Modify by Cambricon
        self.header.splice(
            """
            import torch
            from torch._inductor.codecache import CppWrapperCodeCache

            cpp_wrapper_src = (
            r'''
            """
        )

    self.add_device_include(self.device)

    if V.graph.aot_mode:
        # Modify by Cambricon
        if "mlu" in self.device:
            import torch_mlu

            interface_path = os.path.join(
                torch_mlu.__path__[0], "_inductor/codegen/aoti_runtime", "interface.cpp"
            )
        else:
            interface_path = os.path.join(
                os.path.dirname(__file__), "aoti_runtime", "interface.cpp"
            )
        # end Modify by Cambricon
        if config.aot_inductor.dynamic_linkage:
            # Modify by Cambricon
            # with open(
            #     os.path.join(
            #         os.path.dirname(__file__), "aoti_runtime", "interface.cpp"
            #     )
            with open(interface_path) as f:
                # end Modify by Cambricon
                self.header.splice(f.read())
        else:
            # we produce a separate model header for each model in static linkage
            self.header.splice(f"""#include \"{self.model_class_name_suffix}.h\"""")
        self.header.splice("\n")

    if config.cpp.enable_kernel_profile:
        self.header.splice(
            "#include <torch/csrc/inductor/aoti_runtime/kernel_context_tls.h>"
        )
        self.header.splice(
            """
            namespace torch::aot_inductor {
            thread_local KernelContext* tls_kernel_context = nullptr;
            }
            """
        )


patch = gorilla.Patch(CppWrapperCpu, "write_header", write_header)
gorilla.apply(patch)


def val_to_arg_str_for_prim_type(self, val, type_) -> str:
    # TODO: not using type_ as the first step of refactoring. Will update this later.
    if isinstance(val, bool):
        return "1" if val else "0"
    elif isinstance(val, int):
        # uint64_t is long on Linux, but long long on MacOS and Windows
        return f"{val}LL" if sys.platform in ["darwin", "win32"] else f"{val}L"
    elif isinstance(val, complex):
        return f"c10::complex<double>{{ {self.generate_float_value(val.real)}, {self.generate_float_value(val.imag)} }}"
    elif isinstance(val, str):
        return f'"{val}"'
    elif isinstance(val, (ir.Buffer, ir.ReinterpretView, ir.StorageBox, ir.TensorBox)):
        return val.codegen_reference()
    elif isinstance(val, torch.device):
        return self.codegen_device(val)
    elif isinstance(val, torch.dtype):
        return self.codegen_dtype(val)
    elif isinstance(val, torch.layout):
        return self.codegen_layout(val)
    elif isinstance(val, torch.memory_format):
        return self.codegen_memory_format(val)
    elif isinstance(val, float):
        return self.generate_float_value(val)
    elif isinstance(val, (list, tuple)):
        # FIXME: This happens because type_ is not always properly set to torch.ListType
        return f"{{{', '.join(self.val_to_arg_str(x, None) for x in val)}}}"
    elif isinstance(val, SymTypes):
        return cexpr(val.node.expr)
    elif isinstance(val, sympy.Expr):
        return cexpr(val)
    # Modify by CAMBRICON
    elif val is None:
        if type_ is int:
            return "0L"
        else:
            return repr(val)
    # end Modify by CAMBRICON
    else:
        return repr(val)


patch = gorilla.Patch(
    CppWrapperCpu, "val_to_arg_str_for_prim_type", val_to_arg_str_for_prim_type
)
gorilla.apply(patch)
