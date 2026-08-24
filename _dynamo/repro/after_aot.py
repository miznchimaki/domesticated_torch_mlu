import argparse
import logging
import textwrap
from typing import Any, Callable, Union, Optional
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch._dynamo.debug_utils import (
    _cuda_system_info_comment,
    extra_imports,
    generate_config_string,
    generate_env_vars_string,
    InputWriter,
    NNModuleToString,
)
from torch._dynamo.repro import after_aot
from torch._dynamo.repro.after_aot import maybe_fbcode_instructions
from torch.fx.experimental.symbolic_shapes import fx_placeholder_targets
from torch._inductor.cpp_builder import normalize_path_separator
from ...utils import gorilla


# NOTE by CAMBRICON: Because extra_import takes effect too late, it will cause config to report an error, so a patch is needed instead of setting extra_import.
@gorilla.patch(after_aot)
def generate_compiler_repro_string(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    *,
    stable_output: bool = False,
    save_dir: Optional[str] = None,
    stable_hash: bool = False,
    has_distributed_ops: bool = False,
) -> str:
    if save_dir is not None:
        save_dir = normalize_path_separator(save_dir)
    # Add distributed imports if needed
    distributed_imports = ""
    if has_distributed_ops:
        distributed_imports = textwrap.dedent(
            """
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
        """
        ).strip()

    triton_imports = ""

    if len(kernel_side_table.id_to_kernel) > 0:
        triton_imports = textwrap.dedent(
            """
import triton
import triton.language as tl
        """
        ).strip()

    model_str = textwrap.dedent(
        f"""
{generate_env_vars_string(stable_output=stable_output)}
import torch
# Add by CAMBRICON
import torch_mlu
# end Add by CAMBRICON
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims
{distributed_imports}
{triton_imports}

{generate_config_string(stable_output=stable_output)}

isolate_fails_code_str = None

{extra_imports}

{maybe_fbcode_instructions()}
     """
    )
    model_str += textwrap.dedent(
        """
if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
"""
    )
    if not stable_output:
        model_str += f"# torch version: {torch.version.__version__}\n"
        if hasattr(torch.version, "cuda"):
            model_str += f"# torch cuda version: {torch.version.cuda}\n"
        if hasattr(torch.version, "git_version"):
            model_str += f"# torch git version: {torch.version.git_version}\n\n\n"
        model_str += _cuda_system_info_comment()

    kernel_side_table_prefix = (
        "torch._higher_order_ops.triton_kernel_wrap.kernel_side_table"
    )

    def get_fn_name(kernel: Any) -> str:
        fn_name = (
            # pyrefly: ignore [missing-attribute]
            kernel._fn_name
            if isinstance(kernel, JITFunction)
            else kernel.fn._fn_name
        )
        return fn_name.split(".")[-1]

    def write_kernel_dependencies(
        kernel: Any,
        written_constexpr_vars: set[str],
        written_nested_kernels: set[str],
    ) -> str:
        """Write out global tl.constexpr vars and nested kernel dependencies."""
        result = ""
        jit_fn = kernel if isinstance(kernel, JITFunction) else kernel.fn
        if not getattr(jit_fn, "fn", None) or not getattr(jit_fn, "src", None):
            return result

        fn_globals = getattr(jit_fn.fn, "__globals__", {})
        src = jit_fn.src  # pyrefly: ignore [missing-attribute]
        full_src = src if src.strip().startswith("def ") else "def " + src

        referenced_names: set[str] = set()
        called_names: set[str] = set()
        for node in ast.walk(ast.parse(full_src)):
            if isinstance(node, ast.Name):
                referenced_names.add(node.id)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called_names.add(node.func.id)

        # Write out global tl.constexpr variables
        for name in referenced_names:
            if name in written_constexpr_vars:
                continue
            val = fn_globals.get(name)

            if isinstance(val, TritonConstexpr) and getattr(val, "value", None):
                result += f"{name} = tl.constexpr({val.value})\n"
            elif isinstance(val, (int, float, str, bool)):
                result += f"{name} = {val!r}\n"
            else:
                continue
            written_constexpr_vars.add(name)

        # Write out nested kernel dependencies
        for name in called_names:
            val = fn_globals.get(name)
            if not isinstance(val, JITFunction) or val is jit_fn:
                continue
            nested_fn_name = get_fn_name(val)
            if nested_fn_name in written_nested_kernels:
                continue
            # Mark as written before recursing to prevent cycles
            written_nested_kernels.add(nested_fn_name)
            result += write_kernel_dependencies(
                val, written_constexpr_vars, written_nested_kernels
            )
            result += generate_custom_triton_kernel(val)

        return result

    written_nested_kernels: set[str] = set()
    written_constexpr_vars: set[str] = set()

    model_str += f"{kernel_side_table_prefix}.reset_table()\n"

    for idx in kernel_side_table.id_to_kernel:
        kernel = kernel_side_table.get_kernel(idx)

        try:
            model_str += write_kernel_dependencies(
                kernel, written_constexpr_vars, written_nested_kernels
            )
            fn_name = get_fn_name(kernel)

            unique_name = f"{fn_name}_{idx}"

            kernel_code = generate_custom_triton_kernel(kernel)
            kernel_code = kernel_code.replace(
                f"def {fn_name}(", f"def {unique_name}(", 1
            )
            model_str += kernel_code
            model_str += f"{kernel_side_table_prefix}.add_kernel({unique_name})\n"
        except AttributeError as e:
            model_str += "ERROR: Repro will not work as intended, "
            model_str += f"User defined triton kernel exception: {e}\n"

    # pyrefly: ignore [unbound-name]
    if len(kernel_side_table.constant_args) > 0:
        # pyrefly: ignore [unbound-name]
        model_str += f"{kernel_side_table_prefix}.constant_args={kernel_side_table.constant_args}\n"

    model_str += NNModuleToString.convert(gm)

    writer = InputWriter(save_dir, stable_hash=stable_hash)
    # pyrefly: ignore [implicit-any]
    used_syms = {}

    # Extract from graph placeholders and their corresponding arguments
    placeholder_targets = fx_placeholder_targets(gm)
    for placeholder, arg in zip(placeholder_targets, args):
        # pyrefly: ignore [unbound-name]
        if isinstance(arg, (int, torch.SymInt)):
            writer.symint(placeholder, arg)
        # pyrefly: ignore [unbound-name]
        elif isinstance(arg, torch.Tensor):
            # TODO: improve these names with FQN
            writer.tensor(placeholder, arg)
        elif arg is None:
            writer.const(placeholder)
        else:
            writer.unsupported(placeholder, arg)

        # Extract symbolic variables from the same arguments

        if (
            # pyrefly: ignore [unbound-name]
            isinstance(arg, torch.SymInt)
            # By checking sympy.Symbol, we are excluding any symbolic expressions.
            # TODO: we may need to solve expressions to extract symbol definitions.
            and isinstance(arg.node.expr, sympy.Symbol)
            and arg.node.hint is not None
        ):
            used_syms[str(arg.node)] = arg.node.hint
        # pyrefly: ignore [unbound-name]
        elif isinstance(arg, torch.Tensor):
            # Extract symbolic variables from tensor shapes and strides
            for dim in arg.shape:
                if (
                    # pyrefly: ignore [unbound-name]
                    isinstance(dim, torch.SymInt)
                    and isinstance(dim.node.expr, sympy.Symbol)
                    and dim.node.hint is not None
                ):
                    used_syms[str(dim.node)] = dim.node.hint
            for stride in arg.stride():
                if (
                    # pyrefly: ignore [unbound-name]
                    isinstance(stride, torch.SymInt)
                    and isinstance(stride.node.expr, sympy.Symbol)
                    and stride.node.hint is not None
                ):
                    used_syms[str(stride.node)] = stride.node.hint
            # Extract symbols from storage nbytes (can be a symbolic expression)
            storage = arg.untyped_storage()
            nbytes = storage.nbytes()
            # pyrefly: ignore [unbound-name]
            if isinstance(nbytes, torch.SymInt):
                expr = nbytes.node.expr
                shape_env = nbytes.node.shape_env
                for sym in expr.free_symbols:
                    sym_name = str(sym)
                    if sym_name not in used_syms and shape_env is not None:
                        hint = shape_env.backed_var_to_val.get(sym)
                        if hint is not None:
                            used_syms[sym_name] = int(hint)
    # Add symbolic variable definitions to the top of the generated code
    if used_syms:
        hint_lines = "\n".join(
            f"{name} = {hint}" for name, hint in sorted(used_syms.items())
        )
        model_str = f"{hint_lines}\n\n{model_str}"

    load_args_lines = writer.lines()
    load_args_code = "\n".join(load_args_lines)
    model_str += load_args_code + "\n"

    model_str += "mod = Repro()\n"
    return model_str


@gorilla.patch(after_aot)
def repro_run(options: Any, mod: nn.Module, load_args: Any) -> None:
    from torch._inductor.compile_fx import compile_fx_inner

    mod, args = repro_common(options, mod, load_args)

    from torch.cuda import synchronize

    # Modify by CAMBRICON
    # compiled = compile_fx_inner(mod, args)
    compiled = compile_fx_inner(copy.deepcopy(mod), args)
    # end Modify by CAMBRICON
    assert not isinstance(compiled, str)

    if options.accuracy != "":
        # We don't really respect --accuracy vs --strict-accuracy here, it
        # seems counterintuitive
        if not same_two_models(
            mod,
            compiled,  # type: ignore[arg-type]
            args,
            only_fwd=True,
            ignore_non_fp=config.repro_ignore_non_fp,
        ):
            raise AccuracyError("Bad accuracy detected")
    else:
        need_sync = False

        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.is_cuda:
                need_sync = True
                break

        compiled(list(args))

        if need_sync:
            synchronize()  # ensure segfaults are surfaced
