from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import os
import setuptools
import sys
import sysconfig
import re
import shlex
import warnings
import traceback

import torch
from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension
from torch.utils.file_baton import FileBaton
from typing import List, Optional, Union
import subprocess

COMMON_BANG_CFLAGS = [
    "--bang-mlu-arch=mtp_592",
    "--bang-mlu-arch=mtp_613",
    "--bang-mlu-arch=mtp_617",
]

COMMON_FLAGS = [
    "-O3",
    "-Wall",
    "-fPIC",
    "-std=c++17",
    "-pthread",
]

min_supported_cpython = "0x030A0000"  # Python 3.10 hexcode


def get_pytorch_dir():
    try:
        import torch

        return os.path.dirname(os.path.realpath(torch.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)


pytorch_dir = get_pytorch_dir()


def _prepare_ldflags(extra_ldflags, with_bang, verbose, is_standalone):
    NEUWARE_HOME = _find_neuware_home()
    build_libtorch = os.environ.get("BUILD_LIBTORCH")
    extra_ldflags.append("-L" + pytorch_dir + "/lib")
    extra_ldflags.append("-lc10")
    extra_ldflags.append("-ltorch_cpu")
    if with_bang:
        extra_ldflags.append("-ltorch_mlu")
        if not build_libtorch or build_libtorch.lower() in ("0", "false", "n", "no"):
            extra_ldflags.append("-ltorch_mlu_python")
        extra_ldflags.append("-ltorch_mlu_bangc")
        cur_path = os.path.abspath(__file__)
        torch_mlu_path = os.path.dirname(os.path.dirname(cur_path))
        extra_ldflags.append("-L" + torch_mlu_path + "/csrc/lib")
    extra_ldflags.append("-ltorch")
    if not is_standalone:
        extra_ldflags.append("-ltorch_python")

    if is_standalone and "TBB" in torch.__config__.parallel_info():
        extra_ldflags.append("-ltbb")

    if is_standalone:
        extra_ldflags.append(f"-Wl,-rpath")

    if with_bang:
        if verbose:
            print("Detected MLU files, patching ldflags", file=sys.stderr)
        extra_lib_dir = "lib64"
        if not os.path.exists(_join_neuware_home(extra_lib_dir)) and os.path.exists(
            _join_neuware_home("lib")
        ):
            extra_lib_dir = "lib"
        extra_ldflags.append(f"-L{_join_neuware_home(extra_lib_dir)}")
        if NEUWARE_HOME is not None:
            extra_ldflags.append(f'-L{os.path.join(NEUWARE_HOME, "lib64")}')
    return extra_ldflags


def _find_neuware_home() -> Optional[str]:
    r"""Finds the NEUWARE_HOME install path."""
    # Guess #1
    neuware_home = os.environ.get("NEUWARE_HOME")
    if neuware_home is None:
        # Guess #2
        try:
            with open(os.devnull, "w") as devnull:
                cncc = (
                    subprocess.check_output(["which", "cncc"], stderr=devnull)
                    .decode()
                    .rstrip("\r\n")
                )
                neuware_home = os.path.dirname(os.path.dirname(cncc))
        except Exception:
            # Guess #3
            neuware_home = "/usr/local/neuware_home"
            if not os.path.exists(neuware_home):
                neuware_home = None
    if neuware_home and not torch.mlu.is_available():
        print(f"No NEUWARE runtime is found, using neuware_home='{neuware_home}'")
    return neuware_home


def _is_mlu_file(path: str) -> bool:
    valid_ext = [".mlu"]
    return os.path.splitext(path)[1] in valid_ext


def _join_neuware_home(*paths) -> str:
    r"""
    Joins paths with NEUWARE_HOME, or raises an error if NEUWARE_HOME is not set.

    This is basically a lazy way of raising an error for missing $NEUWARE_HOME
    only once we need to get any MLU-specific path.
    """
    NEUWARE_HOME = _find_neuware_home()
    if NEUWARE_HOME is None:
        raise EnvironmentError(
            "NEUWARE_HOME environment variable is not set. "
            "Please set it to your Cambricon Neuware install root."
        )
    return os.path.join(NEUWARE_HOME, *paths)


def include_paths() -> List[str]:
    """
    Get the include paths required to build a C++ or BANG extension.

    Returns:
        A list of include path strings.
    """
    # Import from Installed Pytorch
    paths = torch.utils.cpp_extension.include_paths(False)
    # Import from Cambricon TORCH_MLU
    cur_path = os.path.abspath(__file__)
    torch_mlu_path = os.path.dirname(os.path.dirname(cur_path))
    paths.append(os.path.join(torch_mlu_path, "csrc", "include"))
    paths.append(
        os.path.join(torch_mlu_path, "csrc", "include", "api", "include", "torch_mlu")
    )
    # Import from Cambricon Neuware
    neuware_home_include = _join_neuware_home("include")
    paths.append(neuware_home_include)
    return paths


def library_paths() -> List[str]:
    r"""
    Get the library paths required to build a C++ or BANG extension.

    Returns:
        A list of library path strings.
    """
    paths = []

    # Import from Installed Pytorch
    paths = torch.utils.cpp_extension.library_paths(False)
    # Import from Cambricon TORCH_MLU
    cur_path = os.path.abspath(__file__)
    torch_mlu_path = os.path.dirname(os.path.dirname(cur_path))
    lib_path = os.path.join(torch_mlu_path, "csrc/lib")
    paths.append(lib_path)
    # Import from Cambricon Neuware
    lib_dir = "lib64"
    paths.append(_join_neuware_home(lib_dir))
    return paths


def _write_ninja_file_and_compile_objects(
    sources: List[str],
    objects,
    cflags,
    post_cflags,
    bang_cflags,
    bang_post_cflags,
    build_directory: str,
    verbose: bool,
    with_bang: Optional[bool],
) -> None:
    torch.utils.cpp_extension.verify_ninja_availability()
    compiler = os.environ.get("CXX", "c++")
    torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version(compiler)
    if with_bang is None:
        with_bang = any(map(_is_mlu_file, sources))
    build_file_path = os.path.join(build_directory, "build.ninja")
    if verbose:
        print(f"Emitting ninja build file {build_file_path}...", file=sys.stderr)
    _write_ninja_file(
        path=build_file_path,
        cflags=cflags,
        post_cflags=post_cflags,
        bang_cflags=bang_cflags,
        bang_post_cflags=bang_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=None,
        library_target=None,
        with_bang=with_bang,
    )
    if verbose:
        print("Compiling objects...", file=sys.stderr)
    torch.utils.cpp_extension._run_ninja_build(
        build_directory,
        verbose,
        # It would be better if we could tell users the name of the extension
        # that failed to build but there isn't a good way to get it here.
        error_prefix="Error compiling objects for extension",
    )


def load(
    name,
    sources: Union[str, List[str]],
    extra_cflags=None,
    extra_bang_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
    verbose=False,
    with_bang: Optional[bool] = None,
    is_python_module=True,
    is_standalone=False,
    keep_intermediates=True,
):
    """
    Load a PyTorch C++ extension just-in-time (JIT).

    To load an extension, a Ninja build file is emitted, which is used to
    compile the given sources into a dynamic library. This library is
    subsequently loaded into the current Python process as a module and
    returned from this function, ready for use.

    By default, the directory to which the build file is emitted and the
    resulting library compiled to is ``<tmp>/torch_extensions/<name>``, where
    ``<tmp>`` is the temporary folder on the current platform and ``<name>``
    the name of the extension. This location can be overridden in two ways.
    First, if the ``TORCH_EXTENSIONS_DIR`` environment variable is set, it
    replaces ``<tmp>/torch_extensions`` and all extensions will be compiled
    into subfolders of this directory. Second, if the ``build_directory``
    argument to this function is supplied, it overrides the entire path, i.e.
    the library will be compiled into that folder directly.

    To compile the sources, the default system compiler (``c++``) is used,
    which can be overridden by setting the ``CXX`` environment variable. To pass
    additional arguments to the compilation process, ``extra_cflags`` or
    ``extra_ldflags`` can be provided. For example, to compile your extension
    with optimizations, pass ``extra_cflags=['-O3']``. You can also use
    ``extra_cflags`` to pass further include directories.

    MLU support with mixed compilation is provided. Simply pass Bang source
    files (``.mlu``) along with other sources. Such files will be
    detected and compiled with cncc rather than the C++ compiler. This includes
    passing the NEUWARE lib64 directory as a library directory, and linking
    ``cnrt``. You can pass additional flags to cncc via
    ``extra_bang_cflags``, just like with ``extra_cflags`` for C++. Various
    heuristics for finding the Cambeicon NEUWARE directory are used, which usually
    work fine. If not, setting the ``NEUWARE_HOME`` environment variable is the
    safest option.

    Args:
        name: The name of the extension to build. This MUST be the same as the
            name of the pybind11 module!
        sources: A list of relative or absolute paths to C++ source files.
        extra_cflags: optional list of compiler flags to forward to the build.
        extra_bang_cflags: optional list of compiler flags to forward to cncc
            when building Bang sources.
        extra_ldflags: optional list of linker flags to forward to the build.
        extra_include_paths: optional list of include directories to forward
            to the build.
        build_directory: optional path to use as build workspace.
        verbose: If ``True``, turns on verbose logging of load steps.
        with_bang: Determines whether Bang headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on the existence of ``.mlu``
            in ``sources``. Set it to `True`` to force Bang headers
            and libraries to be included.
        is_python_module: If ``True`` (default), imports the produced shared
            library as a Python module. If ``False``, behavior depends on
            ``is_standalone``.
        is_standalone: If ``False`` (default) loads the constructed extension
            into the process as a plain dynamic library. If ``True``, build a
            standalone executable.

    Returns:
        If ``is_python_module`` is ``True``:
            Returns the loaded PyTorch extension as a Python module.

        If ``is_python_module`` is ``False`` and ``is_standalone`` is ``False``:
            Returns nothing. (The shared library is loaded into the process as
            a side effect.)

        If ``is_standalone`` is ``True``.
            Return the path to the executable.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torch_mlu.utils.cpp_extension import load
        >>> module = load(
        ...     name='extension',
        ...     sources=['extension.cpp', 'extension_kernel.mlu'],
        ...     extra_cflags=['-O2'],
        ...     verbose=True)
    """
    return _jit_compile(
        name,
        [sources] if isinstance(sources, str) else sources,
        extra_cflags,
        extra_bang_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory
        or torch.utils.cpp_extension._get_build_directory(name, verbose),
        verbose,
        with_bang,
        is_python_module,
        is_standalone,
        keep_intermediates=keep_intermediates,
    )


def load_inline(
    name,
    cpp_sources,
    bang_sources=None,
    functions=None,
    extra_cflags=None,
    extra_bang_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
    verbose=False,
    with_bang=None,
    is_python_module=True,
    with_pytorch_error_handling=True,
    keep_intermediates=True,
    use_pch=False,
):
    r'''
    Load a PyTorch C++ extension just-in-time (JIT) from string sources.

    This function behaves exactly like :func:`load`, but takes its sources as
    strings rather than filenames. These strings are stored to files in the
    build directory, after which the behavior of :func:`load_inline` is
    identical to :func:`load`.

    See `the
    tests <https://github.com/pytorch/pytorch/blob/master/test/test_cpp_extensions_jit.py>`_
    for good examples of using this function.

    Sources may omit two required parts of a typical non-inline C++ extension:
    the necessary header includes, as well as the (pybind11) binding code. More
    precisely, strings passed to ``cpp_sources`` are first concatenated into a
    single ``.cpp`` file. This file is then prepended with ``#include
    <torch/extension.h>``.

    Furthermore, if the ``functions`` argument is supplied, bindings will be
    automatically generated for each function specified. ``functions`` can
    either be a list of function names, or a dictionary mapping from function
    names to docstrings. If a list is given, the name of each function is used
    as its docstring.

    The sources in ``bang_sources`` are concatenated into a separate ``.mlu``
    file and  prepended with ``type_traits`` and ``mlu.h``
    includes. The ``.cpp`` and ``.mlu`` files are compiled
    separately, but ultimately linked into a single library. Note that no
    bindings are generated for functions in ``bang_sources`` per  se. To bind
    to a Bang kernel, you must create a C++ function that calls it, and either
    declare or define this C++ function in one of the ``cpp_sources`` (and
    include its name in ``functions``).

    See :func:`load` for a description of arguments omitted below.

    Args:
        cpp_sources: A string, or list of strings, containing C++ source code.
        bang_sources: A string, or list of strings, containing Bang source code.
        functions: A list of function names for which to generate function
            bindings. If a dictionary is given, it should map function names to
            docstrings (which are otherwise just the function names).
        with_bang: Determines whether Bang headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on whether ``bang_sources`` is
            provided. Set it to ``True`` to force Bang headers
            and libraries to be included.
        with_pytorch_error_handling: Determines whether pytorch error and
            warning macros are handled by pytorch instead of pybind. To do
            this, each function ``foo`` is called via an intermediary ``_safe_foo``
            function. This redirection might cause issues in obscure cases
            of cpp. This flag should be set to ``False`` when this redirect
            causes issues.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from torch.utils.cpp_extension import load_inline
        >>> source = """
        at::Tensor sin_add(at::Tensor x, at::Tensor y) {
          return x.sin() + y.sin();
        }
        """
        >>> module = load_inline(name='inline_extension',
        ...                      cpp_sources=[source],
        ...                      functions=['sin_add'])

    .. note::
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    '''
    build_directory = build_directory or torch.utils.cpp_extension._get_build_directory(
        name, verbose
    )

    if isinstance(cpp_sources, str):
        cpp_sources = [cpp_sources]
    bang_sources = bang_sources or []
    if isinstance(bang_sources, str):
        bang_sources = [bang_sources]

    cpp_sources.insert(0, "#include <torch/extension.h>")

    if use_pch is True:
        # Using PreCompile Header('torch/extension.h') to reduce compile time.
        torch.utils.cpp_extension._check_and_build_extension_h_precompiler_headers(
            extra_cflags, extra_include_paths
        )
    else:
        torch.utils.cpp_extension.remove_extension_h_precompiler_headers()

    # If `functions` is supplied, we create the pybind11 bindings for the user.
    # Here, `functions` is (or becomes, after some processing) a map from
    # function names to function docstrings.
    if functions is not None:
        module_def = []
        module_def.append("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
        if isinstance(functions, str):
            functions = [functions]
        if isinstance(functions, list):
            # Make the function docstring the same as the function name.
            functions = {f: f for f in functions}
        elif not isinstance(functions, dict):
            raise ValueError(
                f"Expected 'functions' to be a list or dict, but was {type(functions)}"
            )
        for function_name, docstring in functions.items():
            if with_pytorch_error_handling:
                module_def.append(
                    f'm.def("{function_name}", torch::wrap_pybind_function({function_name}), "{docstring}");'
                )
            else:
                module_def.append(
                    f'm.def("{function_name}", {function_name}, "{docstring}");'
                )
        module_def.append("}")
        cpp_sources += module_def

    cpp_source_path = os.path.join(build_directory, "main.cpp")
    torch.utils.cpp_extension._maybe_write(cpp_source_path, "\n".join(cpp_sources))

    sources = [cpp_source_path]

    if bang_sources:
        bang_sources.insert(0, "#include <ATen/ATen.h>")
        bang_sources.insert(1, "#include <mlu.h>")
        bang_sources.insert(2, '#include "mlu_op.h"')
        bang_sources.insert(3, '#include "aten/utils/exceptions.h"')
        bang_sources.insert(4, '#include "aten/utils/cnnl_util.h"')
        bang_sources.insert(5, '#include "aten/utils/tensor_util.h"')
        bang_sources.insert(6, '#include "framework/core/MLUStream.h"')

        bang_source_path = os.path.join(build_directory, "bang.mlu")
        torch.utils.cpp_extension._maybe_write(
            bang_source_path, "\n".join(bang_sources)
        )

        sources.append(bang_source_path)

    return _jit_compile(
        name,
        sources,
        extra_cflags,
        extra_bang_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory,
        verbose,
        with_bang,
        is_python_module,
        is_standalone=False,
        keep_intermediates=keep_intermediates,
    )


def _jit_compile(
    name,
    sources,
    extra_cflags,
    extra_bang_cflags,
    extra_ldflags,
    extra_include_paths,
    build_directory: str,
    verbose: bool,
    with_bang: Optional[bool],
    is_python_module,
    is_standalone,
    keep_intermediates=True,
) -> None:
    if is_python_module and is_standalone:
        raise ValueError(
            "`is_python_module` and `is_standalone` are mutually exclusive."
        )

    if with_bang is None:
        with_bang = any(map(_is_mlu_file, sources))

    baton = FileBaton(os.path.join(build_directory, "lock"))
    if baton.try_acquire():
        try:
            _write_ninja_file_and_build_library(
                name=name,
                sources=sources,
                extra_cflags=extra_cflags or [],
                extra_bang_cflags=extra_bang_cflags or [],
                extra_ldflags=extra_ldflags or [],
                extra_include_paths=extra_include_paths or [],
                build_directory=build_directory,
                verbose=verbose,
                with_bang=with_bang,
                is_standalone=is_standalone,
            )
        finally:
            baton.release()
    else:
        baton.wait()

    if verbose:
        print(f"Loading extension module {name}...", file=sys.stderr)

    if is_standalone:
        return torch.utils.cpp_extension._get_exec_path(name, build_directory)
    return torch.utils.cpp_extension._import_module_from_library(
        name, build_directory, is_python_module
    )


def _write_ninja_file_and_build_library(
    name,
    sources: List[str],
    extra_cflags,
    extra_bang_cflags,
    extra_ldflags,
    extra_include_paths,
    build_directory: str,
    verbose: bool,
    with_bang: Optional[bool],
    is_standalone: bool = False,
) -> None:
    torch.utils.cpp_extension.verify_ninja_availability()

    compiler = torch.utils.cpp_extension.get_cxx_compiler()

    torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version(compiler)
    if with_bang is None:
        with_bang = any(map(_is_mlu_file, sources))

    extra_ldflags = _prepare_ldflags(
        extra_ldflags or [], with_bang, verbose, is_standalone
    )
    build_file_path = os.path.join(build_directory, "build.ninja")
    if verbose:
        print(f"Emitting ninja build file {build_file_path}...", file=sys.stderr)
    # NOTE: Emitting a new ninja build file does not cause re-compilation if
    # the sources did not change, so it's ok to re-emit (and it's fast).
    _write_ninja_file_to_build_library(
        path=build_file_path,
        name=name,
        sources=sources,
        extra_cflags=extra_cflags or [],
        extra_bang_cflags=extra_bang_cflags or [],
        extra_ldflags=extra_ldflags or [],
        extra_include_paths=extra_include_paths or [],
        with_bang=with_bang,
        is_standalone=is_standalone,
    )

    if verbose:
        print(f"Building extension module {name}...", file=sys.stderr)
    torch.utils.cpp_extension._run_ninja_build(
        build_directory, verbose, error_prefix=f"Error building extension '{name}'"
    )


def _write_ninja_file_to_build_library(
    path,
    name,
    sources,
    extra_cflags,
    extra_bang_cflags,
    extra_ldflags,
    extra_include_paths,
    with_bang,
    is_standalone,
) -> None:
    extra_cflags = [flag.strip() for flag in extra_cflags]
    extra_bang_cflags = [flag.strip() for flag in extra_bang_cflags]
    extra_ldflags = [flag.strip() for flag in extra_ldflags]
    extra_include_paths = [flag.strip() for flag in extra_include_paths]

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    user_includes = [os.path.abspath(file) for file in extra_include_paths]

    # include_paths() gives us the location of torch/extension.h
    system_includes = include_paths()
    # sysconfig.get_path('include') gives us the location of Python.h
    # Explicitly specify 'posix_prefix' scheme on non-Windows platforms to workaround error on some MacOS
    # installations where default `get_path` points to non-existing `/Library/Python/M.m/include` folder
    python_include_path = sysconfig.get_path("include", "posix_prefix")
    if python_include_path is not None:
        system_includes.append(python_include_path)

    common_cflags = []
    if not is_standalone:
        common_cflags.append(f"-DTORCH_EXTENSION_NAME={name}")
        common_cflags.append("-DTORCH_API_INCLUDE_EXTENSION_H")

    common_cflags += [
        f"{x}" for x in torch.utils.cpp_extension._get_pybind11_abi_build_flags()
    ]

    common_cflags += [f"-I{shlex.quote(include)}" for include in user_includes]
    common_cflags += [f"-isystem {shlex.quote(include)}" for include in system_includes]

    cflags = common_cflags + ["-fPIC", "-std=c++17"] + extra_cflags

    if with_bang:
        bang_flags = common_cflags + _get_bang_arch_flags(extra_bang_cflags)
        bang_flags += ["-fPIC"]
        bang_flags += extra_bang_cflags
        neuware_home = _find_neuware_home()
        bang_flags.append("--neuware-path=" + neuware_home)
        if not any(flag.startswith("-std=") for flag in bang_flags):
            bang_flags.append("-std=c++17")
    else:
        bang_flags = None

    def object_file_path(source_file: str) -> str:
        # '/path/to/file.cpp' -> 'file'
        file_name = os.path.splitext(os.path.basename(source_file))[0]
        if _is_mlu_file(source_file) and with_bang:
            # Use a different object filename in case a C++ and Bang file have
            # the same filename but different extension (.cpp vs. .mlu).
            target = f"{file_name}.mlu.o"
        else:
            target = f"{file_name}.o"
        return target

    objects = [object_file_path(src) for src in sources]
    ldflags = ([] if is_standalone else ["-shared"]) + extra_ldflags

    ext = "" if is_standalone else ".so"
    library_target = f"{name}{ext}"

    _write_ninja_file(
        path=path,
        cflags=cflags,
        post_cflags=None,
        bang_cflags=bang_flags,
        bang_post_cflags=None,
        sources=sources,
        objects=objects,
        ldflags=ldflags,
        library_target=library_target,
        with_bang=with_bang,
    )


def _write_ninja_file(
    path,
    cflags,
    post_cflags,
    bang_cflags,
    bang_post_cflags,
    sources,
    objects,
    ldflags,
    library_target,
    with_bang,
) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `bang_cflags`: list of flags to pass to $cncc. Can be None.
    `bang_postflags`: list of flags to append to the $cncc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_bang`: If we should be compiling with BANG.
    """

    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    bang_cflags = sanitize_flags(bang_cflags)
    bang_post_cflags = sanitize_flags(bang_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    compiler = os.environ.get("CXX", "c++")

    # Version 1.3 is required for the `deps` directive.
    config = ["ninja_required_version = 1.3"]
    config.append(f"cxx = {compiler}")
    if with_bang:
        cncc = _join_neuware_home("bin", "cncc")
        config.append(f"cncc = {cncc}")

    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')

    if with_bang:
        flags.append(f'bang_cflags = {" ".join(bang_cflags)}')
        flags.append(f'bang_post_cflags = {" ".join(bang_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ["rule compile"]
    compile_rule.append(
        "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags"
    )
    compile_rule.append("  depfile = $out.d")
    compile_rule.append("  deps = gcc")

    if with_bang:
        bang_compile_rule = ["rule bang_compile"]
        bang_compile_rule.append(
            f"  command = $cncc -c $in -o $out $bang_cflags $bang_post_cflags"
        )

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_bang_source = _is_mlu_file(source_file) and with_bang
        rule = "bang_compile" if is_bang_source else "compile"
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f"build {object_file}: {rule} {source_file}")

    if library_target is not None:
        link_rule = ["rule link"]
        link_rule.append("  command = $cxx $in $ldflags -o $out")
        link = [f'build {library_target}: link {" ".join(objects)}']
        default = [f"default {library_target}"]
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_bang:
        blocks.append(bang_compile_rule)
    blocks += [link_rule, build, link, default]
    with open(path, "w") as build_file:
        for block in blocks:
            lines = "\n".join(block)
            build_file.write(f"{lines}\n\n")


def MLUExtension(name, sources, *args, **kwargs):
    r"""
    Creates a :class:`setuptools.Extension` for BANG/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a BANG/C++
    extension. This includes the mlu include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

     .. warning::
        The PyTorch python API (as provided in libtorch_python) cannot be built
        with the flag ``py_limited_api=True``.  When this flag is passed, it is
        the user's responsibility in their library to not use APIs from
        libtorch_python (in particular pytorch/python bindings) and to only use
        APIs from libtorch (aten objects, operators and the dispatcher). For
        example, to give access to custom ops from python, the library should
        register the ops through the dispatcher.

        Contrary to CPython setuptools, who does not define -DPy_LIMITED_API
        as a compile flag when py_limited_api is specified as an option for
        the "bdist_wheel" command in ``setup``, PyTorch does! We will specify
        -DPy_LIMITED_API=min_supported_cpython to best enforce consistency,
        safety, and sanity in order to encourage best practices. To target a
        different version, set min_supported_cpython to the hexcode of the
        CPython version of choice.

    Example:
        >>> from setuptools import setup
        >>> from torch_mlu.utils.cpp_extension import BuildExtension, MLUExtension
        >>> setup(
                name='mlu_extension',
                ext_modules=[
                    MLUExtension(
                            name='mlu_extension',
                            sources=['extension.cpp', 'extension_kernel.mlu'],
                            extra_compile_args={'cxx': ['-g'],
                                                'cncc': ['-O2']})
                ],
                cmdclass={
                    'build_ext': BuildExtension
                })
    """
    library_dirs = kwargs.get("library_dirs", [])
    library_dirs += library_paths()
    kwargs["library_dirs"] = library_dirs

    libraries = kwargs.get("libraries", [])
    libraries.append("c10")
    libraries.append("torch")
    libraries.append("torch_cpu")
    if not kwargs.get("py_limited_api", False):
        # torch_python uses more than the python limited api
        libraries.append("torch_python")
    libraries.append("torch_python")

    # Add cambricon torch_mlu library
    libraries.append("torch_mlu")

    kwargs["libraries"] = libraries

    include_dirs = kwargs.get("include_dirs", [])
    include_dirs += include_paths()
    kwargs["include_dirs"] = include_dirs

    kwargs["language"] = "c++"

    return setuptools.Extension(name, sources, *args, **kwargs)


def _get_bang_arch_flags(cflags: Optional[List[str]] = None) -> List[str]:
    r"""
    Determine Bang arch flags to use.

    The added compile flag will be like "--bang-mlu-arch=mtp_592".

    """
    # If cflags is given, there may already be user-provided arch flags in it
    # (from `extra_compile_args`)
    if cflags is not None:
        for flag in cflags:
            if "--bang-mlu-arch" in flag or "--bang-arch" in flag:
                return []

    flags = []
    for bang_cflag in COMMON_BANG_CFLAGS:
        flags.append(bang_cflag)

    flags.append("--no-neuware-version-check")

    return sorted(list(set(flags)))


class BuildExtension(build_ext):
    r"""
     A custom :mod:`setuptools` build extension.

     This :class:`setuptools.build_ext` subclass takes care of passing the
     minimum required compiler flags (e.g. ``-std=c++17``) as well as mixed
     C++/BANG compilation (and support for BANG files in general).

     When using :class:`BuildExtension`, it is allowed to supply a dictionary
     for ``extra_compile_args`` (rather than the usual list) that maps from
     languages (``cxx`` or ``cncc``) to a list of additional compiler flags to
     supply to the compiler. This makes it possible to supply different flags to
     the C++ and BANG compiler during mixed compilation.

    ``use_ninja`` (bool): If ``use_ninja`` is ``True`` (default), then we
     attempt to build using the Ninja backend. Ninja greatly speeds up
     compilation compared to the standard ``setuptools.build_ext``.
     Fallbacks to the standard distutils backend if Ninja is not available.

     .. note::
         By default, the Ninja backend uses #CPUS + 2 workers to build the
         extension. This may use up too many resources on some systems. One
         can control the number of workers by setting the `MAX_JOBS` environment
         variable to a non-negative number.

    """

    @classmethod
    def with_options(cls, **options):
        r"""
        Returns a subclass with alternative constructor that extends any original keyword
        arguments to the original constructor with the given options.
        """

        class cls_with_options(cls):
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs):
        super(BuildExtension, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get("use_ninja", True)
        if self.use_ninja:
            msg = (
                "Attempted to use ninja as the BuildExtension backend but "
                "{}. Falling back to using the slow distutils backend."
            )
            if not torch.utils.cpp_extension.is_ninja_available():
                warnings.warn(msg.format("we could not find ninja."))
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def build_extensions(self):
        _, _ = self._check_abi()
        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx' and 'cncc' when
            # extra_compile_args is a dict. Otherwise, default torch flags do
            # not get passed. Necessary when only one of 'cxx' and 'cncc' is
            # passed to extra_compile_args in MLUExtension, i.e.
            #   MLUExtension(..., extra_compile_args={'cxx': [...]})
            # or
            #   MLUExtension(..., extra_compile_args={'cncc': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ["cxx", "cncc"]:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            self._add_compile_flag(extension, "-DTORCH_API_INCLUDE_EXTENSION_H")

            if extension.py_limited_api:
                # compile any extension that has passed in py_limited_api to the
                # Extension constructor with the Py_LIMITED_API flag set to our
                # min supported CPython version.
                # See https://docs.python.org/3/c-api/stable.html#c.Py_LIMITED_API
                self._add_compile_flag(
                    extension, f"-DPy_LIMITED_API={min_supported_cpython}"
                )

            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)
            self._define_cntoolkit_version(extension)

        # Register .mlu as valid source extensions.
        self.compiler.src_extensions += [".mlu"]
        # Save the original _compile method for later.
        original_compile = self.compiler._compile

        def append_std17_if_no_std_present(cflags) -> None:
            cpp_format_prefix = "-{}="
            cpp_flag_prefix = cpp_format_prefix.format("std")
            cpp_flag = cpp_flag_prefix + "c++17"
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_mlu_flags(cflags):
            cflags = COMMON_FLAGS + cflags + _get_bang_arch_flags(cflags)
            return cflags

        def convert_to_absolute_paths_inplace(paths):
            # Helper function. See Note [Absolute include_dirs].
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_single_compile(
            obj, src, ext, cc_args, extra_postargs, pp_opts
        ) -> None:
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_mlu_file(src):
                    cncc = _join_neuware_home("bin", "cncc")
                    if not isinstance(cncc, list):
                        cncc = [cncc]
                    self.compiler.set_executable("compiler_so", cncc)
                    if isinstance(cflags, dict):
                        cflags = cflags["cncc"]

                    cflags = unix_mlu_flags(cflags)
                    cc_args = ["-I" + _join_neuware_home("include")]
                    pp_opts = cc_args
                    cc_args.append("-c")
                elif isinstance(cflags, dict):
                    cflags = cflags["cxx"]

                append_std17_if_no_std_present(cflags)

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable("compiler_so", original_compiler)

        def unix_wrap_ninja_compile(
            sources,
            output_dir=None,
            macros=None,
            include_dirs=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            depends=None,
        ):
            r"""Compiles sources by outputting a ninja file and running it."""

            # Use absolute path for output_dir so that the object file paths
            # (`objects`) get generated with absolute paths.
            output_dir = os.path.abspath(output_dir)

            # See Note [Absolute include_dirs]
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = self.compiler._setup_compile(
                output_dir, macros, include_dirs, sources, depends, extra_postargs
            )
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            with_bang = any(map(_is_mlu_file, sources))

            # extra_postargs can be either:
            # - a dict mapping cxx/cncc to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs["cxx"]
            else:
                post_cflags = list(extra_postargs)
            append_std17_if_no_std_present(post_cflags)

            bang_post_cflags = None
            bang_cflags = None
            if with_bang:
                bang_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    bang_post_cflags = extra_postargs["cncc"]
                else:
                    bang_post_cflags = list(extra_postargs)
                bang_post_cflags += _get_bang_arch_flags(bang_post_cflags)
                bang_post_cflags += ["-fPIC"]
                neuware_home = _find_neuware_home()
                bang_post_cflags.append("--neuware-path=" + neuware_home)
                if not any(flag.startswith("-std=") for flag in bang_post_cflags):
                    bang_post_cflags.append("-std=c++17")
                bang_post_cflags = unix_mlu_flags(bang_post_cflags)
                append_std17_if_no_std_present(bang_post_cflags)
                bang_cflags = [shlex.quote(f) for f in bang_cflags]
                bang_post_cflags = [shlex.quote(f) for f in bang_post_cflags]

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                post_cflags=[shlex.quote(f) for f in post_cflags],
                bang_cflags=bang_cflags,
                bang_post_cflags=bang_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_bang=with_bang,
            )

            # Return *all* object filenames, not just the ones we just built.
            return objects

        # Monkey-patch the _compile or compile method.
        if self.use_ninja:
            self.compiler.compile = unix_wrap_ninja_compile
        else:
            self.compiler._compile = unix_wrap_single_compile

        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        # Get the original shared library name. For Python 3, this name will be
        # suffixed with "<SOABI>.so", where <SOABI> will be something like
        # cpython-37m-x86_64-linux-gnu.
        ext_filename = super().get_ext_filename(ext_name)
        # If `no_python_abi_suffix` is `True`, we omit the Python 3 ABI
        # component. This makes building shared libraries with setuptools that
        # aren't Python modules nicer.
        if self.no_python_abi_suffix:
            # The parts will be e.g. ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"].
            ext_filename_parts = ext_filename.split(".")
            # Omit the second to last element.
            without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
            ext_filename = ".".join(without_abi)
        return ext_filename

    def _check_abi(self):
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, "compiler_cxx"):
            compiler = self.compiler.compiler_cxx[0]
        else:
            compiler = os.environ.get("CXX", "c++")
        (
            _,
            version,
        ) = torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version(
            compiler
        )
        return compiler, version

    def _add_compile_flag(self, extension, flag):
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    def _define_torch_extension_name(self, extension):
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like torch._C, we take the last part of the string
        # as the library name.
        names = extension.name.split(".")
        name = names[-1]
        define = "-DTORCH_EXTENSION_NAME={}".format(name)
        self._add_compile_flag(extension, define)

    def _add_gnu_cpp_abi_flag(self, extension):
        # Use the same CXX ABI as what PyTorch was compiled with.
        self._add_compile_flag(
            extension,
            "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        )

    def _define_cntoolkit_version(self, extension):
        cntoolkit_version_file = _join_neuware_home("version.txt")
        with open(cntoolkit_version_file) as f:
            content = f.read()
        match = re.search(r"(\d+\.\d+\.\d+)", content)
        if match:
            cntoolkit_version = match.group(1)
        else:
            warnings.warn(
                f"Parse cntoolkit version from {cntoolkit_version_file} failed, skipping cntoolkit version definition."
            )
            return
        cntoolkit_ver_x, cntoolkit_ver_y, cntoolkit_ver_z = map(
            int, cntoolkit_version.split(".")
        )
        self._add_compile_flag(extension, f"-DCNTOOLKIT_MAJOR={cntoolkit_ver_x}")
        self._add_compile_flag(extension, f"-DCNTOOLKIT_MINOR={cntoolkit_ver_y}")
        self._add_compile_flag(extension, f"-DCNTOOLKIT_PATCHLEVEL={cntoolkit_ver_z}")
