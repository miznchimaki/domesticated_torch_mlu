import logging
import os
from typing import Any

import torch.distributed as dist
from torch.utils._triton import has_triton


logger = logging.getLogger(__name__)


class CnshmemLibFinder:
    """
    A class to find path to the CNSHMEM device library.

    Environment variable:

    `CNSHMEM_LIB_DIR` (Optional[str]): The directory where the CNSHMEM device
    library is located. If not provided, it will search for the library in
    common system paths (e.g. /usr/local/neuware/lib64).
    """

    # Class variable to store the found library path for reuse
    found_device_lib_path: str | None = None

    @classmethod
    def find_device_library(cls) -> str:
        """
        Find the path to the CNSHMEM device library.

        Returns:
            str: The path to libcnshmem_device.bc (included).
        """
        if cls.found_device_lib_path is not None:
            return cls.found_device_lib_path

        # First, check if the user has specified a custom library path
        user_lib_dir = os.environ.get("CNSHMEM_LIB_DIR", None)
        if user_lib_dir is not None:
            lib_path = os.path.join(user_lib_dir, "libcnshmem_device.bc")
            if not os.path.exists(lib_path):
                raise RuntimeError(
                    f"CNSHMEM device library not found at specified path: {user_lib_dir}"
                )
            cls.found_device_lib_path = lib_path
            return lib_path

        # Otherwise, search for the library in common system paths
        paths = [
            "/usr/local/neuware/lib64",
            "/usr/local/lib",
            "/usr/lib",
            "/opt/neuware/lib64",
        ]

        # Also check NEUWARE_HOME if set
        neuware_home = os.environ.get("NEUWARE_HOME", None)
        if neuware_home is not None:
            paths.insert(0, os.path.join(neuware_home, "lib64"))

        for path in paths:
            device_lib = os.path.join(path, "libcnshmem_device.bc")
            if os.path.exists(device_lib):
                cls.found_device_lib_path = device_lib
                return device_lib

        raise RuntimeError(f"CNSHMEM device library not found. Searched: {paths}")


class CnshmemKernelRegistry:
    """
    A class to register kernel functions that ** require CNSHMEM initialization **
    """

    # Class variable to store the functions to be initialized
    _to_init: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> None:
        """
        Register a kernel function with the given name.

        Args:
            name (str): The name of the kernel function.
        """
        cls._to_init.setdefault(name)

    @classmethod
    def deregister(cls, name: str) -> None:
        """
        Deregister a kernel function with the given name.

        Args:
            name (str): The name of the kernel function.
        """
        cls._to_init.pop(name, None)

    @classmethod
    def has(cls, name: str) -> bool:
        """
        Check if a kernel function with the given name is registered.

        Args:
            name (str): The name of the kernel function.

        Returns:
            bool: True if the kernel function is registered, False otherwise.
        """
        return name in cls._to_init


def _cnshmem_init_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    """
    A hook function to initialize the CNModule created by `triton.jit` with
    CNSHMEM device context
    """
    from torch._C._distributed_c10d import _cnshmemx_cnmodule_init

    jit_function = kwargs["fn"].jit_function
    fn_name = jit_function.fn.__name__

    # Only initialize CNSHMEM module for kernels registered via @requires_cnshmem
    if CnshmemKernelRegistry.has(fn_name):
        key = kwargs["key"]
        device = kwargs["compile"]["device"]
        jit_function = kwargs["fn"].jit_function
        kernel_cache = jit_function.device_caches[device][0]
        kernel = kernel_cache.get(key, None)
        if kernel is not None:
            kernel.run
            # Initialize CNSHMEM for the CN module
            _cnshmemx_cnmodule_init(kernel.module)
        else:
            logger.warning(
                f"It seems Triton hasn't created a kernel for function {fn_name}. "  # noqa: G004
                "Please report this issue to Triton."
            )


if has_triton():
    from triton.runtime.jit import JITFunction, KernelInterface

    # Create a new Callable class that follows the KernelInterface protocol so
    # that the Callable works with the subscript operator, e.g. `foo[(1, 1)]`
    class GridCallableWithExtern(KernelInterface):
        """
        `KernelInterface` invokes `self.run` in `__getitem__`, i.e. [].  We
        implement a `run` method by directing the call to `JITFunction.run`,
        with added extern_libs kwarg, so that users don't have to pass it
        """

        def __init__(self, jit_func: JITFunction, extern_libs: dict[str, str]) -> None:
            self.jit_func = jit_func
            self.extern_libs = extern_libs

        def run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            # Call the JITFunction.run with added extern_libs kwarg
            return self.jit_func.run(*args, **kwargs, extern_libs=self.extern_libs)


def requires_cnshmem(  # type: ignore[no-untyped-def]
    jit_func,  # JITFunction created by triton.jit
):
    """
    A decorator to register a Triton kernel function that requires CNSHMEM initialization.

    Example usage:
    ```
        @requires_cnshmem
        @triton.jit
        def foo(...):
            ...
    ```

    If you would like to specify a path to the CNSHMEM device library other
    than standard search locations, you can use the following environment
    variable:
    ```
        export CNSHMEM_LIB_DIR=/path/to/neuware/lib64
    ```
    """

    # TODO: use _shmem_triton_utils
    # from torch.distributed._symmetric_memory._shmem_triton_utils import (
    #     build_requires_shmem_decorator,
    # )

    # return build_requires_shmem_decorator(
    #     jit_func=jit_func,
    #     find_device_library=CnshmemLibFinder.find_device_library,
    #     extern_libs_key="libcnshmem_device",
    #     registry=CnshmemKernelRegistry,
    #     init_hook=_cnshmem_init_hook,
    #     error_prefix="@requires_cnshmem",
    # )

    import triton
    from triton.runtime.jit import JITFunction

    if not isinstance(jit_func, JITFunction):
        raise TypeError(f"Expected a JITFunction, but got {type(jit_func)}")

    # Find the CNSHMEM device library
    lib_path = CnshmemLibFinder.find_device_library()

    # TODO: the triton mlu backend only support lib_key == 'libdevice'
    # extern_libs = {"libcnshmem_device": lib_path}
    extern_libs = {"libdevice": lib_path}

    # Register the JITFunction with the kernel registry as "to be initialized"
    CnshmemKernelRegistry.register(jit_func.fn.__name__)

    # Register the CNSHMEM init function as a post-compile hook.
    # [Note] This is a global setting (due to lack of Triton API exposure). To
    # avoid initializing Triton kernels that do not require CNSHMEM, filtering
    # is performed in the hook function itself by checking against
    # CnshmemKernelRegistry.
    triton.knobs.runtime.jit_post_compile_hook = _cnshmem_init_hook

    return GridCallableWithExtern(jit_func, extern_libs)


if has_triton():
    import triton
    import triton.language as tl
    from triton.language import core

    # -------------------------------------------------------
    # Put Operations
    # -------------------------------------------------------

    @triton.jit  # type: ignore[misc]
    def put(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        """
        Put tensor data from local PE to a remote PE.

        This high-level function provides a tensor-aware interface for CNSHMEM put
        operations. It automatically handles type checking and size calculations, making
        the API more ergonomic and type-safe.

        Args:
            dest: Destination tensor on the remote PE. Type must match source.
            source: Source tensor on the local PE containing data to be copied.
            nelems: Number of elements to transfer.
            pe: PE number of the remote PE (0 ≤ pe < cnshmem_n_pes()).

        Notes:
            - Performs compile-time type checking between dest and source tensors.
            - Automatically calculates byte size from tensor type and element count.
            - This is a blocking operation that returns after data has been copied out
              of the source array on the local PE.
            - The operation does not guarantee delivery to the destination PE.
              Use cnshmem_fence() for ordering or cnshmem_quiet() for completion.
        """
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return putmem_extern_wrapper(
            dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe
        )

    @core.extern
    def putmem_extern_wrapper(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        """Low-level extern wrapper for CNSHMEM putmem"""
        return core.extern_elementwise(
            "",
            "",
            [dest, source, size_bytes, pe],
            {
                (
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                    core.dtype("int32"),  # pe number
                ): ("cnshmem_putmem", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit  # type: ignore[misc]
    def put_nbi(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        """
        Put tensor data from local PE to a remote PE, non-blocking.

        Different from the `put` function, this function returns after
        initiating the operation. The operation is considered complete after a
        subsequent call to `quiet`.

        Args:
            dest: Destination tensor on the remote PE. Type must match source.
            source: Source tensor on the local PE containing data to be copied.
            nelems: Number of elements to transfer.
            pe: PE number of the remote PE (0 ≤ pe < cnshmem_n_pes()).

        Notes:
            - Performs compile-time type checking between dest and source tensors.
            - Automatically calculates byte size from tensor type and element count.
        """
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return putmem_nbi_extern_wrapper(
            dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe
        )

    @core.extern
    def putmem_nbi_extern_wrapper(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        """Low-level extern wrapper for CNSHMEM putmem_nbi"""
        return core.extern_elementwise(
            "",
            "",
            [dest, source, size_bytes, pe],
            {
                (
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                    core.dtype("int32"),  # pe number
                ): ("cnshmem_putmem_nbi", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # -------------------------------------------------------
    # Put with Signal Operations
    # -------------------------------------------------------

    @triton.jit  # type: ignore[misc]
    def putmem_signal_block(  # type: ignore[no-untyped-def]
        dst,
        src,
        size_bytes,
        signal,
        sig_val,
        sig_op,
        pe,
    ):  # type: ignore[no-untyped-def]
        """
        Put data to remote PE with atomic signal operation.

        This function copies data from the local PE to the remote PE and then
        atomically updates a signal variable on the remote PE to indicate completion.
        This enables efficient point-to-point synchronization between PEs.

        Args:
            dst (tensor): A tensor on calling PE symmetric to the destination tensor on remote PE.
            src (tensor): Local tensor containing the source data.
            size_bytes (int64): Number of bytes to transfer. Must be positive.
            signal (tensor): Symmetric signal pad with remote PE.
                             Must be 8-byte aligned symmetric memory.
            sig_val (int64): Value to be used in the signal operation.
            sig_op (int32): Signal operation type. Common values:
                           - CNSHMEM_SIGNAL_SET (0): Atomic set operation
                           - CNSHMEM_SIGNAL_ADD (5): Atomic add operation
            pe (int32): PE number of the remote PE (0 ≤ pe < cnshmem_n_pes()).

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a blocking operation that returns after data has been copied out
              of the source array and the signal has been updated on the remote PE.
            - The signal update is performed atomically with respect to other signal
              operations and synchronization routines.
            - The signal variable must be of type uint64_t in symmetric memory.
            - Use with cnshmem_signal_wait_until() for synchronization.
        """
        # Ensure sig_val is 64 bits
        sig_val = 0 << 32 | sig_val
        return putmem_signal_extern_wrapper(
            dst.to(tl.int64),
            src.to(tl.int64),
            size_bytes.to(tl.int64),
            signal.to(tl.int64),
            sig_val.to(tl.uint64),
            sig_op,
            pe,
        )

    @core.extern
    def putmem_signal_extern_wrapper(  # type: ignore[no-untyped-def]
        dst,
        src,
        size_bytes,
        signal,
        sig_val,
        sig_op,
        pe,
        _semantic=None,
    ):  # type: ignore[no-untyped-def]
        """Low-level extern wrapper for CNSHMEM putmem_signal"""
        return core.extern_elementwise(
            "",
            "",
            [dst, src, size_bytes, signal, sig_val, sig_op, pe],
            {
                (
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                    core.dtype("int64"),  # signal ptr
                    core.dtype("uint64"),  # signal value
                    core.dtype("int32"),  # signal operation
                    core.dtype("int32"),  # pe number
                ): ("cnshmem_putmem_signal", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit  # type: ignore[misc]
    def putmem_signal_nbi(  # type: ignore[no-untyped-def]
        dst,
        src,
        size_bytes,
        signal,
        sig_val,
        sig_op,
        pe,
    ):  # type: ignore[no-untyped-def]
        """
        Put data to remote PE with atomic signal operation, non-blocking.

        Different from `putmem_signal_block`, this function returns after
        initiating the operation. The operation is considered complete after a
        subsequent call to `quiet`.

        Args:
            dst (tensor): A tensor on calling PE symmetric to the destination tensor on remote PE.
            src (tensor): Local tensor containing the source data.
            size_bytes (int64): Number of bytes to transfer. Must be positive.
            signal (tensor): Symmetric signal pad with remote PE.
                             Must be 8-byte aligned symmetric memory.
            sig_val (int64): Value to be used in the signal operation.
            sig_op (int32): Signal operation type. Common values:
                           - CNSHMEM_SIGNAL_SET (0): Atomic set operation
                           - CNSHMEM_SIGNAL_ADD (5): Atomic add operation
            pe (int32): PE number of the remote PE (0 ≤ pe < cnshmem_n_pes()).

        Returns:
            int32: Status code (0 for success).
        """
        # Ensure sig_val is 64 bits
        sig_val = 0 << 32 | sig_val
        return putmem_signal_nbi_extern_wrapper(
            dst.to(tl.int64),
            src.to(tl.int64),
            size_bytes.to(tl.int64),
            signal.to(tl.int64),
            sig_val.to(tl.uint64),
            sig_op,
            pe,
        )

    @core.extern
    def putmem_signal_nbi_extern_wrapper(  # type: ignore[no-untyped-def]
        dst,
        src,
        size_bytes,
        signal,
        sig_val,
        sig_op,
        pe,
        _semantic=None,
    ):  # type: ignore[no-untyped-def]
        """Low-level extern wrapper for CNSHMEM putmem_signal_nbi"""
        return core.extern_elementwise(
            "",
            "",
            [dst, src, size_bytes, signal, sig_val, sig_op, pe],
            {
                (
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                    core.dtype("int64"),  # signal ptr
                    core.dtype("uint64"),  # signal value
                    core.dtype("int32"),  # signal operation
                    core.dtype("int32"),  # pe number
                ): ("cnshmem_putmem_signal_nbi", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # -------------------------------------------------------
    # Wait and Signal Operations
    # -------------------------------------------------------

    @triton.jit  # type: ignore[misc]
    def signal_wait_until(signal, cmp, cmp_val):  # type: ignore[no-untyped-def]
        """
        Wait until a signal variable meets a specified condition.

        This function blocks the calling thread until the value at the specified
        signal variable satisfies the given comparison condition. Signal variables
        are special uint64_t symmetric objects used for efficient synchronization
        with signal operations.

        Args:
            signal (tensor): Symmetric signal tensor with remote PE.
                             Must be 8-byte aligned symmetric memory.
            cmp (int32): Comparison operator. Common values:
                        - CNSHMEM_CMP_EQ (0): Wait until signal == cmp_val
                        - CNSHMEM_CMP_NE (1): Wait until signal != cmp_val
                        - CNSHMEM_CMP_GT (2): Wait until signal > cmp_val
                        - CNSHMEM_CMP_GE (3): Wait until signal >= cmp_val
                        - CNSHMEM_CMP_LT (4): Wait until signal < cmp_val
                        - CNSHMEM_CMP_LE (5): Wait until signal <= cmp_val
            cmp_val (int64): Value to compare against.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a blocking operation designed specifically for signal variables.
            - Signal variables are updated atomically by putmem_signal operations.
            - More efficient than wait_until for signal-based synchronization patterns.
            - Ensures the signal update is fully complete before returning.
            - Commonly used with putmem_signal_block for producer-consumer patterns.
        """
        cmp_val = 0 << 32 | cmp_val
        return signal_wait_until_extern_wrapper(
            signal.to(tl.int64), cmp, cmp_val.to(tl.uint64)
        )

    @core.extern
    def signal_wait_until_extern_wrapper(signal, cmp, cmp_val, _semantic=None):  # type: ignore[no-untyped-def]
        """Low-level extern wrapper for CNSHMEM signal_wait_until"""
        return core.extern_elementwise(
            "",
            "",
            [signal, cmp, cmp_val],
            {
                (
                    core.dtype("int64"),  # signal ptr
                    core.dtype("int32"),  # comparison operator
                    core.dtype("uint64"),  # comparison value
                ): ("cnshmem_signal_wait_until", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def signal_op(sig_addr, signal, sig_op, pe, _semantic=None):  # type: ignore[no-untyped-def]
        """
        Perform an atomic signal operation on a remote PE.

        This function atomically updates a signal variable on the specified remote PE
        using the given operation and value. This enables efficient point-to-point
        synchronization and notification between PEs.

        Args:
            sig_addr (int64): Symmetric address of the signal variable (uint64_t) on the remote PE.
                             Must be 8-byte aligned symmetric memory.
            signal (int64): Value to be used in the signal operation.
            sig_op (int32): Signal operation type. Common values:
                           - CNSHMEM_SIGNAL_SET (0): Atomically set sig_addr = signal
                           - CNSHMEM_SIGNAL_ADD (5): Atomically set sig_addr += signal
            pe (int32): PE number of the remote PE (0 ≤ pe < cnshmem_n_pes()).
            _semantic: Optional semantic information for Triton compilation.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a one-sided operation - the remote PE does not need to participate.
            - The signal operation is performed atomically on the remote PE.
            - Can be used with signal_wait_until() on the remote PE for synchronization.
            - The signal variable must be of type uint64_t in symmetric memory.
        """
        return core.extern_elementwise(
            "",
            "",
            [sig_addr, signal, sig_op, pe],
            {
                (
                    core.dtype("int64"),  # signal address ptr
                    core.dtype("int64"),  # signal value
                    core.dtype("int32"),  # signal operation
                    core.dtype("int32"),  # pe number
                ): ("cnshmemx_signal_op", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # -------------------------------------------------------
    # Strided Put Operations
    # -------------------------------------------------------

    @triton.jit  # type: ignore[misc]
    def putmem_strided(  # type: ignore[no-untyped-def]
        dst,
        src,
        segsize,
        dst_stride,
        src_stride,
        segnum,
        pe,
    ):  # type: ignore[no-untyped-def]
        """
        Strided block data transfer to a remote PE.

        Copies segnum segments of segsize bytes each from src to dst, with
        respective strides between consecutive segments.

        Args:
            dst (int64): Destination address on the remote PE.
            src (int64): Source address on the local PE.
            segsize (int32): Size of each segment in bytes.
            dst_stride (int64): Stride between destination segments in bytes.
            src_stride (int64): Stride between source segments in bytes.
            segnum (int32): Number of segments to transfer.
            pe (int32): PE number of the remote PE.

        Notes:
            - This is a blocking operation.
        """
        return putmem_strided_extern_wrapper(
            dst.to(tl.int64),
            src.to(tl.int64),
            segsize,
            dst_stride.to(tl.int64),
            src_stride.to(tl.int64),
            segnum,
            pe,
        )

    @core.extern
    def putmem_strided_extern_wrapper(  # type: ignore[no-untyped-def]
        dst,
        src,
        segsize,
        dst_stride,
        src_stride,
        segnum,
        pe,
        _semantic=None,
    ):  # type: ignore[no-untyped-def]
        """Low-level extern wrapper for CNSHMEM putmem_strided"""
        return core.extern_elementwise(
            "",
            "",
            [dst, src, segsize, dst_stride, src_stride, segnum, pe],
            {
                (
                    core.dtype("int64"),  # dst ptr
                    core.dtype("int64"),  # src ptr
                    core.dtype("int32"),  # segsize
                    core.dtype("int64"),  # dst stride
                    core.dtype("int64"),  # src stride
                    core.dtype("int32"),  # segnum
                    core.dtype("int32"),  # pe number
                ): ("cnshmemx_putmem_strided", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit  # type: ignore[misc]
    def putmem_strided_nbi(  # type: ignore[no-untyped-def]
        dst,
        src,
        segsize,
        dst_stride,
        src_stride,
        segnum,
        pe,
    ):  # type: ignore[no-untyped-def]
        """
        Strided block data transfer to a remote PE, non-blocking.

        Different from `putmem_strided`, this function returns after
        initiating the operation. The operation is considered complete after a
        subsequent call to `quiet`.

        Args:
            dst (int64): Destination address on the remote PE.
            src (int64): Source address on the local PE.
            segsize (int32): Size of each segment in bytes.
            dst_stride (int64): Stride between destination segments in bytes.
            src_stride (int64): Stride between source segments in bytes.
            segnum (int32): Number of segments to transfer.
            pe (int32): PE number of the remote PE.
        """
        return putmem_strided_nbi_extern_wrapper(
            dst.to(tl.int64),
            src.to(tl.int64),
            segsize,
            dst_stride.to(tl.int64),
            src_stride.to(tl.int64),
            segnum,
            pe,
        )

    @core.extern
    def putmem_strided_nbi_extern_wrapper(  # type: ignore[no-untyped-def]
        dst,
        src,
        segsize,
        dst_stride,
        src_stride,
        segnum,
        pe,
        _semantic=None,
    ):  # type: ignore[no-untyped-def]
        """Low-level extern wrapper for CNSHMEM putmem_strided_nbi"""
        return core.extern_elementwise(
            "",
            "",
            [dst, src, segsize, dst_stride, src_stride, segnum, pe],
            {
                (
                    core.dtype("int64"),  # dst ptr
                    core.dtype("int64"),  # src ptr
                    core.dtype("int32"),  # segsize
                    core.dtype("int64"),  # dst stride
                    core.dtype("int64"),  # src stride
                    core.dtype("int32"),  # segnum
                    core.dtype("int32"),  # pe number
                ): ("cnshmemx_putmem_strided_nbi", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # -------------------------------------------------------
    # Memory Ordering Operations
    # -------------------------------------------------------

    @core.extern
    def fence(_semantic=None):  # type: ignore[no-untyped-def]
        """
        Ensure ordering of put operations to each remote PE.

        This function provides a memory fence that ensures point-to-point ordering
        of remote memory operations. Put operations issued before the fence are
        guaranteed to be ordered before put operations issued after the fence,
        when targeting the same remote PE.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This provides weaker ordering guarantees than quiet().
            - Operations to each PE are ordered, but operations to different PEs
              may still be reordered relative to each other.
            - Does not guarantee completion of operations, only ordering.
            - Non-blocking operations are not ordered by fence - use quiet() instead.
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {
                (): ("cnshmem_fence", core.dtype("int32")),
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def pe_fence(_semantic=None):  # type: ignore[no-untyped-def]
        """
        Ensure ordering of put operations to a specific remote PE.

        CNSHMEM-specific: provides PE-scoped fence ordering, which is finer
        granularity than cnshmem_fence().

        Returns:
            int32: Status code (0 for success).
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {
                (): ("cnshmem_pe_fence", core.dtype("int32")),
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def quiet(_semantic=None):  # type: ignore[no-untyped-def]
        """
        Wait for completion of all outstanding put operations.

        This function blocks until all outstanding remote memory operations issued
        by the calling PE have completed. It provides stronger guarantees than
        fence() by ensuring both ordering and completion of all operations.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a blocking operation that waits for completion.
            - Ensures all previous put operations have been delivered to their destinations.
            - Provides global ordering - operations to ALL PEs are ordered.
            - Required to complete non-blocking operations.
            - More expensive than fence() but provides stronger guarantees.
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {
                (): ("cnshmem_quiet", core.dtype("int32")),
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def pe_quiet(_semantic=None):  # type: ignore[no-untyped-def]
        """
        Wait for completion of all outstanding put operations to a specific PE.

        CNSHMEM-specific: provides PE-scoped quiet completion, which is finer
        granularity than cnshmem_quiet().

        Returns:
            int32: Status code (0 for success).
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {
                (): ("cnshmem_pe_quiet", core.dtype("int32")),
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # -------------------------------------------------------
    # Synchronization Operations
    # -------------------------------------------------------

    @core.extern
    def barrier_all(_semantic=None):  # type: ignore[no-untyped-def]
        """
        Synchronize all PEs with completion guarantee.

        This function creates a barrier across all PEs in the CNSHMEM job. It ensures
        that all local and remote memory updates issued before the barrier by any PE
        are completed before any PE exits the barrier.

        Returns:
            int32: Status code (0 for success).

        Notes:
            - This is a collective operation - all PEs must participate.
            - Stronger guarantee than sync_all() - ensures completion of remote operations.
            - Blocks until all PEs reach the barrier AND all memory operations complete.
            - Provides full memory consistency across all PEs.
        """
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("cnshmem_barrier_all", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    # -------------------------------------------------------
    # Utility for inspecting Triton kernels
    # -------------------------------------------------------

    triton_kernels: dict = {}

    def _log_triton_kernel(kernel) -> None:  # type: ignore[no-untyped-def]
        import atexit
        import tempfile

        if dist.is_initialized() and dist.get_rank() != 0:
            return

        def on_exit() -> None:
            logger.info("PTX files:")
            for kernel in triton_kernels:
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    f.write(kernel.asm["ptx"].encode("utf-8"))
                    logger.info(f"+- {kernel.name}: {f.name}")  # noqa: G004

        if len(triton_kernels) == 0:
            atexit.register(on_exit)

        if kernel not in triton_kernels:
            triton_kernels[kernel] = None
