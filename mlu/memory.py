# mypy: allow-untyped-defs
r"""This package adds support for device memory management implemented in MLU."""

import collections
import contextlib
import ctypes
import os
import pickle
import re
import sys
import warnings
from inspect import signature
from typing import Any, Dict, Literal, Optional, Tuple, TYPE_CHECKING, TypedDict, Union
from typing_extensions import deprecated, NotRequired

import torch
import torch_mlu
from torch._utils import _augment_memory_snapshot_stack_traces, _dummy_type
from torch.types import Device

from . import _get_device_index, _lazy_init, is_initialized
from torch.cuda._memory_viz import memory as _memory, segments as _segments


__all__ = [
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "caching_allocator_enable",
    "get_per_process_memory_fraction",
    "set_per_process_memory_fraction",
    "empty_cache",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "host_memory_stats",
    "host_memory_stats_as_nested_dict",
    "reset_accumulated_host_memory_stats",
    "reset_peak_host_memory_stats",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "memory_cached",
    "max_memory_cached",
    "memory_snapshot",
    "memory_summary",
    "mem_get_info",
    "is_linear_memory_enabled",
    "enable_linear_memory",
    "get_allocator_backend",
    "MLUPluggableAllocator",
    "change_current_allocator",
    "MemPool",
    "use_mem_pool",
]


if not hasattr(torch_mlu._MLUC, "_mlu_MLUAllocator"):
    # Define dummy base classes
    torch_mlu._MLUC.__dict__["_mlu_MLUAllocator"] = _dummy_type("_mlu_MLUAllocator")


if not hasattr(torch_mlu._MLUC, "_MemPool"):
    # Define dummy base classes
    torch_mlu._MLUC.__dict__["_MemPool"] = _dummy_type("_MemPool")
    torch_mlu._MLUC.__dict__["_mlu_beginAllocateToPool"] = _dummy_type(
        "_mlu_beginAllocateToPool"
    )
    # TODO: support mlu_beginAllocateCurrentThreadToPool
    torch_mlu._MLUC.__dict__["_mlu_endAllocateCurrentStreamToPool"] = _dummy_type(
        "_mlu_endAllocateCurrentStreamToPool"
    )
    torch_mlu._MLUC.__dict__["_mlu_endAllocateToPool"] = _dummy_type(
        "_mlu_endAllocateToPool"
    )
    torch_mlu._MLUC.__dict__["_mlu_releasePool"] = _dummy_type("_mlu_releasePool")

from torch_mlu._MLUC import (  # noqa: F401
    _mlu_beginAllocateToPool,
    _mlu_MLUAllocator,
    _mlu_endAllocateCurrentStreamToPool,
    _mlu_releasePool,
    _MemPool,
)


def _host_allocator():
    _lazy_init()
    return torch_mlu._MLUC._mlu_mluHostAllocator()


@contextlib.contextmanager
def _free_mutex():
    torch_mlu._MLUC._mlu_lock_mutex()
    try:
        yield
    finally:
        torch_mlu._MLUC._mlu_unlock_mutex()


def caching_allocator_alloc(size, device: Union[Device, int] = None, stream=None):
    r"""Perform a memory allocation using the MLU memory allocator.

    Memory is allocated for a given device and a stream, this
    function is intended to be used for interoperability with other
    frameworks. Allocated memory is released through
    :func:`~torch.mlu.caching_allocator_delete`.

    Args:
        size (int): number of bytes to be allocated.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default MLU device is used.
        stream (torch.mlu.Stream or int, optional): selected stream. If is ``None`` then
            the default stream for the selected device is used.
    """
    if device is None:
        device = torch.mlu.current_device()
    device = _get_device_index(device)
    if stream is None:
        stream = torch.mlu.current_stream(device)
    if isinstance(stream, torch.mlu.Stream):
        stream = stream.mlu_stream
    if not isinstance(stream, int):
        raise TypeError(
            "Invalid type for stream argument, must be "
            "`torch.mlu.Stream` or `int` representing a pointer "
            "to a existing stream"
        )
    with torch.mlu.device(device):
        return torch_mlu._MLUC._mlu_mluCachingAllocator_raw_alloc(size, stream)


def caching_allocator_delete(mem_ptr):
    r"""Delete memory allocated using the MLU memory allocator.

    Memory allocated with :func:`~torch.mlu.caching_allocator_alloc`.
    is freed here. The associated device and stream are tracked inside
    the allocator.

    Args:
        mem_ptr (int): memory address to be freed by the allocator.
    """
    torch_mlu._MLUC._mlu_mluCachingAllocator_raw_delete(mem_ptr)


def caching_allocator_enable(value: bool = True) -> None:
    r"""Enable or disable the MLU memory allocator. On by default."""
    if torch.mlu.is_initialized():
        torch_mlu._MLUC._mlu_mluCachingAllocator_enable(value)


def set_per_process_memory_fraction(
    fraction, device: Union[Device, int] = None
) -> None:
    r"""Set memory fraction for a process.

    The fraction is used to limit an caching allocator to allocated memory on a MLU device.
    The allowed value equals the total visible memory multiplied fraction.
    If trying to allocate more than the allowed value in a process, will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~1. Allowed memory equals total_memory * fraction.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default MLU device is used.
    .. note::
        In general, the total available free memory is less than the total capacity.
    """
    _lazy_init()
    if device is None:
        device = torch.mlu.current_device()
    device = _get_device_index(device)
    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    if fraction < 0 or fraction > 1:
        raise ValueError(f"Invalid fraction value: {fraction}. Allowed range: 0~1")

    torch_mlu._MLUC._mlu_setMemoryFraction(fraction, device)


def get_per_process_memory_fraction(device: Union[Device, int] = None) -> float:
    r"""Get memory fraction for a process.

    Args:
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default MLU device is used.
    Returns:
        memory fraction, in range 0~1. Allowed memory equals total_memory * fraction.
    """
    _lazy_init()
    if device is None:
        device = torch.mlu.current_device()
    device = _get_device_index(device)
    return torch_mlu._MLUC._mlu_getMemoryFraction(device)


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other MLU application and visible in
    `cnmon info`.

    .. note::
        :func:`~torch.mlu.empty_cache` doesn't increase the amount of MLU
        memory available for PyTorch. However, it may help reduce fragmentation
        of MLU memory in certain cases.
    """
    if torch.mlu.is_initialized():
        torch_mlu._MLUC._mlu_emptyCache()


def memory_stats(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Return a dictionary of MLU memory allocator statistics for a given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from ``cnrtMalloc()``.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of active memory blocks.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of inactive, non-releasable memory blocks.
    - ``"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of inactive, non-releasable memory.

    For these core statistics, values are broken down as follows.

    Pool type:

    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool
      (as of October 2019, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of October 2019, for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    In addition to the core statistics, we also provide some simple event
    counters:

    - ``"num_alloc_retries"``: number of failed ``cnrtMalloc`` calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.
    - ``"num_sync_all_streams"``: number of ``synchronize_and_free_events`` calls.
    - ``"num_device_alloc"``: number of MLU allocation calls. This includes both
      cnMemMap and cnrtMalloc.
    - ``"num_device_free"``: number of MLU free calls. This includes both cnMemUnmap
      and cnrtFree.

    The caching allocator can be configured via ENV to not split blocks larger than a
    defined size (see Memory Management section of the documentation).
    This helps avoid memory fragmentation but may have a performance
    penalty. Additional outputs to assist with tuning and evaluating impact:

    - ``"max_split_size"``: blocks above this size will not be split.
    - ``"oversize_allocations.{current,peak,allocated,freed}"``:
      number of over-size allocation requests received by the memory allocator.
    - ``"oversize_segments.{current,peak,allocated,freed}"``:
      number of over-size reserved segments from ``cnrtMalloc()``.

    The caching allocator can be configured via ENV to round memory allocations in order
    to reduce fragmentation. Sometimes the overhead from rounding can be higher than
    the fragmentation it helps reduce. The following stat can be used to check if
    rounding adds too much overhead:

    - ``"requested_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      memory requested by client code, compare this with allocated_bytes to check if
      allocation rounding adds too much overhead.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    result = []

    def _recurse_add_to_result(prefix, obj):
        if isinstance(obj, dict):
            if len(prefix) > 0:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + k, v)
        else:
            result.append((prefix, obj))

    stats = memory_stats_as_nested_dict(device=device)
    _recurse_add_to_result("", stats)
    result.sort()

    return collections.OrderedDict(result)


def memory_stats_as_nested_dict(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Return the result of :func:`~torch.mlu.memory_stats` as a nested dictionary."""
    if not torch.mlu.is_initialized():
        _lazy_init()
    device = _get_device_index(device, optional=True)
    return torch_mlu._MLUC._mlu_memoryStats(device)


def reset_accumulated_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the MLU memory allocator.

    See :func:`~torch.mlu.memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict, as well as
    `"num_alloc_retries"` and `"num_ooms"`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    device = _get_device_index(device, optional=True)
    return torch_mlu._MLUC._mlu_resetAccumulatedMemoryStats(device)


def reset_peak_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Reset the "peak" stats tracked by the MLU memory allocator.

    See :func:`~torch.mlu.memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    device = _get_device_index(device, optional=True)
    return torch_mlu._MLUC._mlu_resetPeakMemoryStats(device)


def host_memory_stats() -> Dict[str, Any]:
    r"""Return a dictionary of pinned (host) allocator statistics.

    Core statistics (host pinned allocator):

    - ``"allocations.{current,peak,allocated,freed}"``:
      pinned blocks owned by the allocator (active + cached). Grows when a new
      block is created via MLU and shrinks when cached blocks are returned.
    - ``"allocated_bytes.{current,peak,allocated,freed}"``:
      bytes of pinned blocks owned by the allocator (active + cached), using
      the rounded block size requested from MLU.
    - ``"active_requests.{current,peak,allocated,freed}"``:
      blocks currently checked out to callers (increments on handout, decrements
      when the block becomes reusable after stream deps finish).
    - ``"active_bytes.{current,peak,allocated,freed}"``:
      bytes corresponding to active blocks.

    Metric type:

    - ``current``: current value.
    - ``peak``: maximum value.
    - ``allocated``: historical total increase.
    - ``freed``: historical total decrease.

    Event/timing counters:

    - ``"num_host_alloc"`` / ``"num_host_free"``: blocks created to grow the
      pool / cached blocks returned to MLU (matches allocations allocated/freed).
    - ``"host_alloc_time.{total,max,min,count,avg}"``: time in MLU alloc calls
      when growing the pool (microseconds).
    - ``"host_free_time.{total,max,min,count,avg}"``: time in MLU free calls
      when cached blocks are returned (microseconds).

    Block sizes are rounded up to the next power of two before calling MLU, so
    byte stats reflect the rounded size. Peak values are aggregated per bucket
    and are a best-effort approximation of the true peak.
    """
    result = []

    def _recurse_add_to_result(prefix, obj):
        if isinstance(obj, dict):
            if len(prefix) > 0:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + k, v)
        else:
            result.append((prefix, obj))

    stats = host_memory_stats_as_nested_dict()
    _recurse_add_to_result("", stats)
    result.sort()

    return collections.OrderedDict(result)


def host_memory_stats_as_nested_dict() -> Dict[str, Any]:
    r"""Return the result of :func:`~torch.mlu.host_memory_stats` as a nested dictionary."""
    if not torch.mlu.is_initialized():
        return {}
    return torch_mlu._MLUC._mlu_hostMemoryStats()


def reset_accumulated_host_memory_stats() -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the host memory allocator.

    See :func:`~torch.mlu.host_memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict.
    """
    return torch_mlu._MLUC._mlu_resetAccumulatedHostMemoryStats()


def reset_peak_host_memory_stats() -> None:
    r"""Reset the "peak" stats tracked by the host memory allocator.

    See :func:`~torch.mlu.host_memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.
    """
    return torch_mlu._MLUC._mlu_resetPeakHostMemoryStats()


def reset_max_memory_allocated(device: Union[Device, int] = None) -> None:
    r"""Reset the starting point in tracking maximum MLU memory occupied by
    tensors for a given device.

    See :func:`~torch.mlu.max_memory_allocated` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch.mlu.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.
    """
    warnings.warn(
        "torch.mlu.reset_max_memory_allocated now calls torch.mlu.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
    )
    return reset_peak_memory_stats(device=device)


def reset_max_memory_cached(device: Union[Device, int] = None) -> None:
    r"""Reset the starting point in tracking maximum MLU memory managed by the caching allocator for a given device.

    See :func:`~torch.mlu.max_memory_cached` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch.mlu.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.
    """
    warnings.warn(
        "torch.mlu.reset_max_memory_cached now calls torch.mlu.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
    )
    return reset_peak_memory_stats(device=device)


def memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Return the current MLU memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Return the maximum MLU memory occupied by tensors in bytes for a given device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.mlu.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Return the current MLU memory managed by the caching allocator in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Return the maximum MLU memory managed by the caching allocator in bytes for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.mlu.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)


@deprecated(
    "`torch.mlu.memory_cached` has been renamed to `torch.mlu.memory_reserved`",
    category=FutureWarning,
)
def memory_cached(device: Union[Device, int] = None) -> int:
    r"""Deprecated; see :func:`~torch.mlu.memory_reserved`."""
    return memory_reserved(device=device)


@deprecated(
    "`torch.mlu.max_memory_cached` has been renamed to `torch.mlu.max_memory_reserved`",
    category=FutureWarning,
)
def max_memory_cached(device: Union[Device, int] = None) -> int:
    r"""Deprecated; see :func:`~torch.mlu.max_memory_reserved`."""
    return max_memory_reserved(device=device)


def memory_snapshot(mempool_id=None, include_traces=True):
    r"""Return a snapshot of the MLU memory allocator state across all devices.

    Interpreting the output of this function requires familiarity with the
    memory allocator internals.

    Args:
        mempool_id: Optional memory pool ID to get snapshot for a specific pool
        include_traces: Whether to include trace entries in the snapshot.
            If True (default), all trace entries are included.
            If False, no trace entries are included (lightweight/fast snapshot).
    """
    if mempool_id is None:
        # pyrefly: ignore [bad-argument-type]
        return torch_mlu._MLUC._mlu_memorySnapshot((0, 0, include_traces))["segments"]
    else:
        return torch_mlu._MLUC._mlu_memorySnapshot(
            # pyrefly: ignore [bad-argument-type]
            (mempool_id[0], mempool_id[1], include_traces)
        )["segments"]


def memory_summary(device: Union[Device, int] = None, abbreviated: bool = False) -> str:
    r"""Return a human-readable printout of the current memory allocator statistics for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
        abbreviated (bool, optional): whether to return an abbreviated summary
            (default: False).
    """
    device = _get_device_index(device, optional=True)
    stats = memory_stats(device=device)

    def _format_size(sz, pref_sz):
        prefixes = ["B  ", "KiB", "MiB", "GiB", "TiB", "PiB"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_sz < 768 * 1024:
                break
            prefix = new_prefix
            sz //= 1024
            pref_sz /= 1024
        return f"{sz:6d} {prefix}"

    def _format_count(cnt, pref_cnt):
        prefixes = [" ", "K", "M"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_cnt < 750 * 1000:
                break
            prefix = new_prefix
            cnt //= 1000
            pref_cnt /= 1000
        return f"{cnt:7d} {prefix} "

    metrics_to_display = [
        ("allocated_bytes", "Allocated memory", _format_size),
        ("active_bytes", "Active memory", _format_size),
        ("requested_bytes", "Requested memory", _format_size),
        ("reserved_bytes", "MLU reserved memory", _format_size),
        ("inactive_split_bytes", "Non-releasable memory", _format_size),
        ("allocation", "Allocations", _format_count),
        ("active", "Active allocs", _format_count),
        ("segment", "MLU reserved segments", _format_count),
        ("inactive_split", "Non-releasable allocs", _format_count),
    ]

    lines = []
    lines.append("=" * 75)
    lines.append(" {_:16} PyTorch MLU memory summary, device ID {device:<17d} ")
    lines.append("-" * 75)
    lines.append(
        "  {_:9} MLU OOMs: {num_ooms:<12d} | {_:6} cnrtMalloc retries: {num_alloc_retries:<8d}  "
    )
    lines.append("=" * 75)
    lines.append(
        "        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  "
    )

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)
        submetrics = [("all", metric_name)]
        if not abbreviated:
            submetrics.append(("large_pool", "      from large pool"))
            submetrics.append(("small_pool", "      from small pool"))

        current_prefval, peak_prefval, allocated_prefval, freed_prefval = (
            None,
            None,
            None,
            None,
        )

        for submetric_key, submetric_name in submetrics:
            prefix = metric_key + "." + submetric_key + "."

            current = stats[prefix + "current"]
            peak = stats[prefix + "peak"]
            allocated = stats[prefix + "allocated"]
            freed = stats[prefix + "freed"]

            if current_prefval is None:
                current_prefval = current
                peak_prefval = peak
                allocated_prefval = allocated
                freed_prefval = freed

            lines.append(
                f" {submetric_name:<21} | {formatter(current, current_prefval)} | {formatter(peak, peak_prefval)} | "
                f"{formatter(allocated, allocated_prefval)} | {formatter(freed, freed_prefval)} "
            )

    metrics_to_display = [
        ("oversize_allocations", "Oversize allocations", _format_count),
        ("oversize_segments", "Oversize MLU segments", _format_count),
    ]

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)

        prefix = metric_key + "."

        current = stats[prefix + "current"]
        peak = stats[prefix + "peak"]
        allocated = stats[prefix + "allocated"]
        freed = stats[prefix + "freed"]

        lines.append(
            f" {metric_name:<21} | {formatter(current, current)} | {formatter(peak, peak)} | "
            f"{formatter(allocated, allocated)} | {formatter(freed, freed)} "
        )

    lines.append("=" * 75)

    fmt_dict = {"_": "", "device": device}
    for k, v in stats.items():
        fmt_dict[k.replace(".", "-")] = v
    return "|" + "|\n|".join(lines).format(**fmt_dict) + "|\n"


def mem_get_info(device: Union[Device, int] = None) -> Tuple[int, int]:
    r"""Return the global free and total MLU memory occupied for a given
    device using cnrtMemGetInfo.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default) or if the device index is not specified.
    """
    if device is None:
        device = torch.mlu.current_device()
    # optional=True allows `device = torch.device('mlu')` for which device.index is None
    device = _get_device_index(device, optional=True)
    return torch_mlu._MLUC._mlu_mem_get_info(device)


linear_memory_enabled = None


def is_linear_memory_enabled():
    r"""
    Returns whether linear memory has enabled or not.
    """
    global linear_memory_enabled
    if linear_memory_enabled is not None:
        return linear_memory_enabled

    mlu_alloc_conf = os.environ.get("PYTORCH_MLU_ALLOC_CONF", "")
    match = re.search(r"use_linear_memory:([^,]*)", mlu_alloc_conf)
    if match:
        linear_memory_enabled = match.group(1) == "True"
    else:
        linear_memory_enabled = False
    return linear_memory_enabled


def enable_linear_memory():
    r"""
    Enable linear memory.
    """
    mlu_alloc_conf = os.environ.get("PYTORCH_MLU_ALLOC_CONF", "")
    if "use_linear_memory" not in mlu_alloc_conf:
        if mlu_alloc_conf and not mlu_alloc_conf.endswith(","):
            mlu_alloc_conf += ","
        os.environ["PYTORCH_MLU_ALLOC_CONF"] = mlu_alloc_conf + "use_linear_memory:True"
    else:
        if not is_linear_memory_enabled():
            warnings.warn(
                "Linear memory has already been disabled, which may cause performance degradation. "
                "You can enable it by setting PYTORCH_MLU_ALLOC_CONF=use_linear_memory:True"
            )


def _record_memory_history_legacy(
    enabled: bool,
    record_context=True,
    trace_alloc_max_entries=1,
    trace_alloc_record_context=False,
    device: Union[Device, int] = None,
    record_context_cpp=False,
    clear_history=False,
    compile_context=False,
    global_record_annotations=False,
    skip_actions=None,
):
    torch_mlu._MLUC._mlu_record_memory_history_legacy(
        enabled,
        record_context,
        trace_alloc_max_entries,
        trace_alloc_record_context,
        record_context_cpp,
        clear_history,
        compile_context,
        global_record_annotations,
        skip_actions if skip_actions is not None else [],
    )


def _record_memory_history(
    enabled: Literal["state", "all"] | None = "all", *args, **kwargs
) -> None:
    """Enable recording of stack traces associated with memory
    allocations, so you can tell what allocated any piece of memory in
    :func:`torch.mlu.memory._snapshot()`.

    In addition to keeping stack traces with each current allocation and free,
    this will also enable recording of a history of all alloc/free events.

    Use :func:`torch.mlu.memory._snapshot()` to retrieve this information,
    and the tools in `_memory_viz.py` to visualize snapshots.

    Buffer behavior
    ---------------

    This will store up to `max_entries` instances of `TraceEntry` when enabled.
    Python trace collection defaults to `sys.maxsize`, meaning long-running
    or indefinitely running jobs should set a reasonable limit to avoid excessive
    memory use. Expect each entry to be several KB.

    Longer running workflows or those with smaller `max_entries` values will only
    store the last accumulated `max_entries` entries, meaning new entries overwrite
    older entries.

    C++ implementation for reference to ring buffer implementation:

    .. code-block:: cpp

        if (record_history) {
          if (alloc_trace->size() < alloc_trace_max_entries_) {
            alloc_trace->emplace_back(te);
          } else {
            (*alloc_trace)[alloc_trace_next++] = te;
            if (alloc_trace_next == alloc_trace_max_entries_) {
              alloc_trace_next = 0;
            }
          }
        }

    Latency impact
    --------------

    The Python trace collection is fast (2us per trace), so you may consider
    enabling this on production jobs if you anticipate ever having to debug
    memory issues.

    C++ trace collection is also fast (~50ns/frame), which for many typical programs
    works out to ~2us per trace, but can vary depending on stack depth.

    Args:
        enabled (Literal[None, "state", "all"], optional):
            `None`, disable recording memory history.
            `"state"`, keep information for currently allocated memory.
            `"all"`, additionally keep a history of all alloc/free calls.
            Defaults to "all".
        context (Literal[None, "state", "alloc", "all"], optional):
            `None`, Do not record any tracebacks.
            `"state"`, Record tracebacks for currently allocated memory.
            `"alloc"`, additionally keep tracebacks for alloc calls.
            `"all"`, additionally keep tracebacks for free calls.
            Defaults to "all".
        stacks (Literal["python", "all"], optional):
            `"python"`, include Python, TorchScript, and inductor frames in tracebacks
            `"all"`, additionally include C++ frames
            Defaults to "all".
        max_entries (int, optional): Keep a maximum of `max_entries`
            alloc/free events in the recorded history recorded.
        clear_history (bool, optional): Clear history when enabling, defaults to False.
        skip_actions (list[str], optional): List of action types to skip when recording
            memory history. This can be used to reduce memory overhead by excluding
            certain types of events from being recorded. Valid action types are:

            - `"alloc"`: Memory allocation events
            - `"free_requested"`: Free requests (memory marked for freeing)
            - `"free_completed"`: Completed free operations (memory actually freed)
            - `"segment_alloc"`: Segment allocation from cnrtMalloc
            - `"segment_free"`: Segment freed back to MLU via cnrtFree
            - `"oom"`: Out-of-memory exceptions
            - `"snapshot"`: Memory snapshot generation events

            For example, to skip recording free_requested events:
            `skip_actions=["free_requested"]`

            Defaults to None (record all actions).

    """
    if isinstance(enabled, bool):
        return _record_memory_history_legacy(enabled, *args, **kwargs)
    else:
        return _record_memory_history_impl(enabled, *args, **kwargs)


def _record_memory_history_impl(
    enabled: Optional[str] = "all",
    context: Optional[str] = "all",
    stacks: str = "all",
    max_entries: int = sys.maxsize,
    device: Union[Device, int] = None,
    clear_history: bool = False,
    compile_context: bool = False,
    global_record_annotations: bool = False,
    skip_actions: Optional[list[str]] = None,
):
    torch_mlu._MLUC._mlu_record_memory_history(
        enabled,
        context,
        stacks,
        max_entries,
        clear_history,
        compile_context,
        global_record_annotations,
        skip_actions if skip_actions is not None else [],
    )


_record_memory_history.__signature__ = signature(_record_memory_history_impl)


def _snapshot(device: Union[Device, int] = None, augment_with_fx_traces=False):
    """Save a snapshot of MLU memory state at the time it was called.

    The state is represented as a dictionary with the following structure.

    .. code-block:: python

        class Snapshot(TypedDict):
            segments: List[Segment]
            device_traces: List[List[TraceEntry]]


        class Segment(TypedDict):
            # Segments are memory returned from a cnrtMalloc call.
            # The size of reserved memory is the sum of all Segments.
            # Segments are cached and reused for future allocations.
            # If the reuse is smaller than the segment, the segment
            # is split into more then one Block.
            # empty_cache() frees Segments that are entirely inactive.
            address: int
            total_size: int  #  cnrtMalloc'd size of segment
            stream: int
            segment_type: Literal["small", "large"]  # 'large' (>1MB)
            allocated_size: int  # size of memory in use
            active_size: int  # size of memory in use or in active_awaiting_free state
            blocks: List[Block]


        class Block(TypedDict):
            # A piece of memory returned from the allocator, or
            # current cached but inactive.
            size: int
            requested_size: int  # size requested during malloc, may be smaller than
            # size due to rounding
            address: int
            state: Literal[
                "active_allocated",  # used by a tensor
                "active_awaiting_free",  # waiting for another stream to finish using
                # this, then it will become free
                "inactive",
            ]  # free for reuse
            frames: List[Frame]  # stack trace from where the allocation occurred


        class Frame(TypedDict):
            filename: str
            line: int
            name: str
            # Optional FX debug fields (present when augment_with_fx_traces=True
            # and the frame corresponds to FX-generated code)
            fx_node_op: str  # FX node operation type (e.g., 'call_function', 'output')
            fx_node_name: str  # FX node name (e.g., 'linear', 'relu_1')
            fx_original_trace: str  # Original model source code stack trace


        class TraceEntry(TypedDict):
            # When `torch.mlu.memory._record_memory_history()` is enabled,
            # the snapshot will contain TraceEntry objects that record each
            # action the allocator took.
            action: Literal[
                "alloc"  # memory allocated
                "free_requested",  # the allocated received a call to free memory
                "free_completed",  # the memory that was requested to be freed is now
                # able to be used in future allocation calls
                "segment_alloc",  # the caching allocator ask cnrtMalloc for more memory
                # and added it as a segment in its cache
                "segment_free",  # the caching allocator called cnrtFree to return memory
                # to cnrt possibly trying free up memory to
                # allocate more segments or because empty_caches was called
                "oom",  # the allocator threw an OOM exception. 'size' is
                # the requested number of bytes that did not succeed
                "snapshot",  # the allocator generated a memory snapshot
                # useful to coorelate a previously taken
                # snapshot with this trace
            ]
            addr: int  # not present for OOM
            frames: List[Frame]
            size: int
            stream: int
            device_free: int  # only present for OOM, the amount of
            # memory cnrt still reports to be free

    Args:
        device: Device to capture snapshot for. If None, captures for current device.
        augment_with_fx_traces: If True, augment stack trace frames with FX debug information
                                that maps generated FX code back to original model source code.
                                This adds fx_node_op, fx_node_name, fx_original_trace, and
                                fx_node_info fields to Frame objects. Default: False.

    Returns:
        The Snapshot dictionary object
    """
    s = torch_mlu._MLUC._mlu_memorySnapshot(None)
    if augment_with_fx_traces:
        s = _augment_memory_snapshot_stack_traces(s)  # type: ignore[assignment, arg-type]
    return s


def _dump_snapshot(filename="dump_snapshot.pickle", augment_with_fx_traces=False):
    """
    Save a pickled version of the `torch.mlu.memory._snapshot()` dictionary to a file.

    This file can be opened by the interactive snapshot viewer at pytorch.org/memory_viz

    Args:
        filename (str, optional): Name of the file to create. Defaults to "dump_snapshot.pickle".
        augment_with_fx_traces (bool, optional): If True, augment the snapshot with FX debug information
                                                  before dumping. This maps generated FX code stack traces
                                                  back to original model source code. Defaults to False.
        verbose (bool, optional): If True and augment_with_fx_traces is True, print verbose debug output
                                  during augmentation. Defaults to False.
    """
    s = _snapshot(augment_with_fx_traces=augment_with_fx_traces)

    with open(filename, "wb") as f:
        pickle.dump(s, f)


def _set_memory_metadata(metadata: str):
    """
    Set custom metadata that will be attached to all subsequent MLU memory allocations.

    This metadata will be recorded in the memory snapshot for all allocations made
    after this call until the metadata is cleared or changed.

    Args:
        metadata (str): Custom metadata string to attach to allocations.
                       Pass an empty string to clear the metadata.
    """
    torch_mlu._MLUC._mlu_setMemoryMetadata(metadata)


def _get_memory_metadata() -> str:
    """
    Get the current custom metadata that is being attached to MLU memory allocations.

    Returns:
        str: The current metadata string, or empty string if no metadata is set.
    """
    return torch_mlu._MLUC._mlu_getMemoryMetadata()


def _save_segment_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        snapshot = _snapshot()
    with open(filename, "w") as f:
        f.write(_segments(snapshot))


def _save_memory_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        snapshot = _snapshot()
    with open(filename, "w") as f:
        f.write(_memory(snapshot))


@deprecated(
    "torch.mlu._set_allocator_settings is deprecated. Use torch._C._accelerator_setAllocatorSettings instead.",
    category=FutureWarning,
)
def _set_allocator_settings(env: str):
    # pyrefly: ignore [missing-attribute]
    # Also parse MLU-specific config options directly, since the device config
    # parser hook mechanism has an ODR issue that prevents the MLU hook from
    # being invoked when libc10.so and libtorch_mlu.so have separate copies
    # of the hook static variable.
    torch_mlu._MLUC._mlu_mluCachingAllocator_parse_config(env)
    return torch._C._accelerator_setAllocatorSettings(env)


def get_allocator_backend() -> str:
    r"""Return a string describing the active allocator backend as set by
    ``PYTORCH_ALLOC_CONF``. Currently available backends are
    ``native`` (PyTorch's native caching allocator) and `cnrtMemAllocAsync`
    (cnrt's built-in asynchronous allocator).
    """
    return torch_mlu._MLUC._mlu_getAllocatorBackend()


class _MLUAllocator:
    r"""Wrapper over internal MLU memory allocators."""

    def __init__(self, allocator: torch_mlu._MLUC._mlu_MLUAllocator):
        self._allocator = allocator

    def allocator(self):
        return self._allocator


class MLUPluggableAllocator(_MLUAllocator):
    r"""MLU memory allocator loaded from a so file."""

    def __init__(self, path_to_so_file: str, alloc_fn_name: str, free_fn_name: str):
        r"""Memory allocators are compiled in .so files and loaded dynamically using ctypes.

        To change the active allocator use the :func:`torch.mlu.memory.change_current_allocator` function.

        Args:
            path_to_so_file(str): Path in the filesystem to the `.so` file containing
                the allocator functions
            alloc_fn_name(str): Name of the function to perform the memory allocation
                in the so file. The signature must be:
                void* alloc_fn_name(ssize_t size, int device, cnrtQueue_t stream);
            free_fn_name(str): Name of the function to perform the memory release
                in the so file. The signature must be:
                void free_fn_name(void* ptr, size_t size, cnrtQueue_t stream);

        .. warning::
            This is currently supported only in unix OSs
        """
        allocator = ctypes.CDLL(path_to_so_file)
        alloc_fn = ctypes.cast(getattr(allocator, alloc_fn_name), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(allocator, free_fn_name), ctypes.c_void_p).value
        if alloc_fn is None:
            raise AssertionError(f"alloc_fn '{alloc_fn_name}' is None")
        if free_fn is None:
            raise AssertionError(f"free_fn '{free_fn_name}' is None")
        self._allocator = torch_mlu._MLUC._mlu_customAllocator(alloc_fn, free_fn)


def change_current_allocator(allocator: _MLUAllocator) -> None:
    r"""Change the currently used memory allocator to be the one provided.

    If the current allocator has already been used/initialized, this function will error.


    Args:
        allocator (torch.mlu.memory._MLUAllocator): allocator to be set as the active one.
    """
    torch_mlu._MLUC._mlu_changeCurrentAllocator(allocator.allocator())


def _get_current_allocator() -> _MLUAllocator:
    r"""Return the allocator being currently used."""
    return _MLUAllocator(torch_mlu._MLUC._mlu_getAllocator())


class MemPool(_MemPool):
    r"""MemPool represents a pool of memory in a caching allocator. Currently,
    it's just the ID of the pool object maintained in the MLUCachingAllocator.

    Args:
        allocator(torch_mlu._MLUC._mlu_MLUAllocator, optional): a
            torch_mlu._MLUC._mlu_MLUAllocator object that can be used to
            define how memory gets allocated in the pool. If :attr:`allocator`
            is ``None`` (default), memory allocation follows the default/
            current configuration of the MLUCachingAllocator.
        use_on_oom(bool): a bool that indicates if this pool can be used
            as a last resort if a memory allocation outside of the pool fails due
            to Out Of Memory. This is False by default.
        no_split(bool): a bool that indicates if this pool should not split a segment.
            This is False by default.
    """

    def __init__(
        self,
        allocator: Optional[_mlu_MLUAllocator] = None,
        use_on_oom: bool = False,
        no_split: bool = False,
    ):
        super().__init__(allocator, True, use_on_oom, no_split)

    @property
    def id(self) -> Tuple[int, int]:
        r"""Returns the ID of this pool as a tuple of two ints."""
        return super().id

    def use_count(self) -> int:
        r"""Returns the reference count of this pool."""
        return super().use_count()

    def snapshot(self, include_traces=True):
        r"""Return a snapshot of the MLU memory allocator pool state across all
        devices.

        Interpreting the output of this function requires familiarity with the
        memory allocator internals.

        Args:
            include_traces: Whether to include trace entries in the snapshot.
                If True (default), all trace entries are included.
                If False, no trace entries are included (lightweight/fast snapshot).
        """
        snapshot = torch.mlu.memory_snapshot(self.id, include_traces=include_traces)
        return snapshot


@contextlib.contextmanager
def use_mem_pool(pool: MemPool, device: Union[Device, int] = None):
    r"""A context manager that routes allocations to a given pool.

    Args:
        pool(torch.mlu.MemPool): a MemPool object to be made active so that
            allocations route to this pool.
        device (torch.device or int, optional): selected device. Uses MemPool on
            the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This context manager makes only current stream's allocations route to
        the given pool. If a new stream is created inside the context manager
        the allocations in that stream will not route to the given pool.
    """
    device_index = (
        torch.mlu.current_device() if device is None else _get_device_index(device)
    )
    _mlu_beginAllocateToPool(device_index, pool.id)
    try:
        yield
    finally:
        _mlu_endAllocateCurrentStreamToPool(device_index, pool.id)
        _mlu_releasePool(device_index, pool.id)
