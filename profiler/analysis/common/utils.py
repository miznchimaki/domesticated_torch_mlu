from typing import List, Optional, Tuple

import logging
import os
import re
import gzip
import json

from json.decoder import JSONDecodeError


def get_logging_level():
    log_level = os.environ.get("TORCH_PROFILER_LOG_LEVEL", "INFO").upper()
    if log_level not in logging._levelToName.values():
        log_level = logging.getLevelName(logging.INFO)
    return log_level


logger = None


def get_logger():
    global logger
    if logger is None:
        logger = logging.getLogger("MLU Profiler Analysis")
        logger.setLevel(get_logging_level())
    return logger


def merge_ranges(src_ranges, is_sorted=False) -> List[Tuple[float, float]]:
    if not src_ranges:
        # return empty list if src_ranges is None or its length is zero.
        return []

    if not is_sorted:
        src_ranges.sort(key=lambda x: x[0])

    merged_ranges = []
    merged_ranges.append(src_ranges[0])
    for src_id in range(1, len(src_ranges)):
        src_range = src_ranges[src_id]
        if src_range[1] > merged_ranges[-1][1]:
            if src_range[0] <= merged_ranges[-1][1]:
                merged_ranges[-1] = (merged_ranges[-1][0], src_range[1])
            else:
                merged_ranges.append((src_range[0], src_range[1]))

    return merged_ranges


RETURN_PATTERN = re.compile(r"(?:void\s+)?([^\(<]+)")


def reduce_name(name: str) -> str:
    if os.getenv("TORCH_MLU_PROFILER_CSV_USE_FULL_NAME", "0").upper() in (
        "1",
        "TRUE",
        "ON",
        "YES",
    ):
        return name
    matched = RETURN_PATTERN.match(name)
    if matched:
        name = matched.group(1)

    return name


def load_json_file(file_path: str):
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} not exits.")

    with open(file_path, "rb") as f:
        data = f.read()
    if file_path.endswith(".gz"):
        data = gzip.decompress(data)

    try:
        trace_json = json.loads(data)
    except JSONDecodeError as e:
        logger.error(
            f"JSONDecodeError: Failed to load json file {file_path}, error: {e}"
        )
        trace_json = None

    return trace_json


class ProfilerConfig:
    """Cached profiler configuration read from environment variables.

    Config is loaded lazily on first access and cached for subsequent reads.

    Environment variable ``TORCH_MLU_PROFILER_DUMP_CONFIG`` controls the
    behaviour:
      - 0: dump csv files (default)
      - 1: dump triton code and calculate IO efficiency
      - 2: dump csv files and add duration to stack
    Multiple values can be combined with commas, e.g. "0,2".
    """

    def __init__(self):
        self._loaded = False
        self._dump_config: List[int] = []
        self._enable_dump_csv = False
        self._enable_dump_triton_code = False
        self._enable_stack_with_duration = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        def to_int_list(s):
            try:
                return [int(v.strip()) for v in s.split(",")]
            except (ValueError, TypeError):
                return [0]

        self._dump_config = to_int_list(
            os.environ.get("TORCH_MLU_PROFILER_DUMP_CONFIG", "0")
        )
        self._enable_dump_csv = 0 in self._dump_config or 2 in self._dump_config
        self._enable_dump_triton_code = 1 in self._dump_config
        self._enable_stack_with_duration = 2 in self._dump_config
        self._loaded = True

    @property
    def dump_config(self) -> List[int]:
        self._ensure_loaded()
        return self._dump_config

    @property
    def enable_dump_csv(self) -> bool:
        self._ensure_loaded()
        return self._enable_dump_csv

    @property
    def enable_dump_triton_code(self) -> bool:
        self._ensure_loaded()
        return self._enable_dump_triton_code

    @property
    def enable_stack_with_duration(self) -> bool:
        self._ensure_loaded()
        return self._enable_stack_with_duration


_profiler_config: Optional[ProfilerConfig] = None


def get_profiler_config() -> ProfilerConfig:
    """Get the cached profiler configuration singleton."""
    global _profiler_config
    if _profiler_config is None:
        _profiler_config = ProfilerConfig()
    return _profiler_config
