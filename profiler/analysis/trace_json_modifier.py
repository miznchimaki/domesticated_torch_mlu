"""Trace JSON modifier for rewriting profiler trace JSON files.

This module provides a unified class for modifying profiler trace JSON files,
allowing various modifications to be applied in a single load/save cycle.
"""

import gzip
import json
import os
import time
from typing import Optional

from .callstack_rebuilder import CallStackRebuilder
from .common import utils
from .common import consts

logger = utils.get_logger()


class TraceJsonModifier:
    """A class for modifying profiler trace JSON files.

    This class provides a unified interface for applying various modifications
    to profiler trace JSON files. It loads the JSON file once, applies enabled
    modifications, and saves the result, avoiding multiple load/save cycles.

    Example:
        modifier = TraceJsonModifier(json_path)
        modifier.process()
    """

    def __init__(self, json_path: str):
        """Initialize the TraceJsonModifier.

        Args:
            json_path: Path to the trace JSON file.
        """
        self.json_path = json_path
        self.trace_data: Optional[dict] = None
        # Efficiency counter event is enabled by environment variable
        self._enable_efficiency_counter_event = (
            float(os.environ.get("TORCH_MLU_PROFILER_TENSOR_PERFORMANCE", "-1.0")) > 0.0
        )

    def load(self) -> bool:
        """Load the trace JSON file.

        Returns:
            True if successful, False otherwise.
        """
        if not os.path.exists(self.json_path):
            logger.error(f"File {self.json_path} does not exist.")
            return False

        try:
            with open(self.json_path, "rb") as f:
                data = f.read()
            if self.json_path.endswith(".gz"):
                data = gzip.decompress(data)

            self.trace_data = json.loads(data)
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Failed to load json file {self.json_path}, error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load file {self.json_path}, error: {e}")
            return False

    def save(self) -> bool:
        """Save the trace JSON file.

        Returns:
            True if successful, False otherwise.
        """
        if self.trace_data is None:
            logger.error("No trace data to save.")
            return False
        if self.json_path.endswith(".gz"):
            with gzip.open(self.json_path, "wt", encoding="utf-8") as f:
                json.dump(self.trace_data, f, indent=2)
        else:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.trace_data, f, indent=2)
        return True

    @staticmethod
    def _sample_clock_gaps(sample_count: int = 5):
        """Sample clock gaps using bracket method to minimize measurement overhead.

        For each sample, reads monotonic -> realtime -> monotonic and
        monotonic_raw -> realtime -> monotonic_raw, then uses the midpoint
        of the two bracket readings to cancel symmetric overhead.
        Returns the sample with the smallest spread (lowest overhead).

        Args:
            sample_count: Number of samples to take.

        Returns:
            Tuple of (realtime_monotonic_gap, realtime_monotonic_raw_gap).
        """
        best_mono_spread = None
        best_mono_gap = None
        best_raw_spread = None
        best_raw_gap = None

        for _ in range(sample_count):
            # Sample realtime-monotonic gap
            mono_1 = time.monotonic_ns()
            rt = time.time_ns()
            mono_2 = time.monotonic_ns()
            spread = mono_2 - mono_1
            mono_mid = (mono_1 + mono_2) // 2
            gap = rt - mono_mid
            if best_mono_spread is None or spread < best_mono_spread:
                best_mono_spread = spread
                best_mono_gap = gap

            # Sample realtime-monotonic_raw gap
            raw_1 = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            rt2 = time.time_ns()
            raw_2 = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            spread = raw_2 - raw_1
            raw_mid = (raw_1 + raw_2) // 2
            gap = rt2 - raw_mid
            if best_raw_spread is None or spread < best_raw_spread:
                best_raw_spread = spread
                best_raw_gap = gap

        return best_mono_gap, best_raw_gap

    def _add_align_time_metadata(self) -> bool:
        """Add time alignment metadata to the trace data.

        Calculates timestamps using realtime, monotonic, and monotonic_raw
        clocks, then adds traceInformation with startTimestamp offsets.

        Returns:
            True if successful, False otherwise.
        """
        if self.trace_data is None:
            logger.error("No trace data loaded.")
            return False

        try:
            # Calculate gaps with bracket method to reduce measurement overhead
            (
                realtime_monotonic_gap,
                realtime_monotonic_raw_gap,
            ) = self._sample_clock_gaps()

            # Get baseTimeNanoseconds from trace data
            base_time_ns = int(self.trace_data.get("baseTimeNanoseconds", 0))

            # Find minimum ts from traceEvents
            trace_events = self.trace_data.get("traceEvents", [])
            min_ts = None
            for evt in trace_events:
                ts = evt.get("ts")
                if ts is not None:
                    if min_ts is None or ts < min_ts:
                        min_ts = ts

            if min_ts is None:
                logger.error("No valid ts found in traceEvents.")
                return False

            # Calculate realtime from trace data
            # realtime = min_ts * 1000 + baseTimeNanoseconds
            realtime_start_ns = int(min_ts * 1000) + base_time_ns

            # Calculate monotonic and monotonic_raw timestamps
            monotonic_start_ns = realtime_start_ns - realtime_monotonic_gap
            monotonic_raw_start_ns = realtime_start_ns - realtime_monotonic_raw_gap

            # Build metadata object
            metadata_obj = {
                "traceInformation": {
                    "startTimestamp": monotonic_start_ns,
                    "rawStartTimestamp": monotonic_raw_start_ns,
                    "utcStartTimestamp": realtime_start_ns,
                }
            }
            self.trace_data["metadata"] = metadata_obj
            return True
        except Exception as e:
            logger.error(f"Skip insert time alignment metadata as {e}")
            return False

    def _add_efficiency_counter_events(self) -> bool:
        """Add efficiency counter events to the trace data.

        Returns:
            True if successful, False otherwise.
        """
        if self.trace_data is None:
            logger.error("No trace data loaded.")
            return False

        efficiency_events = []
        device_start_ts = 0
        trace_events = self.trace_data.get("traceEvents", [])

        for evt in trace_events:
            if evt.get("cat") == "Trace" and "MLU Device Tasks" in evt.get("name", ""):
                device_start_ts = evt.get("ts", 0)

            args = evt.get("args", {})
            if (
                evt.get("cat") == "gpu_user_annotation"
                and evt.get("pid") >= consts.EFFICIENCY_PID_BEGIN
                and args.get("theory_time", None) is not None
                and args.get("duration", 0) > 0
            ):
                compute_efficiency, io_efficiency, op_efficiency = self._get_efficiency(
                    args
                )
                self._append_efficiency_counter_events(
                    efficiency_events,
                    evt.get("ts", 0),
                    evt.get("ts", 0) + evt.get("dur", 0),
                    evt.get("pid", 0),
                    evt.get("tid", 0),
                    compute_efficiency,
                    io_efficiency,
                    op_efficiency,
                )

        if efficiency_events:
            # add max value
            self._append_efficiency_counter_events(
                efficiency_events,
                device_start_ts,
                device_start_ts,
                efficiency_events[0]["pid"],
                efficiency_events[0]["tid"],
                100,
                100,
                100,
            )
            efficiency_events.sort(key=lambda x: x["ts"])
            self.trace_data["traceEvents"].extend(efficiency_events)

        return True

    @staticmethod
    def _create_efficiency_counter_event(
        name: str, ts: float, pid: int, tid: int, efficiency: float
    ) -> dict:
        """Create a single efficiency counter event.

        Args:
            name: The name of the counter.
            ts: The timestamp.
            pid: The process ID.
            tid: The thread ID.
            efficiency: The efficiency value.

        Returns:
            The counter event dictionary.
        """
        return {
            "ph": "C",
            "name": name + " Efficiency(%)",
            "ts": ts,
            "pid": pid,
            "tid": tid,
            "args": {"utils": efficiency},
        }

    def _append_efficiency_counter_events(
        self,
        events: list,
        start_ts: float,
        end_ts: float,
        pid: int,
        tid: int,
        compute_efficiency: float,
        io_efficiency: float,
        op_efficiency: float,
    ) -> None:
        """Append efficiency counter events to the events list.

        Args:
            events: The list to append events to.
            start_ts: Start timestamp.
            end_ts: End timestamp.
            pid: Process ID.
            tid: Thread ID.
            compute_efficiency: Compute efficiency value.
            io_efficiency: IO efficiency value.
            op_efficiency: OP efficiency value.
        """
        events.append(
            self._create_efficiency_counter_event(
                "Compute", start_ts, pid, tid, compute_efficiency
            )
        )
        events.append(
            self._create_efficiency_counter_event("Compute", end_ts, pid, tid, 0.0)
        )
        events.append(
            self._create_efficiency_counter_event(
                "IO", start_ts, pid, tid, io_efficiency
            )
        )
        events.append(
            self._create_efficiency_counter_event("IO", end_ts, pid, tid, 0.0)
        )
        events.append(
            self._create_efficiency_counter_event(
                "OP", start_ts, pid, tid, op_efficiency
            )
        )
        events.append(
            self._create_efficiency_counter_event("OP", end_ts, pid, tid, 0.0)
        )

    @staticmethod
    def _get_efficiency(args: dict) -> tuple:
        """Calculate efficiency values from event args.

        Args:
            args: The event args dictionary.

        Returns:
            Tuple of (compute_efficiency, io_efficiency, op_efficiency).
        """
        duration = args.get("duration")
        theory_time = args.get("theory_time")
        io_time = args.get("io_time")
        tensor_time = args.get("tensor_time")
        vector_time = args.get("vector_time")
        compute_time = args.get("compute_time")

        if compute_time is not None:
            compute_efficiency = compute_time / duration * 100
        else:
            compute_efficiency = (tensor_time + vector_time) / duration * 100
        io_efficiency = io_time / duration * 100
        op_efficiency = theory_time / duration * 100

        return compute_efficiency, io_efficiency, op_efficiency

    def _rebuild_callstacks(self) -> bool:
        """Rebuild callstacks by re-running build_tree without skip_build_tree events.

        MLU profiler adds custom activities (e.g., op_performance_info_activity,
        overlap_activity, communication_activity) that are not real function calls.
        These are marked with ``skip_build_tree`` metadata. Since the current
        PyTorch version does not support skipping these during C++
        build_tree, the parent-child relationships are incorrect, leading to
        wrong "Call stack" fields.

        This method rebuilds the tree excluding skip_build_tree events and
        overwrites the callstacks in the trace data.

        Returns:
            True if successful, False otherwise.
        """
        if self.trace_data is None:
            logger.error("No trace data loaded.")
            return False

        trace_events = self.trace_data.get("traceEvents", [])
        if not trace_events:
            return True

        # Fast path: if with_stack is not enabled, no callstack to rebuild.
        with_stack = self.trace_data.get("with_stack", 0)
        if with_stack != 1 and with_stack != "1":
            return True

        # with_stack=1 but verbose may be False, in which case "Call stack"
        # is not written to events. Check existence before building tree.
        has_callstack = any(
            isinstance(evt.get("args"), dict) and "Call stack" in evt.get("args", {})
            for evt in trace_events
        )
        if not has_callstack:
            return True

        rebuilder = CallStackRebuilder(trace_events)
        rebuilder.build_tree()
        include_duration = utils.get_profiler_config().enable_stack_with_duration
        rebuilder.overwrite_callstacks(trace_events, include_duration)
        return True

    def process(self) -> bool:
        """Load, modify, and save the trace JSON file.

        This is the main entry point that performs all operations in sequence:
        load the file, apply enabled modifications, and save the result.

        Returns:
            True if all operations successful, False otherwise.
        """
        if not self.load():
            return False

        # Time alignment metadata is always added
        self._add_align_time_metadata()

        # Rebuild callstacks to exclude skip_build_tree events
        self._rebuild_callstacks()

        # Efficiency counter events are added based on environment variable
        if self._enable_efficiency_counter_event:
            self._add_efficiency_counter_events()

        return self.save()


def modify_trace_json(json_path: str) -> bool:
    """Modify a trace JSON file with time alignment and efficiency events.

    Args:
        json_path: Path to the trace JSON file.

    Returns:
        True if successful, False otherwise.
    """
    modifier = TraceJsonModifier(json_path)
    return modifier.process()
