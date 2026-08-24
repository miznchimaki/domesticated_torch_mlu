"""Rebuild callstack for profiler trace events.

When MLU profiler adds custom activities (e.g., op_performance_info_activity,
overlap_activity, communication_activity), these activities are not real function
calls but are still included in PyTorch's C++ build_tree process, which
corrupts the parent-child relationships and results in incorrect "Call stack"
fields in the trace JSON.

This module rebuilds the callstack by re-running the build_tree algorithm
while skipping activities marked with "skip_build_tree" metadata, then
recalculating each event's callstack from the corrected tree structure.
"""

from typing import Dict, List, Optional, Set, Tuple

from .common import utils

logger = utils.get_logger()


class _TreeNode:
    """Represents a node in the call tree built from trace events."""

    __slots__ = (
        "event_index",
        "name",
        "cat",
        "start_ts",
        "end_ts",
        "pid",
        "tid",
        "parent",
        "children",
    )

    def __init__(
        self,
        event_index: int,
        name: str,
        cat: str,
        start_ts: float,
        end_ts: float,
        pid: int,
        tid: int,
    ):
        self.event_index = event_index
        self.name = name
        self.cat = cat
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.pid = pid
        self.tid = tid
        self.parent: Optional[_TreeNode] = None
        self.children: List[_TreeNode] = []


class CallStackRebuilder:
    """Rebuilds callstack for trace events by re-running build_tree.

    This class mimics PyTorch's C++ build_tree algorithm
    (torch/csrc/profiler/collection.cpp) but excludes events with
    ``skip_build_tree`` metadata, which represent non-function-call
    annotations added by the MLU profiler.

    It also handles the forward-backward thread relationship: when a
    backward thread event cannot find a parent in its own thread's stack,
    it looks up the paired thread's stack. The pairing is derived from
    ``Sequence number`` — forward and backward events sharing the same
    Sequence number run on different threads, so they identify which two
    JSON tids are a forward-backward pair. Which one is forward is resolved
    lazily during tree building: if the paired thread's stack-top event
    contains the current event, the paired thread is the forward thread.

    After building the tree, it recalculates the "Call stack" field for each
    event by walking the parent chain, matching how KinetoEvent populates
    ``python_stack_`` in ``profiler_kineto.cpp``.

    Usage::

        rebuilder = CallStackRebuilder(trace_events)
        rebuilder.build_tree()
        rebuilder.overwrite_callstacks(trace_events)
    """

    def __init__(self, trace_events: list):
        """Initialize the rebuilder.

        Args:
            trace_events: List of event dicts from the trace JSON's
                ``traceEvents`` array.
        """
        self._trace_events = trace_events
        self._nodes: Dict[int, _TreeNode] = {}
        # Mapping from a JSON tid to the set of paired (forward/backward)
        # JSON tids, derived from shared Sequence numbers.
        self._tid_pairs: Dict[int, Set[int]] = {}

    @staticmethod
    def _is_skip_build_tree(event: dict) -> bool:
        """Check if an event should be skipped during tree building."""
        args = event.get("args")
        if not isinstance(args, dict):
            return False
        val = args.get("skip_build_tree")
        # The C++ side stores it as addMetadata("skip_build_tree", "1"),
        # which produces either an integer 1 or string "1" in JSON.
        return val == 1 or val == "1"

    @staticmethod
    def _is_cpu_op_event(event: dict) -> bool:
        """Check if an event is a CPU-side op that participates in build_tree.

        Only X (complete) events on CPU threads are included, matching the
        C++ build_tree logic which operates on TorchOp and Backend events
        that run on CPU threads.
        """
        if event.get("ph") != "X":
            return False
        cat = event.get("cat", "")
        # CPU ops and python functions participate in the tree.
        return cat in (
            "cpu_op",
            "python_function",
            "user_annotation",
        )

    @staticmethod
    def _get_int_arg(event: dict, key: str, default: int = 0) -> int:
        """Extract an integer value from an event's args."""
        args = event.get("args")
        if not isinstance(args, dict):
            return default
        val = args.get(key, default)
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def _build_tid_pairs(self) -> None:
        """Build the tid pairing from shared Sequence numbers.

        Forward and backward events that correspond to the same autograd
        Node share the same ``Sequence number`` but run on different
        threads (different JSON tids). This method groups events by
        Sequence number and records which tids are paired.

        This does NOT determine which tid is forward or backward — that
        is resolved lazily during tree building by checking containment.
        """
        seq_tids: Dict[int, Set[int]] = {}
        for event in self._trace_events:
            seq_nr = self._get_int_arg(event, "Sequence number", -1)
            if seq_nr < 0:
                continue
            tid = event.get("tid", 0)
            seq_tids.setdefault(seq_nr, set()).add(tid)

        for seq_nr, tids in seq_tids.items():
            if len(tids) < 2:
                continue
            # All tids sharing this Sequence number are pairwise
            # forward-backward candidates.
            for tid in tids:
                self._tid_pairs.setdefault(tid, set()).update(tids - {tid})

    def build_tree(self) -> None:
        """Build the call tree, excluding skip_build_tree events.

        This follows the same algorithm as the C++ build_tree:
        1. Sort all eligible events globally by start time
        2. Maintain a per-tid stack of active events (``stacks``)
        3. When a new event arrives, pop finished events, then find its
           parent: first in its own tid's stack, then in paired tids'
           stacks (checking containment to determine forward direction)
        4. Push the event onto its own tid's stack

        This matches the C++ logic in collection.cpp push_event which first
        looks up ``stacks[event->start_tid_]`` and falls back to
        ``stacks[fwd_tid]`` when forward_tid_ is set.
        """
        # Collect eligible events (CPU ops without skip_build_tree).
        eligible: List[Tuple[int, dict]] = []
        for idx, event in enumerate(self._trace_events):
            if not self._is_cpu_op_event(event):
                continue
            if self._is_skip_build_tree(event):
                continue
            eligible.append((idx, event))

        if not eligible:
            return

        # Build tid pairing from Sequence numbers.
        self._build_tid_pairs()

        # Create tree nodes.
        for idx, event in eligible:
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)
            self._nodes[idx] = _TreeNode(
                event_index=idx,
                name=event.get("name", ""),
                cat=event.get("cat", ""),
                start_ts=ts,
                end_ts=ts + dur,
                pid=event.get("pid", 0),
                tid=event.get("tid", 0),
            )

        # Sort all nodes globally by start time (with larger end time first
        # for ties), matching the C++ stable_sort by start_time_ns_.
        all_nodes = list(self._nodes.values())
        all_nodes.sort(key=lambda n: (n.start_ts, -n.end_ts))

        # Global stack replay, matching C++ build_tree's push_event/pop_event.
        # stacks[tid] -> top-of-stack node for each thread.
        stacks: Dict[int, _TreeNode] = {}

        for node in all_nodes:
            tid = node.tid

            # Pop events from this tid's stack that have ended.
            while tid in stacks and stacks[tid].end_ts <= node.start_ts:
                # Walk up the parent chain to find the nearest ancestor
                # that is still active.
                ancestor = stacks[tid].parent
                while ancestor is not None and ancestor.end_ts <= node.start_ts:
                    ancestor = ancestor.parent
                if ancestor is not None:
                    stacks[tid] = ancestor
                else:
                    del stacks[tid]

            # Find parent: first try own tid's stack.
            parent_node: Optional[_TreeNode] = None
            if tid in stacks:
                parent_node = stacks[tid]
            else:
                # Fallback: try paired tids' stacks. A paired tid that
                # contains this node's time range is the forward thread —
                # this also resolves the forward/backward direction.
                for paired_tid in self._tid_pairs.get(tid, set()):
                    if paired_tid in stacks:
                        candidate = stacks[paired_tid]
                        if (
                            candidate.start_ts <= node.start_ts
                            and candidate.end_ts >= node.end_ts
                        ):
                            parent_node = candidate
                            break

            if parent_node is not None:
                node.parent = parent_node
                parent_node.children.append(node)

            # Push this event onto its own tid's stack.
            if node.end_ts > node.start_ts and node.cat != "user_annotation":
                stacks[tid] = node

    def get_callstack(
        self, event_index: int, include_duration: bool = False
    ) -> Optional[str]:
        """Get the rebuilt callstack for an event.

        The callstack is built by walking up the parent chain and collecting
        parent names, matching how KinetoEvent populates python_stack_ in
        profiler_kineto.cpp (lines 911-916):

            auto parent = result_->parent_.lock();
            while (parent != nullptr) {
                parent->visit_if_base<PyExtraFieldsBase>(
                    [&](const auto&) { python_stack_.push_back(parent->name()); });
                parent = parent->parent_.lock();
            }

        And then serialized as stacksToStr(stack, ";") in AddTensorboardFields.

        Args:
            event_index: Index of the event in the original trace_events list.
            include_duration: If True, append ``duration: [start_ts, end_ts]``
                to each python_function frame in the callstack.

        Returns:
            The callstack string with parent names joined by ";", or None if
            the event is not in the tree (no parents).
        """
        node = self._nodes.get(event_index)
        if node is None:
            return None

        names: List[str] = []
        current = node.parent
        while current is not None:
            if current.cat == "python_function":
                if include_duration:
                    names.append(
                        f"{current.name}, duration: [{current.start_ts},{current.end_ts}]"
                    )
                else:
                    names.append(current.name)
            current = current.parent

        if not names:
            return ""

        return ";".join(names) + ";"

    def overwrite_callstacks(
        self, trace_events: list, include_duration: bool = False
    ) -> int:
        """Overwrite the "Call stack" field in trace events with rebuilt values.

        For each event that has a "Call stack" field in its args, replace
        it with the callstack computed from the rebuilt tree. Events not
        present in the rebuilt tree keep their original "Call stack".

        Args:
            trace_events: The trace events list (modified in-place).
            include_duration: If True, append duration information to each
                python_function frame in the callstack.

        Returns:
            Number of events whose callstack was updated.
        """
        updated = 0
        for idx, event in enumerate(trace_events):
            args = event.get("args")
            if not isinstance(args, dict):
                continue
            if "Call stack" not in args:
                continue

            new_callstack = self.get_callstack(idx, include_duration)
            if new_callstack is not None:
                args["Call stack"] = new_callstack
                updated += 1

        return updated
