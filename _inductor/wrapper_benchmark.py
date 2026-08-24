import torch
from torch._inductor import wrapper_benchmark
from torch._inductor.wrapper_benchmark import (
    BenchmarkCallableType,
    parse_profile_event_list,
    PROFILE_PATH,
)

from ..utils import gorilla


def wrapper_benchmark_parse_profile_event_list(
    benchmark_name: str,
    event_list: torch.autograd.profiler_util.EventList,
    wall_time_ms: float,
    nruns: int,
    device_name: str,
) -> None:
    """
    Parse and generate a report for an event_list.
    """

    def get_self_device_time(
        ev: torch.autograd.profiler_util.EventList,
    ) -> float:
        """
        ev.self_device_time_total is in microsecond. Convert to millisecond.
        """
        return ev.self_device_time_total / 1000 / nruns  # type: ignore[attr-defined]

    all_events: dict[str, list[ProfileEvent]] = defaultdict(list)

    def add_event(
        ev: torch.autograd.profiler_util.EventList,
        category: str,
    ) -> None:
        profile_ev = ProfileEvent(
            category=category,
            key=ev.key,  # type: ignore[attr-defined]
            self_device_time_ms=get_self_device_time(ev),
            count=ev.count / nruns,  # type: ignore[operator] # average across all runs
        )
        all_events[category].append(profile_ev)

    for ev in event_list:
        assert not ev.is_legacy, "Don't support the legacy profiler"
        # Modify by CAMBRICON
        # if ev.device_type == DeviceType.CPU:
        if ev.device_type == DeviceType.CPU or ev.is_user_annotation:
            # end Modify by CAMBRICON
            # ignore the event on CPU side
            continue

        category = "unknown"
        if ev.key.startswith("triton_"):
            if ev.key.startswith("triton_poi"):
                category = "triton_pointwise"
            elif ev.key.startswith("triton_red"):
                category = "triton_reduction"
            elif ev.key.startswith("triton_per"):
                category = "triton_persistent_reduction"
            else:
                category = "triton_unknown"

        add_event(ev, category)

    def report_category(category: str, profile_events: list[ProfileEvent]) -> float:
        if not device_name:
            return 0.0

        from tabulate import tabulate

        profile_events.sort(key=lambda ev: ev.self_device_time_ms, reverse=True)

        rows = []
        total_time = 0.0
        print(f"\n  == {category} category kernels == ")
        for ev in profile_events:
            total_time += ev.self_device_time_ms
            percent = f"{ev.self_device_time_ms / wall_time_ms * 100:.2f}%"
            rows.append([ev.key[:120], ev.self_device_time_ms, ev.count, percent])
        rows.append(
            ["Total", total_time, "", f"{total_time / wall_time_ms * 100:.2f}%"]
        )
        print(
            tabulate(
                rows,
                headers=[
                    "Kernel",
                    f"Self {device_name.upper()} TIME (ms)",
                    "Count",
                    "Percent",
                ],
            )
        )
        return total_time

    def report() -> None:
        category_list = [
            "triton_pointwise",
            "triton_reduction",
            "triton_persistent_reduction",
            "triton_unknown",
            "unknown",
        ]
        assert OrderedSet(all_events.keys()).issubset(
            OrderedSet(category_list)
        ), f"{list(all_events.keys())}"

        per_category_wall_time = {}
        total_device_ms = 0.0
        for category in category_list:
            if category in all_events:
                _time = report_category(category, all_events[category])
                per_category_wall_time[category] = _time
                total_device_ms += _time

        device_busy_percent = f"{total_device_ms / wall_time_ms * 100:.2f}%"
        if device_name:
            print(
                f"\nPercent of time when {device_name.upper()} is busy: {device_busy_percent}"
            )
        else:
            print("No device detected")

        print(f"Total wall time {wall_time_ms:.3f} ms")

        # output such a line so we can gather such line from all compiled modules from all
        # benchmarks and tabulate it!
        # Columns: benchmark_name, pointwise_percent, reduction_percent, persistent_reduction_percent,
        #   unknown_category_percent, device_busy_percent, wall_time_ms
        tabulate_line = f"Output for tabulate: {benchmark_name}"
        for category in category_list:
            percent = (
                f"{per_category_wall_time.get(category, 0.0) / wall_time_ms * 100:.2f}%"
            )
            tabulate_line += f", {percent}"
        tabulate_line += f", {device_busy_percent}, {wall_time_ms:.3f}ms"

        print(tabulate_line)

    report()


patch = gorilla.Patch(
    wrapper_benchmark,
    "parse_profile_event_list",
    wrapper_benchmark_parse_profile_event_list,
)

gorilla.apply(patch)


def perf_profile(
    wall_time_ms: float,
    times: int,
    repeat: int,
    benchmark_name: str,
    benchmark_compiled_module_fn: BenchmarkCallableType,
) -> None:
    with torch.profiler.profile(record_shapes=True) as p:
        benchmark_compiled_module_fn(times=times, repeat=repeat)

    path = PROFILE_PATH
    p.export_chrome_trace(path)
    # Modify by Cambricon
    import torch_mlu
    from torch_mlu.profiler.analysis.api import analyze_data

    try:
        analyze_data(path)
    except Exception as e:
        print(f"Warning: Failed to analyze profiile data: {e}")
    # end Modify by Cambricon
    print(f"Profiling result for a compiled module of benchmark {benchmark_name}:")
    print(f"Chrome trace for the profile is written to {path}")
    event_list = p.key_averages(group_by_input_shape=True)
    print(event_list.table(sort_by="self_device_time_total", row_limit=10))
    parse_profile_event_list(
        benchmark_name, event_list, wall_time_ms, times * repeat, p.use_device or ""
    )


patch = gorilla.Patch(
    wrapper_benchmark,
    "perf_profile",
    perf_profile,
)

gorilla.apply(patch)
