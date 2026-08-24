from .common import consts


def get_efficiency(args):
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


def get_op_performance_header_and_details(events):
    header = [
        "Name",
        "Start Time",
        "Duration(us)",
        "Shape",
        "Type",
        "Correlation IDs",
        "OP Efficiency(%)",
        "Compute Efficiency(%)",
        "IO Efficiency(%)",
        "Theory Time(us)",
        "Tensor Time(us)",
        "Vector Time(us)",
        "IO Time(us)",
        "Compute Time(us)",
        "Theory IO(bytes)",
        "Tensor Compute(ops)",
        "Vector Compute(ops)",
        "Extra",
    ]
    details = []
    for event in events:
        if (
            hasattr(event, "category")
            and event.category == "gpu_user_annotation"
            and event.pid >= consts.EFFICIENCY_PID_BEGIN
            and event.args.get("theory_time", None) is not None
            and event.args.get("duration", 0) > 0
        ):
            compute_efficiency, io_efficiency, op_efficiency = get_efficiency(
                event.args
            )
            theory_time = event.args.get("theory_time")
            io_time = event.args.get("io_time")
            tensor_time = event.args.get("tensor_time")
            vector_time = event.args.get("vector_time")
            compute_time = event.args.get("compute_time")
            details.append(
                [
                    event.name,
                    event.ts,
                    event.duration,
                    event.args.get("shape"),
                    event.args.get("type"),
                    event.args.get("correlation_event_ids"),
                    op_efficiency,
                    compute_efficiency,
                    io_efficiency,
                    theory_time / 1000,  # ns to us
                    tensor_time / 1000,
                    vector_time / 1000,
                    io_time / 1000,
                    compute_time / 1000 if compute_time is not None else None,
                    event.args.get("theory_io"),
                    event.args.get("tensor_compute"),
                    event.args.get("vector_compute"),
                    event.args.get("extra", ""),
                ]
            )

    return header, details
