import os
import argparse

from .op_compare import kernel_compare

# Set the full name for kernel
os.environ["TORCH_MLU_PROFILER_CSV_USE_FULL_NAME"] = "1"


def readArgs():
    parser = argparse.ArgumentParser(description="trace analysis for torch profiler")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="input profiler trace file(s) for trace analysis",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="",
        help="baseline profiler trace file(s) for trace analysis",
    )
    parser.add_argument(
        "--result-dir",
        "--result_dir",
        dest="result_dir",
        type=str,
        default="./result_dir",
        help="Store info to file",
    )
    parser.add_argument(
        "--task-mode",
        type=str,
        choices=["op_compare"],
        default="",
        help="Specify the task mode.",
    )

    args, _ = parser.parse_known_args()

    return args


def main():
    args = readArgs()
    print(args)

    assert args.task_mode, 'task_mode must be set, choices: ["op_compare",]'
    if args.task_mode == "op_compare":
        kernel_compare(args)
    else:
        raise ValueError(f"Unsupported task mode: {args.task_mode}")


if __name__ == "__main__":
    main()
