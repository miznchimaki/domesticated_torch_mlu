"""
MLU Profiler Analysis Tool - Performance Comparison Analyzer
Used to compare operation and kernel performance differences between two torch profiler result files
"""

import os
import json
from typing import Optional, Dict, List, Tuple, Union
from torch_mlu.profiler.analysis.profiler_parser import ProfileData
from torch_mlu.profiler.analysis.common.file_manager import FileManager
from torch_mlu.profiler.analysis.common.path_manager import PathManager
from torch_mlu.profiler.analysis.node import DeviceNode

from .common import validate_json_file

# Header definitions
OP_KERNEL_HEADERS = [
    "top op",
    "op",
    "kernel name",
    "kernel count",
    "dur",
]

OP_KERNEL_WITH_SHAPE_HEADERS = [
    "top op",
    "op",
    "op_input_shapes",
    "op_input_types",
    "kernel name",
    "kernel count",
    "dur",
]

OP_COMPARE_HEADERS = [
    "top op",
    "dev0 top op count",
    "dev1(baseline) top op count",
    "dev0 kernel dur(us)",
    "dev1(baseline) kernel dur(us)",
    "kernel dur diff(us)",
]

OP_COMPARE_WITH_SHAPE_HEADERS = [
    "top op",
    "op",
    "op_input_shapes",
    "op_input_types",
    "dev0 kernel count",
    "dev1(baseline) kernel count",
    "dev0 kernel dur(us)",
    "dev1(baseline) kernel dur(us)",
    "kernel dur diff(us)",
]


class KernelAnalyzer:
    """Kernel performance analyzer class"""

    @staticmethod
    def process_kernel_details(
        details: List[List],
    ) -> Dict[str, List[Union[int, float]]]:
        """
        Process kernel detail data

        Args:
            details: Kernel detail list

        Returns:
            Processed dictionary, key is top op, value is [op count, duration] list
        """
        if not details:
            return {}

        temp_dict = {}

        for item in details:
            try:
                if len(item) < 7:
                    continue
                top_op, top_op_id, *_ = item[:7]
                dur = float(item[6])  # dur is the 7th element

                key = f"{top_op}#{top_op_id}"

                if key in temp_dict:
                    temp_dict[key] += dur
                else:
                    temp_dict[key] = dur
            except (ValueError, TypeError, IndexError):
                continue

        op_post_dict = {}
        for composite_key, dur in temp_dict.items():
            top_op = composite_key.split("#")[0]
            if top_op in op_post_dict:
                op_post_dict[top_op][0] += 1
                op_post_dict[top_op][1] += dur
            else:
                op_post_dict[top_op] = [1, dur]

        return op_post_dict

    @staticmethod
    def process_kernel_post_details(
        details: List[List], enable_input_shape: bool
    ) -> List[List]:
        """
        Process kernel details with shape information

        Args:
            details: Kernel detail list
            enable_input_shape: Whether to enable input shape information

        Returns:
            Processed kernel detail list
        """
        if not details:
            return []

        temp_dict = {}

        for item in details:
            try:
                if len(item) < 7:
                    continue
                top_op, _, op, op_input_shapes, op_input_types, kernel_name, dur = item[
                    :7
                ]
                dur_float = float(dur)

                if enable_input_shape:
                    key = f"{top_op}@{op}@{op_input_shapes}@{op_input_types}@{kernel_name}"
                else:
                    key = f"{top_op}@{op}@{kernel_name}"

                if key in temp_dict:
                    temp_dict[key][1] += dur_float
                    temp_dict[key][0] += 1
                else:
                    temp_dict[key] = [1, dur_float]
            except (ValueError, TypeError, IndexError):
                continue

        result = []
        if enable_input_shape:
            for composite_key, value in temp_dict.items():
                parts = composite_key.split("@")
                if len(parts) >= 5:
                    top_op, op, op_input_shapes, op_input_types, kernel_name = parts[:5]
                    result.append(
                        [
                            top_op,
                            op,
                            op_input_shapes,
                            op_input_types,
                            kernel_name,
                            value[0],
                            value[1],
                        ]
                    )
        else:
            for composite_key, value in temp_dict.items():
                parts = composite_key.split("@")
                if len(parts) >= 3:
                    top_op, op, kernel_name = parts[:3]
                    result.append([top_op, op, kernel_name, value[0], value[1]])

        return result

    @staticmethod
    def compare_dev0_and_dev1_op(
        input_stats: Dict[str, List[Union[int, float]]],
        base_stats: Dict[str, List[Union[int, float]]],
    ) -> Tuple[List[List], float, float]:
        """
        Compare operation statistics between device 0 and device 1

        Args:
            input_stats: Input device statistics
            base_stats: Baseline device statistics

        Returns:
            Comparison result list, device 0 total time, device 1 total time
        """
        profile_list = []
        dev0_total = 0.0
        dev1_total = 0.0
        all_top_ops = set(input_stats.keys()) | set(base_stats.keys())

        for top_op in all_top_ops:
            dev0_count, dev0_dur = input_stats.get(top_op, [0, 0.0])
            dev1_count, dev1_dur = base_stats.get(top_op, [0, 0.0])

            row = [
                top_op,
                dev0_count if top_op in input_stats else "NA",
                dev1_count if top_op in base_stats else "NA",
                dev0_dur if top_op in input_stats else "NA",
                dev1_dur if top_op in base_stats else "NA",
                (dev0_dur - dev1_dur)
                if (top_op in input_stats and top_op in base_stats)
                else "NA",
            ]
            profile_list.append(row)

            dev0_total += dev0_dur if top_op in input_stats else 0
            dev1_total += dev1_dur if top_op in base_stats else 0

        return profile_list, dev0_total, dev1_total

    @staticmethod
    def compare_dev0_and_dev1_op_kernels(
        input_stats: List[List], base_stats: List[List]
    ) -> List[List]:
        """
        Compare kernel statistics between device 0 and device 1

        Args:
            input_stats: Input device kernel statistics
            base_stats: Baseline device kernel statistics

        Returns:
            Comparison result list
        """
        profile_list = []
        temp_dict_dev0 = {}
        temp_dict_dev1 = {}

        # Build mapping for device 0
        for item in input_stats:
            if len(item) < 7:
                continue
            key = f"{item[0]}@{item[1]}@{item[2]}@{item[3]}"
            if key in temp_dict_dev0:
                temp_dict_dev0[key][0] += item[5]
                temp_dict_dev0[key][1] += item[6]
            else:
                temp_dict_dev0[key] = [item[5], item[6]]

        # Build mapping for device 1
        for item in base_stats:
            if len(item) < 7:
                continue
            key = f"{item[0]}@{item[1]}@{item[2]}@{item[3]}"
            if key in temp_dict_dev1:
                temp_dict_dev1[key][0] += item[5]
                temp_dict_dev1[key][1] += item[6]
            else:
                temp_dict_dev1[key] = [item[5], item[6]]

        all_top_ops = set(temp_dict_dev0.keys()) | set(temp_dict_dev1.keys())

        for op_header in all_top_ops:
            dev0_count, dev0_dur = temp_dict_dev0.get(op_header, [0, 0.0])
            dev1_count, dev1_dur = temp_dict_dev1.get(op_header, [0, 0.0])

            parts = op_header.split("@")
            if len(parts) < 4:
                continue
            top_op, op, op_input_shapes, op_input_types = parts[:4]

            row = [
                top_op,
                op,
                op_input_shapes,
                op_input_types,
                dev0_count if op_header in temp_dict_dev0 else "NA",
                dev1_count if op_header in temp_dict_dev1 else "NA",
                dev0_dur if op_header in temp_dict_dev0 else "NA",
                dev1_dur if op_header in temp_dict_dev1 else "NA",
                (dev0_dur - dev1_dur)
                if (op_header in temp_dict_dev0 and op_header in temp_dict_dev1)
                else "NA",
            ]
            profile_list.append(row)

        return profile_list


def kernel_compare(args) -> None:
    """
    Main function: Compare two profiler files

    Args:
        args: Object containing baseline, input, result_dir, enable_comm_op_compare parameters
    """
    base_file = args.baseline
    input_file = args.input
    result_dir = args.result_dir

    # Parameter validation
    if not base_file or not input_file:
        raise ValueError("baseline and input args must be set")

    if not os.path.isfile(base_file):
        raise FileNotFoundError(
            f"baseline arg must be a torch profiler json file: {base_file}"
        )

    if not os.path.isfile(input_file):
        raise FileNotFoundError(
            f"input arg must be a torch profiler json file: {input_file}"
        )

    # Check the json format
    validate_json_file(base_file)
    validate_json_file(input_file)

    # Get torch op and kernel info for input file
    dev0_profile = ProfileData(input_file, enable_dump_csv=False)
    dev0_op_details, dev0_kerners_details = dev0_profile.get_op_details()

    # Get torch op and kernel info for baseline file
    dev1_profile = ProfileData(base_file, enable_dump_csv=False)
    dev1_op_details, dev1_kerners_details = dev1_profile.get_op_details()

    # Post-process baseline and input op info
    analyzer = KernelAnalyzer()
    dev0_op_post_dict = analyzer.process_kernel_details(dev0_op_details)
    dev1_op_post_dict = analyzer.process_kernel_details(dev1_op_details)

    # Compare op info
    (
        compare_profile_list,
        dev0_total_time,
        dev1_total_time,
    ) = analyzer.compare_dev0_and_dev1_op(dev0_op_post_dict, dev1_op_post_dict)

    # Post-process baseline and input kernel info
    dev0_kernel_post_list = analyzer.process_kernel_post_details(dev0_op_details, False)
    dev1_kernel_post_list = analyzer.process_kernel_post_details(dev1_op_details, False)
    dev0_kernel_post_with_shape_list = analyzer.process_kernel_post_details(
        dev0_op_details, True
    )
    dev1_kernel_post_with_shape_list = analyzer.process_kernel_post_details(
        dev1_op_details, True
    )

    # Compare op and kernel info
    compare_kernel_profile_list = analyzer.compare_dev0_and_dev1_op_kernels(
        dev0_kernel_post_with_shape_list, dev1_kernel_post_with_shape_list
    )

    # Generate CSV files
    FileManager.create_csv_file(
        result_dir,
        dev0_kernel_post_with_shape_list,
        "dev0_kernel_details_with_shape.csv",
        OP_KERNEL_WITH_SHAPE_HEADERS,
    )

    FileManager.create_csv_file(
        result_dir,
        dev1_kernel_post_with_shape_list,
        "dev1_kernel_details_with_shape.csv",
        OP_KERNEL_WITH_SHAPE_HEADERS,
    )

    FileManager.create_csv_file(
        result_dir,
        dev0_kernel_post_list,
        "dev0_kernel_details.csv",
        OP_KERNEL_HEADERS,
    )

    FileManager.create_csv_file(
        result_dir,
        dev1_kernel_post_list,
        "dev1_kernel_details.csv",
        OP_KERNEL_HEADERS,
    )

    FileManager.create_csv_file(
        result_dir,
        compare_profile_list,
        "dev0_vs_dev1_op_summary.csv",
        OP_COMPARE_HEADERS,
    )

    FileManager.create_csv_file(
        result_dir,
        compare_kernel_profile_list,
        "dev0_vs_dev1_op_with_shape_summary.csv",
        OP_COMPARE_WITH_SHAPE_HEADERS,
    )

    print("======= dev0 total hardware time(us): ", dev0_total_time)
    print("======= dev1(baseline) total hardware time(us): ", dev1_total_time)
