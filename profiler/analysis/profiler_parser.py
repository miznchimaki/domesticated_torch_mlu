import os
import json
import re
import math
from typing import Dict, List, Optional

from . import trace
from .event_parser import EventParser
from .kernel_parser import KernelParser
from .memory_parser import MemoryParser, MemorySnapshot, MemoryRecord
from .node import OperatorNode, DeviceNode, CommunicationNode, UserAnnotationNode
from .trace import BaseEvent, EventTypes, MemoryEvent, DeviceType, OperatorEvent
from .execution_trace_parser import ExecutionTraceParser
from .data_context import DataContext
from .op_performance_parser import get_op_performance_header_and_details

from .common import utils
from .common import consts
from .common.file_manager import FileManager
from .common.path_manager import PathManager

logger = utils.get_logger()

WORKER_PATTERN = re.compile(
    r""".*/(.*?) # worker name
        \.(\d+)? # optional timestamp like 1619499959628 used as span name
        \.pt\.trace\.json # the ending suffix
        (?:\.gz)?$""",
    re.X,
)  # optional .gz extension


class ProfileData:
    def __init__(
        self,
        file_path: str,
        data_context: Optional[DataContext] = None,
        enable_dump_csv: Optional[bool] = None,
    ):
        # metadatas
        self.file_path = file_path
        self.data_context = data_context
        self.raw_data_path = os.path.dirname(os.path.abspath(file_path))
        self.trace_json = utils.load_json_file(file_path)

        self.et_trace_jsons = []
        if data_context:
            for jf in data_context.execution_trace_files:
                if os.path.exists(jf):
                    et_trace_json = utils.load_json_file(jf)
                    if et_trace_json:
                        self.et_trace_jsons.append(et_trace_json)

        config = utils.get_profiler_config()
        self.enable_dump_csv = (
            config.enable_dump_csv if enable_dump_csv is None else enable_dump_csv
        )
        self.enable_dump_triton_code = config.enable_dump_triton_code

        self.external_id_to_triton_ops: Dict[int, OperatorEvent] = {}

        if self.enable_dump_csv:
            self.output_dir = self._create_output_dir(file_path)

        self.events: List[BaseEvent] = []
        self.all_kernels: List[DeviceNode] = []

        if self.trace_json is not None:
            trace_body = self.trace_json["traceEvents"]
            fwd_bwd_events = []
            for data in trace_body:
                # discard fwdbwd flow event
                if (
                    data.get("cat") != "fwdbwd"
                    and data.get("cat") != "forward_backward"
                ):
                    event = trace.create_event(data)
                    if event is not None:
                        self.events.append(event)
                        is_triton_op = getattr(event, "is_triton_op", False)
                        if is_triton_op:
                            self.external_id_to_triton_ops.update(
                                {event.external_id: event}
                            )

            self.events.sort(key=lambda e: e.ts)

        # Event Parser results
        self.tid2tree: Dict[int, OperatorNode] = None
        self.pl_tid2tree: Dict[int, OperatorNode] = None
        self.memory_snapshot: Optional[MemorySnapshot] = None
        self.comm_node_list: List[CommunicationNode] = None
        self.external_id_to_anno: Dict[int, UserAnnotationNode] = None

    def _create_output_dir(self, file_path: str):
        match = WORKER_PATTERN.match(file_path)
        worker = match.group(1) if match else "default_worker"
        span = match.group(2) if match else "default_span"
        output_dir = os.path.join(
            self.raw_data_path, consts.CAMBRICON_OUTPUT_DIR_NAME, f"{worker}-{span}"
        )
        PathManager.remove_path_safety(output_dir)
        PathManager.make_dir_safety(output_dir)
        return output_dir

    def process(self):
        if self.et_trace_jsons:
            et_trace_parser = ExecutionTraceParser(self.et_trace_jsons)
            et_trace_parser._parse_nodes()
            self.data_context.et_data_dict.update(et_trace_parser.get_rf_id_mapping())
        parser = EventParser()
        (
            self.tid2tree,
            self.pl_tid2tree,
            self.comm_node_list,
            self.external_id_to_anno,
        ) = parser.parse(self.events, self.data_context)

        if self.enable_dump_csv:
            self._save_csv_files()

        if self.enable_dump_triton_code:
            self._save_triton_code_to_json()

    def _save_csv_files(self):
        ops_details = []
        kerners_details = []
        l2_cache = []
        for _, root_node in self.tid2tree.items():
            ops, kernels = root_node.get_operator_and_kernels()
            self.all_kernels.extend(kernels)
            for op in ops:
                ops_details.append(op.data())
            for kernel in kernels:
                kerners_details.append(kernel.data())
                l2_cache.append(kernel.l2_cache_data())

        comm_details = []
        for comm_node in self.comm_node_list:
            if comm_node.external_id in self.external_id_to_anno.keys():
                comm_node.set_op_name(self.external_id_to_anno[comm_node.external_id])
                # else use default name
            comm_details.append(comm_node.data())

        (
            op_performance_header,
            op_performance_details,
        ) = get_op_performance_header_and_details(self.events)

        if op_performance_details:
            FileManager.create_csv_file(
                self.output_dir,
                op_performance_details,
                consts.OP_PERFORMANCE_FILE_NAME,
                op_performance_header,
            )

        if ops_details:
            FileManager.create_csv_file(
                self.output_dir,
                ops_details,
                consts.OPERATOR_DETAILS_FILE_NAME,
                OperatorNode.header(),
            )
        if comm_details:
            FileManager.create_csv_file(
                self.output_dir,
                comm_details,
                consts.COMM_DETAILS_FILE_NAME,
                CommunicationNode.header(),
            )
        if kerners_details:
            FileManager.create_csv_file(
                self.output_dir,
                kerners_details,
                consts.KERNEL_DETAILS_FILE_NAME,
                DeviceNode.header(),
            )
        if l2_cache:
            FileManager.create_csv_file(
                self.output_dir,
                l2_cache,
                consts.L2CACHE_FILE_NAME,
                DeviceNode.l2_cache_header(),
            )

        if self.all_kernels:
            kernel_parser = KernelParser()
            kernel_parser.parse_events(self.all_kernels)
            FileManager.create_csv_file(
                self.output_dir,
                kernel_parser.get_kernel_statistic(),
                consts.KERNEL_STATISTIC_FILE_NAME,
                kernel_parser.kernel_header,
            )

            FileManager.create_csv_file(
                self.output_dir,
                kernel_parser.get_op_kernel_statistic(),
                consts.OP_KERNEL_STATISTIC_FILE_NAME,
                kernel_parser.op_kernel_header,
            )

        memory_events = self._memory_events()
        if memory_events:
            memory_parser = MemoryParser(memory_events)
            self.memory_snapshot = memory_parser.find_memory_nodes(self.tid2tree)

            mlu_memory_records = []
            op_mem_events = []
            alloc = {}
            free = {}
            prev_ts = float("-inf")  # ensure ordered memory records is ordered
            OP_MEM_HEADER = [
                "Operator Name",
                "Size(KB)",
                "Allocation Time",
                "Release Time",
                "Duration(us)",
                "Address",
                "Device Type",
            ]
            for idx, record in enumerate(self.memory_snapshot.memory_records):
                # Only record MLU memory data
                if record.device_type is DeviceType.MLU:
                    mlu_memory_records.append(record.data())

                    # gen data for operator_memory.csv
                    assert prev_ts <= record.ts
                    prev_ts = record.ts
                    addr = record.addr
                    size = record.bytes
                    if record.is_allocation:
                        alloc[addr] = idx
                    else:
                        if addr in alloc:
                            alloc_record = self.memory_snapshot.memory_records[
                                alloc[addr]
                            ]
                            alloc_ts = alloc_record.ts
                            free_ts = record.ts
                            op_mem_events.append(
                                [
                                    alloc_record.full_op_name_or_unknow,  # op name
                                    round(
                                        -size / 1024.0, 3
                                    ),  # free record size is negative
                                    alloc_ts,
                                    free_ts,
                                    round(free_ts - alloc_ts, 3),  # Duration
                                    addr,
                                    alloc_record.device_name,
                                ]
                            )
                            del alloc[addr]
                        else:
                            if addr in free:
                                logger.error(f"Address {addr} is freed multiple times")
                            free[addr] = idx

            for i in alloc.values():
                r = self.memory_snapshot.memory_records[i]
                op_mem_events.append(
                    [
                        r.full_op_name_or_unknow,  # op name
                        round(r.bytes / 1024.0, 3),
                        r.ts,
                        None,
                        None,
                        r.addr,
                        r.device_name,
                    ]
                )

            for i in free.values():
                r = self.memory_snapshot.memory_records[i]
                op_mem_events.append(
                    [
                        r.full_op_name_or_unknow,  # op name
                        round(-r.bytes / 1024.0, 3),
                        None,
                        r.ts,
                        None,
                        r.addr,
                        r.device_name,
                    ]
                )

            if mlu_memory_records:
                FileManager.create_csv_file(
                    self.output_dir,
                    mlu_memory_records,
                    consts.MEMORY_RECORD_FILE_NAME,
                    MemoryRecord.header(),
                )

            if op_mem_events:
                FileManager.create_csv_file(
                    self.output_dir,
                    op_mem_events,
                    consts.OPERATOR_MEMORY_FILE_NAME,
                    OP_MEM_HEADER,
                )

        OperatorNode.include_output_columns = False
        DeviceNode.include_output_columns = False

    def get_op_details(self):
        parser = EventParser()
        (
            self.tid2tree,
            self.pl_tid2tree,
            self.comm_node_list,
            self.external_id_to_anno,
        ) = parser.parse(self.events, self.data_context)

        def get_top_cpu_op(nodes):
            stack = list(nodes)
            while stack:
                node = stack.pop()
                if node.type == "Operator":
                    yield node
                elif hasattr(node, "children"):
                    stack.extend(node.children)

        ops_details = []
        kerners_details = []
        op_count_id = 0
        for _, root_node in self.tid2tree.items():
            _, kernels_origin = root_node.get_operator_and_kernels()
            for kernel in kernels_origin:
                kerners_details.append(kernel.data())

            for op in get_top_cpu_op(root_node.children):
                name = op.name
                kernels = op.get_operator_and_kernels()[1]
                if kernels:
                    for kernel in kernels:
                        kernel_name = kernel.name
                        duration = kernel.duration
                        # Excluding communication operators.
                        if not "c10d::" in kernel.op_name:
                            ops_details.append(
                                [
                                    name,
                                    op_count_id,
                                    kernel.op_name,
                                    kernel.op_input_shapes,
                                    kernel.op_input_types,
                                    kernel_name,
                                    duration,
                                ]
                            )
                    op_count_id += 1
        return ops_details, kerners_details

    def _memory_events(self) -> List[MemoryEvent]:
        memory_events = [e for e in self.events if e.type == EventTypes.MEMORY]
        memory_events.sort(key=lambda e: e.ts)
        return memory_events

    def _save_triton_code_to_json(self):
        def is_float(fstr):
            try:
                float(fstr)
                return True
            except (ValueError, TypeError):
                return False

        def get_kenrel_num_gb_from_code(kernel_code):
            kernel_num_pattern = (
                r"'kernel_num_gb':\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)"
            )
            match = re.search(kernel_num_pattern, kernel_code)
            if match and is_float(match.group(1)):
                kernel_num_gb = float(match.group(1))
                return kernel_num_gb
            return None

        def get_kernel_num_gb(kernel_code, input_dims, input_types):
            if not input_dims or not input_types:
                return get_kenrel_num_gb_from_code(kernel_code)

            tensor_idx = [i for i, v in enumerate(input_types) if "Scalar" not in v]
            tensor_input_types = []
            tensor_input_dims = []
            for i in tensor_idx:
                tensor_input_types.append(input_types[i])
                tensor_input_dims.append(input_dims[i])
            type_size_list = [
                consts.TYPE_SIZE_MAP.get(dtype, 1) for dtype in tensor_input_types
            ]
            input_size_list = [math.prod(shape) for shape in tensor_input_dims]
            if len(type_size_list) != len(input_size_list):
                return get_kenrel_num_gb_from_code(kernel_code)
            total_size = 0
            for i in range(len(type_size_list)):
                total_size += type_size_list[i] * input_size_list[i]
            kernel_num_gb = total_size / (1000**3)
            return kernel_num_gb

        if self.external_id_to_triton_ops:
            trace_json_updated = False
            triton_kernel_external_ids = self.external_id_to_triton_ops.keys()
            for evt in self.trace_json.get("traceEvents"):
                if evt.get("cat") == "kernel":
                    eid = evt.get("args").get("External id")
                    if eid in triton_kernel_external_ids:
                        # triton output code
                        kernel_file = self.external_id_to_triton_ops[
                            eid
                        ].triton_args.kernel_file
                        if kernel_file and os.path.exists(kernel_file):
                            kernel_code = open(
                                kernel_file, "r", encoding="utf-8"
                            ).read()
                            evt["args"]["triton output code"] = kernel_code

                            # kernel_num_gb & io efficiency
                            input_dims = self.external_id_to_triton_ops[eid].input_shape
                            input_types = self.external_id_to_triton_ops[eid].input_type
                            kernel_num_gb = get_kernel_num_gb(
                                kernel_code, input_dims, input_types
                            )
                            if kernel_num_gb:
                                dur_s = evt.get("dur") / 1000 / 1000
                                evt["args"]["kernel num(GB)"] = kernel_num_gb
                                evt["args"]["IO efficiency(GB/s)"] = (
                                    kernel_num_gb / dur_s
                                )
                            trace_json_updated = True

                        forward_to_kernel_args = [
                            "kernel_kwargs",
                            "num_stages",
                            "kernel_flop",
                        ]
                        for arg_name in forward_to_kernel_args:
                            arg_value = getattr(
                                self.external_id_to_triton_ops[eid].triton_args,
                                arg_name,
                            )
                            if arg_value:
                                evt["args"][arg_name.replace("_", " ")] = arg_value
                                trace_json_updated = True

            if trace_json_updated:
                with open(self.file_path, "w") as f:
                    json.dump(self.trace_json, f, indent=2)
