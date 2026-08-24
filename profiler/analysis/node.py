# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
import sys
import json
import torch
from abc import ABC
from typing import List, Optional, Tuple, Dict
from enum import IntEnum

from .common import utils
from .trace import (
    DurationEvent,
    EventTypes,
    KernelEvent,
    ModuleEvent,
    OperatorEvent,
    PLProfileEvent,
)
from .data_context import DataContext
from .execution_trace_parser import split_triton_node_inputs_outputs

logger = utils.get_logger()

ExcludeOpName = ["DataParallel.forward", "DistributedDataParallel.forward"]


class BaseNode(ABC):
    def __init__(
        self,
        name: str,
        start_time: float,
        end_time: float,
        type: str,
        tid: int,
        external_id: Optional[int] = None,
        correlation_id: Optional[int] = None,
    ):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.type = type
        self.tid = tid
        self.external_id = external_id  # For consistency check.
        self.correlation_id = correlation_id

    @staticmethod
    def get_node_argument(event: DurationEvent):
        kwargs = {}
        kwargs["name"] = event.name
        kwargs["start_time"] = event.ts
        kwargs["end_time"] = event.ts + event.duration
        kwargs["type"] = event.type
        kwargs["tid"] = event.tid

        external_id = getattr(event, "external_id", None)
        if external_id is not None:
            kwargs["external_id"] = external_id
        correlation_id = getattr(event, "correlation_id", None)
        if correlation_id is not None:
            kwargs["correlation_id"] = correlation_id

        return kwargs

    @property
    def duration(self) -> float:
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        else:
            return 0


class HostNode(BaseNode):
    def __init__(self, device_duration: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.device_duration = device_duration  # Total time of Kernel, GPU Memcpy, GPU Memset. TODO: parallel multi-stream? # noqa: E501


class OperatorNode(HostNode):
    # Don't use [] as default parameters
    # https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument?page=1&tab=votes#tab-top
    # https://web.archive.org/web/20200221224620/http://effbot.org/zone/default-values.htm
    include_output_columns: bool = False

    def __init__(
        self,
        children=None,
        runtimes=None,
        input_shape: Optional[List[List[int]]] = None,
        input_type: Optional[List[str]] = None,
        input_strides: Optional[List[str]] = None,
        callstack: Optional[str] = None,
        self_host_duration: float = 0,
        self_device_duration: float = 0,
        record_function_id: Optional[int] = None,
        is_triton_op: bool = False,
        triton_inputs_num: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.children: List[OperatorNode] = (
            [] if children is None else children
        )  # OperatorNode and ProfilerStepNode.
        self.runtimes: List[RuntimeNode] = (
            [] if runtimes is None else runtimes
        )  # RuntimeNode
        self.parent_node: Optional[
            OperatorNode
        ] = None  # OperatorNode, only used for 'record_param_comms'
        self.input_shape = input_shape
        self.input_type = input_type
        self.input_strides = input_strides
        self.callstack = callstack
        self.self_host_duration = self_host_duration
        self.self_device_duration = self_device_duration
        self.record_function_id = record_function_id
        self.input_values = None
        self.output_shapes = None
        self.output_types = None
        self.output_strides = None
        self.output_values = None
        self.op_schema = None

        self.is_triton_op = is_triton_op
        self.triton_inputs_num = triton_inputs_num

    @classmethod
    def header(cls):
        headers = [
            "Thread Id",
            "Name",
            "Input Shapes",
            "Input Type",
            "Input Strides",
        ]
        if OperatorNode.include_output_columns:
            headers.extend(
                [
                    "Input Values",
                    "Output Shapes",
                    "Output Type",
                    "Output Strides",
                    "Output Values",
                    "Operator Schema",
                ]
            )
        headers.extend(
            [
                "Start Time",
                "External Id",
                "Call Stack",
                "Host Self Duration(us)",
                "Host Total Duration(us)",
                "Device Self Duration(us)",
                "Device Total Duration(us)",
            ]
        )
        return headers

    def data(self):
        data = [
            self.tid,
            self.name,
            self.input_shape,
            self.input_type,
            self.input_strides,
        ]
        if OperatorNode.include_output_columns:
            data.extend(
                [
                    getattr(self, "input_values", []),
                    getattr(self, "output_shapes", []),
                    getattr(self, "output_types", []),
                    getattr(self, "output_strides", []),
                    getattr(self, "output_values", []),
                    getattr(self, "op_schema", None),
                ]
            )
        data.extend(
            [
                self.start_time,
                self.external_id,
                self.callstack,
                round(self.self_host_duration, 3),
                round(self.end_time - self.start_time, 3),
                round(self.self_device_duration, 3),
                round(self.device_duration, 3),
            ]
        )
        return data

    # Merge MLU overlapping RuntimeNodes for calculating self_host_duration.
    # Cnnl api will call cnrt api, so cnnl api duration contain cnrt's.
    # self_host_duration only need to exclude runtime api once.
    def _merge_runtimes(self):
        merged_runtimes = []
        for rt in self.runtimes:
            if rt.name == "dummy":
                continue
            if not merged_runtimes:
                merged_runtimes.append(rt)
            else:
                if (
                    rt.start_time >= merged_runtimes[-1].start_time
                    and rt.end_time <= merged_runtimes[-1].end_time
                ):
                    # Skip contained runtime api.
                    continue
                else:
                    merged_runtimes.append(rt)
        return merged_runtimes

    def fill_stats(self, data_context: Optional[DataContext] = None):
        # TODO: Replace recursive by using a stack, in case of too deep callstack.
        self.children.sort(key=lambda x: (x.start_time, -x.end_time))
        self.runtimes.sort(
            key=lambda x: (x.start_time, -x.end_time)
            if x.start_time and x.end_time
            else (sys.maxsize, -sys.maxsize - 1)
        )

        if data_context is not None and self.record_function_id is not None:
            et_node = data_context.et_data_dict.get(self.record_function_id, None)
            if et_node is not None:
                if self.is_triton_op:
                    split_triton_node_inputs_outputs(et_node, self.triton_inputs_num)
                self.input_shape = et_node.input_shapes
                self.input_type = et_node.input_types
                self.input_strides = et_node.input_strides
                self.input_values = et_node.input_values
                self.output_shapes = et_node.output_shapes
                self.output_types = et_node.output_types
                self.output_strides = et_node.output_strides
                self.output_values = et_node.output_values
                self.op_schema = et_node.op_schema
                OperatorNode.include_output_columns = True

        for child in self.children:
            child.fill_stats(data_context)
        for rt in self.runtimes:
            rt.fill_stats(self, data_context)

        self.self_host_duration = self.end_time - self.start_time
        for child in self.children:
            self.device_duration += child.device_duration
            self.self_host_duration -= child.end_time - child.start_time

        for rt in self.runtimes:
            self.device_duration += rt.device_duration
            self.self_device_duration += rt.device_duration

        for rt in self._merge_runtimes():
            # From PyTorch 1.8 RC1, cpu_self_time does not include runtime's time.
            # So here we keep consistent with it.
            if rt.end_time is not None and rt.start_time is not None:
                self.self_host_duration -= rt.end_time - rt.start_time

    def get_operator_and_kernels(self):
        ops: List[OperatorNode] = []
        kernels: List[DeviceNode] = []
        for child in self.children:
            child_ops, child_kernels = child.get_operator_and_kernels()
            ops.extend(child_ops)
            kernels.extend(child_kernels)
        for rt in self.runtimes:
            kernels.extend(list(rt.get_kernels()))

        if is_operator_node(self):
            ops.append(self)

        return ops, kernels

    @classmethod
    def create(cls, event: OperatorEvent):
        kwargs = BaseNode.get_node_argument(event)
        return cls(
            input_shape=event.input_shape,
            input_type=event.input_type,
            input_strides=event.input_strides,
            callstack=event.callstack,
            record_function_id=getattr(event, "record_function_id", None),
            is_triton_op=event.is_triton_op,
            triton_inputs_num=event.triton_args.num_inputs,
            **kwargs,
        )


class UserAnnotationNode(HostNode):
    def __init__(
        self,
        parent_node: OperatorNode = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.parent_node = parent_node

    def set_parent(self, parent_node):
        self.parent_node = parent_node

    @classmethod
    def create(cls, event: OperatorEvent):
        kwargs = BaseNode.get_node_argument(event)
        return cls(
            parent_node=None,
            **kwargs,
        )


class CommunicationNode(OperatorNode):
    def __init__(
        self,
        rank=None,
        clique_id=None,
        device_id=None,
        comm_bytes=None,
        comm_type: str = None,
        op_name: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.rank = rank
        self.clique_id = clique_id
        self.device_id = device_id
        self.comm_bytes = comm_bytes
        self.comm_type = comm_type
        self.op_name = op_name
        self.bandwidth = (
            round(self.comm_bytes / self.duration, 3) if self.duration != 0 else None
        )

    @classmethod
    def header(cls):
        return [
            "Thread Id",
            "Name",
            "Start",
            "Communication Bytes",
            "Duration(us)",
            "Bandwidth(MB/s)",
            "Rank",
            "Clique Id",
            "Device Id",
            "Communication Type",
            "Operator Name",
        ]

    def data(self):
        return [
            self.tid,
            self.name,
            self.start_time,
            self.comm_bytes,
            self.duration,
            self.bandwidth,
            self.rank,
            self.clique_id,
            self.device_id,
            self.comm_type,
            self.op_name,
        ]

    def set_op_name(self, anno_node: UserAnnotationNode):
        self.op_name = (
            anno_node.parent_node.name if anno_node.parent_node else self.op_name
        )

    @classmethod
    def create(cls, event: OperatorEvent):
        kwargs = BaseNode.get_node_argument(event)
        return cls(
            rank=event.args.get("rank", None),
            clique_id=event.args.get("clique id", None),
            device_id=event.args.get("device id", None),
            comm_bytes=event.args.get("bytes", None),
            comm_type=event.args.get("type", None),
            op_name=event.args.get("op name", None),
            **kwargs,
        )


class ProfilerStepNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ModuleNode(OperatorNode):
    def __init__(self, module_id: int, python_id: int, python_parent_id: int, **kwargs):
        super().__init__(**kwargs)
        self.module_id = module_id
        self.python_id = python_id
        self.python_parent_id = python_parent_id

    def fill_stats(self, data_context: Optional[DataContext] = None):
        super().fill_stats(data_context)
        self.self_device_duration += get_chilren_self_device_time(self)

    @classmethod
    def create(cls, event: ModuleEvent):
        kwargs = BaseNode.get_node_argument(event)
        kwargs["module_id"] = event.module_id
        kwargs["python_id"] = event.python_id
        kwargs["python_parent_id"] = event.python_parent_id
        # From the time being, the ModuleNode always have external_id to 0.
        # As the result, we need reset the external_id to None to ignore adding the runtime nodes for ModuleNode
        kwargs.pop("external_id", None)
        return cls(**kwargs)


class BackwardNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fill_stats(self, data_context: Optional[DataContext] = None):
        """Override the timestamps and duration for BackwardNode only"""
        self.children.sort(key=lambda x: (x.start_time, -x.end_time))
        self.start_time = self.children[0].start_time
        self.end_time = self.children[-1].end_time

        self.self_host_duration = self.end_time - self.start_time
        for child in self.children:
            self.device_duration += child.device_duration
            self.self_host_duration -= child.end_time - child.start_time


class PLProfileNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def create(cls, event: PLProfileEvent):
        kwargs = BaseNode.get_node_argument(event)
        return cls(**kwargs)


class PLModuleNode(OperatorNode):
    def __init__(self, module_id: int, **kwargs):
        super().__init__(**kwargs)
        self.module_id = module_id

    def fill_stats(self, data_context: Optional[DataContext] = None):
        super().fill_stats(data_context)
        self.self_device_duration += get_chilren_self_device_time(self)

    @classmethod
    def create(cls, event: PLProfileEvent):
        kwargs = BaseNode.get_node_argument(event)
        kwargs["module_id"] = event.module_id
        return cls(**kwargs)


class DataLoaderNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class OptimizerNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RuntimeNode(HostNode):
    def __init__(
        self,
        device_nodes: Optional[List["DeviceNode"]] = None,
        parent_rt_node=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # One runtime could trigger more than one kernel, such as cudaLaunchCooperativeKernelMultiDevice.
        self.device_nodes = (
            sorted(device_nodes, key=lambda x: (x.start_time, -x.end_time))
            if device_nodes
            else None
        )
        # add this member just for use of RuntimeNodes with external_id=0, in order to
        # show kernel's op_node name by the topmost rt_node name rather than 'CallTreeRoot'.
        self.parent_rt_node = parent_rt_node

    def fill_stats(
        self, op_node: OperatorNode = None, data_context: Optional[DataContext] = None
    ):
        if self.device_nodes:
            for device_node in self.device_nodes:
                if op_node:
                    if op_node.name == "CallTreeRoot":
                        if device_node.tasktopo_external_op:
                            # for MLUGraph kernels, use op name in capture stage
                            device_node.op_name = device_node.tasktopo_external_op
                            # set op's input information to device node
                            if data_context is not None:
                                linked_op_info = data_context.op_info_dict.get(
                                    device_node.tasktopo_external_id, None
                                )
                                if linked_op_info is not None:
                                    device_node.op_input_shapes = linked_op_info.shapes
                                    device_node.op_input_types = linked_op_info.dtypes
                                    rf_id = linked_op_info.rf_id
                                    et_node = (
                                        data_context.et_data_dict.get(rf_id, None)
                                        if rf_id is not None
                                        else None
                                    )
                                    if et_node is not None:
                                        device_node.op_input_shapes = (
                                            et_node.input_shapes
                                        )
                                        device_node.op_input_types = et_node.input_types
                                        device_node.op_input_strides = (
                                            et_node.input_strides
                                        )
                                        device_node.op_input_values = (
                                            et_node.input_values
                                        )
                                        device_node.op_output_shapes = (
                                            et_node.output_shapes
                                        )
                                        device_node.op_output_types = (
                                            et_node.output_types
                                        )
                                        device_node.op_output_strides = (
                                            et_node.output_strides
                                        )
                                        device_node.op_output_values = (
                                            et_node.output_values
                                        )
                                        device_node.op_schema = et_node.op_schema
                                        DeviceNode.include_output_columns = True
                                else:
                                    # Setting shape to '[]' other None is to align with OperatorEvent's behavior
                                    device_node.op_input_shapes = []
                        else:
                            # use the topmost runtime node's name to instead 'CallTreeRoot'
                            device_node.op_name = self.name
                            parent = self.parent_rt_node
                            while parent is not None:
                                if parent.name == "DummyZeroRuntimeRoot":
                                    break
                                device_node.op_name = parent.name
                                parent = parent.parent_rt_node
                    else:
                        parent_node = op_node
                        if (
                            op_node.name == "record_param_comms"
                            and op_node.parent_node != None
                        ):
                            parent_node = op_node.parent_node
                        device_node.op_name = parent_node.name
                        device_node.op_input_shapes = parent_node.input_shape
                        device_node.op_input_types = parent_node.input_type
                        device_node.op_input_strides = parent_node.input_strides
                        device_node.op_input_values = parent_node.input_values
                        device_node.op_output_shapes = parent_node.output_shapes
                        device_node.op_output_types = parent_node.output_types
                        device_node.op_output_strides = parent_node.output_strides
                        device_node.op_output_values = parent_node.output_values
                        device_node.op_schema = parent_node.op_schema
                        DeviceNode.include_output_columns = (
                            OperatorNode.include_output_columns
                        )
                        if parent_node.callstack:
                            device_node.callstack = parent_node.callstack
                            DeviceNode.include_callstack = True
                device_duration = device_node.end_time - device_node.start_time
                self.device_duration += device_duration

    def get_kernels(self):
        if self.device_nodes:
            for d in self.device_nodes:
                # Include both KERNEL and MEMCPY events as kernels
                if d.type == EventTypes.KERNEL or d.type == EventTypes.MEMCPY:
                    yield d

    @classmethod
    def create(cls, event, device_nodes: Optional[List["DeviceNode"]]):
        kwargs = BaseNode.get_node_argument(event)
        return cls(device_nodes=device_nodes, **kwargs)


class CounterType(IntEnum):
    IPU = 0
    MEMCORE = 1
    BANDWIDTH = 2
    ACTIVE_IPU = 3


class DeviceUtils:
    def __init__(
        self,
        header: str = None,
        counters: List[str] = None,
        counter_type: CounterType = None,
    ):
        self.header: str = header
        # The counter names may vary across different boards,
        # with priority from high to low.
        self.counters: List[str] = counters
        self.counter_type: CounterType = counter_type


class DeviceNode(BaseNode):
    pmu_header: List[str] = []
    utils_list: List[DeviceUtils] = []
    utils_header: List[str] = []
    total_header: List[str] = []
    header_init: bool = False
    include_output_columns: bool = False
    include_callstack: bool = False

    frequency_map = {}
    bandwidth_map = {}
    core_count_map = {}

    def __init__(
        self,
        device_id: int = None,
        stream_id: int = None,
        context_id: int = None,
        kernel_type: str = None,
        dim: Optional[List[int]] = None,
        tasktopo: int = None,
        tasktopo_node: int = None,
        tasktopo_external_id: int = None,
        tasktopo_external_op: str = None,
        pmus: Dict = {},
        active_corenum: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.op_name = None
        self.op_input_shapes = None
        self.op_input_types = None
        self.op_input_strides = None
        self.op_input_values = None
        self.op_output_shapes = None
        self.op_output_types = None
        self.op_output_strides = None
        self.op_output_values = None
        self.op_schema = None
        self.callstack = None
        self.device_id = device_id
        self.stream_id = stream_id
        self.context_id = context_id
        self.kernel_type = kernel_type
        self.dim = dim
        self.tasktopo = tasktopo
        self.tasktopo_node = tasktopo_node
        self.tasktopo_external_id = tasktopo_external_id
        self.tasktopo_external_op = tasktopo_external_op
        self.active_corenum = active_corenum
        self.pmus_data = []
        self.utils_data = []
        self.total_data = []

        if pmus:
            if not DeviceNode.header_init:
                DeviceNode.utils_list = [
                    DeviceUtils("lt_utils(%)", ["tp_core__lt_cycles"], CounterType.IPU),
                    DeviceUtils(
                        "ct_utils(%)",
                        ["tp_core__csimd_post_cycles", "tp_core__vfu_computing_cycles"],
                        CounterType.IPU,
                    ),
                    DeviceUtils(
                        "dram_read_utils(%)",
                        ["tp_core__dram_read_cycles"],
                        CounterType.IPU,
                    ),
                    DeviceUtils(
                        "dram_write_utils(%)",
                        ["tp_core__dram_write_cycles"],
                        CounterType.IPU,
                    ),
                    DeviceUtils(
                        "bandwidth_read_utils(%)",
                        ["tp_cluster__read_bytes"],
                        CounterType.BANDWIDTH,
                    ),
                    DeviceUtils(
                        "bandwidth_write_utils(%)",
                        ["tp_cluster__write_bytes"],
                        CounterType.BANDWIDTH,
                    ),
                    DeviceUtils(
                        "memcore_io_read_utils(%)",
                        ["tp_memcore__dram_read_cycles"],
                        CounterType.MEMCORE,
                    ),
                    DeviceUtils(
                        "memcore_io_write_utils(%)",
                        ["tp_memcore__dram_write_cycles"],
                        CounterType.MEMCORE,
                    ),
                    DeviceUtils(
                        "mv_utils(%)", ["tp_core__mv_inst_cycles"], CounterType.IPU
                    ),
                    DeviceUtils(
                        "alu_utils(%)", ["tp_core__alu_cycles"], CounterType.IPU
                    ),
                    DeviceUtils(
                        "core_efficiency(%)",
                        ["tp_core__working_total_cycles"],
                        CounterType.ACTIVE_IPU,
                    ),
                ]
                DeviceNode.utils_header = [
                    utils.header for utils in DeviceNode.utils_list
                ]
                DeviceNode.utils_header.append("core_utils(%)")
                DeviceNode.total_header = [
                    "total_approximate_bandwidth_bytes",
                    "total_approximate_ipu_cycles",
                    "total_approximate_memcore_cycles",
                ]
                DeviceNode.pmu_header = list(pmus.keys())
                DeviceNode.header_init = True

            if device_id not in DeviceNode.frequency_map:
                device_prop = torch.mlu.get_device_properties(device_id)
                DeviceNode.frequency_map[device_id] = device_prop.ipu_frequency
                DeviceNode.bandwidth_map[device_id] = device_prop.dram_bandwidth
                DeviceNode.core_count_map[device_id] = device_prop.multi_processor_count
            frequency = DeviceNode.frequency_map.get(device_id)  # MHz
            bandwidth = DeviceNode.bandwidth_map.get(device_id)  # Bytes/us
            core_count = DeviceNode.core_count_map.get(device_id)
            duration = self.end_time - self.start_time  # us
            total_cycles_approximation = frequency * duration * core_count
            total_bandwidth_approximation = bandwidth * duration
            total_active_cycles_approximation = frequency * duration * active_corenum

            get_utils = lambda x, y: round(x / y * 100, 3) if y != 0 else None

            for utils in DeviceNode.utils_list:
                counter_names = [
                    counter for counter in utils.counters if counter in pmus
                ]
                counter_name = counter_names[0] if len(counter_names) > 0 else None
                if counter_name in pmus:
                    total = 0
                    if utils.counter_type == CounterType.IPU:
                        total = total_cycles_approximation
                    elif utils.counter_type == CounterType.MEMCORE:
                        total = total_cycles_approximation / 4
                    elif utils.counter_type == CounterType.BANDWIDTH:
                        total = total_bandwidth_approximation
                    elif utils.counter_type == CounterType.ACTIVE_IPU:
                        total = total_active_cycles_approximation
                    self.utils_data.append(get_utils(pmus[counter_name], total))
                else:
                    self.utils_data.append("N/A")
            if active_corenum > 0:
                self.utils_data.append(get_utils(active_corenum, core_count))
            else:
                self.utils_data.append("N/A")

            self.total_data = [
                round(total_bandwidth_approximation, 3),
                round(total_cycles_approximation, 3),
                round(total_cycles_approximation / 4, 3),
            ]
            self.pmus_data = list(pmus.values())

        # for l2_cache.csv
        self.llc_total = pmus.get("llc__tagram_hit", 0) + pmus.get(
            "llc__tagram_miss", 0
        )
        self.hit_rate = (
            round(pmus.get("llc__tagram_hit") / self.llc_total, 3)
            if self.llc_total
            else "N/A"
        )
        self.viction = pmus.get("llc__eviction", "N/A")

    @classmethod
    def header(cls):
        headers = [
            "Thread Id",
            "Correlation Id",
            "Kernel Name",
            "Operator",
            "Operator Input Shapes",
            "Operator Input Type",
            "Operator Input Strides",
        ]
        if DeviceNode.include_output_columns:
            headers.extend(
                [
                    "Operator Input Values",
                    "Operator Output Shapes",
                    "Operator Output Type",
                    "Operator Output Strides",
                    "Operator Output Values",
                    "Operator Schema",
                ]
            )
        headers.extend(
            [
                "Start Time",
                "Duration(us)",
                "External Id",
                "Device Id",
                "Stream Id",
                "Context Id",
                "Kernel Type",
                "Dims",
                "Tasktopo",
                "Tasktopo Node",
            ]
        )
        if DeviceNode.header_init:
            # has pmu
            headers.append("Active Core Number")
        if DeviceNode.include_callstack:
            headers.append("Operator Call Stack")
        result = (
            headers
            + DeviceNode.utils_header
            + DeviceNode.total_header
            + DeviceNode.pmu_header
        )
        return result

    def data(self):
        data = [
            self.tid,
            self.correlation_id,
            self.name,
            self.op_name,
            self.op_input_shapes,
            self.op_input_types,
            self.op_input_strides,
        ]
        if DeviceNode.include_output_columns:
            data.extend(
                [
                    getattr(self, "op_input_values", []),
                    getattr(self, "op_output_shapes", []),
                    getattr(self, "op_output_types", []),
                    getattr(self, "op_output_strides", []),
                    getattr(self, "op_output_values", []),
                    getattr(self, "op_schema", None),
                ]
            )
        data.extend(
            [
                self.start_time,
                round(self.end_time - self.start_time, 3),
                self.external_id,
                self.device_id,
                self.stream_id,
                self.context_id,
                self.kernel_type,
                self.dim,
                self.tasktopo,
                self.tasktopo_node,
            ]
        )
        if DeviceNode.header_init:
            data.append(self.active_corenum)
        if DeviceNode.include_callstack:
            data.append(self.callstack)
        result = data + self.utils_data + self.total_data + self.pmus_data
        if len(result) < len(DeviceNode.header()):
            # In case some PMU counters are missing in memcpy DeviceNode
            result.extend(["N/A"] * (len(DeviceNode.header()) - len(result)))
        return result

    @classmethod
    def l2_cache_header(cls):
        return [
            "Thread Id",
            "Kernel Name",
            "Stream Id",
            "Correlation Id",
            "Hit Rate",
            "llc__eviction",
        ]

    def l2_cache_data(self):
        return [
            self.tid,
            self.name,
            self.stream_id,
            self.correlation_id,
            self.hit_rate,
            self.viction,
        ]

    @classmethod
    def create(cls, event: KernelEvent):
        kwargs = BaseNode.get_node_argument(event)
        # Handle both KERNEL and MEMCPY events
        if event.type in [EventTypes.KERNEL, EventTypes.MEMCPY]:
            kwargs["device_id"] = event.device_id
            kwargs["stream_id"] = event.stream_id
            kwargs["context_id"] = event.context_id
            if event.type == EventTypes.KERNEL:
                # Kernel-specific attributes
                kwargs["kernel_type"] = event.kernel_type
                kwargs["dim"] = event.dim
                kwargs["tasktopo"] = event.tasktopo
                kwargs["tasktopo_node"] = event.tasktopo_node
                kwargs["tasktopo_external_id"] = event.tasktopo_external_id
                kwargs["tasktopo_external_op"] = event.tasktopo_external_op
                kwargs["pmus"] = event.pmus
                kwargs["active_corenum"] = event.active_corenum
        return cls(**kwargs)


def create_operator_node(event: OperatorEvent):
    if (
        event.name.startswith("enumerate(DataLoader)#")
        and event.name.endswith(".__next__")
        or event.name.startswith("enumerate(DataPipe)#")
    ):
        return DataLoaderNode.create(event)
    elif event.name.startswith("Optimizer.step"):
        return OptimizerNode.create(event)
    elif event.type == EventTypes.USER_ANNOTATION:
        # USER_ANNOTATION is just a label, can't regard as OperatorNode.
        return None
    else:
        return OperatorNode.create(event)


def is_operator_node(node: BaseNode):
    return bool(
        type(node) is OperatorNode
        and node.type == EventTypes.OPERATOR
        and node.name not in ExcludeOpName
        and not node.name.startswith("Optimizer.")
    )  # exclude Optimizer.zero_grad


def get_chilren_self_device_time(node):
    self_device_duration = 0
    for child in node.children:
        if is_operator_node(child):
            self_device_duration += child.device_duration
    return self_device_duration
