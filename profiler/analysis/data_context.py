from dataclasses import dataclass, field
from typing import Dict, List

from .execution_trace_parser import ETNode


@dataclass
class OpInfo:
    shapes: List
    dtypes: List
    rf_id: int


@dataclass
class DataContext:
    # external id -> OpInfo
    op_info_dict: Dict[int, OpInfo] = field(default_factory=dict)

    # list of execution trace file paths
    execution_trace_files: List[str] = field(default_factory=list)

    # rf id -> ETNode
    et_data_dict: Dict[int, ETNode] = field(default_factory=dict)
