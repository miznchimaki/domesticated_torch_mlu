"""
@Copyright (C) [2022-2025] by Cambricon.
@File    :   utils.py
"""
import contextlib
from io import StringIO
import torch, triton
from torch.fx.node import Node
from torch._inductor import config
from ..common import (
    NUMTASKSNAME,
    EVENBSBLOCKNAME,
    get_total_core_num,
    get_core_num_per_cluster,
    is_shape_dynamic,
    FORCE_USE_SM,
)
import sympy
from .. import config as tt_config

NUM_WARPS = 1

TORCH_INT_TYPES = [
    torch.int32,
    torch.int64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.bool,
]


def get_tensor(shape, dtype, device="mlu"):
    if dtype in TORCH_INT_TYPES:
        return torch.randint(
            low=0,
            high=10 if dtype != torch.bool else 2,
            size=shape,
            dtype=dtype,
            device=device,
        )
    else:
        return torch.randn(
            shape,
            dtype=dtype,
            device=device,
        )


def get_tensor_strided(shape, strides, dtype, device="mlu"):
    needed_size = sum((shape - 1) * stride for shape, stride in zip(shape, strides)) + 1
    buffer = get_tensor([needed_size], dtype, device)
    return torch.as_strided(buffer, shape, strides)


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent=0):
        self._lines = []
        self._indent = initial_indent

    def getvalue(self) -> str:
        buf = StringIO()
        for line in self._lines:
            assert isinstance(line, str)
            buf.write(line)
            buf.write("\n")
        return buf.getvalue()

    def clear(self):
        self._lines.clear()

    def __bool__(self):
        return bool(self._lines)

    def prefix(self):
        return " " * (self._indent * self.tabwidth)

    def newline(self):
        self.writeline("\n")

    def writeline(self, line=""):
        if line.strip():
            self._lines.append(f"{self.prefix()}{line}")
        else:
            self._lines.append("")

    def writelines(self, lines=[]):
        for line in lines:
            self.writeline(line)

    def indent(self, offset=1):
        @contextlib.contextmanager
        def ctx():
            self._indent += offset
            try:
                yield
            finally:
                self._indent -= offset

        return ctx()


def stride_order(strides):
    strides = [abs(x) for x in strides]
    ids = list(range(len(strides)))
    ids_nozero = [x for x in ids if strides[x] > 0]

    ids_nozero = sorted(ids_nozero, key=lambda i: [strides[i], -i])
    for ind, stri in enumerate(reversed(strides)):
        if stri == 0:
            ids_nozero.insert(ind, len(strides) - ind - 1)
    return ids_nozero


def can_promote_shared(num_tasks, num_warps=NUM_WARPS):
    if num_warps != 1:
        return False
    if tt_config.numtask_align_up % get_core_num_per_cluster() == 0:
        return True
    if is_shape_dynamic(num_tasks):
        return False
    return min(num_tasks, get_total_core_num()) % get_core_num_per_cluster() == 0


def get_triton_inductor_config(num_tasks):
    assert isinstance(
        num_tasks, int
    ), f"get_triton_inductor_config expect target num_tasks is int type, but get {num_tasks}: {type(num_tasks)}"
    configs = [
        {
            "BS_BLOCK": 1,
            "num_warps": NUM_WARPS,
            "num_stages": 1,
            "force_use_shared_memory": FORCE_USE_SM,
            "can_promote_shared": can_promote_shared(num_tasks),
        },
        {
            "BS_BLOCK": 1,
            "num_warps": NUM_WARPS,
            "num_stages": 3,
            "force_use_shared_memory": FORCE_USE_SM,
            "can_promote_shared": can_promote_shared(num_tasks),
        },
    ]

    extend_config = []
    div2mod = True
    tasks_per_core = triton.cdiv(num_tasks, get_total_core_num())
    if tasks_per_core > 1:
        extend_config.append(
            {
                "BS_BLOCK": tasks_per_core,
                "num_stages": 1,
                "num_warps": NUM_WARPS,
            },
        )
        extend_config.append(
            {
                "BS_BLOCK": tasks_per_core,
                "num_stages": 3,
                "num_warps": NUM_WARPS,
            },
        )
    while tasks_per_core > 2:
        extend_config.append(
            {
                "BS_BLOCK": triton.cdiv(tasks_per_core, 2),
                "num_stages": 1,
                "num_warps": NUM_WARPS,
            },
        )
        if tasks_per_core > 3 and tasks_per_core % 3 == 0 and div2mod:
            extend_config.append(
                {
                    "BS_BLOCK": tasks_per_core // 3,
                    "num_stages": 1,
                    "num_warps": NUM_WARPS,
                },
            )
        div2mod = tasks_per_core % 2 == 0
        tasks_per_core = triton.cdiv(tasks_per_core, 2)
    # Other config prune.
    config_all = configs + extend_config
    for config in config_all:
        config["force_use_shared_memory"] = FORCE_USE_SM
        config["can_promote_shared"] = can_promote_shared(
            num_tasks, config["num_warps"]
        )

    return config_all


# args: *shape, kwargs from config generate by triton_fusion_config_prune
def get_triton_inductor_grid(*args, **kwargs):
    dyn_shapes = args[:-1]
    numtask = dyn_shapes[0]
    if tt_config.numtask_align_up > 1:
        numtask = (
            (numtask + tt_config.numtask_align_up - 1)
            // tt_config.numtask_align_up
            * tt_config.numtask_align_up
        )
    minfn = min
    if is_shape_dynamic(numtask):
        minfn = sympy.Min
    return (minfn(get_total_core_num() // NUM_WARPS, numtask), 1, 1)


def get_triton_inductor_grid_fn():
    ret = get_triton_inductor_grid
    try:
        from torch._inductor.select_algorithm import SymbolicGridFn

        ret = SymbolicGridFn(ret)
    except Exception:
        pass
    return ret


def triton_fusion_config_prune(configs, named_args, **kwargs):
    # If pre test perf, ret all configs to get best local triton perf,
    # if only pre check local triton func, first config is enough.
    if (
        not tt_config.load_in_tensors
        and not tt_config.save_in_out_tensors
        and tt_config.pre_check_triton_kernel
        and not tt_config.pre_test_perf_eager
    ):
        return configs[:1]
    num_tasks = named_args["num_tasks"]
    if num_tasks <= get_total_core_num():
        return configs
    extend_config = []
    div2mod = True
    tasks_per_core = triton.cdiv(num_tasks, get_total_core_num())
    if tasks_per_core > 1:
        extend_config.insert(
            0,
            triton.Config(
                {"BS_BLOCK": tasks_per_core}, num_stages=1, num_warps=NUM_WARPS
            ),
        )
        extend_config.insert(
            0,
            triton.Config(
                {"BS_BLOCK": tasks_per_core}, num_stages=3, num_warps=NUM_WARPS
            ),
        )
    while tasks_per_core > 2:
        extend_config.insert(
            0,
            triton.Config(
                {"BS_BLOCK": triton.cdiv(tasks_per_core, 2)},
                num_stages=1,
                num_warps=NUM_WARPS,
            ),
        )
        if tasks_per_core > 3 and tasks_per_core % 3 == 0 and div2mod:
            extend_config.insert(
                0,
                triton.Config(
                    {"BS_BLOCK": tasks_per_core // 3}, num_stages=1, num_warps=NUM_WARPS
                ),
            )
        div2mod = tasks_per_core % 2 == 0
        tasks_per_core = triton.cdiv(tasks_per_core, 2)
    return configs + extend_config


def get_inputs_outputs(targetnodes: list[Node]) -> tuple[list[Node], list[Node]]:
    inputlist = set()
    outputlist = set()
    for node in targetnodes:
        for inp in node.all_input_nodes:
            if inp not in targetnodes:
                inputlist.add(inp)
        for out in list(node.users):
            if out not in targetnodes:
                outputlist.add(node)
                break
    return list(inputlist), list(outputlist)
