import torch
from torch._inductor.codegen.common import (
    device_op_overrides_dict,
    DeviceOpOverrides,
    register_device_op_overrides,
)

from ...utils import gorilla


def get_device_op_overrides(device: str) -> DeviceOpOverrides:
    assert isinstance(device, str), type(device)

    if not device_op_overrides_dict:
        # Modify by CAMBRICON
        from . import cpu_device_op_overrides, mps_device_op_overrides  # noqa: F401

        # from .cuda import device_op_overrides  # noqa: F401
        # from .mtia import device_op_overrides as mtia_op_overrides  # noqa: F401
        # from .xpu import device_op_overrides as xpu_op_overrides  # noqa: F401
        from torch_mlu._inductor.codegen import device_op_overrides  # noqa: F401

        # end Modify by CAMBRICON

    if device not in device_op_overrides_dict:
        # For backends like TPU that only need no-op overrides (Pallas handles codegen)
        from .cpu_device_op_overrides import CpuDeviceOpOverrides

        register_device_op_overrides(device, CpuDeviceOpOverrides())

    return device_op_overrides_dict[device]


patch = gorilla.Patch(
    torch._inductor.codegen.common,
    "get_device_op_overrides",
    get_device_op_overrides,
)
gorilla.apply(patch)
