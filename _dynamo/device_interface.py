from typing import Callable, Iterable, Optional, Tuple, Type, Union

import torch
from torch._dynamo.device_interface import (
    caching_worker_current_devices,
    caching_worker_device_properties,
    CpuInterface,
    CudaInterface,
    DeviceInterface,
    MpsInterface,
    register_interface_for_device,
    XpuInterface,
)
from ..utils import gorilla

get_mlu_stream: Optional[Callable[[int], int]]
if torch.mlu._is_compiled():
    from torch_mlu._MLUC import _mlu_getCurrentRawStream as get_mlu_stream
else:
    get_mlu_stream = None

_device_initialized = False


class MluInterface(DeviceInterface):
    device = torch.mlu.device
    Event = torch.mlu.Event
    Stream = torch.mlu.Stream

    class Worker:
        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices["mlu"] = device

        @staticmethod
        def current_device() -> int:
            if "mlu" in caching_worker_current_devices:
                return caching_worker_current_devices["mlu"]
            return torch.mlu.current_device()

        @staticmethod
        def get_device_properties(device: torch.types.Device = None):
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == "mlu"
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = MluInterface.Worker.current_device()

            if "mlu" not in caching_worker_device_properties:
                device_prop = [
                    torch.mlu.get_device_properties(i)
                    for i in range(torch.mlu.device_count())
                ]
                caching_worker_device_properties["mlu"] = device_prop

            return caching_worker_device_properties["mlu"][device]

    current_device = staticmethod(torch.mlu.current_device)
    set_device = staticmethod(torch.mlu.set_device)
    device_count = staticmethod(torch.mlu.device_count)
    stream = staticmethod(torch.mlu.stream)  # type: ignore[assignment]
    current_stream = staticmethod(torch.mlu.current_stream)
    set_stream = staticmethod(torch.mlu.set_stream)  # type: ignore[assignment]
    # _set_stream_by_id = staticmethod(torch.mlu._set_stream_by_id)  # type: ignore[assignment]
    synchronize = staticmethod(torch.mlu.synchronize)
    get_device_properties = staticmethod(torch.mlu.get_device_properties)  # type: ignore[assignment]
    get_raw_stream = staticmethod(get_mlu_stream)  # type: ignore[arg-type]
    exchange_device = staticmethod(torch.mlu._exchange_device)  # type: ignore[arg-type]
    maybe_exchange_device = staticmethod(torch.mlu._maybe_exchange_device)  # type: ignore[arg-type]
    memory_allocated = staticmethod(torch.mlu.memory_allocated)
    is_bf16_supported = staticmethod(torch.mlu.is_bf16_supported)  # type: ignore[arg-type]

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return torch.mlu.is_available()

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None):
        # Triton MLU depends on the isa_version parameter. Reuse this
        # interface to pass the isa_version parameter to Triton, thereby
        # reducing the number of monkey patches. When Triton MLU removes
        # its dependency on isa_version, revert this change.
        return torch.mlu.get_device_properties(device).isa_version


@gorilla.patch(torch._dynamo.device_interface)
def init_device_reg() -> None:
    global _device_initialized
    register_interface_for_device("cuda", CudaInterface)
    for i in range(torch.cuda.device_count()):
        register_interface_for_device(f"cuda:{i}", CudaInterface)

    register_interface_for_device("xpu", XpuInterface)
    for i in range(torch.xpu.device_count()):
        register_interface_for_device(f"xpu:{i}", XpuInterface)

    register_interface_for_device("mtia", MtiaInterface)
    for i in range(torch.mtia.device_count()):
        register_interface_for_device(f"mtia:{i}", MtiaInterface)

    register_interface_for_device("cpu", CpuInterface)
    register_interface_for_device("mps", MpsInterface)

    # Add by CAMBRICON
    import torch_mlu
    from torch_mlu._dynamo.device_interface import MluInterface

    register_interface_for_device("mlu", MluInterface)
    for i in range(torch.mlu.device_count()):
        register_interface_for_device(f"mlu:{i}", MluInterface)
    # end Add by CAMBRICON
    _device_initialized = True
