import functools

import torch
from torch.utils._triton import has_triton_package
from . import gorilla


@gorilla.patch(torch.utils._triton)
@functools.cache
def has_triton() -> bool:
    # Modify by CAMBRICON
    # if not has_triton_package():
    #     return False

    # from torch._inductor.config import triton_disable_device_detection

    # if triton_disable_device_detection:
    #     return False

    # from torch._dynamo.device_interface import get_interface_for_device

    # def cuda_extra_check(device_interface: Any) -> bool:
    #     return device_interface.Worker.get_device_properties().major >= 7

    # def cpu_extra_check(device_interface: Any) -> bool:
    #     import triton.backends

    #     return "cpu" in triton.backends.backends

    # def _return_true(device_interface: Any) -> bool:
    #     return True

    # triton_supported_devices = {
    #     "cuda": cuda_extra_check,
    #     "xpu": _return_true,
    #     "cpu": cpu_extra_check,
    #     "mtia": _return_true,
    # }

    # def is_device_compatible_with_triton() -> bool:
    #     for device, extra_check in triton_supported_devices.items():
    #         device_interface = get_interface_for_device(device)
    #         if device_interface.is_available() and extra_check(device_interface):
    #             return True
    #     return False

    # return is_device_compatible_with_triton()
    return has_triton_package()
    # end Modify by CAMBRICON
