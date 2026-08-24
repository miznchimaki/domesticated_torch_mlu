# mypy: allow-untyped-defs
import torch


def get_device(args, kwargs):
    if kwargs.get("device"):
        device = kwargs.get("device")
        if isinstance(device, str):
            device = torch.device(device)
        return device.type

    devices = {arg.device.type for arg in args if isinstance(arg, torch.Tensor)}
    # Modify by CAMBRICON
    # if any(dev == "cuda" for dev in devices):
    #    return "cuda"
    if any(dev == "cuda" or dev == "mlu" for dev in devices):
        return "cuda"
    # end Modify by CAMBRICON
    elif any(dev == "xpu" for dev in devices):
        return "xpu"
    elif any(dev == "hpu" for dev in devices):
        return "hpu"
    elif any(dev == "cpu" for dev in devices):
        return "cpu"
    return None


torch._prims.rng_prims.get_device.__code__ = get_device.__code__
