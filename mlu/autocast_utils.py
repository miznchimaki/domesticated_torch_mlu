import torch
import torch_mlu

__all__ = [
    "get_amp_supported_dtype",
]


def get_amp_supported_dtype():
    return [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]
