# mypy: allow-untyped-defs
# flake8: noqa: B950
"""This module implements the user facing API for flex_attention in PyTorch."""
import torch
from torch import Tensor
import torch.nn.attention.flex_attention as flex_attention

from ..utils import gorilla


def _validate_device(query: Tensor, key: Tensor, value: Tensor) -> None:
    """TODO: Remove once non cuda/cpu devices support is added
    We only need to check query since we have already that q,k,v are on the same device
    """
    if query.device.type == "cpu" and (
        query.requires_grad or key.requires_grad or value.requires_grad
    ):
        raise NotImplementedError(
            "FlexAttention does not support backward on CPU. Please set the input requires_grad to False or use another device."
        )
    # Modify by CAMBRICON
    # See https://github.com/pytorch/pytorch/issues/173071
    # supported_devices = {"cuda", "cpu", "xpu", "hpu"}
    supported_devices = {
        "cuda",
        "cpu",
        "xpu",
        "hpu",
        torch._C._get_privateuse1_backend_name(),
    }
    # end Modify by CAMBRICON
    if query.device.type not in supported_devices:
        raise ValueError(
            "FlexAttention is only supported on CUDA, CPU or HPU devices. "
            f"Found input tensors on {query.device.type} device."
        )


patch = gorilla.Patch(
    flex_attention,
    "_validate_device",
    _validate_device,
)
gorilla.apply(patch)
