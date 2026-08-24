# -*- coding: utf-8 -*-
"""
Patch for torch.as_tensor to handle objects with __cuda_array_interface__.

This patch provides MLU backend support for torch.as_tensor() by handling
objects that implement __cuda_array_interface__ protocol, which is commonly
used by third-party libraries like transformer_engine.

The implementation uses the C++ MLU array interface bindings which follow
PyTorch's tensor_from_cuda_array_interface pattern to preserve data pointers.
"""

import torch
import warnings
from typing import Any, Optional


def _has_cuda_array_interface(obj: Any) -> bool:
    """Check if an object implements __cuda_array_interface__."""
    return hasattr(obj, "__cuda_array_interface__")


def _try_convert_with_dlpack(
    obj: Any, dtype: Optional[torch.dtype] = None, device: Optional[str] = None
) -> Optional[torch.Tensor]:
    """
    Try to convert object using DLPack protocol.

    This is the preferred method for zero-copy tensor conversion.
    """
    if not hasattr(obj, "__dlpack__"):
        return None

    try:
        dlpack_capsule = obj.__dlpack__()
        if dlpack_capsule is not None:
            result = torch.from_dlpack(dlpack_capsule)
            if dtype is not None and result.dtype != dtype:
                result = result.to(dtype)
            if device is not None and result.device != device:
                result = result.to(device)
            return result
    except Exception:
        pass

    return None


def mlu_as_tensor(data, dtype=None, device=None):
    if isinstance(data, torch.Tensor):
        return torch._original_as_tensor(data, dtype=dtype, device=device)

    if hasattr(data, "__dlpack__"):
        try:
            result = torch.from_dlpack(data)
            if dtype is not None and result.dtype != dtype:
                result = result.to(dtype)
            if device is not None and str(result.device) != str(device):
                result = result.to(device)
            return result
        except Exception:
            pass

    if hasattr(data, "__cuda_array_interface__"):
        try:
            import torch_mlu

            result = torch_mlu._MLUC._tensor_from_mlu_array_interface(data, device)
            if dtype is not None and result.dtype != dtype:
                result = result.to(dtype)
            return result
        except Exception as e:
            warnings.warn(
                f"MLU array interface conversion failed: {e}. "
                f"Falling back to default."
            )

    return torch._original_as_tensor(data, dtype=dtype, device=device)


def apply_as_tensor_patch():
    """Apply the monkey-patch for torch.as_tensor to support MLU array interface."""
    if not hasattr(torch, "_original_as_tensor"):
        torch._original_as_tensor = torch.as_tensor
    if torch.as_tensor is mlu_as_tensor:
        return
    torch.as_tensor = mlu_as_tensor
