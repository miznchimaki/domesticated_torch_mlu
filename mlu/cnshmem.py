"""CNSHMEM symmetric memory backend for MLU."""

import torch
import torch.distributed._symmetric_memory as symm_mem


def is_cnshmem_available() -> bool:
    """Check if CNSHMEM library is available at runtime."""
    try:
        # return torch.ops.symm_mem.is_cnshmem_available()
        from torch._C._distributed_c10d import _is_cnshmem_available
    except ImportError:
        return False
    # Check if CNSHMEM is available on current system
    return _is_cnshmem_available()


def _register_cnshmem_backend():
    """Register CNSHMEM as the symmetric memory backend for PrivateUse1.

    Called from torch_mlu/__init__.py during module initialization.
    """
    if is_cnshmem_available():
        from torch._C._distributed_c10d import _SymmetricMemory

        _SymmetricMemory.set_backend("CNSHMEM")


symm_mem.is_cnshmem_available = is_cnshmem_available
