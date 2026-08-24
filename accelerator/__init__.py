import torch
from torch.accelerator import current_accelerator


native_is_available = torch.accelerator.is_available.__code__
def is_available() -> bool:
    r"""Check if the current accelerator is available at runtime: it was build, all the
    required drivers are available and at least one device is visible.
    See :ref:`accelerator<accelerators>` for details.

    Returns:
        bool: A boolean indicating if there is an available :ref:`accelerator<accelerators>`.

    .. note:: This API delegates to the device-specific version of `is_available`.
        On CUDA, when the environment variable ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set,
        this function will NOT poison fork. Otherwise, it will. For more details, see
        :ref:`multiprocessing-poison-fork-note`.

    Example::

        >>> assert torch.accelerator.is_available() "No available accelerators detected."
    """
    # Why not just check "device_count() > 0" like other is_available call?
    # Because device like CUDA have a python implementation of is_available that is
    # non-poisoning and some features like Dataloader rely on it.
    # So we are careful to delegate to the Python version of the accelerator here
    acc = current_accelerator()
    if acc is None:
        return False
    if acc.type == 'mlu':
        return True

    mod = torch.get_device_module(acc)
    return mod.is_available()

# This patch temporarily avoids a circular import error caused by torch.accelerator.is_available 
# accessing torch.mlu before it's registered as a backend. It is applied before autoloading 
# torch_mlu and reverted after autoload to ensure is_available works correctly at runtime.
def apply_accelerator_patch():
    torch.accelerator.is_available.__code__ = is_available.__code__

apply_accelerator_patch()
