# mypy: allow-untyped-defs
import functools
import sys
from typing import Any
from typing_extensions import deprecated

import torch


__all__ = ["autocast", "custom_fwd", "custom_bwd"]


@deprecated(
    "`torch.mlu.amp.autocast(args...)` is deprecated. "
    "Please use `torch.amp.autocast('mlu', args...)` instead.",
    category=FutureWarning,
)
class autocast(torch.amp.autocast_mode.autocast):
    r"""See :class:`torch.autocast`.

    ``torch.mlu.amp.autocast(args...)`` is deprecated. Please use ``torch.amp.autocast("mlu", args...)`` instead.
    """

    # TODO: remove this conditional once we stop supporting Python < 3.13
    # Prior to Python 3.13, inspect.signature could not retrieve the correct
    # signature information for classes decorated with @deprecated (unless
    # the __new__ static method was explicitly defined);
    #
    # However, this issue has been fixed in Python 3.13 and later versions.
    if sys.version_info < (3, 13):

        def __new__(
            cls,
            enabled: bool = True,
            dtype: torch.dtype = torch.float16,
            cache_enabled: bool = True,
        ):
            return super().__new__(cls)

        def __init_subclass__(cls):
            pass

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        cache_enabled: bool = True,
    ):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "mlu"
            self.fast_dtype = dtype
            return
        super().__init__(
            "mlu", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if torch._jit_internal.is_scripting():
            return
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)


# Preserved only for BC reasons
@deprecated(
    "`torch.mlu.amp.autocast_mode._cast(value, dtype)` is deprecated. "
    "Please use `torch.amp.autocast_mode._cast(value, 'mlu', dtype)` instead.",
    category=FutureWarning,
)
def _cast(value, dtype):
    return torch.amp.autocast_mode._cast(value, "mlu", dtype)


@deprecated(
    "`torch.mlu.amp.custom_fwd(args...)` is deprecated. "
    "Please use `torch.amp.custom_fwd(args..., device_type='mlu')` instead.",
    category=FutureWarning,
)
def custom_fwd(fwd=None, *, cast_inputs=None):
    """
    ``torch.mlu.amp.custom_fwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_fwd(args..., device_type='mlu')`` instead.
    """
    return functools.partial(torch.amp.custom_fwd, device_type="mlu")(
        fwd=fwd, cast_inputs=cast_inputs
    )


@deprecated(
    "`torch.mlu.amp.custom_bwd(args...)` is deprecated. "
    "Please use `torch.amp.custom_bwd(args..., device_type='mlu')` instead.",
    category=FutureWarning,
)
def custom_bwd(bwd):
    """
    ``torch.mlu.amp.custom_bwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_bwd(args..., device_type='mlu')`` instead.
    """
    return functools.partial(torch.amp.custom_bwd, device_type="mlu")(bwd)
