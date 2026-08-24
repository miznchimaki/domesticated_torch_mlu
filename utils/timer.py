"""Timer class based on the timeit.Timer class, but torch aware."""
import timeit

import torch
import torch.utils.benchmark as benchmark


native_timer_init = benchmark.Timer.__init__


def timer() -> float:
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()
    return timeit.default_timer()


# Due to import order issues, the accelerator version of timer is selected by
# default during autoload. This patch changes timer to use lazy initialization,
# selecting the appropriate device-specific timer at runtime.
# Long-term, this should be addressed upstream by wrapping timer in a function
# instead of using a global variable, to better support out-of-tree devices.
def patch_timer_init(self, *args, **kwargs):
    if "timer" not in kwargs:
        kwargs["timer"] = timer
    native_timer_init(self, *args, **kwargs)


benchmark.Timer.__init__ = patch_timer_init


def apply_timer_patch():
    benchmark.timer.__code__ = timer.__code__


apply_timer_patch()
