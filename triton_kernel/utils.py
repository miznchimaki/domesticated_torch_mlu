import functools
import torch
import torch_mlu

import triton.backends.mlu.driver as driver


@functools.lru_cache(1)
def get_total_core_num(mlu_current_device=None):
    if torch.mlu.is_available():
        if mlu_current_device is None:
            device_prop = torch.mlu.get_device_properties(torch.mlu.current_device())
        else:
            device_prop = torch.mlu.get_device_properties(mlu_current_device)
        total_cluster_num = device_prop.cluster_count
        total_core_num = total_cluster_num * device_prop.core_num_per_cluster
        return total_core_num
    return 0


@functools.lru_cache(1)
def get_max_nram_size(mlu_current_device=None):
    if torch.mlu.is_available():
        if mlu_current_device is None:
            _devprob = driver.BangUtils().get_device_properties(
                torch.mlu.current_device()
            )
        else:
            _devprob = driver.BangUtils().get_device_properties(mlu_current_device)

        MAX_NRAM_SIZE = _devprob.get("max_nram_size")
        return MAX_NRAM_SIZE
    return 0
