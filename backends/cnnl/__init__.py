import sys
import torch
import torch_mlu
import warnings
from torch.backends import (
    ContextProp, 
    PropModule, 
    __allow_nonbracketed_mutation,
    _FP32Precision,
    _get_fp32_precision_getter,
    _set_fp32_precision_setter,
)
from contextlib import contextmanager

def set_flags(_enabled=None, _benchmark=None, _benchmark_limit=None, _deterministic=None, _allow_tf32=None, _fp32_precision="none",):
    orig_flags = (None,
                  torch_mlu._MLUC._get_cnnl_benchmark(),
                  None,
                  torch_mlu._MLUC._get_cnnl_deterministic(),
                  torch_mlu._MLUC._get_cnnl_allow_tf32(),
                  torch._C._get_fp32_precision_getter("cuda", "all"))
    if _enabled is True:
        warnings.warn("torch.backends.cnnl.enabled is not available on MLU device.")
    if _benchmark is not None:
        torch_mlu._MLUC._set_cnnl_benchmark(_benchmark)
    if _benchmark_limit != 0 and _benchmark_limit is not None:
        warnings.warn("torch.backends.cnnl.benchmark_limit is not available on MLU device.")
    if _deterministic is not None:
        torch_mlu._MLUC._set_cnnl_deterministic(_deterministic)
    if _allow_tf32 is not None:
        torch_mlu._MLUC._set_cnnl_allow_tf32(_allow_tf32)
    if _fp32_precision is not None:
        torch._C._set_fp32_precision_setter("cuda", "all", _fp32_precision)
    return orig_flags

@contextmanager
def flags(enabled=False, benchmark=False, benchmark_limit=0, deterministic=False, allow_tf32=True, fp32_precision="none"):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled, benchmark, benchmark_limit, deterministic, allow_tf32, fp32_precision)
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)

class CnnlModule(PropModule):
    def __init__(self, m, name):
        super(CnnlModule, self).__init__(m, name)

    # Control whether to allow TF32 on part of CNNL ops,
    # same function as `torch.backends.cudnn.allow_tf32`, currently only affect conv.
    deterministic = ContextProp(
        torch_mlu._MLUC._get_cnnl_deterministic, torch_mlu._MLUC._set_cnnl_deterministic
    )
    benchmark = ContextProp(
        torch_mlu._MLUC._get_cnnl_benchmark, torch_mlu._MLUC._set_cnnl_benchmark
    )
    allow_tf32 = ContextProp(
        torch_mlu._MLUC._get_cnnl_allow_tf32, torch_mlu._MLUC._set_cnnl_allow_tf32
    )
    conv = _FP32Precision("cuda", "conv")
    rnn = _FP32Precision("cuda", "rnn")
    fp32_precision = ContextProp(
        _get_fp32_precision_getter("cuda", "all"),
        _set_fp32_precision_setter("cuda", "all"),
    )


