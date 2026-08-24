import functools
from typing import Callable, Optional

import torch
from ..mlu._utils import update_bytecode
from torch._inductor.autotune_process import (
    GPUDeviceBenchmarkMixin,
    TritonBenchmarkRequest,
    autotuning_log,
)
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.codecache import PyCodeCache


def do_bench(
    self,
    fn,
    *input_tensors: torch.Tensor,
    output_tensor: Optional[torch.Tensor] = None,
) -> float:
    device_idx_set = {
        tensor.device.index
        for tensor in [*input_tensors, output_tensor]
        if isinstance(tensor, torch.Tensor)
        and tensor.is_mlu
        and tensor.device.index is not None
    }
    assert len(device_idx_set) <= 1, f"Can not mix devices {device_idx_set}"
    if len(device_idx_set) == 1:
        device_idx = next(iter(device_idx_set))
    else:
        device_idx = torch.mlu.current_device()

    with torch.mlu.device(device_idx):
        out = benchmarker.benchmark_gpu(fn)
        torch.mlu.synchronize()

    return out


update_bytecode(GPUDeviceBenchmarkMixin.do_bench, do_bench)


def make_run_fn(
    self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
) -> Callable[[], None]:
    mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
    autotuning_log.debug(
        "benchmark module key: %s, path: %s",
        self.module_cache_key,
        self.module_path,
    )

    run_method = getattr(mod, self.kernel_name).run
    extra_args = list(self.extra_args)

    # Newer version of triton add warmup argument to JITFunction.run.
    # This code handles backward-compatibility.
    warmup_arg = {}
    import inspect

    if "warmup" in inspect.signature(run_method).parameters:
        warmup_arg["warmup"] = False

    # Modified by Cambricon - replacing CUDA stream with MLU stream
    from torch_mlu._MLUC import _mlu_getCurrentRawStream as get_raw_stream

    return functools.partial(
        run_method,
        *input_tensors,
        output_tensor,
        *self.extra_args,
        grid=self.grid,
        **warmup_arg,
        stream=get_raw_stream(self.output_tensor_meta.device.index),
    )


update_bytecode(TritonBenchmarkRequest.make_run_fn, make_run_fn)
