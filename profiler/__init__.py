import torch

from contextlib import contextmanager

from .profiler import emit_cnpx, insert_hook_for_profiler, tensorboard_trace_handler
torch.autograd.profiler.__setattr__("emit_cnpx", emit_cnpx)

from .register import registry
from torch_mlu._MLUC import _MLURecordFunction

from .get_tensor_performance import get_tensor_performance

# Add torch.profiler.ProfilerActivity.MLU and mapping it to torch.profiler.ProfilerActivity.PrivateUse1
torch.profiler.ProfilerActivity.MLU = torch.profiler.ProfilerActivity.PrivateUse1

def apply__pattern_matcher_patch():
    from ..accelerator import native_is_available
    from ._pattern_matcher import (
        ExtraMLUCopyPattern,
        FP32MatMulPattern,
        MatMulDimInFP16Pattern,
        report_all_anti_patterns
    )
    torch.profiler._pattern_matcher.__setattr__("FP32MatMulPattern", FP32MatMulPattern)
    torch.profiler._pattern_matcher.__setattr__("MatMulDimInFP16Pattern", MatMulDimInFP16Pattern)
    torch.profiler._pattern_matcher.__setattr__("report_all_anti_patterns", report_all_anti_patterns)
    torch.accelerator.is_available.__code__ = native_is_available

insert_hook_for_profiler()

# TODO(fuwenguang): Remove the following code once landing these into the native community.
# Support passing 'self_mlu_xxx's and mapping them to 'self_privateuse1_xxx's
from functools import wraps
def replace_mlu_to_privateuse1(fn):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        if kwargs:
            sort_by = kwargs.get('sort_by', None)
            if sort_by:
                kwargs['sort_by'] = sort_by.replace("mlu", "privateuse1")

            metric = kwargs.get('metric', None)
            if metric:
                kwargs['metric'] = metric.replace("mlu", "privateuse1")
        return fn(*args, **kwargs)

    return wrapper_fn

_build_table_fn = getattr(torch.autograd.profiler_util, "_build_table")
_new_build_table = replace_mlu_to_privateuse1(_build_table_fn)
setattr(torch.autograd.profiler_util, "_build_table", _new_build_table)

_export_stacks = getattr(torch.autograd.profiler_util.EventList, "export_stacks")
_new_export_stacks = replace_mlu_to_privateuse1(_export_stacks)
setattr(torch.autograd.profiler_util.EventList, "export_stacks", _new_export_stacks)


def register_custom_op(op, name=None):
    registry.register_op(op, name)

def register_custom_module(module):
    registry.register_module(module)

__profiler_started = False

@contextmanager
def record_custom_op(name, *args, **kwargs):
    global __profiler_started
    if not __profiler_started:
        yield None
    else:
        with _MLURecordFunction(
            name,
            input_values=list(args),
            keyword_values=kwargs,
        ) as rf:
            yield rf

_orig_start = getattr(torch.profiler.profile, "start")
@wraps(_orig_start)
def _new_start(self, *args, **kwargs):
    registry.wrap()
    _orig_start(self, *args, **kwargs)
    global __profiler_started
    __profiler_started = True
setattr(torch.profiler.profile, "start", _new_start)

_orig_stop = getattr(torch.profiler.profile, "stop")
@wraps(_orig_stop)
def _new_stop(self, *args, **kwargs):
    global __profiler_started
    __profiler_started = False
    registry.unwrap()
    _orig_stop(self, *args, **kwargs)
setattr(torch.profiler.profile, "stop", _new_stop)
