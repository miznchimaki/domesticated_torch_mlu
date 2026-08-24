# remove override after PYTORCH-12898 finish
import functools
import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch._logging import warning_once
from torch._dynamo import polyfills, variables, config
from torch._dynamo.utils import (
    guard_if_dyn,
)
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.ctx_manager import (
    AutocastModeVariable,
    ProfilerContextVariable,
    TorchFunctionDisableVariable,
    ProfilerRecordFunctionContextVariable,
)
from torch._dynamo.variables.functions import bind_args_cached
from torch._dynamo.variables.torch import (
    BaseTorchVariable,
    constant_fold_functions,
    _fsdp_param_group,
    log,
    TorchCtxManagerClassVariable,
    supported_ctx_manager_classes,
)
from torch._dynamo.variables.functions import bind_args_cached

from ...utils import gorilla

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator
supported_ctx_manager_classes[torch.mlu.amp.autocast_mode.autocast] = None


# workaround for PYTORCH-12898
@gorilla.patch(
    torch._dynamo.variables.torch.TorchCtxManagerClassVariable,
    settings=gorilla.Settings(use_replace_references=True),
)
def call_function(
    self,
    tx: "InstructionTranslator",
    args: Sequence[VariableTracker],
    kwargs: "dict[str, VariableTracker]",
) -> "VariableTracker":
    # Modify by CAMBRICON
    # from . import (
    from torch._dynamo.variables import (
        DisabledSavedTensorsHooksVariable,
        DualLevelContextManager,
        FSDPParamGroupUseTrainingStateVariable,
        FxTracebackAnnotateVariable,
        GradIncrementNestingCtxManagerVariable,
        GradInplaceRequiresGradCtxManagerVariable,
        GradModeVariable,
        InferenceModeVariable,
        JvpIncrementNestingCtxManagerVariable,
        SDPAKernelVariable,
        SetFwdGradEnabledContextManager,
        StreamVariable,
        VmapIncrementNestingCtxManagerVariable,
    )

    # end Modify by CAMBRICON

    # Add by CAMBRICON
    def issubclass_override(value, base_class):
        if hasattr(value, "__mro__"):
            inherit_chain = value.__mro__
            for c in inherit_chain:
                if c.__name__ == base_class.__name__:
                    return True
        return False

    # end Add by CAMBRICON

    if self.value is torch.no_grad:
        if len(args) == 1 and isinstance(
            args[0], variables.functions.BaseUserFunctionVariable
        ):
            ctx = GradModeVariable.create(tx, False)
            return ctx.call_function(tx, args, kwargs)
        else:
            return GradModeVariable.create(tx, False)
    elif self.value is torch.enable_grad:
        if len(args) == 1 and isinstance(
            args[0], variables.functions.BaseUserFunctionVariable
        ):
            ctx = GradModeVariable.create(tx, True)
            return ctx.call_function(tx, args, kwargs)
        return GradModeVariable.create(tx, True)
    elif self.value is torch.set_grad_enabled and len(args) == 1:
        return GradModeVariable.create(
            tx, args[0].as_python_constant(), initialized=True
        )
    elif self.value is torch.inference_mode:
        assert len(args) <= 1 and len(kwargs) == 0
        inf_mode = args[0].as_python_constant() if len(args) == 1 else True
        return InferenceModeVariable.create(tx, inf_mode)
    elif self.value in (
        torch.fx.traceback.annotate,
        torch.fx.traceback.annotate.__wrapped__,  # type: ignore[attr-defined]
    ):
        assert len(args) <= 1 and len(kwargs) == 0
        return FxTracebackAnnotateVariable(
            args[0].as_python_constant(), source=self.source
        )
    # Modify by CAMBRICON
    # elif inspect.isclass(self.value) and issubclass(self.value, torch.Stream):
    elif inspect.isclass(self.value) and issubclass_override(self.value, torch.Stream):
        # end Modify by CAMBRICON
        from torch._dynamo.variables.builder import wrap_fx_proxy_cls

        return wrap_fx_proxy_cls(
            StreamVariable,
            tx,
            tx.output.create_proxy(
                "call_function",
                self.value,
                (),
                {},
            ),
        )
    elif self.value in (
        torch.amp.autocast_mode.autocast,
        torch.cuda.amp.autocast,
        torch.cpu.amp.autocast,
    ):
        # pyrefly: ignore [bad-argument-type]
        return AutocastModeVariable.create(self.value, args, kwargs)
    # Add by CAMBRICON
    elif (
        hasattr(self.value, "is_mlu_gpu_migration")
        and self.value.is_mlu_gpu_migration is True
        and hasattr(self.value, "__wrapped__")
        and self.value.__wrapped__
        in (
            torch.amp.autocast_mode.autocast,
            torch.cuda.amp.autocast,
            torch.cpu.amp.autocast,
        )
    ):
        return AutocastModeVariable.create(self.value.__wrapped__, args, kwargs)
    # end Add by CAMBRICON
    elif self.value in (
        torch.profiler.record_function,
        torch.autograd.profiler.record_function,
    ):
        return ProfilerRecordFunctionContextVariable.create(
            func=self.value, record_args=args, record_kwargs=kwargs
        )
    elif self.value in (
        torch.profiler.profile,
        torch.autograd.profiler.profile,
    ):
        warning_once(log, "Profiler function %s will be ignored", self.value)
        return ProfilerContextVariable()
    elif (
        self.value is torch._C.DisableTorchFunctionSubclass
        or self.value is torch._C.DisableTorchFunction
    ):
        assert not (args or kwargs)
        return TorchFunctionDisableVariable.create(
            tx, only_subclass=self.value is torch._C.DisableTorchFunctionSubclass
        )
    elif self.value is torch._functorch.vmap.vmap_increment_nesting:
        assert len(args) == 2
        return VmapIncrementNestingCtxManagerVariable.create(
            tx,
            args,
        )
    elif self.value is torch._functorch.eager_transforms.jvp_increment_nesting:
        assert len(args) == 0
        return JvpIncrementNestingCtxManagerVariable.create(tx)
    elif self.value is torch.autograd.forward_ad._set_fwd_grad_enabled:
        assert len(args) == 1
        return SetFwdGradEnabledContextManager.create(
            tx,
            [guard_if_dyn(x) for x in args],
        )
    elif self.value is torch.autograd.forward_ad.dual_level:
        assert len(args) == 0
        return DualLevelContextManager.create(tx)
    elif self.value is torch._functorch.eager_transforms.grad_increment_nesting:
        assert len(args) == 0
        return GradIncrementNestingCtxManagerVariable.create(tx)
    elif self.value is torch._functorch.eager_transforms.enable_inplace_requires_grad:
        assert len(args) == 1
        return GradInplaceRequiresGradCtxManagerVariable.create(
            tx,
            [guard_if_dyn(x) for x in args],
        )
    elif self.value is torch.autograd.graph.disable_saved_tensors_hooks:
        assert len(args) == 1
        return DisabledSavedTensorsHooksVariable.create(
            tx, args[0].as_python_constant()
        )
    elif (
        _fsdp_param_group is not None
        and self.value is _fsdp_param_group.FSDPParamGroup.use_training_state
    ):
        assert len(args) == 2
        return FSDPParamGroupUseTrainingStateVariable.create(
            tx, args[0], args[1].as_python_constant()
        )
    elif self.value is torch.nn.attention.sdpa_kernel.__wrapped__:  # type: ignore[attr-defined]
        name_to_arg_map = bind_args_cached(
            # pyrefly: ignore[bad-argument-type]
            self.value,
            tx,
            self.source,
            args,
            kwargs,
        )
        backends = name_to_arg_map["backends"].as_python_constant()
        set_priority = name_to_arg_map["set_priority"].as_python_constant()
        return SDPAKernelVariable.create(tx, backends, set_priority)
    # Modify by CAMBRICON: RuntimeError: super(): __class__ cell not found
    # return super().call_function(tx, args, kwargs)
    return super(BaseTorchVariable, self).call_function(tx, args, kwargs)
    # end Modify by CAMBRICON


@functools.lru_cache(None)
def get_mlu_constant_fold_functions():
    import torch_mlu

    constant_fold_functions = [
        torch._C._get_fp32_precision_getter,
        torch_mlu._MLUC._get_cnmatmul_allow_tf32,
        torch_mlu.mlu.is_available,
    ]
    constant_fold_functions = dict.fromkeys(constant_fold_functions)
    return constant_fold_functions


@gorilla.patch(torch._dynamo.variables.torch.BaseTorchVariable)
def can_constant_fold_through(self):
    # Modify by CAMBRICON
    from torch_mlu._dynamo.variables.torch import get_mlu_constant_fold_functions

    # if self.value in constant_fold_functions:
    if (
        self.value in constant_fold_functions
        or (
            hasattr(self.value, "is_mlu_gpu_migration")
            and self.value.is_mlu_gpu_migration is True
            and hasattr(self.value, "__wrapped__")
            and self.value.__wrapped__ in constant_fold_functions
        )
        or self.value in get_mlu_constant_fold_functions()
    ):
        # end Modify by CAMBRICON
        return True
    if (
        self.value is torch.autograd._profiler_enabled
        and config.constant_fold_autograd_profiler_enabled
    ):
        # The relevant flag is enabled only for export. One might wonder
        # why?
        #
        # Actually we would like to not graph break even in the case of
        # Dynamo. But there is a weird-unsolved bug with Kineto + Dynamo
        # when there are distributed jobs that lead to NCCL timeouts. This
        # bug is a rare edege case, but we have not been able to root cause
        # it yet. See https://www.internalfb.com/sevmanager/view/560336 for
        # more details.
        #
        # So is this safe for export? Yes, for export, we do not anticipate
        # JIT tracing in distributed job training, and the weird edge-case
        # interaction with Kineto is not a valid usecase. So, this is ok.
        return True

    return getattr(self.value, "__module__", None) == "math"
