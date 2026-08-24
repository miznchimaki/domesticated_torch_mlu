from collections.abc import Sequence
import os
import traceback
import types

import torch
from torch._dynamo.exc import get_stack_above_dynamo
from torch._dynamo import variables
from torch._dynamo.variables.base import VariableTracker
from ...utils import gorilla
from torch._dynamo import polyfills
import logging


@gorilla.patch(
    torch._dynamo.variables.functions.WrapperUserFunctionVariable,
    settings=gorilla.Settings(use_replace_references=True),
)
def call_function(
    self,
    tx: "InstructionTranslator",
    args: Sequence[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    if hasattr(self.wrapper_obj, "cache_info"):
        target_fn = getattr(self.wrapper_obj, self.attr_to_trace, None)
        module_name = getattr(target_fn, "__module__", "") or ""

        # Modify by CAMBRICON
        # if module_name.split(".", maxsplit=1)[0] != "torch":
        if (
            module_name.split(".", maxsplit=1)[0] != "torch"
            and module_name.split(".", maxsplit=1)[0] != "torch_mlu"
        ):
            # end Modify by CAMBRICON
            frame_summary = tx.frame_summary()
            filename = os.path.basename(frame_summary.filename)
            lineno = frame_summary.lineno
            msg = (
                "Dynamo detected a call to a `functools.lru_cache`-wrapped "
                f"function at '{filename}:{lineno}'. Dynamo ignores the "
                "cache wrapper and directly traces the wrapped function. "
                "Silent incorrectness is only a *potential* risk, not "
                "something we have observed. "
                "Enable TORCH_LOGS=+dynamo for a DEBUG stack trace.\n\n"
                "This call originates from:\n"
                f"{''.join(traceback.format_list([frame_summary]))}"
            )
            torch._dynamo.utils.warn_once(msg)
            dynamo_logger = torch._dynamo.utils.logging.getLogger("torch._dynamo")
            if dynamo_logger.isEnabledFor(logging.DEBUG):
                user_stack = torch._guards.TracingContext.extract_stack()
                user_stack = get_stack_above_dynamo() + user_stack
                frame_loc = (user_stack[-1].filename, user_stack[-1].lineno)
                user_stack_formatted = "".join(traceback.format_list(user_stack))
                user_stack_trace = f"call to a lru_cache wrapped function at: {frame_loc[0]}:{frame_loc[1]}\n"
                user_stack_trace += str(user_stack_formatted)
                dynamo_logger.debug(user_stack_trace)

    all_args = self.self_args() + list(args)
    return variables.UserFunctionVariable(
        polyfills.getattr_and_trace  # type: ignore[arg-type]
    ).call_function(
        tx,
        [self, variables.ConstantVariable(self.attr_to_trace), *all_args],
        kwargs,
    )


@gorilla.patch(
    torch._dynamo.variables.functions.UserFunctionVariable,
)
def get_code(self) -> types.CodeType:
    # Add by CAMBRICON
    if (
        hasattr(self.fn, "is_mlu_gpu_migration")
        and self.fn.is_mlu_gpu_migration is True
        and hasattr(self.fn, "__wrapped__")
    ):
        return self.fn.__wrapped__.__code__
    # end Add by CAMBRICON
    return self.fn.__code__
