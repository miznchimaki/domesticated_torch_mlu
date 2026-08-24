from __future__ import annotations

import math
from copy import deepcopy

from typing import Any, NoReturn, Optional, TYPE_CHECKING, Union

import torch

from torch._guards import (
    Guard,
)

from torch._dynamo.source import (
    LocalSource,
    TypeSource,
)
from torch._dynamo.guards import get_verbose_code_parts

from ..utils import gorilla


@gorilla.patch(torch._dynamo.guards.GuardBuilder)
def id_match_unchecked(
    self, guard: Guard, recompile_hint: Optional[str] = None
) -> None:
    # ___check_obj_id is same as `id(x) == y`
    if isinstance(guard.originating_source, TypeSource):
        # optional optimization to produce cleaner/faster guard code
        return self.TYPE_MATCH(
            Guard(guard.originating_source.base, GuardBuilder.TYPE_MATCH)  # type: ignore[arg-type]
        )

    ref = self.arg_ref(guard)
    val = self.get(guard)
    id_val = self.id_ref(val, guard.name)
    # Modify by Cambricon
    # try:
    #    type_repr = repr(val)
    # except Exception:
    #    # During deepcopy reconstruction or other state transitions,
    #    # objects may be in an incomplete state where repr() fails
    #    type_repr = f"<{type(val).__name__}>"
    from torch_mlu._dynamo import config as dynamo_config

    if dynamo_config.use_guard_repr:
        try:
            type_repr = repr(val)
        except Exception:
            # During deepcopy reconstruction or other state transitions,
            # objects may be in an incomplete state where repr() fails
            type_repr = f"<{type(val).__name__}>"
    else:
        type_repr = f"<{type(val).__name__}>"
    # end Modify by Cambricon
    code = f"___check_obj_id({ref}, {id_val}), type={type_repr}"
    self._set_guard_export_info(guard, [code], provided_func_name="ID_MATCH")
    self.get_guard_manager(guard).add_id_match_guard(
        id_val,
        get_verbose_code_parts(code, guard, recompile_hint),
        guard.user_stack,
    )

    # Keep track of ID_MATCH'd objects. This will be used to modify the
    # cache size logic
    if isinstance(guard.originating_source, LocalSource):
        # TODO(anijain2305) - This is currently restricted to nn.Module objects
        # because many other ID_MATCH'd objects fail - like DeviceMesh.
        # Increase the scope of ID_MATCH'd objects.
        if isinstance(val, torch.nn.Module):
            local_name = guard.originating_source.local_name
            weak_id = self.lookup_weakrefs(val)
            if weak_id is not None:
                self.id_matched_objs[local_name] = weak_id
