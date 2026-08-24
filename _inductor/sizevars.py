from typing import Union
import sympy
from sympy import Expr
import logging
from typing import Optional, Union

import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._sympy.value_ranges import bound_sympy
from torch._inductor import sizevars
from torch._inductor.sizevars import (
    SizeVarAllocator,
)
from ..utils import gorilla

log = logging.getLogger(__name__)


@gorilla.patch(torch._inductor.sizevars.SizeVarAllocator)
def size_hint(
    self,
    expr: Union[Expr, int],
    *,
    fallback: Optional[int] = None,
) -> int:
    if isinstance(expr, SymInt):
        raise TypeError(
            "wrong API usage!, use size_hint from torch.fx.experimental.symbolic_shapes or pass sympy expressions instead"
        )

    out = self.symbolic_hint(
        expr,
        use_user_provided_hint_override=fallback is not None,
    )
    # Add by CAMBRICON
    import torch

    if fallback is None:
        fallback = torch._inductor.config.unbacked_symint_fallback
    # end Add by CAMBRICON
    if not isinstance(out, (int, sympy.Integer)) and fallback is not None:
        # Use the provided heuristic fallback hint
        unbacked_sym_vrs = {
            s: self.shape_env.var_to_range.get(s, None) for s in out.free_symbols
        }
        if all(vr is not None for vr in unbacked_sym_vrs.values()):
            hint_vr = bound_sympy(out, unbacked_sym_vrs)  # type: ignore[arg-type]
            if isinstance(hint_vr.lower, (int, sympy.Integer)):
                fallback = max(fallback, int(hint_vr.lower))
            if isinstance(hint_vr.upper, (int, sympy.Integer)):
                fallback = min(fallback, int(hint_vr.upper))
        return fallback

    try:
        return int(out)
    except Exception:
        log.debug("failed on: %s", out)
        raise


def support_unback_statically_known_multiple_of(
    numerator: Expr, denominator: Union[Expr, int], support_unbacked_symbols=False
) -> bool:
    """
    Return a bool indicating if it is sound to optimize for the numerator being a multiple of the denominator.
    """
    if not support_unbacked_symbols:
        if free_unbacked_symbols(numerator) or free_unbacked_symbols(denominator):
            return False
    expr = sympy.Eq(numerator % denominator, 0)
    alloc = SizeVarAllocator()
    return alloc.is_expr_static_and_true(expr)  # type: ignore[arg-type]
