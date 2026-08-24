import sympy

import torch
from torch.utils._sympy.functions import (
    ToFloat,
    FloorToInt,
    CeilToInt,
    TruncToInt,
    RoundToInt,
)
from torch.utils._sympy.interp import ASSOCIATIVE_OPS
from ... import gorilla


@gorilla.patch(torch.utils._sympy.interp)
def _run_sympy_handler(analysis, args, expr, index_dtype=torch.int64):
    # Special cases
    if isinstance(expr, sympy.Pow) and isinstance(
        expr.args[1], sympy.core.numbers.Half
    ):
        return analysis.sqrt(args[0])
    if isinstance(expr, ToFloat):
        # Modify by CAMBRICON: change torch.float64 to torch.float
        # return analysis.to_dtype(args[0], torch.float64)
        return analysis.to_dtype(args[0], torch.float)
        # end Modify by CAMBRICON

    # These handlers are special because they take an extra dtype argument
    # specifying what they should convert to, and we need to appropriately set
    # this up when we convert from Sympy.  A reasonable default when you
    # are translating is to conservatively do int64, and then narrow these
    # arguments later when you discover you can narrow the index range.  But
    # if you already know that 32-bit indexing is OK, you can directly do the
    # sympy translation with index_dtype=torch.int32
    INDEX_DTYPE_HANDLERS = {
        TruncToInt: "trunc_to_int",
        sympy.floor: "floor_to_int",
        sympy.ceiling: "ceil_to_int",
        FloorToInt: "floor_to_int",
        CeilToInt: "ceil_to_int",
        RoundToInt: "round_to_int",
    }
    if (handler_name := INDEX_DTYPE_HANDLERS.get(expr.func)) is not None:
        return getattr(analysis, handler_name)(*args, index_dtype)

    # Fastpath for n-ary integral addition
    if expr.func is sympy.Add and expr.is_integer and hasattr(analysis, "sym_sum"):
        r = analysis.sym_sum(args)
        log.debug("sym_sum(%s) -> %s", args, r)
        return r

    if hasattr(expr.func, "_torch_handler_name"):
        handler_name = expr.func._torch_handler_name
    else:
        handler_name = handlers()[expr.func]
    handler = getattr(analysis, handler_name)
    try:
        if handler_name in ASSOCIATIVE_OPS:
            if len(args) <= 1:
                raise AssertionError("associative op needs >1 args")
            acc = handler(args[0], args[1])
            for i in range(2, len(args)):
                acc = handler(acc, args[i])
            log.debug("%s(%s) -> %s", handler_name, args, acc)
            return acc
        else:
            r = handler(*args)
            log.debug("%s(%s) -> %s", handler_name, args, r)
            return r
    except NotImplementedError:
        raise
    except Exception:
        log.warning("failed while executing %s(%s)", handler_name, args)
        raise
