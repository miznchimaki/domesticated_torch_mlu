import sys

import torch
from typing import TYPE_CHECKING
from torch._subclasses.fake_tensor import (
    DynamicOutputShapeException,
    FakeTensor,
)
from torch._ops import OpOverload

from torch._subclasses.fake_tensor import FakeTensorMode
from ..utils import gorilla


@gorilla.patch(torch._subclasses.fake_impls)
def nonzero(fake_mode: FakeTensorMode, func: OpOverload, arg: FakeTensor) -> FakeTensor:
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        # Without symints/symfloats, cannot handle this
        raise DynamicOutputShapeException(func)

    if (nnz := arg.nonzero_memo) is None:
        # Avoid importing sympy at a module level
        from torch.fx.experimental.symbolic_shapes import (
            _constrain_range_for_size,
            has_free_symbols,
        )
        from torch.utils._sympy.numbers import IntInfinity
        from torch.utils._sympy.value_ranges import bound_sympy

        if not has_free_symbols(arg.numel()) and arg.numel() == 0:
            # If numel is zero, then the output size must be zero.
            # In this case, we must not allocate an unbacked SymInt,
            # because if we do, it will immediately get refined to
            # zero, but this will be inconsistent with size oblivious
            # tests (which will continue to claim that the unbacked
            # symint cannot equal zero).  We could also unconditionally
            # allocate an unbacked SymInt and not refine its range,
            # but this seems more precise.
            nnz = 0
        else:
            nnz = fake_mode.shape_env.create_unbacked_symint()

            maxval = sys.maxsize - 1

            if not has_free_symbols(arg.numel()):
                maxval = int(arg.numel())
            else:
                prod_node = math.prod(arg.shape).node  # type: ignore[union-attr]
                prod_range = bound_sympy(
                    prod_node.expr, prod_node.shape_env.var_to_range
                )
                if isinstance(prod_range.upper, IntInfinity):
                    maxval = sys.maxsize - 1
                else:
                    maxval = prod_range.upper

            _constrain_range_for_size(nnz, max=maxval)

        arg.nonzero_memo = nnz
    # Modify by CAMBRICON
    if isinstance(arg, torch._subclasses.FakeTensor) and arg.fake_device.type == "mlu":
        return arg.new_empty_strided(
            (nnz, arg.dim()), (arg.dim(), 1), dtype=torch.int64
        )
    # end Modify by CAMBRICON
    return arg.new_empty_strided((nnz, arg.dim()), (1, nnz), dtype=torch.int64)  # type: ignore[return]
