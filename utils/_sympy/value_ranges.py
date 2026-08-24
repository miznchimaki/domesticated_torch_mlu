import torch
from torch.utils._sympy.functions import ToFloat
from torch.utils._sympy.value_ranges import ValueRanges


def to_dtype(a, dtype, src_dtype=None, use_compute_types=True):
    if dtype == torch.float64:
        return ValueRanges.increasing_map(a, ToFloat)
    elif dtype == torch.bool:
        return ValueRanges.unknown_bool()
    elif not dtype.is_floating_point:
        return ValueRanges.unknown_int()
    return ValueRanges.unknown()


torch.utils._sympy.value_ranges.SymPyValueRangeAnalysis.to_dtype.__code__ == to_dtype.__code__
