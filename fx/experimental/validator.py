import torch
from torch.fx.experimental import validator
from ... import gorilla


try:
    import z3

    @gorilla.patch(validator.SympyToZ3)
    def to_dtype(self, x: z3.ArithRef, dtype: torch.dtype) -> z3.ArithRef:
        # Modify by CAMBRICON: change torch.float64 to torch.float
        # if dtype == torch.float64:
        if dtype == torch.float64 or dtype == torch.float32:
            return z3.ToReal(x)
        # end Modify by CAMBRICON
        raise NotImplementedError(f"to_dtype {dtype} NYI")

except ImportError:
    pass
