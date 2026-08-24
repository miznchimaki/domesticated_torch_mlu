# All modification made by Cambricon Corporation: © 2024 Cambricon Corporation
# All rights reserved.
#
# Patch for _tensor_str.py to handle MLU double type tensor printing.
# MLU does not support double for many ops (abs, isfinite, etc.),
# so we need to convert to float for printing compatibility.

from .utils import gorilla
import torch


@gorilla.patch(torch._tensor_str)
def tensor_totype(t):
    """Modified tensor_totype for MLU to handle double type."""
    # Modified by CAMBRICON
    dtype = (
        torch.float
        if (
            t.is_mps
            or (t.is_xpu and not torch.xpu.get_device_properties(t.device).has_fp64)
            or t.is_maia
            or (t.is_mlu and not torch.mlu.get_implicit_double_to_float())
        )
        else torch.double
    )
    # end Modified by CAMBRICON
    return t.to(dtype=dtype)


@gorilla.patch(torch._tensor_str._Formatter)
def __init__(self, tensor):
    # Add by CAMBRICON
    from torch._tensor_str import tensor_totype, PRINT_OPTS

    # end Add by CAMBRICON
    self.floating_dtype = tensor.dtype.is_floating_point
    self.int_mode = True
    self.sci_mode = False
    self.max_width = 1

    with torch.no_grad():
        tensor_view = tensor.reshape(-1)

    if not self.floating_dtype:
        for value in tensor_view:
            value_str = f"{value}"
            self.max_width = max(self.max_width, len(value_str))
    else:
        if tensor.dtype == torch.float4_e2m1fn_x2:  # type: ignore[attr-defined]
            # torch.float4_e2m1fn_x2 is special and does not support the casts necessary
            # to print it, we choose to display the uint8 representation here for
            # convenience of being able to print a tensor.
            # TODO(#146647): extend this to other dtypes without casts defined, such
            # as the bits, uint1..7 and int1..7 dtypes.
            tensor_view = tensor_view.view(torch.uint8)

        # Add by CAMBRICON
        # MLU does not support double for many ops (masked_select, isfinite, etc.)
        # Convert to float for printing compatibility, but only when TORCH_MLU_IMPLICIT_DOUBLE_TO_FLOAT=0
        if (
            tensor.is_mlu
            and tensor.dtype == torch.double
            and not torch.mlu.get_implicit_double_to_float()
        ):
            tensor_view = tensor_view.float()
        # end Add by CAMBRICON

        nonzero_finite_vals = torch.masked_select(
            tensor_view, torch.isfinite(tensor_view) & tensor_view.ne(0)
        )

        if nonzero_finite_vals.numel() == 0:
            # no valid number, do nothing
            return

        if tensor.dtype == torch.float8_e8m0fnu:  # type: ignore[attr-defined]
            # float8_e8m0fnu is special and does not define arithmetic ops,
            # and printing code further in this file assumes the existence
            # of various arithmetic ops to figure out what to print. We hack
            # and convert to float here to make printing work correctly.
            # TODO(#113663): also add the other float8 dtypes here after arithmetic
            # support for them is removed
            nonzero_finite_vals = nonzero_finite_vals.float()

        # Convert to double (or float) for easy calculation. HalfTensor overflows with 1e8, and there's no div() on CPU.
        nonzero_finite_abs = tensor_totype(nonzero_finite_vals.abs())
        nonzero_finite_min = tensor_totype(nonzero_finite_abs.min())
        nonzero_finite_max = tensor_totype(nonzero_finite_abs.max())

        for value in nonzero_finite_vals:
            if value != torch.ceil(value):
                self.int_mode = False
                break

        self.sci_mode = (
            nonzero_finite_max / nonzero_finite_min > 1000.0
            or nonzero_finite_max > 1.0e8
            or nonzero_finite_min < 1.0e-4
            if PRINT_OPTS.sci_mode is None
            else PRINT_OPTS.sci_mode
        )

        if self.int_mode:
            # in int_mode for floats, all numbers are integers, and we append a decimal to nonfinites
            # to indicate that the tensor is of floating type. add 1 to the len to account for this.
            if self.sci_mode:
                for value in nonzero_finite_vals:
                    value_str = f"{{:.{PRINT_OPTS.precision}e}}".format(value)
                    self.max_width = max(self.max_width, len(value_str))
            else:
                for value in nonzero_finite_vals:
                    value_str = f"{value:.0f}"
                    self.max_width = max(self.max_width, len(value_str) + 1)
        else:
            # Check if scientific representation should be used.
            if self.sci_mode:
                for value in nonzero_finite_vals:
                    value_str = f"{{:.{PRINT_OPTS.precision}e}}".format(value)
                    self.max_width = max(self.max_width, len(value_str))
            else:
                for value in nonzero_finite_vals:
                    value_str = f"{{:.{PRINT_OPTS.precision}f}}".format(value)
                    self.max_width = max(self.max_width, len(value_str))
