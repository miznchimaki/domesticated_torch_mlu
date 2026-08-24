import importlib
from typing import Callable
import functools

import torch
import torch.utils._pytree as pytree
import torch._prims_common as utils
from torch._higher_order_ops.utils import (
    _maybe_compile_and_run_fn,
    _set_compilation_env,
    first_slice_copy,
)
from torch._higher_order_ops.associative_scan import (
    wrap_combine_fn_flat,
    generic_associative_scan,
    associative_scan_op,
)


def associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    xs: pytree.PyTree,
    dim: int,
    reverse: bool = False,
    combine_mode: str = "pointwise",
) -> torch.Tensor:
    r"""
    Performs an inclusive scan with an associative combine function.

    .. warning::
        `torch.associative_scan` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    This operator requires runtime code generation and so requires support for
    ``torch.compile``. Further, only CUDA device codegen is supported at the moment.

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, i.e., no lifted arguments are supported at the moment,
            satisfy the associative property and have no side-effects.
        xs (torch.Tensor): The input tensor, or nested pytree of tensors.
            All inputs are expected to have the same shape.
        dim (int): the dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.
        combine_mode (str): A string indicating whether the ``combine_fn`` is ``pointwise`` or ``generic``, default ``pointwise``.
            If ``combine_mode=pointwise``, ``combine_fn`` must be pure, may only contain pointwise operations
            and ``xs`` must be CUDA tensors.
            In all other cases ``combine_mode=generic`` should be used.
            Note: ``combine_mode=pointwise`` is more efficient than ``combine_mode=generic``.


    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y


        cumsum = associative_scan(add, x, dim)

    """
    # TODO: Support lifted arguments in inductor for associative_scan
    # TODO: Support autograd for cases with lifted arguments for combine_mode=pointwise

    # The reason we flatten xs before calling into dynamo is that
    # we want to create a consistent input ordering for combine_fn
    # and we also want to the input ordering matches the output ordering.
    leaves_xs_orig, spec_xs = pytree.tree_flatten(xs)

    def _validate_input(cfn, lxs, d, r, cm):
        # Basic arguments check
        if not callable(cfn):
            raise ValueError(f"Combine_fn must be a callable, but got {cfn}")
        if not isinstance(d, int):
            raise ValueError("Dim must be an int, but got " + str(type(d)))
        if not isinstance(r, bool):
            raise RuntimeError("Reverse must be a bool, but got " + str(type(r)))
        if cm not in ["pointwise", "generic"]:
            raise ValueError(
                f"Combine_mode must either 'pointwise' or 'generic', but got {cm}"
            )
        # Modify by CAMBRICON
        # if cm == "pointwise" and not all(l.device.type in ("cuda", "xpu") for l in lxs):
        #    raise ValueError(
        #        "For combine_mode='pointwise', all input tensors need to be on CUDA or XPU"
        #    )
        if cm == "pointwise" and not all(l.device.type in ("mlu") for l in lxs):
            raise ValueError(
                "For combine_mode='pointwise', all input tensors need to be on MLU"
            )
        # end Modify by CAMBRICON

        # Checks for xs
        if len(lxs) == 0:
            raise ValueError("Expected at least 1 xs leaf")
        if any(not isinstance(x, torch.Tensor) for x in lxs):
            raise ValueError("xs leaves must be a Tensor")
        if any(x.is_sparse for x in lxs):
            raise ValueError(
                "xs leaves must dense Tensors, consider using `to_dense()`"
            )
        if any(x.ndim <= d for x in lxs):
            raise ValueError(
                "All xs leaves must at least have 'dim' number of dimensions and scan dimension > 0"
            )
        if any(x.shape[d] == 0 for x in lxs):
            raise ValueError(
                "All xs leaves must at least have 'dim' number of dimensions and scan dimension > 0"
            )

    ndim = leaves_xs_orig[0].ndim
    dim = utils.canonicalize_dim(ndim, dim)

    _validate_input(combine_fn, leaves_xs_orig, dim, reverse, combine_mode)

    # Move scan dim to 0 and always perform scan on dim 0
    leaves_xs = [torch.movedim(elem, dim, 0) for elem in leaves_xs_orig]

    if reverse:
        leaves_xs = [torch.flip(elem, [0]) for elem in leaves_xs]

    if combine_mode == "generic":
        # The generic_associative_scan implementation calls the combine_fn with a `batch` along the scan dimension
        # For example, consider:
        # def add(x: torch.Tensor, y: torch.Tensor):
        #     return x + y
        # leaves = torch.tensor([[0.0, 1.0, 2.0, 3.0]
        #                        [0.0, 1.0, 2.0, 3.0]])
        # which has shape 2 x 4;
        # dim = 1;
        # In the first iteration of `_scan` the combine_fn gets invoked with
        # combine_fn([torch.tensor([[0.0, 2.0],
        #                           [0.0, 2.0]])],
        #            [torch.tensor([[1.0, 3.0],
        #                           [1.0, 3.0]])])
        # The arguments are of shape 2 x 2, but can be evaluated in parallel along the scan dimension.
        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=torch.vmap(
                combine_fn,
                in_dims=(
                    pytree.tree_unflatten([0] * len(leaves_xs), spec_xs),
                    pytree.tree_unflatten([0] * len(leaves_xs), spec_xs),
                ),
                out_dims=0,
            ),
            spec=spec_xs,
            num_leaves=len(leaves_xs),
        )
        out = generic_associative_scan(combine_fn, leaves_xs, additional_inputs=())
        out = pytree.tree_unflatten(out, spec_xs)
    else:
        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=combine_fn,
            spec=spec_xs,
            num_leaves=len(leaves_xs),
        )

        def run_flattened_associative_scan(combine_fn, leaves_xs):
            return associative_scan_op(combine_fn, leaves_xs, additional_inputs=())

        out = _maybe_compile_and_run_fn(
            run_flattened_associative_scan,
            combine_fn,
            leaves_xs,
        )

    if reverse:
        out = pytree.tree_map(lambda elem: elem.flip([0]), out)

    out = pytree.tree_map(lambda elem: torch.movedim(elem, 0, dim), out)

    return out


# TODO(luohaizhao): torch._higher_order_ops.associative_scan is a function object and overrid module object, so
# use importlib to get module object. Any better idea?
associative_scan_module = importlib.import_module(
    "torch._higher_order_ops.associative_scan"
)
associative_scan_module.associative_scan = associative_scan
torch._higher_order_ops.associative_scan = associative_scan
