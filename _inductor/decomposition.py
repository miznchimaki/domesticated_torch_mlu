from typing import Optional

import torch
import torch._prims_common as utils
from torch import Tensor
from torch._decomp import remove_decompositions
from torch._decomp.decompositions import (
    pw_cast_for_opmath,
    upsample_compute_output_size,
    _upsample_nearest,
    matmul,
)
from torch._inductor.decomposition import decompositions
from torch._inductor.lowering import lowerings, make_fallback
from torch._prims_common import (
    DimsSequenceType,
    TensorLikeType,
)
from torch._prims_common.wrappers import out_wrapper


from . import decomposition_utils
from ..utils import gorilla

aten = torch.ops.aten
DispatchKey = torch._C.DispatchKey


def remove_py_kernels(aten_ops):
    dispatch_keys = [DispatchKey.Autograd, DispatchKey.CompositeImplicitAutograd]
    for op in aten_ops:
        if hasattr(op, "py_kernels"):
            for key in dispatch_keys:
                op.py_kernels.pop(key, None)


# Exclude below decompositions to get a better performance for now.
# The following blacklist will be gradually reduced to incorporate more operators.
mlu_python_override_to_exclude = [
    aten.native_batch_norm.default,
    aten.upsample_bicubic2d.vec,
    aten.upsample_bilinear2d.default,
    aten.upsample_nearest2d.default,
]
remove_py_kernels(mlu_python_override_to_exclude)


remove_decompositions(decompositions, decomposition_utils.mlu_decomps_to_exclude)
remove_decompositions(decompositions, decomposition_utils.mlu_libdevice_to_exclude)
decomposition_utils.make_fallbacks(decomposition_utils.mlu_decomps_to_exclude)
decomposition_utils.make_fallbacks(
    [
        torch.ops.torch_mlu.fused_mm.default,
        torch.ops.torch_mlu.fused_bmm.default,
    ]
)


# Add by CAMBRICON
# The logic is consistent with cnnl_native_batch_norm(aten/operators/cnnl/native_batch_norm.cpp)
def input_shape_helper(x, num_features, dim):
    if dim > 5:
        x = x.reshape({x.size(0), num_features, -1})
    dim = x.dim()
    # PT do not have NLC channel last format currently, so we go NHWC
    if 3 == dim:
        x = x.unsqueeze(3)
    return x


def recover_output_shape_helper(y, sizes, dim):
    if 3 == dim or dim > 5:
        y = y.squeeze(3)
    if dim > 5:
        y = y.reshape(sizes)
    return y


# remove the patch after native_batch_norm was removed from mlu_decomps_to_exclude
# aten.native_batch_norm is removed from the decomposition list.
# therefore, graph will invoke cnnl_native_batch_norm(cnnl/native_batch_norm.cpp).
# fx_graph derive the size and stride of the output in lowering phase
# using decompotision_fn in decomposition_list(decomposition_list[aten.native_batch_norm]),
# The output of decomposition_fn is continuous, while the output of
# cnnl_native_batch_norm is discontinuous in some case.
def native_batch_norm_helper(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
    functional: bool,
) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    reduction_dims = [0] + list(range(2, input.dim()))
    computation_dtype = utils.get_computation_dtype(input.dtype)
    new_running_mean = running_mean
    new_running_var = running_var
    if training:
        computation_dtype = utils.get_computation_dtype(input.dtype)
        input_acc = input.to(dtype=computation_dtype)
        biased_var, mean = torch.var_mean(
            input_acc, dim=reduction_dims, correction=0, keepdim=True
        )
        rstd = torch.rsqrt(biased_var + eps)

        output = (input - mean) * rstd

        save_mean = torch.squeeze(mean, reduction_dims)
        save_rstd = torch.squeeze(rstd, reduction_dims)
        if running_mean is not None:
            new_running_mean = momentum * save_mean + (1 - momentum) * running_mean
            if not functional:
                running_mean.copy_(new_running_mean)
        if running_var is not None:
            n = input.numel() / input.shape[1]
            # This doesn't strictly match eager's numerics, which accumulates var sum and then directly applies the correction
            # But... that would require re-implementing var here, for negligible numerics gain on a tensor whose
            # numerics probably don't matter.
            squeezed_var = torch.squeeze(biased_var, reduction_dims)
            unbiased_var = squeezed_var * (n / (n - 1))
            new_running_var = momentum * unbiased_var + (1 - momentum) * running_var
            if not functional:
                running_var.copy_(new_running_var)
    else:
        if running_mean is None or running_var is None:
            raise AssertionError(
                "running_mean and running_var must not be None in eval mode"
            )
        running_mean = running_mean.to(dtype=computation_dtype, copy=True)
        new_running_mean = running_mean
        running_var = running_var.to(dtype=computation_dtype, copy=True)
        new_running_var = running_var
        mean = running_mean
        invstd = 1 / (torch.sqrt(running_var + eps))
        # Very annoying inconsistency where CPU and CUDA give different shapes
        if input.device.type != "cpu":
            save_mean = running_mean
            save_rstd = invstd
        else:
            save_mean = input.new_zeros((0,))
            save_rstd = input.new_zeros((0,))
        mean = _unsqueeze_to_dim(mean, input.dim() - 1)
        invstd = _unsqueeze_to_dim(invstd, input.dim() - 1)
        output = (input - mean) * invstd

    if weight is not None:
        weight = weight.flatten()
        weight = _unsqueeze_to_dim(weight, input.dim() - 1)
        output = output * weight

    if bias is not None:
        bias = bias.flatten()
        bias = _unsqueeze_to_dim(bias, input.dim() - 1)
        output = output + bias

    if input.device.type == "cpu":
        save_mean = save_mean.to(dtype=input.dtype)
        save_rstd = save_rstd.to(dtype=input.dtype)

    # Modify by CAMBRICON
    if (
        isinstance(input, torch._subclasses.FakeTensor)
        and input.fake_device.type == "mlu"
    ):
        import torch_mlu
        from torch_mlu._inductor.decomposition import (
            input_shape_helper,
            recover_output_shape_helper,
        )

        orig_dim = input.dim()
        num_features = input.size(1)
        orig_size = input.size()
        input_t = input_shape_helper(input, num_features, orig_dim)

        memory_format = (
            torch.contiguous_format
            if orig_dim == 2
            else torch_mlu._MLUC._get_channels_last_memory_format(input_t.dim())
        )
        input_t = input_t.contiguous(memory_format=memory_format)
        output = output.reshape(input_t.size())
        output = output.contiguous(memory_format=memory_format)
        output = recover_output_shape_helper(output, orig_size, orig_dim)
    # end Modify by CAMBRICON

    return (
        output.to(dtype=input.dtype),
        save_mean,
        save_rstd,
        new_running_mean,
        new_running_var,
    )


patch = gorilla.Patch(
    torch._decomp.decompositions, "native_batch_norm_helper", native_batch_norm_helper
)
gorilla.apply(patch)


# remove the patch after native_batch_norm_backward was removed from mlu_decomps_to_exclude
def native_batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_invstd: Optional[Tensor],
    train: bool,
    eps: float,
    output_mask: list[bool],
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    input_dtype = input.dtype
    if weight is not None:
        weight_dtype = weight.dtype
    else:
        weight_dtype = input_dtype
    computation_dtype = utils.get_computation_dtype(input.dtype)
    (
        grad_out_cast,
        input_cast,
        weight_cast,
        running_mean_cast,
        running_var_cast,
        save_mean_cast,
        save_invstd_cast,
    ) = (
        x.to(computation_dtype) if x is not None else x
        for x in (
            grad_out,
            input,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
        )
    )
    input_shape = input.shape
    input_rank = input.dim()
    if input_rank < 2:
        raise AssertionError(f"rank of the input must be at least 2, got {input_rank}")

    axis = 1
    num_features = prod(list(input_shape)) / input_shape[axis]
    mean = save_mean_cast
    invstd = save_invstd_cast
    if train:
        if mean is None or invstd is None:
            raise AssertionError("mean and invstd must not be None in training mode")

    else:
        if running_mean_cast is None or running_var_cast is None:
            raise AssertionError(
                "running_mean_cast and running_var_cast must not be None in eval mode"
            )
        mean = running_mean_cast
        invstd = torch.rsqrt(running_var_cast + eps)

    broadcast_mask: list[int] = [1] * input_rank
    broadcast_mask[axis] = input_shape[axis]

    reduction_axes: list[int] = []
    for i in range(input_rank):
        if i != axis:
            reduction_axes.append(i)

    mean = _broadcast_batch_norm_backward(mean, broadcast_mask)  # type: ignore[arg-type]
    norm = 1.0 / num_features
    grad_output_sum = torch.sum(grad_out_cast, reduction_axes)  # type: ignore[arg-type]
    dot_p = torch.sum(grad_out_cast * (input_cast - mean), reduction_axes)  # type: ignore[operator]

    grad_mean = _broadcast_batch_norm_backward(grad_output_sum * norm, broadcast_mask)
    proj_scale = _broadcast_batch_norm_backward(
        torch.mul(dot_p * norm, invstd * invstd),  # type: ignore[operator]
        broadcast_mask,
    )

    if weight_cast is None:
        grad_scale = _broadcast_batch_norm_backward(invstd, broadcast_mask) * 1.0  # type: ignore[arg-type]
    else:
        grad_scale = _broadcast_batch_norm_backward(
            invstd * weight_cast, broadcast_mask
        )

    if train:
        proj = (input_cast - mean) * proj_scale  # type: ignore[operator]
        grad_input = ((grad_out_cast - proj) - grad_mean) * grad_scale
    else:
        grad_input = grad_out_cast * grad_scale

    if output_mask[1]:
        grad_weight = dot_p * invstd
    else:
        grad_weight = None  # "None" doesn't work with vjp, should use zeros for vjp

    if output_mask[2]:
        grad_bias = grad_output_sum
    else:
        grad_bias = None  # "None" doesn't work with vjp, should use zeros for vjp

    # Add by CAMBRICON
    if (
        isinstance(input, torch._subclasses.FakeTensor)
        and input.fake_device.type == "mlu"
    ):
        import torch_mlu
        from torch_mlu._inductor.decomposition import (
            input_shape_helper,
            recover_output_shape_helper,
        )

        orig_dim = input.dim()
        num_features = input.size(1)
        orig_size = input.size()
        input_t = input_shape_helper(input, num_features, orig_dim)
        memory_format = (
            torch.contiguous_format
            if orig_dim == 2
            else torch_mlu._MLUC._get_channels_last_memory_format(input_t.dim())
        )
        input_t = input_t.contiguous(memory_format=memory_format)
        grad_input = grad_input.as_strided(input_t.size(), input_t.stride())
        if output_mask[0]:
            grad_input = recover_output_shape_helper(grad_input, orig_size, orig_dim)
    # end Add by CAMBRICON

    return (
        grad_input.to(input_dtype),
        _maybe_cast(grad_weight, weight_dtype),
        _maybe_cast(grad_bias, weight_dtype),
    )


patch = gorilla.Patch(
    torch._decomp.decompositions,
    "native_batch_norm_backward",
    native_batch_norm_backward,
)
gorilla.apply(patch)


@pw_cast_for_opmath
def _upsample_linear(
    input: Tensor,
    output_size: list[int],
    align_corners: bool,
    scales: list[Optional[float]],
) -> Tensor:
    # get dimensions of original image
    n_channels = input.shape[1]
    inp_sizes = input.shape[2:]
    n_dims = len(inp_sizes)

    _, dtype = utils.elementwise_dtypes(
        input,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )

    def get_values(inp_size, out_size, scales, nsqueeze):
        # First Calculate scaling factor
        scale_factor = _compute_scale(inp_size, out_size, align_corners, scales)
        # We have to create arange with int64 dtype and use .to in order to avoid
        # additional kernels creation in inductor and get a perf slowdown
        i = torch.arange(out_size, device=input.device).to(dtype=dtype)

        x_f32 = _compute_source_index(scale_factor, i, align_corners).clamp(min=0.0)
        x_f32 = x_f32.reshape(x_f32.shape[0], *[1] * (nsqueeze))
        x = x_f32.to(torch.int64)
        xp1 = (x + 1).clamp(max=inp_size - 1)
        return x_f32, x, xp1

    values = [
        get_values(inp_size, out_size, scales, n_dims - 1 - i)
        for i, (inp_size, out_size, scales) in enumerate(
            zip(inp_sizes, output_size, scales)
        )
    ]
    xs_f32, xs, xp1s = list(zip(*values))

    vs = []
    for a in product(*[[0, 1]] * n_dims):
        idx = [None, None] + [xs[k] if a[k] == 0 else xp1s[k] for k in range(n_dims)]
        v = aten._unsafe_index(input, idx)
        v = _maybe_convert_to_dtype(v, dtype)
        vs.append(v)

    for i in reversed(range(n_dims)):
        xscale = (xs_f32[i] - xs[i]).clamp(0.0, 1.0).to(dtype)
        vs = [
            # x1 * (1 - alpha) + x2 * alpha == x1 + (x2 - x1) * alpha
            v1 + torch.mul(v2 - v1, xscale)
            for v1, v2 in zip(vs[::2], vs[1::2])
        ]

    if len(vs) != 1:
        raise AssertionError(f"Expected vs to have exactly 1 element, got {len(vs)}")
    result = vs[0]

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(input)

    # following "heuristic: only use channels_last path when it's faster than the contiguous path"
    if input.device.type == "cuda" and n_channels < 16:
        memory_format = torch.contiguous_format

    if not isinstance(result, torch.Tensor):
        raise AssertionError(
            f"Expected result to be a Tensor, got {type(result).__name__}"
        )

    # Modify by CAMBRICON
    # result = result.contiguous(memory_format=memory_format)
    if (
        isinstance(input, torch._subclasses.FakeTensor)
        and input.fake_device.type == "mlu"
    ):

        def get_channels_last_strides_1d(sizes):
            n_sizes = len(sizes)
            strides = [0] * n_sizes
            if n_sizes == 3:
                strides[1] = 1
                strides[2] = sizes[1]
                strides[0] = strides[2] * sizes[2]
            elif n_sizes == 2:
                strides[0] = 1
                strides[1] = sizes[0]
            else:
                return RuntimeError(
                    "get_channels_last_strides_1d does not support shape as ", sizes
                )
            return strides

        if n_dims == 1:
            result = result.as_strided(
                result.size(), get_channels_last_strides_1d(result.size())
            )
        elif n_dims == 2:
            result = result.contiguous(memory_format=torch.channels_last)
        elif n_dims == 3:
            result = result.contiguous(memory_format=torch.channels_last_3d)
    else:
        result = result.contiguous(memory_format=memory_format)
    # end Modify by CAMBRICON

    if not input.is_floating_point():
        result = result.round()

    return result


patch = gorilla.Patch(
    torch._decomp.decompositions,
    "_upsample_linear",
    _upsample_linear,
)
gorilla.apply(patch)


@out_wrapper()
@pw_cast_for_opmath
def upsample_bicubic2d_default(
    input: Tensor,
    output_size: tuple[int, int],
    align_corners: bool,
    scale_h: Optional[float] = None,
    scale_w: Optional[float] = None,
) -> Tensor:
    # get dimensions of original image
    _, _, in_h, in_w = input.shape

    # Calculate horizontal and vertical scaling factor
    h_scale_factor = _compute_scale(in_h, output_size[0], align_corners, scale_h)
    w_scale_factor = _compute_scale(in_w, output_size[1], align_corners, scale_w)

    _, dtype = utils.elementwise_dtypes(
        input, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )

    # We have to create arange with int64 dtype and use .to in order to avoid
    # additional kernels creation in inductor and get a perf slowdown
    i = torch.arange(output_size[0], device=input.device).to(dtype=dtype)
    j = torch.arange(output_size[1], device=input.device).to(dtype=dtype)

    x_float = _compute_source_index(w_scale_factor, j, align_corners)
    y_float = _compute_source_index(h_scale_factor, i, align_corners)
    y_float = y_float.unsqueeze(-1)

    x = x_float.floor()
    y = y_float.floor()

    # We should also clamp xscale/yscale
    # See guard_index_and_lambda in UpSample.h
    yscale = (y_float - y).clamp(0.0, 1.0)
    xscale = (x_float - x).clamp(0.0, 1.0)
    x = x.to(torch.int64)
    y = y.to(torch.int64)

    iys_ofs = (y - 1, y, y + 1, y + 2)
    ixs_ofs = (x - 1, x, x + 1, x + 2)

    weights_x = _upsample_get_cubic_coefficients(xscale)
    weights_y = _upsample_get_cubic_coefficients(yscale)

    weights_precision_x, weights_precision_y = None, None
    if input.dtype == torch.uint8:
        weights_precision_x = _compute_weight_precision(weights_x)
        weights_precision_y = _compute_weight_precision(weights_y)

        weights_x = [
            (w * (1 << weights_precision_x) + torch.sign(w) * 0.5).to(torch.int16)
            for w in weights_x
        ]
        weights_y = [
            (w * (1 << weights_precision_y) + torch.sign(w) * 0.5).to(torch.int16)
            for w in weights_y
        ]

    def load_bounded(ys, xs):
        y_idx = torch.clamp(ys, 0, in_h - 1)
        x_idx = torch.clamp(xs, 0, in_w - 1)
        v = aten._unsafe_index(input, [None, None, y_idx, x_idx])
        return v

    def get_x_interp(y):
        src_x = tuple(load_bounded(y, x_ofs) for x_ofs in ixs_ofs)
        if input.dtype == torch.uint8:
            if weights_precision_x is None:
                raise AssertionError(
                    "weights_precision_x must not be None for uint8 input"
                )
            return _sum_tensors_uint8(src_x, weights_x, weights_precision_x)
        return _sum_tensors(c1 * c2 for (c1, c2) in zip(src_x, weights_x))

    src_y = tuple(get_x_interp(y_ofs) for y_ofs in iys_ofs)
    if input.dtype == torch.uint8:
        if weights_precision_y is None:
            raise AssertionError("weights_precision_y must not be None for uint8 input")
        result = _sum_tensors_uint8(src_y, weights_y, weights_precision_y)
    else:
        result = _sum_tensors(c1 * c2 for (c1, c2) in zip(src_y, weights_y))

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(input)

    # Modify by CAMBRICON
    # result = result.contiguous(memory_format=memory_format)
    if (
        isinstance(result, torch._subclasses.FakeTensor)
        and input.fake_device.type == "mlu"
    ):
        result = result.contiguous(memory_format=torch.channels_last)
    else:
        result = result.contiguous(memory_format=memory_format)
    # end Modify by CAMBRICON
    return result


patch = gorilla.Patch(
    torch._decomp.decompositions,
    "upsample_bicubic2d_default",
    upsample_bicubic2d_default,
)
gorilla.apply(patch)


def flip(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType:
    if not isinstance(dims, tuple) and not isinstance(dims, list):
        raise ValueError("dims has to be a sequence of ints")
    dims = utils.canonicalize_dims(a.ndim, dims)  # type: ignore[assignment]
    utils.validate_no_repeating_dims(dims)

    # Add by CAMBRICON
    if isinstance(a, torch._subclasses.FakeTensor) and a.fake_device.type == "mlu":
        out = prims.rev(a, dims)
        return torch.empty_like(out, memory_format=utils.suggest_memory_format(a))
    # end Add by CAMBRICON

    return prims.rev(a, dims)


patch = gorilla.Patch(
    torch._refs,
    "flip",
    flip,
)
gorilla.apply(patch)


def _softmax_backward_data(
    grad_output: Tensor, output: Tensor, dim: int, input_dtype: torch.dtype
):
    new_grad_output = grad_output * output
    grad_input = new_grad_output - output * torch.sum(
        new_grad_output, dim=dim, keepdim=True
    )

    # CPU kernel doesn't respect input_dtype, but following check doesn't work for meta tensor
    # if grad_output.device == torch.device("cpu"):
    #     return grad_input.contiguous()
    # Modify by CAMBRICON
    # return _cast_grad_to_input_dtype(grad_output, grad_input, input_dtype).contiguous()
    if (
        isinstance(grad_output, torch._subclasses.FakeTensor)
        and grad_output.fake_device.type == "mlu"
    ):
        memory_format = utils.suggest_memory_format(grad_output)
    else:
        memory_format = torch.contiguous_format

    return _cast_grad_to_input_dtype(grad_output, grad_input, input_dtype).contiguous(
        memory_format=memory_format
    )
    # end Modify by CAMBRICON


patch = gorilla.Patch(
    torch._decomp.decompositions,
    "_softmax_backward_data",
    _softmax_backward_data,
)
gorilla.apply(patch)


def _log_softmax(x: Tensor, dim: int, half_to_float: bool):
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    # eager log_softmax returns a contiguous tensor. Ensure that decomp also
    # returns a contiguous tensor.
    # Modify by CAMBRICON
    if isinstance(x, torch._subclasses.FakeTensor) and x.fake_device.type == "mlu":
        memory_format = utils.suggest_memory_format(x)
    # end Modify by CAMBRICON
    x = x.contiguous()
    if half_to_float:
        if x.dtype != torch.half:
            raise AssertionError(
                f"half_to_float is True but x.dtype is {x.dtype}, expected torch.half"
            )
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        x, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    x = x.to(computation_dtype)
    if guard_or_false(x.numel() == 0):
        shifted = x
    else:
        x_max = torch.amax(x, dim, keepdim=True)
        shifted = x - x_max
    shifted_logsumexp = torch.log(torch.sum(torch.exp(shifted), dim, keepdim=True))
    result = shifted - shifted_logsumexp
    if not half_to_float:
        result = result.to(result_dtype)

    # Modify by CAMBRICON
    if (
        isinstance(result, torch._subclasses.FakeTensor)
        and result.fake_device.type == "mlu"
    ):
        result = result.to(memory_format=memory_format)
    # end Modify by CAMBRICON
    return result


patch = gorilla.Patch(
    torch._decomp.decompositions,
    "_log_softmax",
    _log_softmax,
)
gorilla.apply(patch)


def _log_softmax_backward_data(
    grad_output: Tensor, output: Tensor, dim: int, input_dtype: torch.dtype
):
    grad_input = grad_output - torch.exp(output) * torch.sum(
        grad_output, dim=dim, keepdim=True
    )

    # Modify by CAMBRICON
    if (
        isinstance(grad_output, torch._subclasses.FakeTensor)
        and grad_output.fake_device.type == "mlu"
    ):
        grad_input = _cast_grad_to_input_dtype(grad_output, grad_input, input_dtype)
        return grad_input.to(memory_format=utils.suggest_memory_format(grad_output))
    # end Modify by CAMBRICON
    return _cast_grad_to_input_dtype(grad_output, grad_input, input_dtype)


patch = gorilla.Patch(
    torch._decomp.decompositions,
    "_log_softmax_backward_data",
    _log_softmax_backward_data,
)
gorilla.apply(patch)


# We can not add the aten.upsample_nearest2d.vec to mlu_python_override_to_exclude,
# because that will break dynamic shape function.
def _upsample_nearest_vec(
    input: Tensor,
    output_size: Optional[list[int]],
    scale_factors: Optional[list[float]],
) -> Tensor:
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    scales = (
        scale_factors if scale_factors else [None] * len(osize)  # type: ignore[list-item]
    )
    # Modify by CAMBRICON for perf, because the perf of decomposed ops are poor
    # return _upsample_nearest(input, osize, scales)
    if len(scales) == 2:
        return aten.upsample_nearest2d.default(input, osize, *scales)
    else:
        return _upsample_nearest(input, osize, scales)
    # end Modify by CAMBRICON


patch = gorilla.Patch(
    torch._decomp.decompositions,
    "_upsample_nearest_vec",
    _upsample_nearest_vec,
)
gorilla.apply(patch)


# We can not add the aten.upsample_bilinear2d.vec to mlu_python_override_to_exclude,
# because that will break dynamic shape function.
def _upsample_linear_vec(input, output_size, align_corners, scale_factors):
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    scales = scale_factors if scale_factors else [None] * len(osize)
    # Modify by CAMBRICON for perf, because the perf of decomposed ops are poor
    # return _upsample_linear(input, osize, align_corners, scales)
    if len(scales) == 2:
        return aten.upsample_bilinear2d.default(input, osize, align_corners, *scales)
    else:
        return _upsample_linear(input, osize, align_corners, scales)
    # end Modify by CAMBRICON


patch = gorilla.Patch(
    torch._decomp.decompositions,
    "_upsample_linear_vec",
    _upsample_linear_vec,
)
gorilla.apply(patch)


def _weight_norm_interface(v, g, dim=0):
    # https://github.com/pytorch/pytorch/blob/852f8526c52190125446adc9a6ecbcc28fb66182/aten/src/ATen/native/WeightNorm.cpp#L58
    keep_dim = tuple(i for i in range(len(v.shape)) if i != dim)
    # align with cuda behavior, keep norm in 'float' when g is 'bfloat16'
    # Modify by CAMBRICON
    # norm_dtype = torch.float if g.dtype == torch.bfloat16 else None
    # norm = v.norm(2, keep_dim, keepdim=True, dtype=norm_dtype)
    # return v * (g / norm.to(g.dtype)), norm
    if isinstance(v, torch._subclasses.FakeTensor) and v.fake_device.type == "mlu":
        # When _weight_norm_interface is in the mlu_decomps_to_exclude list, it will not be decomposed, we only need to ensure that the shape inference is correct.
        from torch_mlu._inductor.decomposition_utils import mlu_decomps_to_exclude

        assert aten._weight_norm_interface in mlu_decomps_to_exclude

        norm_dtype = (
            torch.float
            if g.dtype == torch.bfloat16 or g.dtype == torch.float16
            else g.dtype
        )
        w = torch.empty_like(v, memory_format=torch.contiguous_format)
        norm = torch.empty_strided(
            g.size(), g.stride(), dtype=norm_dtype, device=g.device
        )

        return w, norm
    else:
        norm_dtype = torch.float if g.dtype == torch.bfloat16 else None
        norm = v.norm(2, keep_dim, keepdim=True, dtype=norm_dtype)
        return v * (g / norm.to(g.dtype)), norm
    # end Modify by CAMBRICON


patch = gorilla.Patch(
    torch._decomp.decompositions,
    "_weight_norm_interface",
    _weight_norm_interface,
)
gorilla.apply(patch)


def should_fold(tensor1: torch.Tensor, tensor2: torch.Tensor, is_out: bool) -> bool:
    # For comments of the logic of this function see eager in /native/LinearAlgebra.cpp

    t1, t2 = (tensor1, tensor2) if tensor1.ndim >= tensor2.ndim else (tensor2, tensor1)

    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if not (t1.ndim >= 3 and t2.ndim <= 2):
        return False
    if t2.requires_grad and not is_out:
        return True
    if tensor1.ndim == 2:
        return False
    # Modify by CAMBRICON, align with torch_mlu/csrc/aten/operators/cnnl/matmul.cpp:should_fold
    if tensor1.device.type == "mlu" or tensor2.device.type == "mlu":
        return True
    # end Modify by CAMBRICON
    if guard_or_false(t1.numel() == 0):
        return True

    t1_shape = t1.shape
    t1_stride = t1.stride()

    # Check the contiguous, we can skip the dim with size of 1
    # as aten: https://github.com/pytorch/pytorch/blob/e201460f8aa1510b4c4686627d57b69756c4b916/aten/src/ATen/TensorGeometry.cpp#L17
    expected_stride = [1]
    for size in reversed(t1_shape[1:]):
        expected_stride.append(size * expected_stride[-1])
    return all(
        guard_or_false(size == 1) or guard_or_false(left == right)
        for left, right, size in zip(
            t1_stride, list(reversed(expected_stride)), t1_shape
        )
    )


patch = gorilla.Patch(
    torch._decomp.decompositions,
    "should_fold",
    should_fold,
)
gorilla.apply(patch)


def native_layer_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: list[int],
    mean: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    output_mask: list[bool],
) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    input_shape = input.shape
    input_ndim = input.dim()
    computation_dtype = utils.get_computation_dtype(input.dtype)
    grad_out_cast, input_cast, weight_cast, bias_cast = (
        x.to(computation_dtype, memory_format=torch.contiguous_format)
        if x is not None
        else x
        for x in (grad_out, input, weight, bias)
    )
    if grad_out_cast is None:
        raise AssertionError("grad_out_cast should not be None")

    axis = input_ndim - len(normalized_shape)
    inner_dims = input_shape[axis:]
    outer_dims = input_shape[:axis]
    inner_dim_indices: list[int] = []
    outer_dim_indices: list[int] = []
    for i in range(input_ndim):
        if i >= axis:
            inner_dim_indices.append(i)
        else:
            outer_dim_indices.append(i)

    N = prod(inner_dims)  # type: ignore[arg-type]
    M = prod(outer_dims)  # type: ignore[arg-type]
    from torch.fx.experimental.symbolic_shapes import statically_known_true

    if statically_known_true(M == 0) or statically_known_true(N == 0):
        return (
            input.new_zeros(input_shape) if output_mask[0] else None,
            input.new_zeros(input_shape[axis:]) if output_mask[1] else None,
            input.new_zeros(input_shape[axis:]) if output_mask[2] else None,
        )
    mean = _unsqueeze_to_dim(mean, input_cast.dim())  # type: ignore[union-attr]
    rstd = _unsqueeze_to_dim(rstd, input_cast.dim())  # type: ignore[union-attr]
    if input_cast is None:
        raise AssertionError("input_cast should not be None")
    x_hat = (input_cast - mean) * rstd
    if weight_cast is not None:
        grad_x_hat = grad_out_cast * weight_cast
    else:
        grad_x_hat = grad_out_cast
    a = grad_x_hat * N
    b = torch.sum(grad_x_hat, inner_dim_indices, True)
    c1 = torch.mul(grad_x_hat, x_hat)
    c2 = torch.sum(c1, inner_dim_indices, True)
    c3 = torch.mul(x_hat, c2)

    inner = a - b - c3
    d_input: Optional[Tensor] = None
    d_weight: Optional[Tensor] = None
    d_bias: Optional[Tensor] = None
    if output_mask[0]:
        d_input = (rstd / N) * inner

    if output_mask[1] and weight_cast is not None:
        if len(outer_dim_indices) > 0:
            d_weight = torch.sum(grad_out_cast * x_hat, outer_dim_indices, False)
        else:
            d_weight = grad_out_cast * x_hat

    if output_mask[2] and bias_cast is not None:
        if len(outer_dim_indices) > 0:
            d_bias = torch.sum(grad_out_cast, outer_dim_indices, False)
        else:
            d_bias = grad_out_cast.clone()

    # Add by CAMBRICON
    from torch._inductor.lowering import fallbacks

    if (
        input.device.type == "mlu"
        and aten.native_layer_norm_backward.default in fallbacks
    ):
        d_input = (
            d_input.contiguous(memory_format=torch.contiguous_format)
            if d_input is not None
            else None
        )
        d_weight = (
            d_weight.contiguous(memory_format=torch.contiguous_format)
            if d_weight is not None
            else None
        )
        d_bias = (
            d_bias.contiguous(memory_format=torch.contiguous_format)
            if d_bias is not None
            else None
        )

    # end Add by CAMBRICON

    return (
        _maybe_cast(d_input, input.dtype),
        _maybe_cast(d_weight, weight.dtype if weight is not None else None),
        _maybe_cast(d_bias, bias.dtype if bias is not None else None),
    )


patch = gorilla.Patch(
    torch._decomp.decompositions,
    "native_layer_norm_backward",
    native_layer_norm_backward,
)
gorilla.apply(patch)


def silu(x: torch.Tensor) -> torch.Tensor:
    # Modify by CAMBRICON, align with torch_mlu eager and improve perf
    # return x / (1 + x.neg().exp())
    return x * x.sigmoid()
    # end Modify by CAMBRICON


patch = gorilla.Patch(
    torch._inductor.decomposition,
    "silu",
    silu,
)
gorilla.apply(patch)

# Because MLU eager register AutogradPrivateUse1 impl for this op, which do not support
# dynamic shape but has higher priority than CompositeImplicitAutograd decompose impl
aten.matmul.default.py_impl(DispatchKey.AutogradPrivateUse1)(matmul)
aten.matmul.out.py_impl(DispatchKey.AutogradPrivateUse1)(matmul)
