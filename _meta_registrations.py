from collections.abc import Sequence
from typing import List, Optional, Union
import torch
from torch import Tensor
from torch._decomp import (
    _add_op_to_registry,
    _convert_out_params,
    meta_table,
)
from torch._prims_common.wrappers import out_wrapper

from torch.utils import _pytree as pytree
from .utils import gorilla

aten = torch.ops.aten
_meta_lib_dont_use_me_use_register_meta = torch.library.Library("aten", "IMPL", "Meta")


def register_meta(ops):
    def wrapper(fn):
        fn = _convert_out_params(fn)

        def register(op):
            if op in meta_table:
                meta_table.pop(op)
            _add_op_to_registry(meta_table, op, fn)

        pytree.tree_map_(register, ops)

        op_overloads = []
        for op in ops:
            if isinstance(op, torch._ops.OpOverloadPacket):
                for op_overload in op.overloads():
                    op_overload = getattr(op, op_overload)
                    op_overloads.append(op_overload)
            elif isinstance(op, torch._ops.OpOverload):
                op_overloads.append(op)

        for op_overload in op_overloads:
            op_overload.py_impl(torch._C.DispatchKey.Meta)(fn)
            _meta_lib_dont_use_me_use_register_meta.impl(op_overload, fn)

        return fn

    return wrapper


@torch.library.register_fake("torch_mlu::grouped_gemm")
def grouped_gemm_fake(
    a_list: List[Tensor],
    b_list: List[Tensor],
    c_list: Optional[List[Tensor]] = None,
    bias_list: Optional[List[Tensor]] = None,
    alpha_list: Optional[List[float]] = None,
    beta_list: Optional[List[float]] = None,
    trans_a: Optional[bool] = None,
    trans_b: Optional[bool] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    if out is not None:
        return out
    else:
        total_m = total_numel = 0
        common_n = None
        is_same_n = True
        for a_i, b_i in zip(a_list, b_list):
            if trans_a:
                a_i = a_i.transpose(-2, -1)
            if trans_b:
                b_i = b_i.transpose(-2, -1)
            total_m += a_i.size(0)
            total_numel += a_i.size(0) * b_i.size(1)
            if common_n is None:
                common_n = b_i.size(1)
            elif common_n != b_i.size(1):
                is_same_n = False
        if is_same_n:
            out_shape = (total_m, common_n)
        else:
            out_shape = (total_numel,)
        new_out = torch.empty(
            out_shape, dtype=a_i.dtype, device=a_i.device, requires_grad=False
        )
        return new_out


def meta__scaled_dot_product_fused_attention_overrideable(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_bias: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float | None = None,
):
    B = query.size(0)
    H_Q = query.size(1)
    S_Q = query.size(2)
    S_KV = key.size(2)
    D_V = value.size(-1)

    # Modify by CAMBRICON
    # res_shape = (B, H_Q, S_Q, D_V)
    res_shape = (B * S_Q, H_Q, D_V)
    # res = alloc_with_matching_layout(query, res_shape)
    res = (
        torch.empty(res_shape, dtype=query.dtype, device=query.device)
        .view(B, S_Q, H_Q, D_V)
        .transpose(1, 2)
    )

    # logsum_exp = torch.empty(
    #     (B, H_Q, S_Q),
    #     dtype=torch.float,
    #     device=query.device,
    # )
    logsum_exp = (
        torch.empty(
            (H_Q, B * S_Q),
            dtype=torch.float,
            device=query.device,
        )
        .view(H_Q, B, S_Q)
        .transpose(0, 1)
    )
    # end Modify by CAMBRICON

    # See Note [Seed and Offset]
    seed = torch.empty((), dtype=torch.long, device="meta")
    offset = torch.empty((), dtype=torch.long, device="meta")

    return (
        res,
        logsum_exp,
        None,
        None,
        S_Q,
        S_KV,
        seed,
        offset,
        None,
    )


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta__scaled_dot_product_fused_attention_overrideable",
    meta__scaled_dot_product_fused_attention_overrideable,
)
gorilla.apply(patch)


@torch.library.register_fake("torch_mlu::fused_mm")
def fused_mm_fake(
    mat1: Tensor,
    mat2: Tensor,
    activation: Optional[str] = None,
    bias: Optional[Tensor] = None,
    activation_param: Optional[float] = None,
    is_training: bool = True,
) -> Tensor:
    assert (
        mat1.dim() == 2 and mat2.dim() == 2
    ), "fused mm only support 2D Tensor (got {}D and {}D tensors)".format(
        mat1.dim(), mat2.dim()
    )
    assert (
        mat1.shape[1] == mat2.shape[0]
    ), "mat1.shape[1] ({}) must equal to mat2.shape[0] ({})".format(
        mat1.shape[1], mat2.shape[0]
    )
    out_shape = [mat1.shape[0], mat2.shape[1]]
    new_out = torch.empty(
        out_shape,
        dtype=mat1.dtype,
        device=mat1.device,
        memory_format=torch.contiguous_format,
    )
    return new_out


@torch.library.register_fake("torch_mlu::fused_bmm")
def fused_bmm_fake(
    mat1: Tensor,
    mat2: Tensor,
    activation: Optional[str] = None,
    bias: Optional[Tensor] = None,
    activation_param: Optional[float] = None,
    is_training: bool = True,
) -> Tensor:
    assert (
        mat1.dim() == 3 and mat2.dim() == 3
    ), f"fused bmm only support 3D Tensor, got mat1.dim()={mat1.dim()}, mat2.dim()={mat2.dim()}"
    assert (
        mat1.shape[2] == mat2.shape[1]
    ), f"mat1.shape[2] ({mat1.shape[2]}) must equal to mat2.shape[1] ({mat2.shape[1]})"
    assert (
        mat1.shape[0] == mat2.shape[0]
    ), f"mat1.shape[0] ({mat1.shape[0]}) must equal to mat2.shape[0] ({mat2.shape[0]})"
    out_shape = [mat1.shape[0], mat1.shape[1], mat2.shape[2]]
    new_out = torch.empty(out_shape, dtype=mat1.dtype, device=mat1.device)
    return new_out


@torch.library.register_fake("torch_mlu::fused_convolution")
def fused_convolution_fake(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    mode,
    slope,
) -> Tensor:
    assert mode in (
        "relu",
        "leaky_relu",
    ), f"fused_convolution only supports mode 'relu' and 'leaky_relu'"
    conv_out = torch.ops.aten.convolution.default(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    return conv_out


# func: _scaled_dot_product_fused_attention_overrideable_backward
# (Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor attn_bias, bool[4] grad_input_mask, Tensor out, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, Tensor philox_seed, Tensor philox_offset, *, float? scale=None)
# -> (Tensor grad_query, Tensor grad_key, Tensor grad_value, Tensor grad_attn_bias)
@register_meta(
    [
        aten._scaled_dot_product_fused_attention_overrideable_backward,
    ]
)
def meta__scaled_dot_product_fused_attention_overrideable_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_bias: Optional[Tensor],
    grad_input_mask: list[bool],
    out: Tensor,
    logsumexp: Tensor,
    cum_seq_q: Tensor,
    cum_seq_k: Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: Tensor,
    philox_offset: Tensor,
    scale: Optional[float] = None,
):
    q_t = query.transpose(1, 2).contiguous()
    k_t = key.transpose(1, 2).contiguous()
    v_t = value.transpose(1, 2).contiguous()
    grad_q = torch.empty_like(q_t).transpose(1, 2)
    grad_k = torch.empty_like(q_t).transpose(1, 2)
    grad_v = torch.empty_like(q_t).transpose(1, 2)
    grad_bias = None
    if attn_bias is not None and grad_input_mask[3]:
        grad_bias = torch.empty_like(bias)

    return grad_q, grad_k, grad_v, grad_bias


def get_channels_last_strides_2d(sizes):
    # same as c10/core/MemoryFormat.h
    strides = [0] * len(sizes)
    if len(sizes) == 4:
        strides[1] = 1
        strides[3] = sizes[1]
        strides[2] = strides[3] * sizes[3]
        strides[0] = strides[2] * sizes[2]
    elif len(sizes) == 3:
        strides[0] = 1
        strides[2] = sizes[0]
        strides[1] = strides[2] * sizes[2]
    else:
        raise RuntimeError("channel_last_strides_2d doesn't support size")

    return strides


def get_channels_last_strides_3d(sizes):
    # same as c10/core/MemoryFormat.h
    strides = [0] * len(sizes)
    if len(sizes) == 5:
        strides[1] = 1
        strides[4] = sizes[1]
        strides[3] = strides[4] * sizes[4]
        strides[2] = strides[3] * sizes[3]
        strides[0] = strides[2] * sizes[2]
    elif len(sizes) == 4:
        strides[0] = 1
        strides[3] = sizes[0]
        strides[2] = strides[3] * sizes[3]
        strides[1] = strides[2] * sizes[2]
    else:
        raise RuntimeError("channel_last_strides_2d doesn't support size")
    return strides


def meta_max_pool2d_with_indices(
    input,
    kernel_size,
    stride=(),
    padding=(0,),
    dilation=(1,),
    ceil_mode=False,
):
    (
        nInputPlane,
        outputHeight,
        outputWidth,
    ) = max_pool2d_checks_and_compute_shape(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )

    nbatch = input.size(-4) if input.dim() == 4 else 1

    # Modify by CAMBRICON
    if (
        isinstance(input, torch._subclasses.FakeTensor)
        and input.fake_device.type == "mlu"
    ):
        if input.dim() == 3:
            from torch_mlu._meta_registrations import get_channels_last_strides_2d

            size = [nInputPlane, outputHeight, outputWidth]
            output_strides = get_channels_last_strides_2d(size)
            return (
                torch.empty_strided(
                    size,
                    output_strides,
                    dtype=input.dtype,
                    device=input.device,
                ),
                torch.empty_strided(
                    size,
                    output_strides,
                    dtype=torch.int64,
                    device=input.device,
                ),
            )
        else:
            size = [nbatch, nInputPlane, outputHeight, outputWidth]
            return (
                torch.empty(
                    size,
                    dtype=input.dtype,
                    device=input.device,
                    memory_format=torch.channels_last,
                ),
                torch.empty(
                    size,
                    dtype=torch.int64,
                    device=input.device,
                    memory_format=torch.channels_last,
                ),
            )
    # end Modify by CAMBRICON

    memory_format = utils.suggest_memory_format(input)
    if input.dim() == 3:
        size = [nInputPlane, outputHeight, outputWidth]
    else:
        size = [nbatch, nInputPlane, outputHeight, outputWidth]
    return (
        torch.empty(
            size,
            dtype=input.dtype,
            device=input.device,
            memory_format=memory_format,
        ),
        torch.empty(
            size,
            dtype=torch.int64,
            device=input.device,
            memory_format=memory_format,
        ),
    )


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_max_pool2d_with_indices",
    meta_max_pool2d_with_indices,
)
gorilla.apply(patch)


def meta_max_pool2d_with_indices_backward(
    grad_output,
    self,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    indices,
):
    (
        nInputPlane,
        outputHeight,
        outputWidth,
    ) = max_pool2d_checks_and_compute_shape(
        self, kernel_size, stride, padding, dilation, ceil_mode
    )

    torch._check(
        self.dtype == grad_output.dtype,
        lambda: f"Expected dtype {self.dtype} for `gradOutput` but got dtype {grad_output.dtype}",
    )

    nOutputPlane = nInputPlane
    ndim = self.ndim

    def _check_dim_size(t):
        check_dim_size(t, ndim, ndim - 3, nOutputPlane)
        check_dim_size(t, ndim, ndim - 2, outputHeight)
        check_dim_size(t, ndim, ndim - 1, outputWidth)

    _check_dim_size(grad_output)
    _check_dim_size(indices)

    # Modify by CAMBRICON
    if (
        isinstance(self, torch._subclasses.FakeTensor)
        and self.fake_device.type == "mlu"
    ):
        if self.dim() == 3:
            from torch_mlu._meta_registrations import get_channels_last_strides_2d

            strides = get_channels_last_strides_2d(self.shape)
            return torch.empty_strided(
                self.shape,
                strides,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            return torch.empty(
                self.shape,
                dtype=self.dtype,
                device=self.device,
                memory_format=torch.channels_last,
            )
    # end Modify by CAMBRICON

    memory_format = utils.suggest_memory_format(self)
    return torch.empty(
        self.shape,
        dtype=self.dtype,
        device=self.device,
        memory_format=memory_format,
    )


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_max_pool2d_with_indices_backward",
    meta_max_pool2d_with_indices_backward,
)
gorilla.apply(patch)


def _pad2d_common(input, padding, *, is_reflection):
    dim_w = 2
    dim_h = 1
    dim_slices = 0
    nbatch = 1

    _padding_check_valid_input(input, padding, dim=2)

    ndim = input.ndim
    if ndim == 4:
        nbatch = input.size(0)
        dim_w += 1
        dim_h += 1
        dim_slices += 1

    pad_l, pad_r, pad_t, pad_b = padding

    nplane = input.size(dim_slices)
    input_h = input.size(dim_h)
    input_w = input.size(dim_w)
    output_h = input_h + pad_t + pad_b
    output_w = input_w + pad_l + pad_r

    if is_reflection:
        torch._check(
            pad_l < input_w and pad_r < input_w,
            lambda: (
                f"Argument #4: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_l}, {pad_r}) at dimension {dim_w} of input {input.shape}"
            ),
        )
        torch._check(
            pad_t < input_h and pad_b < input_h,
            lambda: (
                f"Argument #6: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_t}, {pad_b}) at dimension {dim_h} of input {input.shape}"
            ),
        )

    torch._check(
        output_w >= 1 or output_h >= 1,
        lambda: (
            f"input (H: {input_h} W: {input_w}) is too small. "
            f"Calculated output H: {output_h} W: {output_w}"
        ),
    )

    if input.ndim == 3:
        return input.new_empty((nplane, output_h, output_w))
    else:
        # Modify by CAMBRICON
        if (
            isinstance(input, torch._subclasses.FakeTensor)
            and input.fake_device.type == "mlu"
        ):
            from torch._prims_common import suggest_memory_format

            return input.new_empty((nbatch, nplane, output_h, output_w)).contiguous(
                memory_format=suggest_memory_format(input)
            )
        # end Modify by CAMBRICON

        return input.new_empty((nbatch, nplane, output_h, output_w))


patch = gorilla.Patch(
    torch._meta_registrations,
    "_pad2d_common",
    _pad2d_common,
)
gorilla.apply(patch)


def meta_conv(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    is_transposed: bool,
    output_padding: list[int],
    groups: int,
):
    # Modify by CAMBRICON
    if (
        isinstance(input_tensor, torch._subclasses.FakeTensor)
        and input_tensor.fake_device.type == "mlu"
    ):
        import torch_mlu

        k = weight.dim()
        # convert conv1d to conv2d
        if k == 3:
            if len(stride) == 1:
                stride.insert(0, 1)
                padding.insert(0, 0)
                dilation.insert(0, 1)
                output_padding.insert(0, 0)

            input_tensor = input_tensor.unsqueeze(2)
            weight = weight.unsqueeze(2)

        shape_out = calc_conv_nd_return_shape(
            input_tensor,
            weight,
            stride,
            padding,
            dilation,
            is_transposed,
            groups,
            output_padding if is_transposed else None,
        )

        input_channels_dim = 1
        output_channels_dim = 1
        if input_tensor.size(input_channels_dim) == 0:
            shape_out[output_channels_dim] = 0

        memory_format = torch_mlu._MLUC._get_channels_last_memory_format(
            input_tensor.dim()
        )
        out = input_tensor.new_empty(shape_out).to(memory_format=memory_format)

        if k == 3:
            out = out.squeeze(2)
        return out
    # end Modify by CAMBRICON
    shape_out = calc_conv_nd_return_shape(
        input_tensor,
        weight,
        stride,
        padding,
        dilation,
        is_transposed,
        groups,
        output_padding if is_transposed else None,
    )

    from torch.fx.experimental.symbolic_shapes import guard_or_false

    input_channels_dim = 1
    output_channels_dim = 1
    if guard_or_false(input_tensor.size(input_channels_dim) == 0):
        shape_out[output_channels_dim] = 0

    out = input_tensor.new_empty(shape_out)
    return out


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_conv",
    meta_conv,
)
gorilla.apply(patch)


def calc_conv_nd_return_shape(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride: list[int] | int,
    padding: list[int] | int,
    dilation: list[int] | int,
    is_transposed: bool,
    groups: int,
    output_padding: list[int] | int | None = None,
):
    def _formula(ln: int, p: int, d: int, k: int, s: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output

        See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
        Returns:
            The output length
        """
        return (ln + 2 * p - d * (k - 1) - 1) // s + 1

    def _formula_transposed(ln: int, p: int, d: int, k: int, s: int, op: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output
        if transposed convolution is used.
        See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
            op: output padding in that dim

        Returns:
            The output length
        """
        return (ln - 1) * s - 2 * p + d * (k - 1) + op + 1

    kernel_size = weight.shape[2:]
    dims = input_tensor.shape[2:]
    if is_transposed:
        out_channels = groups * weight.shape[1]
    else:
        out_channels = weight.shape[0]
        if weight.shape[1] * groups != input_tensor.shape[1]:
            raise RuntimeError("Invalid channel dimensions")

    ret_shape = [input_tensor.shape[0], out_channels]
    if isinstance(stride, IntLike):
        # pyrefly: ignore [bad-assignment]
        stride = [stride] * len(dims)
    elif len(stride) == 1:
        stride = [stride[0]] * len(dims)

    if isinstance(padding, IntLike):
        # pyrefly: ignore [bad-assignment]
        padding = [padding] * len(dims)
    elif len(padding) == 1:
        padding = [padding[0]] * len(dims)

    if isinstance(dilation, IntLike):
        # pyrefly: ignore [bad-assignment]
        dilation = [dilation] * len(dims)
    elif len(dilation) == 1:
        dilation = [dilation[0]] * len(dims)

    output_padding_list: list[int] | None = None
    if output_padding:
        if isinstance(output_padding, IntLike):
            # pyrefly: ignore [bad-assignment]
            output_padding_list = [output_padding] * len(dims)
        elif len(output_padding) == 1:
            output_padding_list = [output_padding[0]] * len(dims)
        else:
            output_padding_list = output_padding

    for i in range(len(dims)):
        # If output_padding is present, we are dealing with a transposed convolution
        if output_padding_list:
            ret_shape.append(
                _formula_transposed(
                    dims[i],
                    # pyrefly: ignore [bad-index]
                    padding[i],
                    # pyrefly: ignore [bad-index, index-error]
                    # pyrefly: ignore [bad-index, index-error]
                    dilation[i],
                    kernel_size[i],
                    # pyrefly: ignore [bad-index, index-error]
                    stride[i],
                    output_padding_list[i],
                )
            )
        else:
            ret_shape.append(
                # pyrefly: ignore [bad-index, index-error]
                _formula(dims[i], padding[i], dilation[i], kernel_size[i], stride[i])
            )
    # NOTE: Backend behavior for zero-sized spatial dimensions is inconsistent.
    # CUDA (cuDNN) handles zero-sized outputs gracefully by short-circuiting,
    # but other backends fail: CPU rejects it, ROCm/miopen returns
    # miopenStatusBadParm, and MPS asserts "Placeholder tensor is empty".
    # We only allow zero-sized outputs on CUDA with cuDNN (not ROCm/HIP).
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.fx.experimental.symbolic_shapes import sym_or

    device = (
        input_tensor.fake_device
        if isinstance(input_tensor, FakeTensor)
        else input_tensor.device
    )

    # ROCm also reports device.type as "cuda", but miopen doesn't support zero-sized outputs

    # Modify by CAMBRICON
    # is_cudnn = device.type == "cuda" and torch.version.hip is None
    is_cudnn = device.type == "mlu" and torch.version.hip is None
    # end Modify by CAMBRICON
    if not is_cudnn:
        torch._check(
            sym_or(*[x > 0 for x in ret_shape[2:]]),
            lambda: f"Given input size per channel: {list(dims)}. "
            f"Calculated output size per channel: {ret_shape[2:]}. "
            f"Output size is too small",
        )

    return ret_shape


patch = gorilla.Patch(
    torch._meta_registrations,
    "calc_conv_nd_return_shape",
    calc_conv_nd_return_shape,
)
gorilla.apply(patch)


def meta_convolution_backward(
    grad_output_,
    input_,
    weight_,
    bias_sizes_opt,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    # High level logic taken from slow_conv3d_backward_cpu which should
    # be representative of all convolution_backward impls

    backend_grad_input = None
    backend_grad_weight = None
    backend_grad_bias = None

    # Modify by CAMBRICON
    if (
        isinstance(input_, torch._subclasses.FakeTensor)
        and input_.fake_device.type == "mlu"
    ):
        import torch_mlu

        k = weight_.dim()
        if input_.size(0) == 0 or input_.size(1) == 0:
            if output_mask[0]:
                backend_grad_input = torch.zeros_like(input_)
            if output_mask[1]:
                backend_grad_weight = torch.zeros_like(weight_)
            if output_mask[2]:
                backend_grad_bias = weight_.new_empty(bias_sizes_opt)
            return (backend_grad_input, backend_grad_weight, backend_grad_bias)

        if k == 3:
            if len(stride) == 1:
                stride.insert(0, 1)
                padding.insert(0, 0)
                dilation.insert(0, 1)
                output_padding.insert(0, 0)

            input_ = input_.unsqueeze(2)
            grad_output_ = grad_output_.unsqueeze(2)
            weight_ = weight_.unsqueeze(2)

        memory_format = torch_mlu._MLUC._get_channels_last_memory_format(input_.dim())

        if output_mask[0]:
            backend_grad_input = grad_output_.new_empty(input_.size()).to(
                memory_format=memory_format
            )
        if output_mask[1]:
            backend_grad_weight = torch.empty_like(weight_, memory_format=memory_format)
        if output_mask[2]:
            backend_grad_bias = grad_output_.new_empty(bias_sizes_opt)

        if k == 3:
            if output_mask[0]:
                backend_grad_input = backend_grad_input.squeeze(2)
            if output_mask[1]:
                backend_grad_weight = backend_grad_weight.squeeze(2)
        return (backend_grad_input, backend_grad_weight, backend_grad_bias)

    # end Modify by CAMBRICON
    # Backend layout expectation: GPU backends (CUDA via cudnn_conv_suggest_memory_format,
    # MPS via mps_conv_use_channels_last) return channels_last outputs when either input
    # tensor is channels_last. This must be matched here to avoid stride assertion failures
    # in inductor when the predicted strides don't match actual backend output strides.
    # See: https://github.com/pytorch/pytorch/issues/171622
    #
    # Memory format inference rules (matching backend behavior):
    #   - grad_input format: derived from grad_output and weight
    #   - grad_weight format: derived from input and grad_output
    def _conv_memory_format(t1, t2):
        # Match the logic in cudnn_conv_suggest_memory_format and mps_conv_use_channels_last:
        # Use channels_last if either tensor suggests it
        fmt1 = suggest_memory_format(t1)
        fmt2 = suggest_memory_format(t2)
        if fmt1 == torch.channels_last or fmt2 == torch.channels_last:
            return torch.channels_last
        if fmt1 == torch.channels_last_3d or fmt2 == torch.channels_last_3d:
            return torch.channels_last_3d
        return torch.contiguous_format

    if output_mask[0]:
        memory_format = _conv_memory_format(grad_output_, weight_)
        backend_grad_input = grad_output_.new_empty(input_.size()).to(
            memory_format=memory_format
        )
    if output_mask[1]:
        memory_format = _conv_memory_format(input_, grad_output_)
        backend_grad_weight = grad_output_.new_empty(weight_.size()).to(
            memory_format=memory_format
        )
    if output_mask[2]:
        backend_grad_bias = grad_output_.new_empty(bias_sizes_opt)

    return (backend_grad_input, backend_grad_weight, backend_grad_bias)


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_convolution_backward",
    meta_convolution_backward,
)
gorilla.apply(patch)


def upsample_nearest2d(input, output_size, scales_h=None, scales_w=None):
    torch._check(
        input.numel() != 0 or multiply_integers(input.size()[1:]),
        lambda: f"Non-empty 4D data tensor expected but got a tensor with sizes {input.size()}",
    )
    full_output_size = upsample_common_check(
        input.size(), output_size, num_spatial_dims=2
    )
    output = input.new_empty(full_output_size)

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(input)

    # following "heuristic: only use channels_last path when it's faster than the contiguous path"
    _, n_channels, _, _ = input.shape
    if input.device.type == "cuda" and n_channels < 4:
        memory_format = torch.contiguous_format

    # Modify by CAMBRICON
    if (
        isinstance(input, torch._subclasses.FakeTensor)
        and input.fake_device.type == "mlu"
    ):
        output = output.contiguous(memory_format=torch.channels_last)
        return output
    # end Modify by CAMBRICON

    output = output.contiguous(memory_format=memory_format)

    return output


patch = gorilla.Patch(
    torch._meta_registrations,
    "upsample_nearest2d",
    upsample_nearest2d,
)
gorilla.apply(patch)


def meta_avg_pool2d(
    input,
    kernel_size,
    stride=(),
    padding=(0,),
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    def unpack(name, val):
        torch._check(
            len(val) in [1, 2],
            lambda: f"avg_pool2d: {name} must either be a single int, or a tuple of two ints",
        )
        H = val[0]
        W = H if len(val) == 1 else val[1]
        return H, W

    kH, kW = unpack("kernel_size", kernel_size)
    torch._check(
        len(stride) in [0, 1, 2],
        lambda: "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
    )
    torch._check(
        input.dtype not in [torch.uint8, torch.uint16, torch.uint32, torch.uint64],
        lambda: f""""avg_pool2d" not implemented for '{input.dtype.__str__()}'""",
    )
    if len(stride) == 0:
        dH, dW = kH, kW
    elif len(stride) == 1:
        dH, dW = stride[0], stride[0]
    else:
        dH, dW = unpack("stride", stride)

    padH, padW = unpack("padding", padding)

    torch._check(
        divisor_override is None or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    nbatch = input.size(-4) if input.dim() == 4 else 1
    nInputPlane = input.size(-3)
    inputHeight = input.size(-2)
    inputWidth = input.size(-1)

    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, 1, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, 1, ceil_mode)

    memory_format = utils.suggest_memory_format(input)
    pool2d_shape_check(
        input,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        1,
        1,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        memory_format,
    )

    # Modify by CAMBRICON
    if (
        isinstance(input, torch._subclasses.FakeTensor)
        and input.fake_device.type == "mlu"
    ):
        if input.dim() == 3:
            from torch_mlu._meta_registrations import get_channels_last_strides_2d

            size = [nInputPlane, outputHeight, outputWidth]
            output_strides = get_channels_last_strides_2d(size)
            return torch.empty_strided(
                size,
                output_strides,
                dtype=input.dtype,
                device=input.device,
            )
        else:
            size = [nbatch, nInputPlane, outputHeight, outputWidth]
            return torch.empty(
                size,
                dtype=input.dtype,
                device=input.device,
                memory_format=torch.channels_last,
            )
    # end Modify by CAMBRICON
    if input.dim() == 3:
        size = [nInputPlane, outputHeight, outputWidth]
    else:
        size = [nbatch, nInputPlane, outputHeight, outputWidth]
    return torch.empty(
        size,
        dtype=input.dtype,
        device=input.device,
        memory_format=memory_format,
    )


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_avg_pool2d",
    meta_avg_pool2d,
)
gorilla.apply(patch)


def meta_avg_pool2d_backward(
    gradOutput_,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    # From aten/src/ATen/native/AveragePool2d.cpp structured kernel meta func.
    torch._check(
        len(kernel_size) == 1 or len(kernel_size) == 2,
        lambda: "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints",
    )
    kH = kernel_size[0]
    kW = kH if len(kernel_size) == 1 else kernel_size[1]
    torch._check(
        len(stride) == 0 or len(stride) == 1 or len(stride) == 2,
        lambda: "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
    )
    dH = kH if len(stride) == 0 else stride[0]
    dW = kW if len(stride) == 0 else dH if len(stride) == 1 else stride[1]
    torch._check(
        len(padding) == 1 or len(padding) == 2,
        lambda: "avg_pool2d: padding must either be a single int, or a tuple of two ints",
    )
    padH = padding[0]
    padW = padH if len(padding) == 1 else padding[1]

    torch._check(
        divisor_override is None or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    input_size = input.shape
    nbatch = input_size[-4] if input.dim() == 4 else 1
    nInputPlane = input_size[-3]
    inputHeight = input_size[-2]
    inputWidth = input_size[-1]

    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, 1, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, 1, ceil_mode)

    mem_format = utils.suggest_memory_format(input)

    avg_pool2d_backward_shape_check(
        input,
        gradOutput_,
        nbatch,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        mem_format,
    )

    # Modify by CAMBRICON
    if (
        isinstance(input, torch._subclasses.FakeTensor)
        and input.fake_device.type == "mlu"
    ):
        if input.dim() == 3:
            from torch_mlu._meta_registrations import get_channels_last_strides_2d

            return torch.empty_strided(
                input_size,
                get_channels_last_strides_2d(input_size),
                dtype=input.dtype,
                device=input.device,
            )
        else:
            return torch.empty(
                input_size,
                dtype=input.dtype,
                device=input.device,
                memory_format=torch.channels_last,
            )
    # end Modify by CAMBRICON
    return torch.empty(
        input_size,
        dtype=input.dtype,
        device=input.device,
        memory_format=mem_format,
    )


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_avg_pool2d_backward",
    meta_avg_pool2d_backward,
)
gorilla.apply(patch)


def meta_adaptive_max_pool2d(input, output_size):
    ndim = input.ndim
    torch._check(
        ndim in (3, 4),
        lambda: f"adaptive_max_pool2d(): Expected 3D or 4D tensor, but got: {input.shape}",
    )
    for i in range(1, ndim):
        torch._check(
            input.size(i) > 0,
            lambda: (
                f"adaptive_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
                f"but input has sizes {input.shape} with dimension {i} being empty"
            ),
        )

    torch._check(
        len(output_size) == 2,
        lambda: "adaptive_max_pool2d(): internal error: output_size.size() must be 2",
    )

    dimH = 1
    sizeB = 1
    sizeD = 0

    if input.ndim == 4:
        sizeB = input.size(0)
        dimH += 1

    sizeD = input.size(dimH - 1)
    osizeH, osizeW = output_size

    # Add by CAMBRICON
    if (
        isinstance(input, torch._subclasses.FakeTensor)
        and input.fake_device.type == "mlu"
    ):
        out_shape = (sizeB, sizeD, osizeH, osizeW)
        out = input.new_empty(out_shape).to(memory_format=torch.channels_last)
        indices = input.new_empty(out_shape, dtype=torch.int64).to(
            memory_format=torch.channels_last
        )
        if input.ndim == 3:
            out = out.squeeze(0)
            indices = indices.squeeze(0)
        return out, indices
    # end Add by CAMBRICON

    if input.ndim == 3:
        out_shape = (sizeD, osizeH, osizeW)
        out = input.new_empty(out_shape)
        indices = input.new_empty(out_shape, dtype=torch.int64)
        return out, indices
    else:
        out_shape = (sizeB, sizeD, osizeH, osizeW)  # type: ignore[assignment]
        memory_format = utils.suggest_memory_format(input)
        out = input.new_empty(out_shape).to(memory_format=memory_format)
        indices = input.new_empty(out_shape, dtype=torch.int64).to(
            memory_format=memory_format
        )
        return out, indices


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_adaptive_max_pool2d",
    meta_adaptive_max_pool2d,
)
gorilla.apply(patch)


def meta_adaptive_max_pool2d_backward(grad_output, input, indices):
    ndim = grad_output.ndim
    torch._check(
        ndim in (3, 4),
        lambda: f"adaptive_max_pooling2d_backward(): Expected 3D or 4D grad_output, but got: {grad_output.shape}",
    )

    _adaptive_pool_empty_output_check(grad_output, "adaptive_max_pool2d_backward")

    torch._check(
        input.dtype == grad_output.dtype,
        lambda: f"expected dtype {input.dtype} for `grad_output` but got dtype {grad_output.dtype}",
    )

    # Add by CAMBRICON
    if (
        isinstance(input, torch._subclasses.FakeTensor)
        and input.fake_device.type == "mlu"
    ):
        orig_input_ndim = input.ndim
        if orig_input_ndim == 3:
            input = input.unsqueeze(0)
        out = input.new_empty(input.shape).to(memory_format=torch.channels_last)
        if orig_input_ndim == 3:
            out = out.squeeze(0)
        return out
    # end Add by CAMBRICON

    memory_format = utils.suggest_memory_format(input)
    return input.new_empty(input.shape).to(memory_format=memory_format)


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_adaptive_max_pool2d_backward",
    meta_adaptive_max_pool2d_backward,
)
gorilla.apply(patch)


def meta_adaptive_avg_pool2d(self, output_size):
    torch._check(
        self.ndim == 3 or self.ndim == 4,
        lambda: f"Expected 3D or 4D tensor, but got {self.shape}",
    )
    output_shape = self.shape[:-2] + tuple(output_size)
    # Add by CAMBRICON
    if self.device.type == "mlu" or (
        isinstance(self, torch._subclasses.FakeTensor)
        and self.fake_device.type == "mlu"
    ):
        import torch_mlu

        orig_self_ndim = self.ndim
        if orig_self_ndim == 3:
            output_shape = [1, *output_shape]
            self = self.unsqueeze(0)

        memory_format = torch_mlu._MLUC._get_channels_last_memory_format(self.dim())
        out = torch.empty(
            output_shape,
            dtype=self.dtype,
            device=self.device,
            memory_format=memory_format,
        )
        if orig_self_ndim == 3:
            out = out.squeeze(0)
        return out
    # end Add by CAMBRICON

    memory_format = utils.suggest_memory_format(self)
    # need to set memory_format to preserve the memory format of the input
    # channel last input should have channel last output
    return torch.empty(
        output_shape,
        dtype=self.dtype,
        device=self.device,
        memory_format=memory_format,
    )


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_adaptive_avg_pool2d",
    meta_adaptive_avg_pool2d,
)
gorilla.apply(patch)


def meta__adaptive_avg_pool2d_backward(grad_out, self):
    ndim = grad_out.ndim
    for i in range(1, ndim):
        torch._check(
            grad_out.size(i) > 0,
            lambda: f"adaptive_avg_pool2d_backward(): Expected grad_output to have non-zero \
                      size for non-batch dimensions, {grad_out.shape} with dimension {i} being empty",
        )
    torch._check(
        ndim == 3 or ndim == 4,
        lambda: f"adaptive_avg_pool2d_backward(): Expected 3D or 4D tensor, but got {self.shape}",
    )
    torch._check(
        self.dtype == grad_out.dtype,
        lambda: f"expected dtype {self.dtype} for `grad_output` but got dtype {grad_out.dtype}",
    )
    # Add by CAMBRICON
    if self.device.type == "mlu" or (
        isinstance(self, torch._subclasses.FakeTensor)
        and self.fake_device.type == "mlu"
    ):
        import torch_mlu

        orig_self_ndim = self.ndim
        if orig_self_ndim == 3:
            self = self.unsqueeze(0)
        memory_format = torch_mlu._MLUC._get_channels_last_memory_format(self.dim())
        grad_input = torch.empty_like(self, memory_format=memory_format)
        if orig_self_ndim == 3:
            grad_input = grad_input.squeeze(0)
        return grad_input
    # end Add by CAMBRICON

    memory_format = torch.contiguous_format
    if is_channels_last(self):
        memory_format = torch.channels_last
    return self.new_empty(self.shape).to(memory_format=memory_format)


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta__adaptive_avg_pool2d_backward",
    meta__adaptive_avg_pool2d_backward,
)
gorilla.apply(patch)


def meta_adaptive_avg_pool3d(self, output_size):
    torch._check(
        self.ndim == 4 or self.ndim == 5,
        lambda: f"Expected 4D or 5D tensor, but got {self.shape}",
    )
    # Add by CAMBRICON
    if (
        isinstance(self, torch._subclasses.FakeTensor)
        and self.fake_device.type == "mlu"
    ):
        import torch_mlu

        out_shape = self.shape[:-3] + tuple(output_size)
        orig_self_ndim = self.ndim
        if orig_self_ndim == 4:
            out_shape = (1,) + out_shape
            self = self.unsqueeze(0)

        memory_format = torch_mlu._MLUC._get_channels_last_memory_format(self.dim())
        out = torch.empty(
            out_shape, dtype=self.dtype, device=self.device, memory_format=memory_format
        )
        if orig_self_ndim == 4:
            out = out.squeeze(0)
        return out
    # end Add by CAMBRICON

    return self.new_empty(self.shape[:-3] + tuple(output_size))


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_adaptive_avg_pool3d",
    meta_adaptive_avg_pool3d,
)
gorilla.apply(patch)


def meta__adaptive_avg_pool3d_backward(grad_output, self):
    _adaptive_pool_empty_output_check(grad_output, "adaptive_avg_pool3d_backward")
    # Add by CAMBRICON
    if (
        isinstance(self, torch._subclasses.FakeTensor)
        and self.fake_device.type == "mlu"
    ):
        import torch_mlu

        orig_self_ndim = self.ndim
        if orig_self_ndim == 4:
            self = self.unsqueeze(0)

        memory_format = torch_mlu._MLUC._get_channels_last_memory_format(self.dim())
        grad_input = self.new_empty(self.shape).to(memory_format=memory_format)
        if orig_self_ndim == 4:
            grad_input = grad_input.squeeze(0)
        return grad_input
    # end Add by CAMBRICON
    return torch.empty_like(self, memory_format=torch.legacy_contiguous_format)


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta__adaptive_avg_pool3d_backward",
    meta__adaptive_avg_pool3d_backward,
)
gorilla.apply(patch)


def grid_sampler_2d_backward_meta(
    grad_output,
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
    output_mask,
):
    input_requires_grad = output_mask[0]
    if input_requires_grad:
        # Modify by CAMBRICON
        if (
            isinstance(input, torch._subclasses.FakeTensor)
            and input.fake_device.type == "mlu"
        ):
            import torch_mlu

            memory_format = torch_mlu._MLUC._get_channels_last_memory_format(
                input.dim()
            )
            grad_input = torch.zeros_like(input, memory_format=memory_format)
        else:
            grad_input = torch.zeros_like(input, memory_format=torch.contiguous_format)
        # end Modify by CAMBRICON
    else:
        grad_input = None
    grad_grid = torch.empty_like(grid, memory_format=torch.contiguous_format)
    return (grad_input, grad_grid)


patch = gorilla.Patch(
    torch._meta_registrations,
    "grid_sampler_2d_backward_meta",
    grid_sampler_2d_backward_meta,
)
gorilla.apply(patch)


def meta_repeat(self, repeats):
    torch._check(
        len(repeats) >= self.dim(),
        lambda: "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor",
    )
    for i, rep in enumerate(repeats):
        torch._check(
            rep >= 0,
            lambda: f"Repeats cannot be negative, found {rep} at index {i}",
        )
    # Add new leading dimensions to the tensor if the
    # number of target dimensions is larger than the
    # number of source dimensions.
    num_new_dimensions = len(repeats) - self.dim()
    padded_size = (1,) * num_new_dimensions + tuple(self.shape)
    target_size = [padded_size[i] * repeats[i] for i in range(len(repeats))]

    # Modify by CAMBRICON
    if (
        isinstance(self, torch._subclasses.FakeTensor)
        and self.fake_device.type == "mlu"
        and num_new_dimensions == 0
    ):
        return torch.empty(
            target_size,
            dtype=self.dtype,
            device=self.device,
            memory_format=utils.suggest_memory_format(self),
        )
    # end Modify by CAMBRICON
    return self.new_empty(target_size)


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_repeat",
    meta_repeat,
)
gorilla.apply(patch)


def softmax(x: Tensor, dim: int, half_to_float: bool) -> Tensor:
    if half_to_float:
        if x.dtype not in [torch.half, torch.bfloat16]:
            raise AssertionError(
                f"half_to_float is True but x.dtype is {x.dtype}, expected half or bfloat16"
            )

    computation_dtype, result_dtype = utils.elementwise_dtypes(
        x, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )

    result_dtype = result_dtype if not half_to_float else computation_dtype
    # Modify by CAMBRICON
    if isinstance(x, torch._subclasses.FakeTensor) and x.fake_device.type == "mlu":
        res = torch.empty_like(
            x, dtype=result_dtype, memory_format=utils.suggest_memory_format(x)
        )
        return res
    # end Modify by CAMBRICON

    res = torch.empty_like(x, dtype=result_dtype, memory_format=torch.contiguous_format)
    return res


patch = gorilla.Patch(
    torch._meta_registrations,
    "softmax",
    softmax,
)
gorilla.apply(patch)

# def meta_pad2d_backward(grad_output, self, padding):
#     dim_w = 2
#     dim_h = 1
#     dim_plane = 0
#
#     self_shape = self.shape
#     if self.dim() == 4:
#         dim_w += 1
#         dim_h += 1
#         dim_plane += 1
#
#     pad_l, pad_r, pad_t, pad_b = padding
#
#     input_h = self_shape[dim_h]
#     input_w = self_shape[dim_w]
#     output_h = input_h + pad_t + pad_b
#     output_w = input_w + pad_l + pad_r
#
#     torch._check(
#         output_w == grad_output.size(dim_w),
#         lambda: f"grad_output width unexpected. Expected: {output_w}, Got: {grad_output.size(dim_w)}",
#     )
#     torch._check(
#         output_h == grad_output.size(dim_h),
#         lambda: f"grad_output height unexpected. Expected: {output_h}, Got: {grad_output.size(dim_h)}",
#     )
#     if (
#         isinstance(self, torch._subclasses.FakeTensor)
#         and self.fake_device.type == "mlu"
#     ):
#         self_ = self.unsqueeze(0) if self.ndim == 3 else self
#         out = torch.empty_like(self_, memory_format=torch.channels_last)
#         out = out.resize_as_(self)
#         if self.ndim == 3:
#             out = out.squeeze(0)
#         return out
#     return self.new_empty(self.shape)
#
#
# patch = gorilla.Patch(
#     torch._meta_registrations,
#     "meta_pad2d_backward",
#     meta_pad2d_backward,
# )
# gorilla.apply(patch)


@register_meta(
    [
        aten.upsample_bilinear2d_backward.default,
    ]
)
def meta_upsample_bilinear2d_backward(
    grad_output: Tensor,
    output_size: Sequence[Union[int, torch.SymInt]],
    input_size: Sequence[Union[int, torch.SymInt]],
    align_corners,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    return grad_output.new_empty(input_size).to(memory_format=torch.channels_last)


@out_wrapper("grad_input")
def meta_avg_pool3d_backward(
    grad_output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    torch._check(
        len(kernel_size) in (1, 3),
        lambda: "avg_pool3d: kernel_size must be a single int, or a tuple of three ints",
    )
    kT = kernel_size[0]
    kH = kT if len(kernel_size) == 1 else kernel_size[1]
    kW = kT if len(kernel_size) == 1 else kernel_size[2]

    torch._check(
        not stride or len(stride) in (1, 3),
        lambda: "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints",
    )
    dT = kT if not stride else stride[0]
    dH = kH if not stride else (dT if len(stride) == 1 else stride[1])
    dW = kW if not stride else (dT if len(stride) == 1 else stride[2])

    torch._check(
        len(padding) in (1, 3),
        lambda: "avg_pool3d: padding must be a single int, or a tuple of three ints",
    )
    padT = padding[0]
    padH = padT if len(padding) == 1 else padding[1]
    padW = padT if len(padding) == 1 else padding[2]

    torch._check(
        input.ndim in (4, 5),
        lambda: "non-empty 4D or 5D (batch mode) tensor expected for input",
    )

    torch._check(
        not divisor_override or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    nslices = input.size(-4)
    itime = input.size(-3)
    iheight = input.size(-2)
    iwidth = input.size(-1)

    otime_for_shape_check = pooling_output_shape(itime, kT, padT, dT, 1, ceil_mode)
    oheight_for_shape_check = pooling_output_shape(iheight, kH, padH, dH, 1, ceil_mode)
    owidth_for_shape_check = pooling_output_shape(iwidth, kW, padW, dW, 1, ceil_mode)

    avg_pool3d_backward_shape_check(
        input,
        grad_output,
        nslices,
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        padT,
        padH,
        padW,
        itime,
        iheight,
        iwidth,
        otime_for_shape_check,
        oheight_for_shape_check,
        owidth_for_shape_check,
        "avg_pool3d_backward()",
    )

    # Modify by CAMBRICON
    if (
        isinstance(input, torch._subclasses.FakeTensor)
        and input.fake_device.type == "mlu"
    ):
        if input.ndim == 4:
            from torch_mlu._meta_registrations import get_channels_last_strides_3d

            return torch.empty_strided(
                input.shape,
                get_channels_last_strides_3d(input.shape),
                dtype=input.dtype,
                device=input.device,
            )
        else:
            return input.new_empty(input.shape).to(memory_format=torch.channels_last_3d)
    # end Modify by CAMBRICON
    return input.new_empty(input.shape)


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_avg_pool3d_backward",
    meta_avg_pool3d_backward,
)
gorilla.apply(patch)


@out_wrapper()
def _constant_pad_nd_meta(input, pad, value=0):
    # same checks as decomposition in torch/_refs/__init__.py:constant_pad_nd()
    torch._check(
        len(pad) % 2 == 0,
        lambda: f"Length of pad must be even but instead it equals {len(pad)}",
    )

    input_sizes = input.shape
    l_inp = len(input_sizes)
    l_pad = len(pad) // 2
    l_diff = l_inp - l_pad

    torch._check(
        l_inp >= l_pad,
        lambda: "Length of pad should be no more than twice the number of "
        f"dimensions of the input. Pad length is {len(pad)} while the input has "
        f"{l_inp} dimensions.",
    )
    # Modify by CAMBRICON
    if (
        isinstance(input, torch._subclasses.FakeTensor)
        and input.fake_device.type == "mlu"
    ):
        memory_format = utils.suggest_memory_format(input)
        input_contiguous = input.contiguous(memory_format=memory_format)
        all_pads_is_zero = not any(p != 0 for p in pad)
        if all_pads_is_zero:
            return input_contiguous.clone()
    else:
        if all(isinstance(p, utils.IntWithoutSymInt) and p <= 0 for p in pad):
            c_input = input
            for i in range(l_diff, l_inp):
                pad_idx = 2 * (l_inp - i - 1)
                if pad[pad_idx] < 0:
                    c_input = c_input.narrow(
                        i, -pad[pad_idx], c_input.shape[i] + pad[pad_idx]
                    )

                if pad[pad_idx + 1] < 0:
                    c_input = c_input.narrow(i, 0, c_input.shape[i] + pad[pad_idx + 1])

            return c_input.clone()

    # end Modify by CAMBRICON

    new_shape = list(input_sizes[:l_diff])
    for i in range(l_pad):
        pad_idx = len(pad) - ((i + 1) * 2)
        new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1]
        torch._check(
            new_dim >= 0,
            lambda: f"The input size {input_sizes[l_diff + i]}, plus negative padding "
            f"{pad[pad_idx]} and {pad[pad_idx + 1]} resulted in a negative output size, "
            f"which is invalid. Check dimension {l_diff + i} of your input.",
        )
        new_shape.append(new_dim)

    return torch.empty(
        new_shape,
        dtype=input.dtype,
        device=input.device,
        requires_grad=input.requires_grad,
        memory_format=suggest_memory_format(input),
    )


patch = gorilla.Patch(
    torch._meta_registrations,
    "_constant_pad_nd_meta",
    _constant_pad_nd_meta,
)
gorilla.apply(patch)


@out_wrapper("out", "indices")
def meta_max_pool3d_with_indices(
    input,
    kernel_size,
    stride=(),
    padding=(0,),
    dilation=(1,),
    ceil_mode=False,
):
    torch._check(
        len(kernel_size) in (1, 3),
        lambda: "max_pool3d: kernel_size must either be a single int, or a tuple of three ints",
    )
    kT = kernel_size[0]
    kH = kT if len(kernel_size) == 1 else kernel_size[1]
    kW = kT if len(kernel_size) == 1 else kernel_size[2]

    torch._check(
        not stride or len(stride) in (1, 3),
        lambda: "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints",
    )
    dT = kT if not stride else stride[0]
    dH = kH if not stride else (dT if len(stride) == 1 else stride[1])
    dW = kW if not stride else (dT if len(stride) == 1 else stride[2])

    torch._check(
        len(padding) in (1, 3),
        lambda: "max_pool3d: padding must either be a single int, or a tuple of three ints",
    )
    pT = padding[0]
    pH = pT if len(padding) == 1 else padding[1]
    pW = pT if len(padding) == 1 else padding[2]

    torch._check(
        len(dilation) in (1, 3),
        lambda: "max_pool3d: dilation must be either a single int, or a tuple of three ints",
    )
    dilationT = dilation[0]
    dilationH = dilationT if len(dilation) == 1 else dilation[1]
    dilationW = dilationT if len(dilation) == 1 else dilation[2]

    torch._check(
        input.ndim in (4, 5),
        lambda: "non-empty 4D or 5D (batch mode) tensor expected for input",
    )

    nbatch = input.size(-5) if input.ndim == 5 else 1
    nslices = input.size(-4)
    itime = input.size(-3)
    iheight = input.size(-2)
    iwidth = input.size(-1)

    otime = pooling_output_shape(itime, kT, pT, dT, dilationT, ceil_mode)
    oheight = pooling_output_shape(iheight, kH, pH, dH, dilationH, ceil_mode)
    owidth = pooling_output_shape(iwidth, kW, pW, dW, dilationW, ceil_mode)

    pool3d_shape_check(
        input,
        nslices,
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        pT,
        pH,
        pW,
        dilationT,
        dilationH,
        dilationW,
        itime,
        iheight,
        iwidth,
        otime,
        oheight,
        owidth,
        "max_pool3d_with_indices()",
    )

    # Modify by CAMBRICON
    if (
        isinstance(input, torch._subclasses.FakeTensor)
        and input.fake_device.type == "mlu"
    ):
        out_shape = (nbatch, nslices, otime, oheight, owidth)  # type: ignore[assignment]

        out = input.new_empty(out_shape).to(memory_format=torch.channels_last_3d)
        indices = input.new_empty(out_shape, dtype=torch.int64).to(
            memory_format=torch.channels_last_3d
        )
        if input.ndim == 4:
            out = out.squeeze(0)
            indices = indices.squeeze(0)

        return out, indices
    # end Modify by CAMBRICON

    # channels_last_3d only applies to 5D tensors (C++ enforces this)
    channels_last = (
        input.ndim == 5 and utils.suggest_memory_format(input) == torch.channels_last_3d
    )
    if input.ndim == 4:
        out_shape = (nslices, otime, oheight, owidth)
    else:
        out_shape = (nbatch, nslices, otime, oheight, owidth)  # type: ignore[assignment]

    out = input.new_empty(out_shape)
    indices = input.new_empty(out_shape, dtype=torch.int64)

    if channels_last:
        out = out.to(memory_format=torch.channels_last_3d)
        indices = indices.to(memory_format=torch.channels_last_3d)

    return out, indices


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_max_pool3d_with_indices",
    meta_max_pool3d_with_indices,
)
gorilla.apply(patch)


@out_wrapper("grad_input")
def meta_max_pool3d_with_indices_backward(
    grad_output,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    indices,
):
    torch._check(
        len(kernel_size) in (1, 3),
        lambda: "max_pool3d: kernel_size must either be a single int, or a tuple of three ints",
    )
    kT = kernel_size[0]
    kH = kT if len(kernel_size) == 1 else kernel_size[1]
    kW = kT if len(kernel_size) == 1 else kernel_size[2]

    torch._check(
        not stride or len(stride) in (1, 3),
        lambda: "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints",
    )
    dT = kT if not stride else stride[0]
    dH = kH if not stride else (dT if len(stride) == 1 else stride[1])
    dW = kW if not stride else (dT if len(stride) == 1 else stride[2])

    torch._check(
        len(padding) in (1, 3),
        lambda: "max_pool3d: padding must either be a single int, or a tuple of three ints",
    )
    pT = padding[0]
    pH = pT if len(padding) == 1 else padding[1]
    pW = pT if len(padding) == 1 else padding[2]

    torch._check(
        len(dilation) in (1, 3),
        lambda: "max_pool3d: dilation must be either a single int, or a tuple of three ints",
    )
    dilationT = dilation[0]
    dilationH = dilationT if len(dilation) == 1 else dilation[1]
    dilationW = dilationT if len(dilation) == 1 else dilation[2]

    torch._check(
        input.ndim in (4, 5),
        lambda: "non-empty 4D or 5D (batch mode) tensor expected for input",
    )

    nslices = input.size(-4)
    itime = input.size(-3)
    iheight = input.size(-2)
    iwidth = input.size(-1)

    otime = grad_output.size(-3)
    oheight = grad_output.size(-2)
    owidth = grad_output.size(-1)

    max_pool3d_backward_shape_check(
        input,
        grad_output,
        indices,
        nslices,
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        pT,
        pH,
        pW,
        dilationT,
        dilationH,
        dilationW,
        itime,
        iheight,
        iwidth,
        otime,
        oheight,
        owidth,
        "max_pool3d_with_indices_backward()",
    )

    # Modify by CAMBRICON
    if (
        isinstance(input, torch._subclasses.FakeTensor)
        and input.fake_device.type == "mlu"
    ):
        nbatch = input.size(-5) if input.ndim == 5 else 1
        out_shape = [nbatch, nslices, itime, iheight, iwidth]
        grad_input = input.new_empty(out_shape).to(memory_format=torch.channels_last_3d)
        if input.ndim == 4:
            grad_input = grad_input.squeeze(0)
    # end Modify by CAMBRICON
    # channels_last_3d only applies to 5D tensors (C++ enforces this)
    channels_last = (
        input.ndim == 5 and utils.suggest_memory_format(input) == torch.channels_last_3d
    )

    grad_input = input.new_empty(input.shape)

    if channels_last:
        grad_input = grad_input.to(memory_format=torch.channels_last_3d)

    return grad_input


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_max_pool3d_with_indices_backward",
    meta_max_pool3d_with_indices_backward,
)
gorilla.apply(patch)


def upsample_nearest2d_backward(
    grad_output: Tensor,
    output_size: Sequence[int | torch.SymInt],
    input_size: Sequence[int | torch.SymInt],
    scales_h: float | None = None,
    scales_w: float | None = None,
):
    full_output_size = upsample_common_check(
        input_size, output_size, num_spatial_dims=2
    )
    torch._check(
        grad_output.ndim == 4,
        lambda: f"Expected grad_output to be a tensor of dimension 4 but got: dimension {grad_output.ndim}",
    )
    for i in range(4):
        torch._check(
            grad_output.size(i) == full_output_size[i],
            lambda: (
                f"Expected grad_output to have the same shape as output;"
                f" output.size({i}) = {full_output_size[i]}"
                f" but got grad_output.size({i}) = {grad_output.size(i)}"
            ),
        )

    # Add by CAMBRICON
    if (
        isinstance(grad_output, torch._subclasses.FakeTensor)
        and grad_output.fake_device.type == "mlu"
    ):
        return grad_output.new_empty(input_size).to(
            memory_format=torch.channels_last
        )  # type: ignore[call-overload]
    # end Add by CAMBRICON
    return grad_output.new_empty(input_size).to(
        memory_format=utils.suggest_memory_format(grad_output)
    )  # type: ignore[call-overload]


patch = gorilla.Patch(
    torch._meta_registrations,
    "upsample_nearest2d_backward",
    upsample_nearest2d_backward,
)
gorilla.apply(patch)


def meta_index_Tensor(self, indices):
    torch._check(bool(indices), lambda: "at least one index must be provided")
    # aten::index is the internal advanced indexing implementation
    # checkIndexTensorTypes and expandTensors
    result: list[Tensor | None] = []
    for i, index in enumerate(indices):
        if index is not None:
            torch._check(
                index.dtype in [torch.long, torch.int, torch.int8, torch.bool],
                lambda: "tensors used as indices must be long, int, byte or bool tensors",
            )
            if index.dtype in [torch.int8, torch.bool]:
                nonzero = index.nonzero()
                k = len(result)
                torch._check_index(
                    k + index.ndim <= self.ndim,
                    lambda: f"too many indices for tensor of dimension {self.ndim}",
                )
                for j in range(index.ndim):
                    torch._check_index(
                        index.shape[j] == self.shape[k + j],
                        lambda: f"The shape of the mask {index.shape} at index {i} "
                        f"does not match the shape of the indexed tensor {self.shape} at index {k + j}",
                    )
                    result.append(nonzero.select(1, j))
            else:
                result.append(index)
        else:
            result.append(index)
    indices = result
    torch._check(
        len(indices) <= self.ndim,
        lambda: f"too many indices for tensor of dimension {self.ndim} (got {len(indices)})",
    )
    # expand_outplace
    import torch._refs as refs  # avoid import cycle in mypy

    indices = list(refs._maybe_broadcast(*indices))
    # add missing null tensors
    while len(indices) < self.ndim:
        indices.append(None)

    # hasContiguousSubspace
    #   true if all non-null tensors are adjacent
    # See:
    # https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    # https://stackoverflow.com/questions/53841497/why-does-numpy-mixed-basic-advanced-indexing-depend-on-slice-adjacency
    state = 0
    has_contiguous_subspace = False
    for index in indices:
        if state == 0:
            if index is not None:
                state = 1
        elif state == 1:
            if index is None:
                state = 2
        else:
            if index is not None:
                break
    else:
        has_contiguous_subspace = True

    # transposeToFront
    # This is the logic that causes the newly inserted dimensions to show up
    # at the beginning of the tensor, if they're not contiguous
    if not has_contiguous_subspace:
        dims = []
        transposed_indices = []
        for i, index in enumerate(indices):
            if index is not None:
                dims.append(i)
                transposed_indices.append(index)
        for i, index in enumerate(indices):
            if index is None:
                dims.append(i)
                transposed_indices.append(index)
        self = self.permute(dims)
        indices = transposed_indices

    # AdvancedIndex::AdvancedIndex
    # Now we can assume the indices have contiguous subspace
    # This is simplified from AdvancedIndex which goes to more effort
    # to put the input and indices in a form so that TensorIterator can
    # take them.  If we write a ref for this, probably that logic should
    # get implemented
    before_shape: list[int] = []
    after_shape: list[int] = []
    replacement_shape: list[int] = []
    for dim, index in enumerate(indices):
        if index is None:
            if replacement_shape:
                after_shape.append(self.shape[dim])
            else:
                before_shape.append(self.shape[dim])
        else:
            replacement_shape = list(index.shape)

    def _restride_src(self):
        """
        This follows restride_src in TensorAdvancedIndexing.cpp
        """
        shape = before_shape + replacement_shape + after_shape
        strides = list(self.stride())
        # pyrefly: ignore [unsupported-operation]
        strides[len(before_shape) : len(self.shape) - len(after_shape)] = [0] * len(
            replacement_shape
        )
        return self.as_strided(shape, strides)

    out = self.new_empty(before_shape + replacement_shape + after_shape)
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if guard_or_false(self.numel() == 0):
        # No need to worry about the output strides if self is empty.
        return out

    # Try to follow eager to decide the output stride based on self.
    # Note that perm here is the reverse of the 'perm_' decided by
    # TensorIteratorBase::reorder_dimensions
    restrided_self = _restride_src(self)
    perm, _ = utils.compute_elementwise_output_logical_to_physical_perm(restrided_self)

    # Follow TensorIteratorBase::allocate_or_resize_outputs
    if list(perm) != list(range(len(perm))):
        perm_shape = utils.apply_perm(out.shape, perm)
        new_stride = utils.make_contiguous_strides_for(perm_shape)
        new_stride = utils.apply_perm(new_stride, utils.invert_perm(perm))
        out = out.as_strided(out.size(), new_stride)
    # Modify by CAMBRICON
    if (
        isinstance(self, torch._subclasses.FakeTensor)
        and self.fake_device.type == "mlu"
    ):
        out = torch.empty_like(out, memory_format=torch.contiguous_format)
    # end Modify by CAMBRICON
    return out


patch = gorilla.Patch(
    torch._meta_registrations,
    "meta_index_Tensor",
    meta_index_Tensor,
)
gorilla.apply(patch)
