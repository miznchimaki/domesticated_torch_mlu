import torch
import torch_mlu
import triton
import triton.language as tl
from triton.runtime import libentry

from .utils import get_total_core_num, get_max_nram_size


def assert_shape_within_int32(shape, name=""):
    INT32_MAX = 2**31 - 1
    if any(dim > INT32_MAX for dim in shape):
        raise ValueError(
            f"{name + ': ' if name else ''}Tensor shape {shape} exceeds INT32_MAX ({INT32_MAX})"
        )


def replication_pad3d_mlu(input: torch.Tensor, padding: list[int]) -> torch.Tensor:
    """
    input: (N, C, D, H, W) or (C, D, H, W)
    padding (int): the size of the padding.
     {padding_left}, {padding_right},
     {padding_top}, {padding_bottom},
     {padding_front}, {padding_back}
    """
    # do check and generate out
    assert input.device.type == "mlu", "Expected input to be on MLU device"

    assert input.ndim == 5 or input.ndim == 4, "Only 5D or 4D tensors supported"
    N, C, D, H, W = (1, *input.shape) if input.ndim == 4 else input.shape
    assert (
        C != 0 and D != 0 and H != 0 and W != 0
    ), f"Expected 4D or 5D (batch mode) tensor with possibly 0 batch size and \
         other non-zero dimensions for input, but got: {input.shape}"

    assert len(padding) == 6, "padding size is expected to be 6"

    (
        pad_w_before,
        pad_w_after,
        pad_h_before,
        pad_h_after,
        pad_d_before,
        pad_d_after,
    ) = padding

    D_out = D + pad_d_before + pad_d_after
    H_out = H + pad_h_before + pad_h_after
    W_out = W + pad_w_before + pad_w_after
    assert (
        W_out >= 1 and H_out >= 1 and D_out >= 1
    ), f"input (D: {D}, H: {H}, W: {W}) is too small, Calculated output D: \
         {D_out}, H: {H_out}, W: {W_out}"

    # special padding
    if all(p == 0 for p in padding):
        return input.clone().contiguous()

    # generator out
    # handle as memory_format: N, D, H, W, C
    out = torch.empty(
        (N, C, D_out, H_out, W_out),
        dtype=input.dtype,
        device=input.device,
        memory_format=torch.channels_last_3d,
    )
    replication_pad3d_out_mlu(input, padding, out=out)
    # memory_format of cuda_out is torch.contiguous(N, C, D, H, W)
    out = out.contiguous()
    if input.ndim == 4:
        out = out.squeeze(0)
    return out


def replication_pad3d_out_mlu(
    input: torch.Tensor, padding: list[int], *, out: torch.Tensor
) -> torch.Tensor:
    assert len(padding) == 6
    (
        pad_w_before,
        pad_w_after,
        pad_h_before,
        pad_h_after,
        pad_d_before,
        pad_d_after,
    ) = padding

    N, C, D, H, W = (1, *input.shape) if input.ndim == 4 else input.shape

    N_out, C_out, D_out, H_out, W_out = (1, *out.shape) if out.ndim == 4 else out.shape

    if input.numel() == 0:
        return

    # 单维度不能超过INT32_MAX
    assert_shape_within_int32(input.shape, name="Input tensor")
    assert_shape_within_int32(out.shape, name="Output tensor")

    input_ = input.unsqueeze(0) if input.ndim == 4 else input

    if not input_.is_contiguous(memory_format=torch.channels_last_3d):
        input_ = input_.contiguous(memory_format=torch.channels_last_3d)

    assert out.is_contiguous(
        memory_format=torch.channels_last_3d
    ), "Out Tensor must be channels_last_3d"

    # 计算网格大小
    def grid_fn(meta):
        num_blocks_w = triton.cdiv(W_out, meta["BLOCK_W"])
        # 为什么这里不用算C，因为C只在核内拆分，不在核间拆分
        # num_blocks_c = triton.cdiv(C, meta["BLOCK_C"])
        total_tasks = N * D_out * H_out * num_blocks_w
        return (min(total_tasks, get_total_core_num()),)

    from .replication_pad3d_kernel import replication_pad3d_kernel

    replication_pad3d_kernel[grid_fn](
        input_,
        out,
        N,
        D,
        H,
        W,
        C,
        D_out,
        H_out,
        W_out,
        pad_d_before,
        pad_h_before,
        pad_w_before,
    )
    return out
