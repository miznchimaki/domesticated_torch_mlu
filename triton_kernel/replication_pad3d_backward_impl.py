import torch
import torch_mlu
import triton
import triton.language as tl
from triton.runtime import libentry

from .utils import get_total_core_num, get_max_nram_size


def get_best_w(N, D_out, H_out, W_in, total_core_num, w_max):
    # must satisfy (BLOCK_W + pad_w_before + pad_w_after) * C * sizeof(dtype) * 2 <= MAX_NRAM_SIZE
    if w_max < W_in and w_max > 1:
        num_blocks_w = (W_in + w_max - 1) // w_max
        w_max = (W_in + num_blocks_w - 1) // num_blocks_w
    elif w_max >= W_in:
        w_max = W_in
    else:
        return 1

    # 避免任务份数过少带来的负载不均衡
    num_blocks_w = (W_in + w_max - 1) // w_max
    w_rem = W_in % w_max
    while (
        N * D_out * H_out * num_blocks_w < total_core_num
        and num_blocks_w >= 1
        and w_max > 1
    ):
        w_max = w_max // 2
        w_rem = W_in % w_max
        num_blocks_w = (W_in + w_max - 1) // w_max
    # 避免尾数段不均衡
    if w_rem > 0 and w_max // w_rem > 2:
        w_max = w_max // 2
        w_rem = W_in % w_max
        num_blocks_w = (W_in + w_max - 1) // w_max
    return w_max


def get_best_w_and_c(
    N, D_out, H_out, W_out, C, element_size, total_core_num, MAX_NRAM_SIZE
):
    if C * element_size > MAX_NRAM_SIZE:
        c_max = MAX_NRAM_SIZE // element_size
        # c_max > 1 and c_max < C:
        num_blocks_c = (C + c_max - 1) // c_max
        c_max = (C + num_blocks_c - 1) // num_blocks_c
        return (1, c_max)
    else:
        # 如果默认可以放得下C的话，这里的W的选取十分关键，影响负载均衡
        # 比较典型的，W_out = 1000, 选取BLOCK_W = 999, 这个效率很差
        w_max = MAX_NRAM_SIZE // (C * element_size)  # w_max >= 1
        UINT16_MAX = 2**16 - 1
        w_max = min(UINT16_MAX, w_max)  # load segment_num limitation
        if w_max < W_out and w_max > 1:
            num_blocks_w = (W_out + w_max - 1) // w_max
            w_max = (W_out + num_blocks_w - 1) // num_blocks_w
        elif w_max >= W_out:
            w_max = W_out
        else:
            # w_max = 1
            return (1, C)

        # 避免任务份数过少带来的负载不均衡
        num_blocks_w = (W_out + w_max - 1) // w_max
        w_rem = W_out % w_max
        while (
            N * D_out * H_out * num_blocks_w < total_core_num
            and num_blocks_w >= 1
            and w_max > 1
        ):
            w_max = w_max // 2
            w_rem = W_out % w_max
            num_blocks_w = (W_out + w_max - 1) // w_max
        # 避免尾数段不均衡
        if w_rem > 0 and w_max // w_rem > 2:
            w_max = w_max // 2
            w_rem = W_out % w_max
            num_blocks_w = (W_out + w_max - 1) // w_max
        return (w_max, C)


def assert_shape_within_int32(shape, name=""):
    INT32_MAX = 2**31 - 1
    if any(dim > INT32_MAX for dim in shape):
        raise ValueError(
            f"{name + ': ' if name else ''}Tensor shape {shape} exceeds INT32_MAX ({INT32_MAX})"
        )


def replication_pad3d_backward_mlu(
    grad_output: torch.Tensor, input: torch.Tensor, padding: list[int]
) -> torch.Tensor:
    """
    grad_output: 输出梯度张量，形状为 (N, C, D_out, H_out, W_out) 或 (C, D_out, H_out, W_out)
    input: 正向的输入
    padding: 填充参数
             [pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after]
    input_shape: 原始输入形状 (N, C, D_in, H_in, W_in) 或 (C, D_in, H_in, W_in)
    """
    # 参数检查
    assert grad_output.device.type == "mlu", "Expected grad_output to be on MLU device"
    assert input.device.type == "mlu", "Expected input to be on MLU device"

    # shapeAndGradOutputCheck3d
    assert len(padding) == 6, "padding Size is expected to be 6"
    (
        pad_w_before,
        pad_w_after,
        pad_h_before,
        pad_h_after,
        pad_d_before,
        pad_d_after,
    ) = padding

    assert input.ndim == 4 or input.ndim == 5, "Expected 4D or 5D tensor"
    N_in, C_in, D_in, H_in, W_in = (1, *input.shape) if input.ndim == 4 else input.shape
    assert (
        C_in != 0 and D_in != 0 and H_in != 0 and W_in != 0
    ), f"Expected 4D or 5D (batch mode) tensor with possibly 0 batch size and \
         other non-zero dimensions for input, but got: {input.shape}"

    D_out = D_in + pad_d_before + pad_d_after
    H_out = H_in + pad_h_before + pad_h_after
    W_out = W_in + pad_w_before + pad_w_after
    assert (
        W_out >= 1 and H_out >= 1 and D_out >= 1
    ), f"input (D: {D}, H: {H}, W: {W}) is too small, Calculated output D: \
         {D_out}, H: {H_out}, W: {W_out}"
    # do not support large tensor on cuda, but supported on mlu
    assert (
        grad_output.ndim == 4 or grad_output.ndim == 5
    ), "Expected 4D or 5D grad_output tensor"

    # 验证形状一致性
    N_gout, C_gout, D_gout, H_gout, W_gout = (
        (1, *grad_output.shape) if grad_output.ndim == 4 else grad_output.shape
    )
    assert (
        N_in == N_gout
    ), f"gradOutput batch unexpected. Expected: {N_in}, Got {N_gout}"
    assert (
        C_in == C_gout
    ), f"gradOutput channel unexpected. Expected: {C_in}, Got {C_gout}"
    assert (
        D_out == D_gout
    ), f"gradOutput depth unexpected. Expected: {D_out}, Got {D_gout}"
    assert (
        H_out == H_gout
    ), f"gradOutput height unexpected. Expected: {H_out}, Got {H_gout}"
    assert (
        W_out == W_gout
    ), f"gradOutput width unexpected. Expected: {W_out}, Got {W_gout}"

    # special padding
    if all(p == 0 for p in padding):
        return grad_output.clone().contiguous()

    # 创建输入梯度张量（全零初始化）
    if input.ndim == 4:
        N, C_gin, D_gin, H_gin, W_gin = 1, *input.shape  # gin meas grad_input
    else:
        N, C_gin, D_gin, H_gin, W_gin = input.shape
    if input.ndim == 4:
        grad_input = torch.empty(
            (1, C_gin, D_gin, H_gin, W_gin),
            dtype=input.dtype,
            device=input.device,
            memory_format=torch.channels_last_3d,
        )
    else:
        grad_input = torch.empty(
            (N, C_gin, D_gin, H_gin, W_gin),
            dtype=input.dtype,
            device=input.device,
            memory_format=torch.channels_last_3d,
        )

    # 调用反向传播核心函数
    replication_pad3d_backward_grad_input_mlu(
        grad_output, input, padding, grad_input=grad_input
    )

    # 调整输出形状
    grad_input = grad_input.contiguous()
    if input.ndim == 4:
        return grad_input.squeeze(0)
    return grad_input


def replication_pad3d_backward_grad_input_mlu(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    padding: list[int],
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    (
        pad_w_before,
        pad_w_after,
        pad_h_before,
        pad_h_after,
        pad_d_before,
        pad_d_after,
    ) = padding

    N_in, C_in, D_in, H_in, W_in = (1, *input.shape) if input.ndim == 4 else input.shape

    # 验证形状一致性
    N_gout, C_gout, D_gout, H_gout, W_gout = (
        (1, *grad_output.shape) if grad_output.ndim == 4 else grad_output.shape
    )
    N_gin, C_gin, D_gin, H_gin, W_gin = (
        (1, *grad_input.shape) if grad_input.ndim == 4 else grad_input.shape
    )
    assert N_in == N_gin, f"gradInput batch unexpected. Expected: {N_in}, Got {N_gin}"
    assert C_in == C_gin, f"gradInput channel unexpected. Expected: {C_in}, Got {C_gin}"
    assert D_in == D_gin, f"gradInput depth unexpected. Expected: {D_in}, Got {D_gin}"
    assert H_in == H_gin, f"gradInput height unexpected. Expected: {H_in}, Got {H_gin}"
    assert W_in == W_gin, f"gradInput width unexpected. Expected: {W_in}, Got {W_gin}"

    if grad_input.numel() == 0:
        return

    # 单维度不能超过INT32_MAX
    assert_shape_within_int32(grad_input.shape, name="Input tensor")
    assert_shape_within_int32(grad_output.shape, name="Output tensor")

    # 确保内存格式正确
    assert grad_input.is_contiguous(
        memory_format=torch.channels_last_3d
    ), "gradInput Tensor must be channels_last_3d"

    grad_input.zero_()

    grad_output_ = grad_output.unsqueeze(0) if grad_output.ndim == 4 else grad_output
    if not grad_output_.is_contiguous(memory_format=torch.channels_last_3d):
        grad_output_ = grad_output_.contiguous(memory_format=torch.channels_last_3d)

    MAX_NRAM_SIZE = int(get_max_nram_size() * 0.8)
    element_size = grad_input.element_size()
    total_core_num = get_total_core_num()
    INT32_MAX = 2**31 - 1
    is_large_tensor = (
        grad_input.numel() * element_size > INT32_MAX
        or grad_output_.numel() * element_size > INT32_MAX
    )
    UINT16_MAX = 2**16 - 1
    opt_one_w = max(0, pad_w_before) + max(0, pad_w_after) + 1
    # load: BLOCK_W + max(0, pad_w_before) + max(0, pad_w_right) <= 65535
    # sumpool: max(pad_w_before) + 1 < 65535
    # opt requires MAX_NRAM_SIZE can hold one point and pad
    is_opt_kernel = (
        opt_one_w < UINT16_MAX
        and opt_one_w * (C_gout * element_size * 2) <= MAX_NRAM_SIZE
    )

    from .replication_pad3d_backward_kernel import (
        replication_pad3d_backward_normal_kernel,
        replication_pad3d_backward_opt_kernel,
        replication_pad3d_backward_opt_kernel_neg_left,
        replication_pad3d_backward_opt_kernel_neg_right,
        replication_pad3d_backward_opt_kernel_neg_both,
    )

    if not is_opt_kernel:
        # 通用kernel，拆分grad_output_
        w_max, c_max = get_best_w_and_c(
            N_gout,
            D_gout,
            H_gout,
            W_gout,
            C_gout,
            element_size,
            total_core_num,
            MAX_NRAM_SIZE,
        )

        BLOCK_W = max(1, min(w_max, W_gout))
        BLOCK_C = max(1, min(c_max, C_gout))

        # 计算网格大小 (不超过total_core_num个核心)
        num_blocks_w = (W_gout + BLOCK_W - 1) // BLOCK_W
        total_valid_core = min(total_core_num, N_gout * D_gout * H_gout * num_blocks_w)

        replication_pad3d_backward_normal_kernel[(total_valid_core,)](
            grad_output_,
            grad_input,
            N_gin,
            D_gin,
            H_gin,
            W_gin,
            C_gin,
            D_gout,
            H_gout,
            W_gout,
            pad_d_before,
            pad_h_before,
            pad_w_before,
            is_large_tensor,
            BLOCK_W,
            BLOCK_C,
        )
    else:
        # 优化kernle
        # 为什么一定要启用四个kernel:
        # pad_w_before, pad_w_after是tl.constexpr类型，用来描述切片的长度，
        # 这个值不能是负值，不能在运行期判断

        # 确保(max_w, C)对应的描述的内存块必然是连续的, 为了atomic指令的维度折叠
        # 优化kernel,拆分W_gin
        # NEED_NRAM_SIZE
        #     = (BLOCK_W * 2 + max(0, pad_w_before) + max(0, pad_w_after)) * C_gout * element_size
        w_max = (
            MAX_NRAM_SIZE // (C_gout * element_size * 2)
            - max(0, pad_w_before)
            - max(0, pad_w_after)
        )  # w_max >= 1
        # load limitation
        # is_opt_kernel guarantee (UINT16_MAX-max(0, pad_w_before)-max(0, pad_w_after)-1) > 0
        w_max = min(UINT16_MAX - max(0, pad_w_before) - max(0, pad_w_after) - 1, w_max)
        valid_W_gin = W_gin + min(0, pad_w_before) + min(0, pad_w_after)
        w_max = get_best_w(N_gout, D_gout, H_gout, valid_W_gin, total_core_num, w_max)

        # 确保能够实现维度折叠，BLOCK_C=C
        BLOCK_C = C_gout
        BLOCK_W = max(1, min(w_max, valid_W_gin))

        # 计算网格大小 (不超过total_core_num个核心)
        num_blocks_w = (valid_W_gin + BLOCK_W - 1) // BLOCK_W
        total_valid_core = min(total_core_num, N_gout * D_gout * H_gout * num_blocks_w)

        if pad_w_before >= 0 and pad_w_after >= 0:
            replication_pad3d_backward_opt_kernel[(total_valid_core,)](
                grad_output_,
                grad_input,
                N_gin,
                D_gin,
                H_gin,
                W_gin,
                C_gin,
                D_gout,
                H_gout,
                W_gout,
                pad_d_before,
                pad_h_before,
                pad_w_before,
                pad_w_after,
                is_large_tensor,
                BLOCK_W,
                BLOCK_C,
            )
        elif pad_w_before < 0 and pad_w_after >= 0:
            replication_pad3d_backward_opt_kernel_neg_left[(total_valid_core,)](
                grad_output_,
                grad_input,
                N_gin,
                D_gin,
                H_gin,
                W_gin,
                C_gin,
                D_gout,
                H_gout,
                W_gout,
                pad_d_before,
                pad_h_before,
                pad_w_before,
                pad_w_after,
                is_large_tensor,
                BLOCK_W,
                BLOCK_C,
            )
        elif pad_w_before >= 0 and pad_w_after < 0:
            replication_pad3d_backward_opt_kernel_neg_right[(total_valid_core,)](
                grad_output_,
                grad_input,
                N_gin,
                D_gin,
                H_gin,
                W_gin,
                C_gin,
                D_gout,
                H_gout,
                W_gout,
                pad_d_before,
                pad_h_before,
                pad_w_before,
                pad_w_after,
                is_large_tensor,
                BLOCK_W,
                BLOCK_C,
            )
        else:
            # pad_w_before < 0 and pad_w_after < 0
            replication_pad3d_backward_opt_kernel_neg_both[(total_valid_core,)](
                grad_output_,
                grad_input,
                N_gin,
                D_gin,
                H_gin,
                W_gin,
                C_gin,
                D_gout,
                H_gout,
                W_gout,
                pad_d_before,
                pad_h_before,
                pad_w_before,
                pad_w_after,
                is_large_tensor,
                BLOCK_W,
                BLOCK_C,
            )

    return grad_input
