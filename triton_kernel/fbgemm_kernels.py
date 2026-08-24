from typing import List, Optional, Tuple, Union
import math
import numpy as np
import torch
import triton  # @manual
import triton.language as tl  # @manual
from triton.runtime import fast_libentry
from torch.library import custom_op
from torch._tensor import Tensor

from .utils import get_total_core_num
from .triton_decorator_config import triton_kernel_decorator


# from: https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/triton/jagged/triton_jagged_tensor_ops.py#L630
def _jagged_offsets_to_dense_indice(
    offsets: List[torch.Tensor], dense_strides: List[int], dense_sizes: List[int]
) -> torch.Tensor:
    output_offset = torch.zeros(len(offsets[-1]) - 1, device="cpu", dtype=torch.int32)

    offsets_cpu = []

    for offset in offsets:
        offsets_cpu.append(offset.cpu())

    for i in range(0, len(offsets_cpu[-1]) - 1):
        idx = i
        result = 0

        # flag to check if current offset is in the range of dense
        in_range = True
        for j in range(len(offsets_cpu) - 2, -1, -1):
            left = 0
            right = offsets_cpu[j].size(0)

            # binary search found the corresponding offset group of current index
            while left < right:
                mid = left + (right - left) // 2

                if offsets_cpu[j][mid] > idx:
                    right = mid
                else:
                    left = mid + 1

            cur_val = idx - offsets_cpu[j][left - 1]

            if dense_sizes and cur_val >= dense_sizes[j + 1]:
                in_range = False
                break

            result += cur_val * dense_strides[j + 1]
            idx = left - 1

        if in_range:
            result += idx * dense_strides[0]

            # another out of output dense range case
            if dense_sizes and idx > dense_sizes[0]:
                result = -1
            output_offset[i] = result
        else:
            output_offset[i] = -1

    return output_offset.to("mlu")


@triton.jit
def tensor_elementwise_add(x, y):
    return x + y


@triton.jit
def tensor_elementwise_mul(x, y):
    return x * y


# 为heuristics_direct模式设计的BLOCK_ROW,BLOCK_COL,BLOCK_SIZE选择函数
# 基于实验数据优化得到的阈值，详见测试报告：
#   wiki pageId=532248599
def get_block_row(M):
    """根据M选择BLOCK_ROW"""
    if M > 79:
        return 128
    elif M > 32:
        return 64
    else:
        return 16


def get_block_col(N):
    """根据N选择BLOCK_COL"""
    if N == 1:
        return 1
    elif N > 1000:
        return 512
    elif N > 100:
        return 128
    elif N > 50:
        return 64
    else:
        return 16


def get_block_size(MN):
    """根据M*N选择BLOCK_SIZE"""
    if MN >= 417792:
        return 16384
    else:
        return 4096


def get_autotune_config():
    base_configs = [
        triton.Config({"BLOCK_ROW": row, "BLOCK_COL": col})
        for row in [16, 64, 128, 256, 512, 1024]
        for col in [1, 16, 32, 64, 128, 256, 512]
    ]
    return base_configs


def dense_to_jagged_filter_configs(configs, named_args, **kwargs):
    row_size = named_args["M"]
    col_size = named_args["N"]
    valid_configs = []
    for cfg in configs:
        BLOCK_ROW = cfg.kwargs["BLOCK_ROW"]
        BLOCK_COL = cfg.kwargs["BLOCK_COL"]

        # Dimensional folding when col_size is 1
        if col_size == 1:
            if BLOCK_COL != 1:
                continue
        else:
            COL_ALIGN = triton.next_power_of_2(col_size)
            if BLOCK_COL > max(COL_ALIGN, 16):
                continue

        ROW_ALIGN = triton.next_power_of_2(row_size)
        if BLOCK_ROW > max(ROW_ALIGN, 16):
            continue

        valid_configs.append(cfg)
    return valid_configs


@triton_kernel_decorator(
    # autotune配置
    autotune_configs=get_autotune_config(),
    autotune_key=["M", "N", "operation_function"],
    prune_configs={"early_config_prune": dense_to_jagged_filter_configs},
    # autotune模式专用的heuristics
    # 注意：这里的heuristics会在autotune之前应用，用于设置autotune过程中需要的参数
    heuristics_autotune={
        "num_stages": lambda args: 0 if args["operation_function"] is None else 5,
        "IS_LARGE_TENSOR": lambda named_args: (
            named_args["output_jagged_ptr"].numel()
            * named_args["output_jagged_ptr"].element_size()
            > np.iinfo(np.int32).max
            or named_args["input_dense_ptr"].numel()
            * named_args["input_dense_ptr"].element_size()
            > np.iinfo(np.int32).max
        ),
    },
    # direct模式专用的heuristics
    # 不开启autotune时，直接使用这些启发式规则决定参数
    heuristics_direct={
        # 根据输入规模动态选择BLOCK_ROW
        "BLOCK_ROW": lambda args: get_block_row(args["M"]),
        # 根据列数动态选择BLOCK_COL
        "BLOCK_COL": lambda args: get_block_col(args["N"]),
        # 直接模式下的固定参数
        "num_stages": lambda args: 0 if args["operation_function"] is None else 5,
        "IS_LARGE_TENSOR": lambda named_args: (  # 同样需要判断大张量
            named_args["output_jagged_ptr"].numel()
            * named_args["output_jagged_ptr"].element_size()
            > np.iinfo(np.int32).max
            or named_args["input_dense_ptr"].numel()
            * named_args["input_dense_ptr"].element_size()
            > np.iinfo(np.int32).max
        ),
    },
    fast_libentry=True,
)
def dense_to_jagged_kernel(
    output_jagged_ptr,  # [total_length, D]
    jagged_offsets_ptr,  # [B + 1]
    num_batches,
    input_dense_ptr,
    dense_indices_ptr,
    operation_jagged_values_ptr,
    dense_col_stride,
    dense_row_stride,
    dense_matrix_stride,
    M,
    N,
    jagged_values_row_size,
    jagged_values_row_stride,
    jagged_values_col_stride,
    operation_function,
    JAGGED_DIM: tl.constexpr,
    BLOCK_COL: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    IS_LARGE_TENSOR: tl.constexpr,
):
    if IS_LARGE_TENSOR:
        jagged_values_row_size = tl.full([], jagged_values_row_size, dtype=tl.int64)
        jagged_values_row_stride = tl.full([], jagged_values_row_stride, dtype=tl.int64)
        jagged_values_col_stride = tl.full([], jagged_values_col_stride, dtype=tl.int64)
        dense_col_stride = tl.full([], dense_col_stride, dtype=tl.int64)
        dense_row_stride = tl.full([], dense_row_stride, dtype=tl.int64)
        dense_matrix_stride = tl.full([], dense_matrix_stride, dtype=tl.int64)
        M = tl.full([], M, dtype=tl.int64)
        N = tl.full([], N, dtype=tl.int64)

    block_id = tl.program_id(0)
    num_blocks = tl.num_programs(axis=0)

    # Round-robin dispatch: each block_id handles pid = block_id, block_id + num_blocks, ...
    for batch_idx in range(block_id, num_batches, num_blocks):
        # Load jagged start/end
        idx_offset = tl.load(jagged_offsets_ptr + batch_idx + tl.arange(0, 2))
        start_offset = tl.minimum(idx_offset[0], jagged_values_row_size)
        end_offset = tl.minimum(idx_offset[1], jagged_values_row_size)
        cur_row_size = end_offset - start_offset

        jagged_boundary_col = jagged_values_row_stride
        jagged_boundary_row = cur_row_size
        dense_boundary_col = tl.minimum(jagged_boundary_col, N)
        dense_boundary_row = tl.minimum(cur_row_size, M)

        out_ptr = output_jagged_ptr + start_offset * jagged_values_row_stride
        op_ptr = (
            operation_jagged_values_ptr + start_offset * jagged_values_row_stride
            if operation_function is not None
            else None
        )

        # Resolve dense base pointer
        dense_ptr = input_dense_ptr
        if JAGGED_DIM > 2:
            dense_indice = tl.load(dense_indices_ptr + batch_idx)
            if dense_indice == -1:
                if IS_LARGE_TENSOR:
                    dense_boundary_col = tl.full([], -1, dtype=tl.int64)
                else:
                    dense_boundary_col = -1
            else:
                dense_ptr += dense_indice
        else:
            dense_ptr += batch_idx * dense_matrix_stride

        offset_row = tl.arange(0, BLOCK_ROW)
        offset_col = tl.arange(0, BLOCK_COL)
        for _i in range(0, cur_row_size, BLOCK_ROW):
            row = offset_row + _i
            for _j in range(0, dense_boundary_col, BLOCK_COL):
                col = offset_col + _j
                block_offset = (
                    row[:, None] * dense_row_stride + col[None, :] * dense_col_stride
                )
                out_block_offset = (
                    row[:, None] * jagged_values_row_stride
                    + col[None, :] * jagged_values_col_stride
                )
                dense_mask = (row[:, None] < dense_boundary_row) & (
                    col[None, :] < dense_boundary_col
                )
                jagged_mask = (row[:, None] < jagged_boundary_row) & (
                    col[None, :] < jagged_boundary_col
                )
                dense_values = tl.load(
                    dense_ptr + block_offset, mask=dense_mask, other=0
                )
                if operation_function is not None:
                    op_values = tl.load(
                        op_ptr + out_block_offset, mask=jagged_mask, other=0
                    )
                    if operation_function == "add":
                        dense_values = tensor_elementwise_add(dense_values, op_values)
                    else:
                        # mul
                        dense_values = tensor_elementwise_mul(dense_values, op_values)
                tl.store(out_ptr + out_block_offset, dense_values, mask=jagged_mask)


def get_mini_batch_autotune_config():
    base_configs = [
        triton.Config({"BLOCK_SIZE": block_size})
        for block_size in [512, 4096, 16384, 32768]
    ]
    return base_configs


@triton_kernel_decorator(
    autotune_configs=get_mini_batch_autotune_config(),
    autotune_key=["M", "N", "operation_function"],
    heuristics_autotune={
        "num_stages": lambda args: 0 if args["operation_function"] is None else 5,
        "IS_LARGE_TENSOR": lambda named_args: (
            named_args["output_jagged_ptr"].numel()
            * named_args["output_jagged_ptr"].element_size()
            > np.iinfo(np.int32).max
            or named_args["input_dense_ptr"].numel()
            * named_args["input_dense_ptr"].element_size()
            > np.iinfo(np.int32).max
        ),
    },
    heuristics_direct={
        "BLOCK_SIZE": lambda args: get_block_size(args["M"] * args["N"]),
        "num_stages": lambda args: 0 if args["operation_function"] is None else 5,
        "IS_LARGE_TENSOR": lambda named_args: (
            named_args["output_jagged_ptr"].numel()
            * named_args["output_jagged_ptr"].element_size()
            > np.iinfo(np.int32).max
            or named_args["input_dense_ptr"].numel()
            * named_args["input_dense_ptr"].element_size()
            > np.iinfo(np.int32).max
        ),
    },
    fast_libentry=True,
)
def dense_to_jagged_kernel_mini_batch(
    output_jagged_ptr,  # [total_length, D]
    jagged_offsets_ptr,  # [B + 1]
    num_batches,
    input_dense_ptr,
    dense_indices_ptr,
    operation_jagged_values_ptr,
    dense_matrix_stride,
    M,
    N,
    jagged_values_row_size,
    jagged_values_row_stride,
    operation_function,
    JAGGED_DIM: tl.constexpr,
    processor_count: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_LARGE_TENSOR: tl.constexpr,
):
    if IS_LARGE_TENSOR:
        jagged_values_row_size = tl.full([], jagged_values_row_size, dtype=tl.int64)
        jagged_values_row_stride = tl.full([], jagged_values_row_stride, dtype=tl.int64)
        dense_matrix_stride = tl.full([], dense_matrix_stride, dtype=tl.int64)
        M = tl.full([], M, dtype=tl.int64)
        N = tl.full([], N, dtype=tl.int64)

    block_id = tl.program_id(0)
    num_blocks = tl.num_programs(axis=0)
    jagged_offsets_block = tl.arange(0, processor_count + 1)
    idx_offset = tl.load(
        jagged_offsets_ptr + jagged_offsets_block,
        mask=jagged_offsets_block < num_batches + 1,
    )
    for batch_idx in range(num_batches):
        start_offset = tl.minimum(idx_offset[batch_idx], jagged_values_row_size)
        end_offset = tl.minimum(idx_offset[batch_idx + 1], jagged_values_row_size)
        cur_row_size = end_offset - start_offset
        out_ptr = output_jagged_ptr + start_offset * jagged_values_row_stride
        op_ptr = (
            operation_jagged_values_ptr + start_offset * jagged_values_row_stride
            if operation_function is not None
            else None
        )
        dense_boundary_col = N
        dense_boundary_row = tl.minimum(cur_row_size, M)
        jagged_boundary_col = jagged_values_row_stride
        jagged_boundary_row = cur_row_size
        base_input_ptr = input_dense_ptr
        if JAGGED_DIM > 2:
            dense_indice = tl.load(dense_indices_ptr + batch_idx)
            if dense_indice == -1:
                if IS_LARGE_TENSOR:
                    dense_boundary_col = tl.full([], -1, dtype=tl.int64)
                else:
                    dense_boundary_col = -1
            else:
                base_input_ptr += dense_indice
        else:
            base_input_ptr = input_dense_ptr + batch_idx * dense_matrix_stride

        block_size = tl.arange(0, BLOCK_SIZE)
        rows_per_core = cur_row_size // num_blocks
        remainder = cur_row_size % num_blocks
        row_start = block_id * rows_per_core + tl.minimum(block_id, remainder)
        row_end = row_start + rows_per_core + (1 if block_id < remainder else 0)
        for row_offset in range(row_start * N, row_end * N, BLOCK_SIZE):
            offset = row_offset + block_size
            dense_mask = offset < dense_boundary_row * dense_boundary_col
            jagged_mask = offset < jagged_boundary_row * jagged_boundary_col
            dense_values = tl.load(base_input_ptr + offset, mask=dense_mask, other=0)
            if operation_function is not None:
                op_values = tl.load(op_ptr + offset, mask=jagged_mask, other=0)
                if operation_function == "add":
                    dense_values = tensor_elementwise_add(dense_values, op_values)
                else:
                    # mul
                    dense_values = tensor_elementwise_mul(dense_values, op_values)
            tl.store(out_ptr + offset, dense_values, mask=jagged_mask)


def dense_to_jagged(
    dense: torch.Tensor,
    jagged_offsets: List[torch.Tensor],
    total_L: Union[int, None] = None,
    operation_function: Union[str, None] = None,
    operation_jagged_values: Union[torch.Tensor, None] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    device = dense.device
    dtype = dense.dtype
    len_jagged_offsets = len(jagged_offsets)
    dense_dim = dense.dim()

    # check
    assert dense_dim == len_jagged_offsets + 1 or dense_dim == len_jagged_offsets + 2, (
        f"the dim of dense, {dense.dim()} does not match "
        f"the length of jagged_offsets, {jagged_offsets}."
    )

    # compute shape
    D_folded = dense_dim == len_jagged_offsets + 1
    dense_view = dense.unsqueeze(-1) if D_folded else dense
    dense_view_size = dense_view.shape
    D = dense_view_size[-1]

    last_offset = (
        jagged_offsets[-1]
        if jagged_offsets[-1].is_contiguous()
        else jagged_offsets[-1].contiguous()
    )
    num_batches = last_offset.numel() - 1
    *_, M, N = dense_view_size
    if total_L is not None:
        total_rows = total_L
    else:
        total_rows = last_offset[-1].item()
    output_jagged_shape = (total_rows,) if D_folded else (total_rows, D)

    # zero element check
    if dense.numel() == 0 or math.prod(output_jagged_shape) == 0:
        return (
            torch.zeros(output_jagged_shape, device=device, dtype=dtype),
            jagged_offsets,
        )

    assert (
        total_L is None or operation_function is None
    ), f"total_L and operation_function cannot be valid at same time."

    # (TODO) PYTORCH-17017]: if total_rows > last_offset[-1].item(),
    # the pad portion would be set to a random number instead of 0
    output_jagged_values = torch.empty(output_jagged_shape, device=device, dtype=dtype)

    if operation_function is not None:
        assert operation_jagged_values.shape == output_jagged_values.shape

    output_jagged_values_view = (
        output_jagged_values.unsqueeze(-1) if D_folded else output_jagged_values
    )
    output_jagged_values_view_size = output_jagged_values_view.shape
    output_jagged_values_view_stride = output_jagged_values_view.stride()
    JAGGED_DIM = len_jagged_offsets + 1
    dense_indices = None
    jagged_values_row_size = output_jagged_values_view_size[0]
    (
        jagged_values_row_stride,
        jagged_values_col_stride,
    ) = output_jagged_values_view_stride[-2:]

    processor_count = get_total_core_num(device.index)

    BURST_SIZE = 128
    IS_MINI_BATCH = (
        num_batches <= processor_count
        and M >= processor_count
        and M * N * dense_view.element_size() >= processor_count * BURST_SIZE
    )

    if IS_MINI_BATCH and not dense_view.is_contiguous():
        # handle contiguous of dense before compute dense_indice
        dense_view = dense_view.contiguous()

    dense_view_stride = dense_view.stride()
    dense_matrix_stride, dense_row_stride, dense_col_stride = dense_view_stride[-3:]

    if len_jagged_offsets > 1:
        dense_indices = _jagged_offsets_to_dense_indice(
            jagged_offsets,
            dense_view_stride[:-2],
            dense_view_size[:-2],
        )

    if IS_MINI_BATCH:
        grid = (processor_count,)
        dense_to_jagged_kernel_mini_batch[grid](
            output_jagged_values,
            last_offset,
            num_batches,
            dense_view,
            dense_indices,
            operation_jagged_values if operation_function else output_jagged_values,
            dense_matrix_stride,
            M,
            N,
            jagged_values_row_size,
            jagged_values_row_stride,
            operation_function,
            JAGGED_DIM,
            processor_count=processor_count,
        )
    else:
        grid = (min(processor_count, num_batches),)
        dense_to_jagged_kernel[grid](
            output_jagged_values,
            last_offset,
            num_batches,
            dense,
            dense_indices,
            operation_jagged_values if operation_function else output_jagged_values,
            dense_col_stride,
            dense_row_stride,
            dense_matrix_stride,
            M,
            N,
            jagged_values_row_size,
            jagged_values_row_stride,
            jagged_values_col_stride,
            operation_function,
            JAGGED_DIM,
        )
    return output_jagged_values, jagged_offsets


def jagged_to_dense_filter_configs(configs, named_args, **kwargs):
    dense_matrix_stride = named_args["dense_matrix_stride"]
    dense_row_stride = named_args["dense_row_stride"]
    dense_col_stride = named_args["dense_col_stride"]
    valid_configs = []
    for cfg in configs:
        BLOCK_ROW = cfg.kwargs["BLOCK_ROW"]
        BLOCK_COL = cfg.kwargs["BLOCK_COL"]

        col_size = dense_row_stride // dense_col_stride
        # Dimensional folding when col_size is 1
        if col_size == 1:
            if BLOCK_COL != 1:
                continue
        else:
            COL_ALIGN = triton.next_power_of_2(col_size)
            if BLOCK_COL > max(COL_ALIGN, 16):
                continue

        row_size = dense_matrix_stride // dense_row_stride
        ROW_ALIGN = triton.next_power_of_2(row_size)
        if BLOCK_ROW > max(ROW_ALIGN, 16):
            continue

        valid_configs.append(cfg)
    return valid_configs


@triton_kernel_decorator(
    autotune_configs=get_autotune_config(),
    autotune_key=[
        "dense_matrix_stride",
        "dense_row_stride",
        "operation_function",
    ],
    prune_configs={"early_config_prune": jagged_to_dense_filter_configs},
    heuristics_autotune={
        "IS_LARGE_TENSOR": lambda named_args: (
            named_args["jagged_values_ptr"].numel()
            * named_args["jagged_values_ptr"].element_size()
            > np.iinfo(np.int32).max
            or named_args["dense_ptr"].numel() * named_args["dense_ptr"].element_size()
            > np.iinfo(np.int32).max
        ),
        "num_stages": lambda args: 0 if args["operation_dense"] is None else 5,
    },
    heuristics_direct={
        "BLOCK_ROW": (
            lambda args: get_block_row(
                args["dense_matrix_stride"] // args["dense_row_stride"]
            )
        ),
        "BLOCK_COL": lambda args: get_block_col(args["dense_row_stride"]),
        "num_stages": lambda args: 0 if args["operation_dense"] is None else 5,
        "IS_LARGE_TENSOR": lambda named_args: (
            named_args["jagged_values_ptr"].numel()
            * named_args["jagged_values_ptr"].element_size()
            > np.iinfo(np.int32).max
            or named_args["dense_ptr"].numel() * named_args["dense_ptr"].element_size()
            > np.iinfo(np.int32).max
        ),
    },
    fast_libentry=True,
)
def jagged_to_dense_2d_kernel(
    jagged_values_ptr,
    jagged_offsets_ptr,
    jagged_row_stride,
    dense_ptr,
    padded_value,
    dense_col_stride,
    dense_row_stride,
    dense_matrix_stride,
    num_batches,
    operation_function,
    operation_dense,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
    IS_LARGE_TENSOR: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(axis=0)
    task_start = pid
    task_end = num_batches

    if IS_LARGE_TENSOR:
        jagged_row_stride = tl.full([], jagged_row_stride, dtype=tl.int64)
        dense_col_stride = tl.full([], dense_col_stride, dtype=tl.int64)
        dense_row_stride = tl.full([], dense_row_stride, dtype=tl.int64)
        dense_matrix_stride = tl.full([], dense_matrix_stride, dtype=tl.int64)

    for batch_idx in range(task_start, task_end, step):
        idx_offset = tl.load(jagged_offsets_ptr + batch_idx + tl.arange(0, 2))
        start_offset = idx_offset[0]
        end_offset = idx_offset[1]
        cur_jagged_tensor_row_size = end_offset - start_offset

        base_dense_ptr = dense_ptr + batch_idx * dense_matrix_stride
        base_jagged_values_ptr = jagged_values_ptr + start_offset * dense_row_stride
        if operation_function is not None:
            base_operation_dense = operation_dense + batch_idx * dense_matrix_stride

        dense_col_size = dense_row_stride
        dense_row_size = dense_matrix_stride // dense_row_stride
        offset_row = tl.arange(0, BLOCK_ROW)
        offset_col = tl.arange(0, BLOCK_COL)
        for _i in range(0, dense_row_size, BLOCK_ROW):
            row = offset_row + _i
            for _j in range(0, dense_col_size, BLOCK_COL):
                col = offset_col + _j
                block_offset = (
                    row[:, None] * dense_row_stride + col[None, :] * dense_col_stride
                )
                dense_mask = (row[:, None] < dense_row_size) & (
                    col[None, :] < dense_col_size
                )
                jagged_mask = (row[:, None] < cur_jagged_tensor_row_size) & (
                    col[None, :] < jagged_row_stride
                )
                jagged_val = tl.load(
                    base_jagged_values_ptr + block_offset,
                    mask=jagged_mask,
                    other=padded_value,
                )
                if operation_function is not None:
                    operation_dense_val = tl.load(
                        base_operation_dense + block_offset, mask=dense_mask, other=0.0
                    )
                    if operation_function == "add":
                        jagged_val = tensor_elementwise_add(
                            operation_dense_val, jagged_val
                        )
                    else:
                        jagged_val = tensor_elementwise_mul(
                            operation_dense_val, jagged_val
                        )
                tl.store(base_dense_ptr + block_offset, jagged_val, mask=dense_mask)


@triton_kernel_decorator(
    autotune_configs=get_mini_batch_autotune_config(),
    autotune_key=["dense_row_stride", "dense_matrix_stride", "operation_function"],
    heuristics_autotune={
        "num_stages": lambda args: 0 if args["operation_function"] is None else 5,
        "IS_LARGE_TENSOR": lambda named_args: (
            named_args["jagged_values_ptr"].numel()
            * named_args["jagged_values_ptr"].element_size()
            > np.iinfo(np.int32).max
            or named_args["dense_ptr"].numel() * named_args["dense_ptr"].element_size()
            > np.iinfo(np.int32).max
        ),
    },
    heuristics_direct={
        "BLOCK_SIZE": lambda args: get_block_size(args["dense_matrix_stride"]),
        "num_stages": lambda args: 0 if args["operation_function"] is None else 5,
        "IS_LARGE_TENSOR": lambda named_args: (
            named_args["jagged_values_ptr"].numel()
            * named_args["jagged_values_ptr"].element_size()
            > np.iinfo(np.int32).max
            or named_args["dense_ptr"].numel() * named_args["dense_ptr"].element_size()
            > np.iinfo(np.int32).max
        ),
    },
    fast_libentry=True,
)
def jagged_to_dense_2d_kernel_mini_batch(
    jagged_values_ptr,
    jagged_offsets_ptr,
    jagged_row_stride,
    dense_ptr,
    padded_value,
    dense_col_stride,
    dense_row_stride,
    dense_matrix_stride,
    num_batches,
    operation_function,
    operation_dense,
    processor_count: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_LARGE_TENSOR: tl.constexpr,
):
    if IS_LARGE_TENSOR:
        jagged_row_stride = tl.full([], jagged_row_stride, dtype=tl.int64)
        dense_col_stride = tl.full([], dense_col_stride, dtype=tl.int64)
        dense_row_stride = tl.full([], dense_row_stride, dtype=tl.int64)
        dense_matrix_stride = tl.full([], dense_matrix_stride, dtype=tl.int64)

    block_id = tl.program_id(0)
    num_blocks = tl.num_programs(axis=0)
    jagged_offsets_block = tl.arange(0, processor_count + 1)
    idx_offset = tl.load(
        jagged_offsets_ptr + jagged_offsets_block,
        mask=jagged_offsets_block < num_batches + 1,
        other=0,
    )
    for batch_idx in range(num_batches):
        start_offset = idx_offset[batch_idx]
        end_offset = idx_offset[batch_idx + 1]
        cur_row_size = end_offset - start_offset

        base_dense_ptr = dense_ptr + batch_idx * dense_matrix_stride
        base_jagged_values_ptr = jagged_values_ptr + start_offset * dense_row_stride
        if operation_function is not None:
            base_operation_dense = operation_dense + batch_idx * dense_matrix_stride

        N = dense_row_stride
        M = dense_matrix_stride // dense_row_stride

        block_offset = tl.arange(0, BLOCK_SIZE)
        rows_per_core = M // num_blocks
        remainder = M % num_blocks
        row_start = block_id * rows_per_core + tl.minimum(block_id, remainder)
        row_end = row_start + rows_per_core + (1 if block_id < remainder else 0)
        for row_offset in range(row_start * N, row_end * N, BLOCK_SIZE):
            offset = row_offset + block_offset
            dense_mask = offset < dense_matrix_stride
            jagged_mask = offset < cur_row_size * N
            jagged_val = tl.load(
                base_jagged_values_ptr + offset,
                mask=jagged_mask,
                other=padded_value,
            )
            dense_values = tl.load(base_dense_ptr + offset, mask=dense_mask, other=0)
            if operation_function is not None:
                operation_dense_val = tl.load(
                    base_operation_dense + offset, mask=dense_mask, other=0.0
                )
                if operation_function == "add":
                    jagged_val = tensor_elementwise_add(operation_dense_val, jagged_val)
                else:
                    # mul
                    jagged_val = tensor_elementwise_mul(operation_dense_val, jagged_val)
            tl.store(base_dense_ptr + offset, jagged_val, mask=dense_mask)


@triton_kernel_decorator(
    autotune_configs=get_autotune_config(),
    autotune_key=["dense_matrix_stride", "dense_row_stride"],
    prune_configs={"early_config_prune": jagged_to_dense_filter_configs},
    heuristics_autotune={
        "IS_LARGE_TENSOR": lambda named_args: (
            named_args["jagged_values_ptr"].numel()
            * named_args["jagged_values_ptr"].element_size()
            > np.iinfo(np.int32).max
            or named_args["dense_ptr"].numel() * named_args["dense_ptr"].element_size()
            > np.iinfo(np.int32).max
        ),
    },
    heuristics_direct={
        "BLOCK_ROW": (
            lambda args: get_block_row(
                args["dense_matrix_stride"] // args["dense_row_stride"]
            )
        ),
        "BLOCK_COL": lambda args: get_block_col(args["dense_row_stride"]),
        "IS_LARGE_TENSOR": lambda named_args: (
            named_args["jagged_values_ptr"].numel()
            * named_args["jagged_values_ptr"].element_size()
            > np.iinfo(np.int32).max
            or named_args["dense_ptr"].numel() * named_args["dense_ptr"].element_size()
            > np.iinfo(np.int32).max
        ),
    },
    fast_libentry=True,
)
def jagged_to_dense_kernel(
    jagged_values_ptr,
    jagged_offsets_ptr,
    jagged_row_stride,
    dense_ptr,
    dense_indices_ptr,
    dense_col_stride,
    dense_row_stride,
    dense_matrix_stride,
    total_tasks,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
    IS_LARGE_TENSOR: tl.constexpr,
):
    pid = tl.program_id(0)
    total_pgm = tl.num_programs(0)
    task_per_pgm = total_tasks // total_pgm
    rem_tasks = total_tasks % total_pgm

    task_num = task_per_pgm + (pid < rem_tasks)
    task_start = pid * task_per_pgm + (pid if pid < rem_tasks else rem_tasks)
    task_end = task_start + task_num

    if IS_LARGE_TENSOR:
        jagged_row_stride = tl.full([], jagged_row_stride, dtype=tl.int64)
        dense_col_stride = tl.full([], dense_col_stride, dtype=tl.int64)
        dense_row_stride = tl.full([], dense_row_stride, dtype=tl.int64)
        dense_matrix_stride = tl.full([], dense_matrix_stride, dtype=tl.int64)

    for task_id in range(task_start, task_end):
        # (TODO)perf improve: load all offsets and dense_indice at once
        offsets = tl.load(jagged_offsets_ptr + task_id + tl.arange(0, 2))
        begin = offsets[0]
        end = offsets[1]
        dense_indice = tl.load(dense_indices_ptr + task_id)
        # if the dense_indice is -1 which mean it's a truncation case
        # in that case we don't need to do anything since the dense
        # initialize with padded value
        if dense_indice != -1:
            base_output_dense_ptr = dense_ptr + dense_indice
            base_jagged_values_ptr = jagged_values_ptr + begin * jagged_row_stride

            N = dense_row_stride
            M = tl.minimum(dense_matrix_stride // dense_row_stride, end - begin)

            offset_row = tl.arange(0, BLOCK_ROW)
            offset_col = tl.arange(0, BLOCK_COL)
            for _i in range(0, M, BLOCK_ROW):
                row = offset_row + _i
                for _j in range(0, N, BLOCK_COL):
                    col = offset_col + _j
                    block_offset = (
                        row[:, None] * dense_row_stride
                        + col[None, :] * dense_col_stride
                    )
                    mask = (row[:, None] < M) & (col[None, :] < N)
                    jagged_val = tl.load(
                        base_jagged_values_ptr + block_offset, mask=mask, other=0
                    )
                    tl.store(
                        base_output_dense_ptr + block_offset, jagged_val, mask=mask
                    )


def jagged_to_dense(
    jagged_values: torch.Tensor,
    jagged_offsets: List[torch.Tensor],
    jagged_max_lengths: List[int],
    padding_value: float = 0.0,
    operation_function: Union[str, None] = None,
    operation_dense: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    num_jagged_dim = len(jagged_offsets)
    jagged_values_dim = jagged_values.dim()
    device = jagged_values.device
    dtype = jagged_values.dtype

    assert (
        num_jagged_dim >= 1 and num_jagged_dim <= 5
    ), "num_jagged_dim must be no greater than 5 and no less than 1"
    assert (
        len(jagged_max_lengths) == num_jagged_dim
    ), f"len(jagged_max_length), {len(jagged_max_length)} != len(jagged_offsets), {num_jagged_dim}"
    assert jagged_values_dim == 1 or jagged_values_dim == 2

    outer_dense_size = jagged_offsets[0].size(0) - 1
    inner_dense_size = jagged_values.size(-1)
    jagged_max_lengths_tuple = tuple(jagged_max_lengths)
    inner_shape = (inner_dense_size,) if jagged_values_dim == 2 else ()
    output_shape = (outer_dense_size,) + jagged_max_lengths_tuple + inner_shape

    # zero element check
    if math.prod(output_shape) == 0:
        return torch.empty(output_shape, device=device, dtype=dtype)

    # for better performance, make input tensor contiguous to avoid discrete IO
    if not jagged_values.is_contiguous():
        jagged_values = jagged_values.contiguous()
    # avoid modifying the original list
    last_offset = (
        jagged_offsets[-1]
        if jagged_offsets[-1].is_contiguous()
        else jagged_offsets[-1].contiguous()
    )

    if num_jagged_dim > 1:
        output_dense = torch.full(
            output_shape,
            padding_value,
            device=device,
            dtype=dtype,
        )
    else:
        output_dense = torch.empty(
            output_shape,
            device=device,
            dtype=dtype,
        )

    if operation_function is not None:
        assert operation_dense.shape == output_dense.shape

    output_dense_view = (
        output_dense.unsqueeze(-1) if jagged_values_dim == 1 else output_dense
    )

    output_dense_view_stride = output_dense_view.stride()
    output_dense_view_size = output_dense_view.shape
    dense_matrix_stride, dense_row_stride, dense_col_stride = output_dense_view_stride[
        -3:
    ]

    processor_count = get_total_core_num(device.index)
    total_tasks = (
        outer_dense_size if num_jagged_dim == 1 else (last_offset.size(0) - 1)
    )  # split jagged_tensor

    if num_jagged_dim > 1:
        grid = (min(total_tasks, processor_count),)
        dense_indices = _jagged_offsets_to_dense_indice(
            jagged_offsets,
            output_dense_view_stride[:-2],
            output_dense_view_size[:-2],
        )
        # (TODO)perf improve: split dense_tensor if operation_function is not None
        jagged_to_dense_kernel[grid](
            jagged_values,
            last_offset,
            jagged_values.stride(0),
            output_dense_view,
            dense_indices,
            dense_col_stride,
            dense_row_stride,
            dense_matrix_stride,
            total_tasks,
        )

        if operation_function is not None:
            if operation_function == "add":
                output_dense = output_dense + operation_dense
            else:
                output_dense = output_dense * operation_dense  # "mul"
    else:
        BURST_SIZE = 128
        if (
            total_tasks < processor_count
            and output_dense_view_size[-2] >= processor_count
            and dense_matrix_stride * output_dense_view.element_size()
            >= processor_count * BURST_SIZE
        ):
            grid = (processor_count,)
            jagged_to_dense_2d_kernel_mini_batch[grid](
                jagged_values,
                last_offset,
                jagged_values.stride(0),
                output_dense_view,
                padding_value,
                dense_col_stride,
                dense_row_stride,
                dense_matrix_stride,
                total_tasks,
                operation_function,
                operation_dense if operation_function else None,
                processor_count=processor_count,
            )
        else:
            grid = (min(total_tasks, processor_count),)
            jagged_to_dense_2d_kernel[grid](
                jagged_values,
                last_offset,
                jagged_values.stride(0),
                output_dense_view,
                padding_value,
                dense_col_stride,
                dense_row_stride,
                dense_matrix_stride,
                total_tasks,
                operation_function,
                operation_dense if operation_function else None,
            )
    return output_dense
