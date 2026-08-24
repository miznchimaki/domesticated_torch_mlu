import torch
import torch_mlu
import triton
import triton.language as tl
from triton.runtime import libentry

from .utils import get_total_core_num, get_max_nram_size


@libentry()
@triton.jit
def replication_pad3d_backward_normal_kernel(
    grad_output_ptr,
    grad_input_ptr,
    N,
    D_in,
    H_in,
    W_in,
    C,
    D_out,
    H_out,
    W_out,
    pad_d_before,
    pad_h_before,
    pad_w_before,
    is_large_tensor: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    if is_large_tensor:
        N = tl.full([], N, dtype=tl.int64)
        C = tl.full([], C, dtype=tl.int64)

    # 计算W维度的块数量
    num_blocks_w = tl.cdiv(W_out, BLOCK_W)
    # 总任务数
    total_tasks = N * D_out * H_out * num_blocks_w

    # 获取当前核心ID和总核心数
    core_id = tl.program_id(0)
    total_cores = tl.num_programs(0)

    # 计算每个核心处理的任务量
    tasks_per_core = tl.cdiv(total_tasks, total_cores)
    start_task = core_id * tasks_per_core
    end_task = tl.minimum(start_task + tasks_per_core, total_tasks)

    # 预计算步长以优化指针计算
    grad_input_n_stride = D_in * H_in * W_in * C
    grad_input_d_stride = H_in * W_in * C
    grad_input_h_stride = W_in * C

    grad_output_n_stride = D_out * H_out * W_out * C
    grad_output_d_stride = H_out * W_out * C
    grad_output_h_stride = W_out * C

    # 定义片上存储空间
    w_offsets = tl.arange(0, BLOCK_W)
    c_offsets = tl.arange(0, BLOCK_C)
    w_mask = w_offsets < BLOCK_W
    c_mask = c_offsets < BLOCK_C

    # 遍历分配给当前核心的所有任务
    for task_idx in range(start_task, end_task):
        # 将任务索引分解为(n, d, h, block_w)
        block_w_idx = task_idx % num_blocks_w
        h_idx = (task_idx // num_blocks_w) % H_out
        d_idx = (task_idx // (num_blocks_w * H_out)) % D_out
        n_idx = task_idx // (num_blocks_w * H_out * D_out)

        # 计算当前W块的起始位置和长度
        w_start = block_w_idx * BLOCK_W
        w_end = tl.minimum(w_start + BLOCK_W, W_out)
        w_len = w_end - w_start

        # 计算输入索引（使用clamp处理边界）
        d_in = tl.minimum(tl.maximum(d_idx - pad_d_before, 0), D_in - 1)
        h_in = tl.minimum(tl.maximum(h_idx - pad_h_before, 0), H_in - 1)

        # 计算输入和输出的基础指针
        grad_output_base = (
            grad_output_ptr
            + n_idx * grad_output_n_stride
            + d_idx * grad_output_d_stride
            + h_idx * grad_output_h_stride
        )

        grad_input_base = (
            grad_input_ptr
            + n_idx * grad_input_n_stride
            + d_in * grad_input_d_stride
            + h_in * grad_input_h_stride
        )

        # 处理所有通道块
        for c_start in range(0, C, BLOCK_C):
            # 计算当前通道块的实际长度
            c_len = tl.minimum(BLOCK_C, C - c_start)
            c_valid = c_offsets < c_len

            # 构造输出梯度指针矩阵 [BLOCK_W, BLOCK_C]
            grad_output_ptrs = grad_output_base + (
                (w_start + w_offsets)[:, None] * C + c_start + c_offsets[None, :]
            )

            # 加载输出梯度数据 [BLOCK_W, BLOCK_C]
            grad_data = tl.load(
                grad_output_ptrs,
                mask=(w_offsets < w_len)[:, None] & c_valid[None, :],
                other=0,
            )

            # 逐行处理，每行包含 BLOCK_C 个连续元素
            for w_i in range(BLOCK_W):
                # 只处理有效行
                if w_i < w_len:
                    # 计算当前行的指针和掩码
                    w_in_index = tl.minimum(
                        tl.maximum(w_start + w_i - pad_w_before, 0), W_in - 1
                    )
                    row_ptrs = grad_input_base + w_in_index * C + c_start + c_offsets
                    row_mask = c_mask & c_valid
                    row_data = grad_data[w_i, :]

                    # 执行原子加操作（整行连续内存）
                    tl.atomic_add(row_ptrs, row_data, mask=row_mask)


@libentry()
@triton.jit
def replication_pad3d_backward_opt_kernel(
    grad_output_ptr,
    grad_input_ptr,
    N,
    D_in,
    H_in,
    W_in,
    C,
    D_out,
    H_out,
    W_out,
    pad_d_before,
    pad_h_before,
    pad_w_before: tl.constexpr,
    pad_w_after: tl.constexpr,
    is_large_tensor: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    if is_large_tensor:
        N = tl.full([], N, dtype=tl.int64)
        C = tl.full([], C, dtype=tl.int64)

    # 计算W维度的块数量
    num_blocks_w = tl.cdiv(W_in, BLOCK_W)
    # 总任务数
    total_tasks = N * D_out * H_out * num_blocks_w

    # 获取当前核心ID和总核心数
    core_id = tl.program_id(0)
    total_cores = tl.num_programs(0)

    # 计算每个核心处理的任务量
    tasks_per_core = tl.cdiv(total_tasks, total_cores)
    start_task = core_id * tasks_per_core
    end_task = tl.minimum(start_task + tasks_per_core, total_tasks)

    # 预计算步长
    grad_input_n_stride = D_in * H_in * W_in * C
    grad_input_d_stride = H_in * W_in * C
    grad_input_h_stride = W_in * C

    grad_output_n_stride = D_out * H_out * W_out * C
    grad_output_d_stride = H_out * W_out * C
    grad_output_h_stride = W_out * C

    # 定义片上存储空间
    w_offsets = tl.arange(0, BLOCK_W)
    c_offsets = tl.arange(0, BLOCK_C)
    w_mask = w_offsets < BLOCK_W
    c_mask = c_offsets < BLOCK_C

    # 定义加载范围 - 使用纯编译时常量表达式
    load_w_offsets = tl.arange(0, BLOCK_W + pad_w_before + pad_w_after)
    load_w_mask = load_w_offsets < (BLOCK_W + pad_w_before + pad_w_after)

    # 遍历分配给当前核心的所有任务
    for task_idx in range(start_task, end_task):
        # 将任务索引分解为(n, d, h, block_w)
        block_w_idx = task_idx % num_blocks_w
        h_idx = (task_idx // num_blocks_w) % H_out
        d_idx = (task_idx // (num_blocks_w * H_out)) % D_out
        n_idx = task_idx // (num_blocks_w * H_out * D_out)

        # 计算当前W块的起始位置和实际长度
        w_start = block_w_idx * BLOCK_W
        w_end = tl.minimum(w_start + BLOCK_W, W_in)
        w_in_len = w_end - w_start

        # 检查是否需要处理边界padding
        need_pad_w_before = (w_start == 0) and (pad_w_before > 0)
        need_pad_w_after = (w_end == W_in) and (pad_w_after > 0)

        # 计算输入索引（clamp处理D/H边界）
        d_in = tl.minimum(tl.maximum(d_idx - pad_d_before, 0), D_in - 1)
        h_in = tl.minimum(tl.maximum(h_idx - pad_h_before, 0), H_in - 1)

        # 计算基础指针
        grad_output_base = (
            grad_output_ptr
            + n_idx * grad_output_n_stride
            + d_idx * grad_output_d_stride
            + h_idx * grad_output_h_stride
        )
        grad_input_base = (
            grad_input_ptr
            + n_idx * grad_input_n_stride
            + d_in * grad_input_d_stride
            + h_in * grad_input_h_stride
        )

        # 计算输出加载位置
        w_out_start = tl.where(need_pad_w_before, 0, pad_w_before + w_start)
        w_out_len = w_in_len + (
            pad_w_before * need_pad_w_before.to(tl.int32)
            + pad_w_after * need_pad_w_after.to(tl.int32)
        )
        w_out_end = tl.where(need_pad_w_after, W_out, w_out_start + w_out_len)

        # 计算加载位置
        w_out_indices = w_out_start + load_w_offsets
        w_out_valid = w_out_indices < w_out_end

        # 加载输出梯度 [BLOCK_W + pad_w_before + pad_w_after, BLOCK_C]
        grad_output_ptrs = grad_output_base + (
            w_out_indices[:, None] * C + c_offsets[None, :]
        )
        grad_data = tl.load(
            grad_output_ptrs,
            mask=w_out_valid[:, None] & load_w_mask[:, None] & c_mask[None, :],
            other=0.0,
        )

        # 1.=== 处理左padding ===
        if need_pad_w_before:
            # 聚合左padding梯度
            grad_data[pad_w_before, :] = tl.sum(
                grad_data[0 : pad_w_before + 1, :], axis=0
            )

        # 2.=== 处理右padding ===
        if need_pad_w_after:
            left_pad_end = tl.where(need_pad_w_before, pad_w_before, 0)
            # assert w_in_len >= 1
            # 聚合右padding梯度
            right_pad_grad_idx = left_pad_end + w_in_len - 1
            grad_data[right_pad_grad_idx, :] = tl.sum(
                grad_data[right_pad_grad_idx : right_pad_grad_idx + 1 + pad_w_after, :],
                axis=0,
            )

        # 3.=== 处理中间部分 ===
        if need_pad_w_before:
            center_data = grad_data[pad_w_before : pad_w_before + BLOCK_W, :]
        else:
            center_data = grad_data[0:BLOCK_W, :]

        # 原子加到输入对应位置
        center_ptrs = grad_input_base + (
            (w_start + w_offsets)[:, None] * C + c_offsets[None, :]
        )
        center_mask = (
            w_mask[:, None] & c_mask[None, :] & (w_offsets < w_in_len)[:, None]
        )
        tl.atomic_add(center_ptrs, center_data, mask=center_mask)


@libentry()
@triton.jit
def replication_pad3d_backward_opt_kernel_neg_left(
    grad_output_ptr,
    grad_input_ptr,
    N,
    D_in,
    H_in,
    W_in,
    C,
    D_out,
    H_out,
    W_out,
    pad_d_before,
    pad_h_before,
    pad_w_before,
    pad_w_after: tl.constexpr,
    is_large_tensor: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    if is_large_tensor:
        N = tl.full([], N, dtype=tl.int64)
        C = tl.full([], C, dtype=tl.int64)

    # 计算W维度的块数量
    real_W_in = W_in + pad_w_before
    num_blocks_w = tl.cdiv(real_W_in, BLOCK_W)
    # 总任务数
    total_tasks = N * D_out * H_out * num_blocks_w

    # 获取当前核心ID和总核心数
    core_id = tl.program_id(0)
    total_cores = tl.num_programs(0)

    # 计算每个核心处理的任务量
    tasks_per_core = tl.cdiv(total_tasks, total_cores)
    start_task = core_id * tasks_per_core
    end_task = tl.minimum(start_task + tasks_per_core, total_tasks)

    # 预计算步长
    grad_input_n_stride = D_in * H_in * W_in * C
    grad_input_d_stride = H_in * W_in * C
    grad_input_h_stride = W_in * C

    grad_output_n_stride = D_out * H_out * W_out * C
    grad_output_d_stride = H_out * W_out * C
    grad_output_h_stride = W_out * C

    # 定义片上存储空间
    w_offsets = tl.arange(0, BLOCK_W)
    c_offsets = tl.arange(0, BLOCK_C)
    w_mask = w_offsets < BLOCK_W
    c_mask = c_offsets < BLOCK_C

    # 定义加载范围 - 使用纯编译时常量表达式
    load_w_offsets = tl.arange(0, BLOCK_W + pad_w_after)
    load_w_mask = load_w_offsets < (BLOCK_W + pad_w_after)

    # 遍历分配给当前核心的所有任务
    for task_idx in range(start_task, end_task):
        # 将任务索引分解为(n, d, h, block_w)
        block_w_idx = task_idx % num_blocks_w
        h_idx = (task_idx // num_blocks_w) % H_out
        d_idx = (task_idx // (num_blocks_w * H_out)) % D_out
        n_idx = task_idx // (num_blocks_w * H_out * D_out)

        # 计算当前W块的起始位置和实际长度
        w_start = block_w_idx * BLOCK_W - pad_w_before
        w_end = tl.minimum(w_start + BLOCK_W, W_in)
        w_in_len = w_end - w_start

        # 检查是否需要处理边界padding
        need_pad_w_after = (w_end == W_in) and (pad_w_after > 0)

        # 计算输入索引（clamp处理D/H边界）
        d_in = tl.minimum(tl.maximum(d_idx - pad_d_before, 0), D_in - 1)
        h_in = tl.minimum(tl.maximum(h_idx - pad_h_before, 0), H_in - 1)

        # 计算基础指针
        grad_output_base = (
            grad_output_ptr
            + n_idx * grad_output_n_stride
            + d_idx * grad_output_d_stride
            + h_idx * grad_output_h_stride
        )
        grad_input_base = (
            grad_input_ptr
            + n_idx * grad_input_n_stride
            + d_in * grad_input_d_stride
            + h_in * grad_input_h_stride
        )

        # 计算输出加载位置
        w_out_start = w_start + pad_w_before
        w_out_len = w_in_len + (pad_w_after * need_pad_w_after.to(tl.int32))
        w_out_end = tl.where(need_pad_w_after, W_out, w_out_start + w_out_len)

        # 计算加载位置
        w_out_indices = w_out_start + load_w_offsets
        w_out_valid = w_out_indices < w_out_end

        # 加载输出梯度 [BLOCK_W + pad_w_after, BLOCK_C]
        grad_output_ptrs = grad_output_base + (
            w_out_indices[:, None] * C + c_offsets[None, :]
        )
        grad_data = tl.load(
            grad_output_ptrs,
            mask=w_out_valid[:, None] & load_w_mask[:, None] & c_mask[None, :],
            other=0.0,
        )

        # 1.=== 处理右padding ===
        if need_pad_w_after:
            # assert w_in_len >= 1
            # 聚合右padding梯度
            right_pad_grad_idx = w_in_len - 1
            grad_data[right_pad_grad_idx, :] = tl.sum(
                grad_data[right_pad_grad_idx : right_pad_grad_idx + 1 + pad_w_after, :],
                axis=0,
            )

        # 2.=== 处理中间部分 ===
        center_data = grad_data[0:BLOCK_W, :]

        # 原子加到输入对应位置
        center_ptrs = grad_input_base + (
            (w_start + w_offsets)[:, None] * C + c_offsets[None, :]
        )
        center_mask = (
            w_mask[:, None] & c_mask[None, :] & (w_offsets < w_in_len)[:, None]
        )
        tl.atomic_add(center_ptrs, center_data, mask=center_mask)


@libentry()
@triton.jit
def replication_pad3d_backward_opt_kernel_neg_right(
    grad_output_ptr,
    grad_input_ptr,
    N,
    D_in,
    H_in,
    W_in,
    C,
    D_out,
    H_out,
    W_out,
    pad_d_before,
    pad_h_before,
    pad_w_before: tl.constexpr,
    pad_w_after,
    is_large_tensor: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    if is_large_tensor:
        N = tl.full([], N, dtype=tl.int64)
        C = tl.full([], C, dtype=tl.int64)

    # 计算W维度的块数量
    real_W_in = W_in + pad_w_after
    num_blocks_w = tl.cdiv(real_W_in, BLOCK_W)
    # 总任务数
    total_tasks = N * D_out * H_out * num_blocks_w

    # 获取当前核心ID和总核心数
    core_id = tl.program_id(0)
    total_cores = tl.num_programs(0)

    # 计算每个核心处理的任务量
    tasks_per_core = tl.cdiv(total_tasks, total_cores)
    start_task = core_id * tasks_per_core
    end_task = tl.minimum(start_task + tasks_per_core, total_tasks)

    # 预计算步长
    grad_input_n_stride = D_in * H_in * W_in * C
    grad_input_d_stride = H_in * W_in * C
    grad_input_h_stride = W_in * C

    grad_output_n_stride = D_out * H_out * W_out * C
    grad_output_d_stride = H_out * W_out * C
    grad_output_h_stride = W_out * C

    # 定义片上存储空间
    w_offsets = tl.arange(0, BLOCK_W)
    c_offsets = tl.arange(0, BLOCK_C)
    w_mask = w_offsets < BLOCK_W
    c_mask = c_offsets < BLOCK_C

    # 定义加载范围 - 使用纯编译时常量表达式
    load_w_offsets = tl.arange(0, BLOCK_W + pad_w_before)
    load_w_mask = load_w_offsets < (BLOCK_W + pad_w_before)

    # 遍历分配给当前核心的所有任务
    for task_idx in range(start_task, end_task):
        # 将任务索引分解为(n, d, h, block_w)
        block_w_idx = task_idx % num_blocks_w
        h_idx = (task_idx // num_blocks_w) % H_out
        d_idx = (task_idx // (num_blocks_w * H_out)) % D_out
        n_idx = task_idx // (num_blocks_w * H_out * D_out)

        # 计算当前W块的起始位置和实际长度
        w_start = block_w_idx * BLOCK_W
        w_end = tl.minimum(w_start + BLOCK_W, real_W_in)
        w_in_len = w_end - w_start

        # 检查是否需要处理边界padding
        need_pad_w_before = (w_start == 0) and (pad_w_before > 0)

        # 计算输入索引（clamp处理D/H边界）
        d_in = tl.minimum(tl.maximum(d_idx - pad_d_before, 0), D_in - 1)
        h_in = tl.minimum(tl.maximum(h_idx - pad_h_before, 0), H_in - 1)

        # 计算基础指针
        grad_output_base = (
            grad_output_ptr
            + n_idx * grad_output_n_stride
            + d_idx * grad_output_d_stride
            + h_idx * grad_output_h_stride
        )
        grad_input_base = (
            grad_input_ptr
            + n_idx * grad_input_n_stride
            + d_in * grad_input_d_stride
            + h_in * grad_input_h_stride
        )

        # 计算输出加载位置
        w_out_start = tl.where(need_pad_w_before, 0, pad_w_before + w_start)
        w_out_len = w_in_len + (pad_w_before * need_pad_w_before.to(tl.int32))
        w_out_end = w_out_start + w_out_len

        # 计算加载位置
        w_out_indices = w_out_start + load_w_offsets
        w_out_valid = w_out_indices < w_out_end

        # 加载输出梯度 [BLOCK_W + pad_w_before, BLOCK_C]
        grad_output_ptrs = grad_output_base + (
            w_out_indices[:, None] * C + c_offsets[None, :]
        )
        grad_data = tl.load(
            grad_output_ptrs,
            mask=w_out_valid[:, None] & load_w_mask[:, None] & c_mask[None, :],
            other=0.0,
        )

        # 1.=== 处理左padding ===
        if need_pad_w_before:
            # 聚合左padding梯度
            grad_data[pad_w_before, :] = tl.sum(
                grad_data[0 : pad_w_before + 1, :], axis=0
            )

        # 2.=== 处理中间部分 ===
        if need_pad_w_before:
            center_data = grad_data[pad_w_before : pad_w_before + BLOCK_W, :]
        else:
            center_data = grad_data[0:BLOCK_W, :]

        # 原子加到输入对应位置
        center_ptrs = grad_input_base + (
            (w_start + w_offsets)[:, None] * C + c_offsets[None, :]
        )
        center_mask = (
            w_mask[:, None] & c_mask[None, :] & (w_offsets < w_in_len)[:, None]
        )
        tl.atomic_add(center_ptrs, center_data, mask=center_mask)


@libentry()
@triton.jit
def replication_pad3d_backward_opt_kernel_neg_both(
    grad_output_ptr,
    grad_input_ptr,
    N,
    D_in,
    H_in,
    W_in,
    C,
    D_out,
    H_out,
    W_out,
    pad_d_before,
    pad_h_before,
    pad_w_before,
    pad_w_after,
    is_large_tensor: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    if is_large_tensor:
        N = tl.full([], N, dtype=tl.int64)
        C = tl.full([], C, dtype=tl.int64)

    # 计算W维度的块数量
    real_W_in = W_in + pad_w_before + pad_w_after
    num_blocks_w = tl.cdiv(real_W_in, BLOCK_W)
    # 总任务数
    total_tasks = N * D_out * H_out * num_blocks_w

    # 获取当前核心ID和总核心数
    core_id = tl.program_id(0)
    total_cores = tl.num_programs(0)

    # 计算每个核心处理的任务量
    tasks_per_core = tl.cdiv(total_tasks, total_cores)
    start_task = core_id * tasks_per_core
    end_task = tl.minimum(start_task + tasks_per_core, total_tasks)

    # 预计算步长
    grad_input_n_stride = D_in * H_in * W_in * C
    grad_input_d_stride = H_in * W_in * C
    grad_input_h_stride = W_in * C

    grad_output_n_stride = D_out * H_out * W_out * C
    grad_output_d_stride = H_out * W_out * C
    grad_output_h_stride = W_out * C

    # 定义片上存储空间
    w_offsets = tl.arange(0, BLOCK_W)
    c_offsets = tl.arange(0, BLOCK_C)
    w_mask = w_offsets < BLOCK_W
    c_mask = c_offsets < BLOCK_C

    # 定义加载范围 - 使用纯编译时常量表达式
    load_w_offsets = tl.arange(0, BLOCK_W)
    load_w_mask = load_w_offsets < BLOCK_W

    # 遍历分配给当前核心的所有任务
    for task_idx in range(start_task, end_task):
        # 将任务索引分解为(n, d, h, block_w)
        block_w_idx = task_idx % num_blocks_w
        h_idx = (task_idx // num_blocks_w) % H_out
        d_idx = (task_idx // (num_blocks_w * H_out)) % D_out
        n_idx = task_idx // (num_blocks_w * H_out * D_out)

        # 计算当前W块的起始位置和实际长度
        w_start = block_w_idx * BLOCK_W - pad_w_before
        w_end = tl.minimum(w_start + BLOCK_W, W_in + pad_w_after)
        w_in_len = w_end - w_start

        # 检查是否需要处理边界padding
        need_pad_w_before = (w_start == 0) and (pad_w_before > 0)
        need_pad_w_after = (w_end == W_in) and (pad_w_after > 0)

        # 计算输入索引（clamp处理D/H边界）
        d_in = tl.minimum(tl.maximum(d_idx - pad_d_before, 0), D_in - 1)
        h_in = tl.minimum(tl.maximum(h_idx - pad_h_before, 0), H_in - 1)

        # 计算基础指针
        grad_output_base = (
            grad_output_ptr
            + n_idx * grad_output_n_stride
            + d_idx * grad_output_d_stride
            + h_idx * grad_output_h_stride
        )
        grad_input_base = (
            grad_input_ptr
            + n_idx * grad_input_n_stride
            + d_in * grad_input_d_stride
            + h_in * grad_input_h_stride
        )

        # 计算输出加载位置
        w_out_start = w_start + pad_w_before
        w_out_len = w_in_len
        w_out_end = w_out_start + w_out_len

        # 计算加载位置
        w_out_indices = w_out_start + load_w_offsets
        w_out_valid = w_out_indices < w_out_end

        # 加载输出梯度 [BLOCK_W, BLOCK_C]
        grad_output_ptrs = grad_output_base + (
            w_out_indices[:, None] * C + c_offsets[None, :]
        )
        grad_data = tl.load(
            grad_output_ptrs,
            mask=w_out_valid[:, None] & load_w_mask[:, None] & c_mask[None, :],
            other=0.0,
        )

        # 1.=== 处理中间部分 ===
        center_data = grad_data[0:BLOCK_W, :]

        # 原子加到输入对应位置
        center_ptrs = grad_input_base + (
            (w_start + w_offsets)[:, None] * C + c_offsets[None, :]
        )
        center_mask = (
            w_mask[:, None] & c_mask[None, :] & (w_offsets < w_in_len)[:, None]
        )
        tl.atomic_add(center_ptrs, center_data, mask=center_mask)
