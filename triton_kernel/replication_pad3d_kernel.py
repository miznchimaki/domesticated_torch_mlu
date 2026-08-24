import torch
import torch_mlu
import triton
import triton.language as tl
from triton.runtime import libentry

from .utils import get_total_core_num, get_max_nram_size


def get_autotune_config():
    base_configs = [
        triton.Config(
            {"BLOCK_W": w, "BLOCK_C": c},
            num_stages=0,
        )
        for w in [128, 256, 512, 1024]
        for c in [128, 256, 512]
    ]
    return base_configs


def filter_configs(configs, named_args, **kwargs):
    """Filter configs based on hardware constraints"""
    # MAX_ELEMS_PER_BLOCK = 1024 * 1024 # 1M elements
    MAX_NRAM_SIZE = int(get_max_nram_size() * 0.8)
    input = named_args["input_ptr"]
    W_out = named_args["W_out"]
    C = named_args["C"]
    element_size = input.element_size()
    valid_configs = []
    for cfg in configs:
        BLOCK_W = cfg.kwargs["BLOCK_W"]
        BLOCK_C = cfg.kwargs["BLOCK_C"]
        total_size = BLOCK_W * BLOCK_C * element_size
        if total_size > MAX_NRAM_SIZE:
            continue
        W_align = triton.next_power_of_2(W_out)
        C_align = triton.next_power_of_2(C)
        if BLOCK_W > max(W_align, 128) or BLOCK_C > max(C_align, 128):
            continue
        valid_configs.append(cfg)
    return valid_configs


def get_heuristics():
    # for get better performance, relax the scope of large tensors
    UINT32_MAX = 2**32 - 1
    return {
        "is_large_tensor": lambda named_args: (
            named_args["input_ptr"].numel() * named_args["input_ptr"].element_size()
            > UINT32_MAX
            or named_args["output_ptr"].numel()
            * named_args["output_ptr"].element_size()
            > UINT32_MAX
        )
    }


@libentry()
@triton.autotune(
    configs=get_autotune_config(),
    key=["D_out", "C"],
    prune_configs_by={"early_config_prune": filter_configs},
)
@triton.heuristics(get_heuristics())
@triton.jit
def replication_pad3d_kernel(
    input_ptr,
    output_ptr,
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
    # 总任务数 = N * D_out * H_out * num_blocks_w
    total_tasks = N * D_out * H_out * num_blocks_w
    # 获取当前核心ID和总核心数
    core_id = tl.program_id(0)
    total_cores = tl.num_programs(0)

    # 计算每个核心处理的任务量
    tasks_per_core = tl.cdiv(total_tasks, total_cores)
    start_task = core_id * tasks_per_core
    end_task = tl.minimum(start_task + tasks_per_core, total_tasks)

    # 预计算步长以优化指针计算
    input_n_stride = D_in * H_in * W_in * C
    input_d_stride = H_in * W_in * C
    input_h_stride = W_in * C

    output_n_stride = D_out * H_out * W_out * C
    output_d_stride = H_out * W_out * C
    output_h_stride = W_out * C

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
        w_valid = w_offsets < w_len

        # 计算输入索引（使用clamp处理边界）
        d_in = tl.minimum(tl.maximum(d_idx - pad_d_before, 0), D_in - 1)
        h_in = tl.minimum(tl.maximum(h_idx - pad_h_before, 0), H_in - 1)

        # 计算输入和输出的基础指针
        input_base = (
            input_ptr
            + n_idx * input_n_stride
            + d_in * input_d_stride
            + h_in * input_h_stride
        )
        output_base = (
            output_ptr
            + n_idx * output_n_stride
            + d_idx * output_d_stride
            + h_idx * output_h_stride
        )

        # 处理所有通道块
        for c_start in range(0, C, BLOCK_C):
            # 计算当前通道块的实际长度
            c_len = tl.minimum(BLOCK_C, C - c_start)
            c_valid = c_offsets < c_len

            # 计算W维度的输入索引（考虑padding）
            w_in_indices = tl.minimum(
                tl.maximum(w_start + w_offsets - pad_w_before, 0), W_in - 1
            )

            # 构造输入指针矩阵 [BLOCK_W, BLOCK_C]
            input_ptrs = input_base + (
                w_in_indices[:, None] * C + c_start + c_offsets[None, :]
            )

            # 加载数据到片上 [BLOCK_W, BLOCK_C]
            data_mask = (
                w_mask[:, None] & w_valid[:, None] & c_mask[None, :] & c_valid[None, :]
            )
            data = tl.load(input_ptrs, mask=data_mask, other=0)

            # 构造输出指针矩阵 [BLOCK_W, BLOCK_C]
            output_ptrs = output_base + (
                (w_start + w_offsets)[:, None] * C + c_start + c_offsets[None, :]
            )

            # 只存储有效数据
            tl.store(output_ptrs, data, mask=data_mask)
