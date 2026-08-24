"""
@Copyright (C) [2022-2025] by Cambricon.
@File    :   config.py
"""

import sys
import os
from pathlib import Path
from torch._inductor.runtime.runtime_utils import cache_dir
from torch_mlu._inductor import config as torch_mlu_config


LOGGER_CACHE = {}


def get_simple_logger(logger_name: str = "logging.INFO"):
    import logging

    def _get_log_level_from_env(default_level: int) -> int:
        import os

        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        env_level = os.getenv("TRITONFUSION_LOG_LEVEL", "INFO").upper()
        if env_level not in level_map:
            logging.warning(
                "can't find log level in TRITON_FUSION_LOG_LEVEL, set to default level: INFO"
            )
        return level_map.get(env_level, default_level)

    logger_name = logger_name.removeprefix("torch_mlu._inductor.fx_passes.")

    # early return.
    if logger_name in LOGGER_CACHE:
        return LOGGER_CACHE[logger_name]

    # create and setting logger.
    logger = logging.getLogger(logger_name)

    # default logging level set to logging.INFO
    log_level = _get_log_level_from_env(logging.INFO)
    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    LOGGER_CACHE[logger_name] = logger
    return logger


logger = get_simple_logger(__name__)

# Blacklist for some supported but not well performenced op names.
skipped_fusing_ops = []
skipped_fusing_ops += torch_mlu_config.tritonfusion.skipped_fusing_ops
logger.debug(f"Config get skipped_fusing_ops: {skipped_fusing_ops}")

# Root op type names for fusion start.
fusing_start_ops = [
    x.strip() for x in os.environ.get("TRITONFUSION_START_OPS", "").split(",") if x
] + [
    "aten.cat.default",
    "aten.mm.default",
    "aten.addmm.default",
    "aten.bmm.default",
    "aten.var_mean.correction",
]
fusing_start_ops = list(
    set([x for x in fusing_start_ops if x not in skipped_fusing_ops])
)
logger.debug(f"Config get fusing_start_ops: {fusing_start_ops}")

# Max/Min fused nodes limit.
max_fused_nodes_num = os.environ.get("TRITONFUSION_MAX_FUSE_NUM", None)
max_fused_nodes_num = (
    int(max_fused_nodes_num) if max_fused_nodes_num is not None else max_fused_nodes_num
)

min_fused_nodes_num = os.environ.get("TRITONFUSION_MIN_FUSE_NUM", 1)
min_fused_nodes_num = int(min_fused_nodes_num)
assert min_fused_nodes_num >= 1, "require at least 1 node in subgraph"
logger.debug(
    f"Config get max_fused_nodes_num: {max_fused_nodes_num}    min_fused_nodes_num: {min_fused_nodes_num}"
)


numtask_align_up = int(os.environ.get("TRITONFUSION_ALIGN_TASKNUM", 4))
logger.debug(f"Config get numtask_align_up: {numtask_align_up}")


# For tests only.
test_fallback_kernel = os.environ.get("TRITONFUSION_TEST_FALLBACK", "0") == "1"
logger.debug(f"Config get test_fallback_kernel: {test_fallback_kernel}")

# For pre test perf compare with eager.
pre_test_perf_eager = os.environ.get("TRITONFUSION_PRE_TEST_PERF_EAGER", "0") == "1"
logger.debug(f"Config get pre_test_perf_eager: {pre_test_perf_eager}")

# For pre check triton kernel.
pre_check_triton_kernel = os.environ.get("TRITONFUSION_PRE_CHECK_TRITON", "1") == "1"
logger.debug(f"Config get pre_check_triton_kernel: {pre_check_triton_kernel}")

cache_dir_path = os.environ.get(
    "TRITONFUSION_CACHE_DIR", os.path.join(cache_dir(), "tritonfusion")
)
logger.debug(f"Config get cache_dir_path: {cache_dir_path}")

target_batch_size = os.environ.get("TRITONFUSION_TARGET_BATCHSIZE", None)
logger.debug(f"Config get target_batch_size: {target_batch_size}")

TRITONFUSION_SAVE_TENSOR_ENV = "TRITONFUSION_SAVE_TENSORS"
save_in_out_tensors = os.environ.get(TRITONFUSION_SAVE_TENSOR_ENV, "0") == "1"
logger.debug(f"Config get save_in_out_tensors: {save_in_out_tensors}")

TRITONFUSION_LOAD_TENSOR_ENV = "TRITONFUSION_LOAD_TENSORS"
load_in_tensors = os.environ.get(TRITONFUSION_LOAD_TENSOR_ENV, "0") == "1"
logger.debug(f"Config get load_in_tensors: {load_in_tensors}")
