import copy
from itertools import product
import os
import sys
import functools
import logging
import math
import numpy as np

from typing import Any, Dict, List, Optional, Union, Literal

import torch
from torch._inductor.runtime.runtime_utils import (
    conditional_product,
    ceildiv,
    get_first_attr,
    get_max_y_grid,
    triton_hash_to_path_key,
)
from torch._inductor.runtime.triton_compat import (
    ASTSource,
    CompiledKernel,
    Config,
    GPUTarget,
    OutOfResources,
    PTXASError,
    triton,
    HAS_WARP_SPEC,
)
from torch._inductor.runtime.triton_heuristics import (
    cached_autotune,
    CachingAutotuner,
    HeuristicType,
    NoTritonConfigsError,
    TritonCompileResult,
    _get_config,
    GridExpr,
    config_to_dict,
    CompileResult,
    _KernelType,
    triton_config_reduction,
)
from torch._inductor.runtime import triton_helpers
from torch.utils._triton import has_triton_package
from torch._inductor.triton_bundler import TritonBundler
from torch._inductor.runtime.benchmarking import benchmarker
from ...utils import gorilla

log = logging.getLogger(__name__)


def get_total_block_size(cfg):
    block = 1
    for label in "XYZ":
        key = f"{label}BLOCK"
        if key in cfg.kwargs:
            block *= cfg.kwargs[key]
    return block


def extract_nram_info(compile_result, oom_log):
    import re

    if compile_result:
        binary = compile_result.kernel
        nram_size_regex = re.compile("mlu\.alloc\(\).*memref<(\d+)x.(\d+), 101>.*")
        mluir = binary.asm["mluiropt"]
        nram_info = re.search(nram_size_regex, mluir)
        if nram_info:
            length, type_size = nram_info.groups()
            length, type_size = int(length), int(type_size)
            nram_size = int(length * type_size / 8)
        else:
            nram_size = None
    else:
        nram_size_regex = re.compile("out of resource: NRAM, Required: (\d+),")
        nram_size = int(re.search(nram_size_regex, oom_log).group(1))
    return nram_size


def is_oom_for_config(
    config: Config, oom_reduction_configs: Optional[list[Config]], is_reduction: bool
):
    """Check if a configuration is likely to cause OOM based on historical OOMs."""

    def _is_special_case(current: tuple, oom: tuple) -> bool:
        if math.prod(current) != math.prod(oom):
            return False
        # Case 1: 2D configs where highest dim is smaller but product same
        if len(current) == 2 and current[0] < oom[0]:
            return True
        # Case 2: 3D configs where first two dims are smaller
        if len(current) == 3 and all(current[i] < oom[i] for i in (0, 1)):
            return True
        return False

    block_sizes = tuple(config.kwargs.values())
    for oom_config in reversed(oom_reduction_configs):
        oom_block_size = tuple(oom_config.kwargs.values())

        # Direct size comparison
        if all(a >= b for a, b in zip(block_sizes, oom_block_size)):
            return True

        # Special handling for non-reduction kernels
        if not is_reduction and _is_special_case(block_sizes, oom_block_size):
            return True
    return False


def sorted_configs(configs, reverse=False):
    if len(configs) <= 1:
        return configs

    return sorted(
        configs, key=lambda c: tuple([v for _, v in c.kwargs.items()]), reverse=reverse
    )


def calculate_metrics(
    xnumel, ynumel, znumel, rnumel, xblock, yblock, zblock, rblock, core_num
):
    """Calculate block number, max loop count, and max workload.  Handles 1D, 2D, and 3D cases."""
    grid_x = ceildiv(xnumel, xblock) if xnumel is not None else 1
    grid_y = ceildiv(ynumel, yblock) if ynumel is not None else 1
    grid_z = ceildiv(znumel, zblock) if znumel is not None else 1
    grid_r = ceildiv(rnumel, rblock) if rnumel is not None else 1
    block_num = grid_x * grid_y * grid_z * grid_r
    actual_core_num = min(core_num, block_num)
    max_loop_count = ceildiv(block_num, actual_core_num)
    total_blocks = xblock * (yblock or 1) * (zblock or 1) * (rblock or 1)
    max_workload = core_num * max_loop_count * total_blocks
    return block_num, max_loop_count, max_workload


def estimate_nram_usage(config: Config, coefficients: list[float]) -> int:
    """Estimate NRAM usage based on config and pre-calculated coefficients."""
    x = config.kwargs.get("XBLOCK", 1)
    y = config.kwargs.get("YBLOCK", 1)
    r = config.kwargs.get("R0_BLOCK", 1)
    y *= r  # Only one of y and r is valid

    if len(coefficients) == 1:
        return int(x * coefficients[0])
    elif len(coefficients) == 3:
        return int((coefficients[0] * x + coefficients[1]) * y + coefficients[2] * x)
    else:
        raise ValueError("Invalid number of coefficients for NRAM estimation.")


def select_base_configs(configs, is_pointwise):
    # does not support len of size_hints is 3
    dim = len(configs[0].kwargs.keys())
    base_configs = []
    if dim == 1:
        base_configs.append(configs[-1])
    elif dim == 2:
        if is_pointwise:
            base_block_size = [
                (1024, 1024),
                (8, 32768),
                (16384, 16),
            ]
            base_configs = [
                Config({"XBLOCK": x, "YBLOCK": y}, num_warps=1, num_stages=3)
                for x, y in base_block_size
            ]
        else:
            # we only need to find 3 configs for sampling method
            x_size_record = []
            r_size_record = []
            for c in configs:
                x, r = list(c.kwargs.values())
                if x not in x_size_record and r not in r_size_record:
                    x_size_record.append(x)
                    r_size_record.append(r)
                    base_configs.append(c)
                    if len(base_configs) == 3:
                        break

            if len(base_configs) < 3:
                for c in configs:
                    if c not in base_configs:
                        base_configs.append(c)
                    if len(base_configs) == 3:
                        break
            assert len(base_configs) == 3

    return base_configs


def filter_configs(
    cls: CachingAutotuner,
    configs: Optional[list[Config]] = None,
    size_hints: dict[str, int] = None,
    compiler_reserve_nram=32 * 1024,
) -> list[Config]:
    from torch._dynamo.utils import dynamo_timed

    with dynamo_timed("filter configs"):
        return _filter_configs(cls, configs, size_hints, compiler_reserve_nram)


def _filter_configs(
    cls: CachingAutotuner,
    configs: Optional[list[Config]] = None,
    size_hints: dict[str, int] = None,
    compiler_reserve_nram=32 * 1024,
) -> list[Config]:
    """Filters Triton configs based on NRAM usage.

    Args:
        cls: Autotuner instance containing configs and device properties.
        compiler_reserve_nram: Reserved NRAM for compiler (default 16KB).

    Returns:
        Filtered configs sorted by performance metrics, with NRAM-safe configs first.
    """
    is_pointwise = cls.heuristic_type == HeuristicType.POINTWISE
    if size_hints is None:
        size_hints = cls.size_hints
    dim = len(size_hints)
    # only support pointwise 1d, 2d, 3d and reduction 2d, 3d
    if (is_pointwise and dim not in [1, 2, 3]) or (
        not is_pointwise and dim not in [2, 3]
    ):
        raise ValueError(f"Invalid size_hints dimension: {dim}.")

    if not configs:
        configs = cls.configs
    nram_limit = cls.device_props.onchip_mem_size - compiler_reserve_nram
    core_num = cls.device_props.multi_processor_count
    candidate_configs = []
    exceeding_configs = []

    if is_pointwise:
        logging.debug("Using Genesis API for pointwise NRAM estimation")

        configs = sorted_configs(configs)
        compile_result_map = {}
        cls.analysis_types = "nram"
        compile_count = 0
        for config in configs:
            if is_oom_for_config(config, exceeding_configs, True):
                exceeding_configs.append(config)
            try:
                block_size_equal_one = [
                    (prefix, block_size)
                    for prefix, block_size in config.kwargs.items()
                    if block_size == 1
                ]
                compile_key = (
                    tuple(block_size_equal_one) if block_size_equal_one else "default"
                )
                compile_result = compile_result_map.get(compile_key, None)
                if not compile_result:
                    compile_count += 1
                    compile_result = cls._precompile_config(config)
                    compile_result_map[compile_key] = compile_result
                options = {
                    "num_warps": config.num_warps,
                    "num_stages": config.num_stages,
                    "kernel_name": cls.inductor_meta["kernel_name"],
                    "restrict_ptr_hint": True,
                    "onchip_mem_analysis": "nram",
                }
                nram_size = compile_result.onchip_mem_cal(config.kwargs, options)
            except (RuntimeError, KeyError) as e:
                candidate_configs.append(config)
                continue
            except triton.CompilationError as e:
                continue
            logging.debug(
                f"estimate nram: {cls.inductor_meta['kernel_name']}, {config.kwargs}, num_stage={config.num_stages}, nram_size={nram_size}"
            )
            if nram_size < nram_limit:
                candidate_configs.append(config)
            else:
                exceeding_configs.append(config)
        cls.analysis_types = ""
        logging.debug(
            f"precompile {compile_count} times for {len(configs)} configs in nram estimate phase"
        )
    else:
        logging.debug("Using Legacy API for reduction NRAM estimation")
        nram_info = []
        base_configs = select_base_configs(configs, is_pointwise)
        for config in base_configs:
            oom_log = None
            try:
                single_compile_result = cls._precompile_config(config)
            except OutOfResources as e:
                single_compile_result = None
                oom_log = str(e)
            except Exception as e:
                raise e
            nram_size = extract_nram_info(single_compile_result, oom_log)
            if nram_size is None:
                break
            nram_info.append((config, nram_size))

        if len(nram_info) != 3:
            return configs  # Insufficient nram_info for filtering
        A = np.array(
            [
                [
                    c.kwargs["XBLOCK"] * c.kwargs["R0_BLOCK"],
                    c.kwargs["R0_BLOCK"],
                    c.kwargs["XBLOCK"],
                ]
                for c, _ in nram_info
            ]
        )
        b = np.array([nram_size for _, nram_size in nram_info])
        coefficients = np.linalg.lstsq(A, b, rcond=None)[0]

        for config in configs:
            estimated_nram = estimate_nram_usage(config, coefficients)
            logging.debug(
                f"estimate nram: {cls.inductor_meta['kernel_name']}, {config.kwargs}, {estimated_nram}"
            )
            if estimated_nram < nram_limit:
                candidate_configs.append(config)
            else:
                exceeding_configs.append(config)

    # Fallback when no valid configs found
    if not candidate_configs:
        return configs
    xnumel = size_hints.get("x", None)
    ynumel = size_hints.get("y", None)
    znumel = size_hints.get("z", None)
    rnumel = size_hints.get("r0_", None)
    total_numel = math.prod(size_hints.values())

    def compute_normalization_constants():
        loops, workloads, xblocks, yblocks, zblocks = [], [], [], [], []
        for cfg in candidate_configs:
            xblock = cfg.kwargs.get("XBLOCK", 1)
            yblock = cfg.kwargs.get("YBLOCK", 1)
            zblock = cfg.kwargs.get("ZBLOCK", 1)
            _, loop, workload = calculate_metrics(
                xnumel, ynumel, znumel, 1, xblock, yblock, zblock, 1, core_num
            )
            loops.append(loop)
            workloads.append(workload)
            xblocks.append(xblock)
            yblocks.append(yblock)
            zblocks.append(zblock)

        # Use the 95th percentile as the reference for normalization.
        loop_q95 = np.percentile(loops, 95)
        workload_q95 = np.percentile(workloads, 95)
        xblock_q95 = np.percentile(xblocks, 95)
        yblock_q95 = np.percentile(yblocks, 95)
        zblock_q95 = np.percentile(zblocks, 95)

        return loop_q95, xblock_q95, yblock_q95, zblock_q95, workload_q95

    def make_pointwise_sort_key():
        def _pointwise_sort_key(config: Config):
            xblock = config.kwargs.get("XBLOCK", 1)
            yblock = config.kwargs.get("YBLOCK", 1)
            zblock = config.kwargs.get("ZBLOCK", 1)

            _, loop_count, workload = calculate_metrics(
                xnumel, ynumel, znumel, 1, xblock, yblock, zblock, 1, core_num
            )

            block_score = 0
            if dim == 2 or dim == 3:
                block_size = [zblock, yblock, xblock]
                block_q95 = [zblock_q95, yblock_q95, xblock_q95]
                block_score = min(
                    math.log2(block_size[-1]) / math.log2(block_q95[-1]),
                    1.0,
                )

                if xblock == 1 or yblock == 1:
                    block_score -= 0.1
                if xnumel <= 64 and xblock == xnumel:
                    block_score += 0.1
                if ynumel <= 64 and yblock == ynumel:
                    block_score += 0.1

            loop_score = max(
                1.0 - math.log2(loop_count + 1) / math.log2(loop_q95 + 1), 0.0
            )
            workload_score = max(1.0 - workload / workload_q95, 0.0)

            score = 0.5 * loop_score + 0.3 * block_score + 0.2 * workload_score

            numel_block_pairs = [(xnumel, xblock)]
            if dim >= 2:
                numel_block_pairs.append((ynumel, yblock))
            if dim >= 3:
                numel_block_pairs.append((znumel, zblock))

            num_splits = 1
            valid = True

            for numel, block in numel_block_pairs:
                if numel % block != 0:
                    valid = False
                    break
                num_splits *= numel // block

            if valid and num_splits % (core_num / 4) == 0:
                score += 0.1

            logging.debug(
                f"config={config.kwargs}, score={score:.4f}, loop={loop_count}, loop_score={loop_score}, block_score={block_score}, workload={workload}, workload_score={workload_score}"
            )
            return (score, xblock, yblock, zblock)

        return _pointwise_sort_key

    def _reduction_sort_key(config: Config) -> tuple:
        xblock_size = config.kwargs.get("XBLOCK", 1)
        rblock_size = config.kwargs.get("R0_BLOCK", 1)
        _, max_loop_count, max_workload = calculate_metrics(
            xnumel, 1, 1, rnumel, xblock_size, 1, 1, rblock_size, core_num
        )
        metrics = (
            -max_loop_count,
            -max_workload,
            rblock_size,
            xblock_size,
        )
        logging.debug(f"config is {config.kwargs}, and metrics is {metrics}")
        return metrics

    def select_top_k_with_diversity(
        configs, k=6, diversity_key="XBLOCK", monitor_key="YBLOCK"
    ):
        """
        Configs selection based on diversity and power-of-two properties

        Algorithm Strategy:
        1. For 3D config (with ZBLOCK): Use diversity + monitoring strategy to select top k-2 configs
        2. For 2D config: Use basic diversity strategy to select top k-2 configs
        3. From remaining configs, select 2 config with power-of-two properties
        """
        diverse = []  # Configs selected by diversity criteria
        seen_diversity = set()
        monitor_count = {}
        fallback_0 = []  # Configs not selected in first stage
        diverse_count = 0

        is_3d_config = len(configs[0].kwargs) == 3
        if is_3d_config:
            for cfg in configs:
                key_val = cfg.kwargs.get(diversity_key)
                monitor_val = cfg.kwargs.get(monitor_key)
                if diverse_count < k - 2:
                    monitor_repeated = monitor_count.get(monitor_val, 0) >= 2
                    if key_val not in seen_diversity:
                        if not monitor_repeated:
                            diverse.append(cfg)
                            seen_diversity.add(key_val)
                            monitor_count[monitor_val] = (
                                monitor_count.get(monitor_val, 0) + 1
                            )
                        else:
                            fallback_0.append(cfg)
                        diverse_count += 1
                    else:
                        fallback_0.append(cfg)
                else:
                    fallback_0.append(cfg)
        else:
            for cfg in configs:
                key_val = cfg.kwargs.get(diversity_key)
                if len(diverse) < k - 2 and key_val not in seen_diversity:
                    diverse.append(cfg)
                    seen_diversity.add(key_val)
                else:
                    fallback_0.append(cfg)

        # Stage 2: Power-of-Two Config Selection
        power_of_two_configs = []  # Configs with power-of-two properties
        fallback_1 = []  # Final unselected configs

        def is_power_of_two(n):
            return n > 1 and (n & (n - 1)) == 0

        def zblock_yblock_power_of_two(cfg):
            return is_power_of_two(cfg.kwargs.get("ZBLOCK")) and is_power_of_two(
                cfg.kwargs.get("YBLOCK")
            )

        if is_3d_config:
            has_all_power_of_two = False
            for cfg in fallback_0:
                if len(power_of_two_configs) < 2:
                    if has_all_power_of_two and zblock_yblock_power_of_two(cfg):
                        power_of_two_configs.append(cfg)
                    if not has_all_power_of_two and all(
                        is_power_of_two(value) for value in cfg.kwargs.values()
                    ):
                        power_of_two_configs.append(cfg)
                        has_all_power_of_two = True
                    fallback_1.append(cfg)
                else:
                    fallback_1.append(cfg)
        else:
            for cfg in fallback_0:
                if len(power_of_two_configs) < 2 and all(
                    is_power_of_two(value) for value in cfg.kwargs.values()
                ):
                    power_of_two_configs.append(cfg)
                else:
                    fallback_1.append(cfg)

        return diverse + power_of_two_configs + fallback_1

    if is_pointwise:
        (
            loop_q95,
            xblock_q95,
            yblock_q95,
            zblock_q95,
            workload_q95,
        ) = compute_normalization_constants()
        sort_key = make_pointwise_sort_key()
    else:
        sort_key = _reduction_sort_key

    candidate_configs = sorted(candidate_configs, key=sort_key, reverse=True)
    candidate_configs = select_top_k_with_diversity(candidate_configs)
    # exceeding_configs = sorted(exceeding_configs, key=sort_key, reverse=True)
    # candidate_configs = candidate_configs + exceeding_configs
    logging.debug([c.kwargs for c in candidate_configs])
    return candidate_configs


@gorilla.patch(CachingAutotuner)
def _precompile_config(self, cfg: Config) -> CompileResult[_KernelType]:
    """Ahead of time compile a given autotuner config."""
    compile_meta = self._create_compile_meta(cfg)
    # Add by CAMBRICON
    from torch.mlu.memory import is_linear_memory_enabled
    from torch_mlu._inductor import config as inductor_config

    if inductor_config.debug_tunning:
        print("compile kernel:", self.fn)

    compile_meta = copy.deepcopy(self.triton_meta)
    cfg_kwargs = cfg.kwargs
    if self.device_props.type == "hip":
        cfg_kwargs = {**cfg_kwargs}
        for k in ("matrix_instr_nonkdim", "waves_per_eu", "kpack"):
            if k in cfg_kwargs:
                compile_meta[k] = cfg_kwargs.pop(k)
    compile_meta["constants"].update(cfg_kwargs)
    for i in self.fn.constexprs:
        arg_name = self.fn.arg_names[i]
        if arg_name not in compile_meta["constants"] and (
            arg_name == "num_warps" or arg_name == "num_stages"
        ):
            compile_meta["constants"][arg_name] = getattr(cfg, arg_name)
    compile_meta["num_warps"] = cfg.num_warps
    compile_meta["num_stages"] = cfg.num_stages
    if HAS_WARP_SPEC:
        compile_meta["num_consumer_groups"] = getattr(cfg, "num_consumer_groups", 0)
        compile_meta["num_buffers_warp_spec"] = getattr(cfg, "num_buffers_warp_spec", 0)
    compile_meta["debug"] = self.inductor_meta.get(
        "assert_indirect_indexing", True
    ) and not self.inductor_meta.get("is_hip", False)

    # device type will be "hip" rather than "cuda" here
    compile_meta["device_type"] = self.device_props.type
    compile_meta["cc"] = self.device_props.cc

    compile_meta["silence"] = True
    compile_meta["is_linear"] = (
        is_linear_memory_enabled()
        if self.device_props.supports_linear_memory
        else False
    )
    compile_meta["restrict_ptr_hint"] = True
    compile_meta["isa_version"] = self.device_props.cc
    analysis_types = self.analysis_types if hasattr(self, "analysis_types") else ""
    # end Add by CAMBRICON

    if self.device_props.type == "cpu":
        triton_helpers.set_driver_to_cpu()
    else:
        triton_helpers.set_driver_to_gpu()

    if not ASTSource:
        raise RuntimeError("Installed triton version too old, please upgrade")

    compile_args = (
        ASTSource(
            self.fn,
            # Modify by Cambricon
            # compile_meta["signature"],
            copy.deepcopy(compile_meta["signature"]),
            # end Modify by Cambricon
            compile_meta["constants"],
            compile_meta["configs"][0],
        ),
    )

    # Modify by CAMBRICON
    # if self.device_props.type == "mtia":
    #     from mtia.host_runtime.torch_mtia.acc_flags import (  # type: ignore[import-not-found]
    #         build_codename,
    #     )

    #     arch = build_codename()
    # else:
    #     arch = compile_meta["cc"]
    # target = GPUTarget(
    #     compile_meta["device_type"],
    #     arch,
    #     cc_warp_size(compile_meta["cc"]),
    # )
    target = GPUTarget(
        compile_meta["device_type"],
        compile_meta["cc"],
        1,
    )

    # options = self._create_compile_options(cfg, compile_meta)
    options = {
        "num_warps": compile_meta["num_warps"],
        "num_stages": compile_meta["num_stages"],
        "debug": compile_meta["debug"],
        "sanitize_overflow": False,  # turn off additional asserts added for overflow checks
        "silence": compile_meta["silence"],
        "isa_version": compile_meta["isa_version"],
        "is_linear": compile_meta["is_linear"],
        "restrict_ptr_hint": compile_meta["restrict_ptr_hint"],
        "onchip_mem_analysis": analysis_types,
    }
    if self.device_props.type == "hip":
        if "waves_per_eu" in compile_meta:
            options["waves_per_eu"] = compile_meta["waves_per_eu"]
        if "matrix_instr_nonkdim" in compile_meta:
            options["matrix_instr_nonkdim"] = compile_meta["matrix_instr_nonkdim"]

    compile_kwargs = {
        "target": target,
        "options": options,
    }

    if analysis_types:
        result = triton.compile(*compile_args, **compile_kwargs)
        return result
    # end Modify by CAMBRICON

    try:
        binary = triton.compile(*compile_args, **compile_kwargs)

    # Add by CAMBRICON
    except OutOfResources as e:
        raise e
    # end Add by CAMBRICON
    except Exception:
        log.exception(
            "Triton compilation failed: %s\n%s\nmetadata: %s",
            self.inductor_meta.get("kernel_name", "triton_"),
            self.fn.src,
            compile_meta,
        )
        raise

    # Simulate JIT Hook call
    if (
        torch._inductor.config.run_jit_post_compile_hook
        and knobs
        and getattr(knobs.runtime, "jit_post_compile_hook", None)
    ):
        try:
            hook = knobs.runtime.jit_post_compile_hook

            # base args everyone should get
            call_kwargs = dict(
                key=getattr(self.fn, "cache_key", self.kernel_hash or str(self.fn)),
                repr=getattr(self.fn, "src", None),
                fn=self.fn,
                compile=binary,
                is_manual_warmup=False,
                already_compiled=True,
            )

            # only add inductor_args if the hook takes it
            sig = inspect.signature(hook)
            params = sig.parameters
            if "inductor_args" in params and "config_args" in self.inductor_meta:
                call_kwargs["inductor_args"] = self.inductor_meta["config_args"]

            hook(**call_kwargs)
        except Exception:
            log.exception("jit_post_compile_hook failed")
    TritonBundler.put(
        triton_hash_to_path_key(binary.hash), self.triton_meta.get("device", 0)
    )
    # If the binary has a cubin file to directly launch, save it on the binary
    static_launcher = StaticTritonCompileResult.can_statically_launch(
        binary, self.inductor_meta, self.triton_meta, self.heuristic_type
    )

    if static_launcher is not None:
        result = StaticTritonCompileResult(
            static_launcher, cfg, compile_meta, self.inductor_meta
        )
        return result

    return TritonCompileResult(binary, cfg, compile_meta, self.inductor_meta)


@gorilla.patch(CachingAutotuner)
def _precompile_worker(self):
    # Add by CAMBRICON
    from torch_mlu._inductor import config
    from torch_mlu._inductor.runtime.triton_heuristics import (
        extract_nram_info,
        filter_configs,
        is_oom_for_config,
        select_base_configs,
        sorted_configs,
    )
    from torch._dynamo.utils import dynamo_timed

    # end Add by CAMBRICON

    if self.compile_results:
        for result in self.compile_results:
            TritonBundler.put(
                triton_hash_to_path_key(result.kernel.hash),  # type: ignore[attr-defined]
                self.triton_meta.get("device", 0),
            )
        return
    assert not self.launchers
    if not self.configs:
        raise NoTritonConfigsError("No triton configs are available")

    # Add by CAMBRICON
    configs = sorted_configs(self.configs)
    is_pointwise = self.heuristic_type == HeuristicType.POINTWISE
    enable_filter_configs = config.filter_configs and (
        (is_pointwise and len(configs) >= 2)
        or (self.heuristic_type is HeuristicType.REDUCTION and len(configs) > 3)
    )
    if enable_filter_configs:
        configs = filter_configs(self)

    oom_config_record = []
    # end Add by CAMBRICON
    compile_results = []
    exc = None
    # Modify by CAMBRICON
    # for c in self.configs:
    for c in configs:
        if config.debug_tunning:
            print("current config:", c)
        if is_oom_for_config(c, oom_config_record, not is_pointwise):
            oom_config_record.append(c)
            continue
        try:
            compile_results.append(self._precompile_config(c))

        # except (OutOfResources, PTXASError, IntelGPUError) as e:
        except (OutOfResources, PTXASError) as e:
            exc = e
        if enable_filter_configs and len(compile_results) == (
            8 if len(c.kwargs.values()) == 3 else 6
        ):
            break
    # end Modify by CAMBRICON

    if len(compile_results) == 0:
        raise NoTritonConfigsError(
            f"No valid triton configs. {type(exc).__name__}: {exc}"
        )
    self.compile_results = compile_results
    self.configs = None


@gorilla.patch(CachingAutotuner)
def save_gpu_kernel(self, stream, launcher):
    key = self.inductor_meta.get("kernel_name", None)  # unique kernel name
    assert key is not None, "kernel_name can not be None"
    params = {
        "mangled_name": (
            launcher.bin.metadata.name
            if hasattr(launcher.bin.metadata, "name")
            else launcher.bin.metadata["name"]
        ),
        "num_warps": (
            launcher.bin.num_warps
            if hasattr(launcher.bin, "num_warps")
            else launcher.bin.metadata.num_warps
        ),
        "shared_mem": (
            launcher.bin.shared
            if hasattr(launcher.bin, "shared")
            else launcher.bin.metadata.shared
        ),
        "stream": stream,
        # User defined triton kernels will have arbitrary kwarg names
        "config": config_to_dict(launcher.config),
        "inductor_meta": self.inductor_meta,
        "triton_meta": self.triton_meta,
        "def_args": launcher.def_args,
        "call_args": launcher.call_args,
        "global_scratch": launcher.global_scratch,
        "profile_scratch": launcher.profile_scratch,
    }
    if self.device_props.type == "xpu":
        # On the XPU backend, threads_per_warp is not always 32.
        # For Intel GEMM Triton kernels, it can be 16.
        # This information must be preserved so that the Cpp wrapper
        # can launch the kernel with the correct configuration.
        params["threads_per_warp"] = getattr(
            launcher.bin.metadata, "threads_per_warp", 32
        )

    from torch._inductor import config
    from torch._inductor.codecache import CudaKernelParamCache

    # Modify by CAMBRICON
    # bin_type = {"hip": "hsaco", "xpu": XPU_KERNEL_FORMAT}.get(
    #     self.device_props.type, "cubin"
    # )
    bin_type = {"hip": "hsaco", "xpu": XPU_KERNEL_FORMAT, "mlu": "cnbin"}.get(
        self.device_props.type, "cubin"
    )
    binary = launcher.bin.asm[bin_type]

    # ROCm multi-arch: capture LLVM IR
    if torch.version.hip and config.aot_inductor.emit_multi_arch_kernel:
        # Multi-arch ROCm: Capture LLVM IR for cross-architecture compilation
        asm_type = "ll"

        # llir is the key to obtain LLVM IR from triton
        asm = launcher.bin.asm.get("llir", None)

        # CRITICAL: Multi-arch compilation cannot proceed without LLVM IR
        # Fail fast with clear error message pointing to the issue
        if not asm:
            available_keys = list(launcher.bin.asm.keys())
            raise RuntimeError(
                f"ROCm multi-arch requires LLVM IR, but none found. "
                f"Available keys: {available_keys}. "
                f"Triton may need to be patched to emit LLVM IR."
            )

    # Everything else: capture architecture-specific assembly
    else:
        # asm_type = {"hip": "amdgcn", "cuda": "ptx", "xpu": "spv"}.get(self.device_props.type)
        asm_type = {"hip": "amdgcn", "cuda": "ptx", "xpu": "spv", "mlu": "mlisa"}.get(
            self.device_props.type, None
        )
        # asm = launcher.bin.asm.get(asm_type, None)
        asm = launcher.bin.asm.get(asm_type, None)
    # end Modify by CAMBRICON

    CudaKernelParamCache.set(key, params, binary, bin_type, asm, asm_type)
    self.cuda_kernel_saved = True


@gorilla.patch(CachingAutotuner)
def bench(self, launcher, *args, with_profiler=False, **kwargs):
    """Measure the performance of a given launcher"""
    # we don't skip configs with spilled registers when auto-tuning custom
    # (user-written) Triton kernels, as (i) we don't have any knowledge or
    # control over the kernel code; (ii) there is empirical evidence that
    # for some (complicated) custom Triton kernels, a register-spilling
    # config may yield the best latency.
    if (
        not self.custom_kernel
        and launcher.n_spills is not None
        and launcher.n_spills
        > self.inductor_meta.get("spill_threshold", 32 if torch.version.hip else 16)
    ):
        log.debug(
            "Skip config %s because of register spilling: %d",
            launcher.config,
            launcher.n_spills,
        )
        return float("inf")

    device_interface = self.get_device_interface()
    stream = device_interface.get_raw_stream(device_interface.current_device())

    cpu_copies = self.copy_args_to_cpu_if_needed(*args, **kwargs)

    def kernel_call():
        cloned_args, cloned_kwargs = self.maybe_clone_args(cpu_copies, *args, **kwargs)
        # reset to zero before evaluating any config
        self.reset_to_zero_args(*args, **kwargs)
        kernel_name = self.inductor_meta.get("kernel_name", "triton kernel")
        if autograd_profiler._is_profiler_enabled:
            profiler_kwargs = self.get_profiler_kwargs(stream, launcher)
            with torch._C._profiler._RecordFunctionFast(
                kernel_name,
                cloned_args,
                profiler_kwargs,
            ):
                try:
                    launcher(
                        *cloned_args,
                        **cloned_kwargs,
                        stream=stream,
                    )
                except Exception:
                    log.error("Failed during launch %s: ", kernel_name)
                    raise

        else:
            try:
                launcher(
                    *cloned_args,
                    **cloned_kwargs,
                    stream=stream,
                )
            except Exception:
                log.error("Failed during launch %s: ", kernel_name)
                raise
        self.restore_args_from_cpu(cpu_copies)

    # only use profiler when not already in a profiler instance
    if with_profiler and not autograd_profiler._is_profiler_enabled:
        from torch._inductor.utils import do_bench_using_profiling

        return do_bench_using_profiling(kernel_call, warmup=10, rep=40)

    benchmark_kwargs = (
        # Modify by CAMBRICON: warmup=2, rep=4
        # {} if self.device_props.type == "cpu" else {"rep": 40, "is_vetted_benchmarking": True}
        {}
        if self.device_props.type == "cpu"
        else {"warmup": 2, "rep": 4, "is_vetted_benchmarking": True}
        # end Modify by CAMBRICON
    )
    return benchmarker.benchmark(
        fn=kernel_call,
        device=self.device_props.type,
        **benchmark_kwargs,  # type: ignore[arg-type]
    )


@functools.lru_cache(None)
def get_all_factors_and_candidates(n):
    if n is None:
        return []
    if n <= 32:
        heuristics_list = list(range(2, math.ceil(n / 2), 2))
    elif n <= 64:
        heuristics_list = list(range(4, math.ceil(n / 2), 4))
    elif n <= 128:
        heuristics_list = list(range(8, math.ceil(n / 2), 8))
    elif n < 512:
        heuristics_list = list(range(64, math.ceil(n / 2), 64)) + [16, 32]
    elif n < 1024:
        heuristics_list = list(range(128, math.ceil(n / 2), 128)) + [16, 32, 64]
    elif n <= 2048:
        heuristics_list = list(range(256, math.ceil(n / 2), 256)) + [16, 32, 128]
    elif n <= 16384:
        heuristics_list = [32, 256, 512] + list(range(1024, math.ceil(n / 2), 1024))
    elif n < 65536:
        heuristics_list = [64, 128, 1024, 2048, 4096, 8192, 12288, 16384]
    else:
        heuristics_list = [128, 1024, 2048, 4096, 8192, 12288, 16384]
    heuristics_list.append(1)
    if n <= 32768:
        heuristics_list.append(n)
    heuristics_list = sorted(heuristics_list, reverse=True)
    result_list = []
    for candidate in heuristics_list:
        remainder = n % candidate
        pad_num = candidate - remainder
        if remainder and pad_num / n > 0.5 and len(result_list) > 2:
            continue
        else:
            result_list.append(candidate)
    return result_list


@functools.lru_cache(None)
def get_all_factors_and_candidates_for_pointwise_1d_and_2d(n):
    if n is None:
        return []
    elif n <= 32:
        heuristics_list = list(range(2, math.ceil(n / 2) + 1, 2))
    elif n <= 64:
        heuristics_list = list(range(4, math.ceil(n / 2) + 1, 4))
    elif n <= 128:
        heuristics_list = list(range(8, math.ceil(n / 2), 8))
    elif n <= 512:
        heuristics_list = list(range(64, math.ceil(n / 2), 64)) + [16, 32]
    elif n <= 1024:
        heuristics_list = list(range(128, math.ceil(n / 2), 128)) + [16, 32, 64]
    elif n <= 2048:
        heuristics_list = list(range(256, math.ceil(n / 2), 256)) + [16, 32, 64, 128]
    elif n <= 16384:
        heuristics_list = [32, 256, 512] + list(range(1024, math.ceil(n / 2), 1024))
    elif n < 65536:
        heuristics_list = [128, 256, 512] + list(range(1024, 16385, 1024))
    else:
        heuristics_list = [128, 2048, 4096, 8192, 16384, 32768]
    heuristics_list.append(1)
    if n <= 32768:
        heuristics_list.append(n)
    heuristics_list = sorted(heuristics_list, reverse=True)
    result_list = []
    for candidate in heuristics_list:
        remainder = n % candidate
        pad_num = candidate - remainder
        if remainder and pad_num / n > 0.5 and len(result_list) > 2:
            continue
        else:
            result_list.append(candidate)
    return result_list


@functools.lru_cache(None)
def get_all_factors_and_candidates_for_pointwise_3d(n):
    if n is None:
        return []
    if n <= 32:
        heuristics_list = list(range(2, math.ceil(n / 2), 4)) + [1]
    elif n <= 64:
        heuristics_list = list(range(4, math.ceil(n / 2), 8)) + [2]
    elif n <= 128:
        heuristics_list = list(range(8, math.ceil(n / 2), 16)) + [4]
    elif n <= 512:
        heuristics_list = list(range(64, math.ceil(n / 2), 64)) + [8]
    elif n <= 2048:
        heuristics_list = list(range(256, math.ceil(n / 2), 256)) + [16, 32]
    elif n <= 16384:
        heuristics_list = [32] + list(range(1024, math.ceil(n / 2), 2048))
    elif n < 65536:
        heuristics_list = [128, 2048, 8192, 16384]
    else:
        heuristics_list = [128, 2048, 8192, 32768]
    if n <= 32768:
        heuristics_list.append(n)
    heuristics_list = sorted(heuristics_list, reverse=True)
    result_list = []
    for candidate in heuristics_list:
        remainder = n % candidate
        pad_num = candidate - remainder
        if remainder and pad_num / n > 0.5 and len(result_list) > 2:
            continue
        else:
            result_list.append(candidate)
    return result_list


@functools.lru_cache(None)
def get_all_factors_and_candidates_v1(n, include_pow2=True):
    """
    Returns a list of suitable block candidates for n.
    Features:
      - Prioritizes blocks that evenly divide n
      - Supplementally adds sparse power-of-two values (sampled at intervals)
      - Ensures results are representative and non-redundant
    """
    if n is None or n <= 0:
        return []

    # ---- 1. Basic factors (numbers that evenly divide n)
    divisors = sorted({i for i in range(1, n + 1) if n % i == 0})
    divisors = [d for d in divisors if 1 < d <= n // 2]

    # ---- 2. Sparse power-of-two candidates
    pow2 = []
    if include_pow2:
        max_pow = int(math.log2(max(n, 2)))
        # Sample power-of-two values at intervals (e.g., 2, 8, 32, 128, 512, ...)
        pow2 = [2**i for i in range(1, max_pow + 1, 2)]
        pow2 = [p for p in pow2 if p <= n // 2]

    # ---- 3. Combine and remove duplicates
    candidates = sorted(set(divisors + pow2))

    # ---- 4. add 1 and n
    if 1 not in candidates:
        candidates.insert(0, 1)
    if n not in candidates and n <= 32768:
        candidates.append(n)

    candidates = [candidate for candidate in candidates if candidate <= 32768]
    result_list = []
    for candidate in candidates:
        remainder = n % candidate
        pad_num = candidate - remainder
        if remainder and pad_num / n > 0.5 and len(result_list) > 2:
            continue
        else:
            result_list.append(candidate)

    result_list = sorted(result_list, reverse=True)
    return result_list


@functools.lru_cache(None)
def get_all_factors_and_candidates_v2(n):
    """
    Returns a list of suitable block candidates for n.
    Features:
      - Prioritizes blocks that evenly divide n
      - Supplementally adds sparse power-of-two values (sampled at intervals)
      - Ensures results are representative and non-redundant
    """
    if n is None or n <= 0:
        return []

    # ---- 1. Basic factors (numbers that evenly divide n)
    divisors = sorted([d for d in range(1, n + 1) if n % d == 0])

    # ---- 2. all power-of-two candidates
    pow2_list = []
    k = 1
    while (val := 2**k) <= n // 2:
        pow2_list.append(val)
        k += 1

    # ---- 3. Combine and remove duplicates
    candidates = sorted(set(divisors + pow2_list))

    # ---- 4. add 1 and n
    if 1 not in candidates:
        candidates.insert(0, 1)
    if n not in candidates and n <= 32768:
        candidates.append(n)

    candidates = [candidate for candidate in candidates if candidate <= 32768]
    result_list = []
    for candidate in candidates:
        remainder = n % candidate
        pad_num = candidate - remainder
        if remainder and pad_num / n > 0.5 and len(result_list) > 2:
            continue
        else:
            result_list.append(candidate)

    result_list = sorted(result_list, reverse=True)
    return result_list


def all_candidate_blocks(size_hints, use_configs_filter=False, is_pointwise_3d=False):
    max_length = 1048576
    minimal_length = 256
    all_candidates = []
    dim_candidates = {}

    assert len(size_hints) in [1, 2, 3]
    gen_candidates_func = get_all_factors_and_candidates_v2

    prefixs = size_hints.keys()
    for prefix in prefixs:
        dim_candidates[prefix] = gen_candidates_func(size_hints[prefix])

    for blocks in product(*[dim_candidates[prefix] for prefix in prefixs]):
        length = conditional_product(*blocks)
        if length >= max_length:
            continue
        if len(prefixs) > 1 and len(all_candidates) > 1 and length < minimal_length:
            continue
        all_candidates.append(dict(zip(prefixs, blocks)))
    return all_candidates


@gorilla.patch(torch._inductor.runtime.triton_heuristics)
def pointwise(
    size_hints,
    triton_meta,
    tile_hint=None,
    filename=None,
    min_elem_per_thread=0,
    inductor_meta=None,
    return_configs=False,
):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    # Add by CAMBRICON
    from torch_mlu._inductor import config
    from torch_mlu._inductor.runtime.triton_heuristics import (
        all_candidate_blocks,
    )

    # end Add by CAMBRICON

    inductor_meta = {} if inductor_meta is None else inductor_meta
    # Modify by CAMBRICON
    # configs = _handle_combo_kernel_per_subkernel_blocks(
    #     size_hints,
    #     inductor_meta,
    #     triton_meta,
    #     filename=filename,
    #     tile_hint=tile_hint,
    #     min_elem_per_thread=min_elem_per_thread,
    # )
    # if configs is not None:
    #     return cached_autotune(
    #         None,
    #         configs,
    #         triton_meta=triton_meta,
    #         inductor_meta=inductor_meta,
    #         heuristic_type=HeuristicType.POINTWISE,
    #         filename=filename,
    #     )

    # assert not inductor_meta.get("no_x_dim")

    # numel = functools.reduce(operator.mul, size_hints.values())
    # bs = max(256, min(numel // 128, 1024))

    # hinted_configs = autotune_hints_to_configs(
    #     inductor_meta.get("autotune_hints", OrderedSet()),
    #     size_hints,
    #     bs,
    #     triton_meta["device"],
    # )

    # triton_config_with_settings = functools.partial(
    #     triton_config, min_elem_per_thread=min_elem_per_thread
    # )

    # configs = None
    # if len(size_hints) == 1:
    #     if not inductor_meta.get("autotune_pointwise", True) and not (
    #         inductor_meta.get("max_autotune")
    #         or inductor_meta.get("max_autotune_pointwise")
    #     ):
    #         configs = [triton_config_with_settings(size_hints, bs)]
    #     else:
    #         configs = [
    #             triton_config_with_settings(size_hints, bs, num_elements_per_warp=256),
    #             triton_config_with_settings(
    #                 size_hints, bs // 2, num_elements_per_warp=64
    #             ),
    #             *hinted_configs,
    #         ]
    #         # Additional configs appended for ROCm builds
    #         if torch.version.hip:
    #             if inductor_meta.get("max_autotune_pointwise"):
    #                 configs.extend(
    #                     [
    #                         triton_config_with_settings(
    #                             size_hints, TRITON_MAX_BLOCK["X"], waves_per_eu=2
    #                         ),
    #                         triton_config_with_settings(
    #                             size_hints,
    #                             4096,  # wrt: better than the max_block for some kernel
    #                         ),
    #                         triton_config_with_settings(
    #                             size_hints,
    #                             2048,
    #                             num_warps=8,
    #                             num_stages=2,
    #                             waves_per_eu=1,  # 20% improvement
    #                         ),
    #                     ]
    #                 )
    #             if inductor_meta.get("atomic_add_found"):
    #                 configs.extend(
    #                     [
    #                         triton_config_with_settings(
    #                             size_hints,
    #                             64,
    #                             num_warps=1,
    #                             num_stages=1,  # 250% improvement
    #                         )
    #                     ]
    #                 )
    #         if torch.xpu.is_available():
    #             configs.extend(
    #                 [  # intel-xpu-backend-for-triton #5133
    #                     triton_config_with_settings(size_hints, 32),
    #                 ]
    #             )
    # if len(size_hints) == 2:
    #     # Only avoiding tuning on TileHint.SQUARE if not on ROCm builds
    #     # ROCm has observed improvement by diverging here
    #     if (
    #         not inductor_meta.get("autotune_pointwise", True)
    #         or (
    #             torch.version.hip is None
    #             and tile_hint == TileHint.SQUARE
    #             and torch.version.xpu is None
    #         )
    #     ) and not (
    #         inductor_meta.get("max_autotune")
    #         or inductor_meta.get("max_autotune_pointwise")
    #     ):
    #         configs = [triton_config_with_settings(size_hints, 32, 32)]
    #     else:
    #         configs = [
    #             triton_config_with_settings(size_hints, 32, 32),
    #             triton_config_with_settings(size_hints, 64, 64),  # ~8% better for fp16
    #             triton_config_with_settings(size_hints, 256, 16),
    #             triton_config_with_settings(size_hints, 16, 256),
    #             triton_config_with_settings(size_hints, bs, 1),
    #             triton_config_with_settings(size_hints, 1, bs),
    #             *hinted_configs,
    #         ]
    #         # Additional configs appended for ROCm builds
    #         if torch.version.hip:
    #             configs.extend(
    #                 [
    #                     triton_config_with_settings(
    #                         size_hints, 64, 32
    #                     ),  # better for some kernels
    #                     triton_config_with_settings(
    #                         size_hints, 128, 16
    #                     ),  # +10% for some kernels
    #                     triton_config_with_settings(
    #                         size_hints, 128, 32
    #                     ),  # additional 10% more
    #                     triton_config_with_settings(
    #                         size_hints, 32, 512
    #                     ),  # +30% for some kernels
    #                 ]
    #             )
    #         if torch.xpu.is_available():
    #             configs.extend(
    #                 [
    #                     # intel-xpu-backend-for-triton #5198
    #                     triton_config_with_settings(size_hints, 32, 32, num_warps=8),
    #                     # intel-xpu-backend-for-triton #5199
    #                     triton_config_with_settings(size_hints, 4, 256),
    #                 ]
    #             )
    # if len(size_hints) == 3:
    #     if not (
    #         inductor_meta.get("max_autotune_pointwise") or torch.xpu.is_available()
    #     ):
    #         configs = [triton_config_with_settings(size_hints, 16, 16, 16)]
    #     else:
    #         configs = [
    #             triton_config_with_settings(size_hints, 16, 16, 16),
    #             triton_config_with_settings(size_hints, 64, 8, 8),
    #             triton_config_with_settings(size_hints, 8, 64, 8),
    #             triton_config_with_settings(size_hints, 8, 8, 64),
    #             triton_config_with_settings(size_hints, bs, 1, 1),
    #             triton_config_with_settings(size_hints, 1, bs, 1),
    #             triton_config_with_settings(size_hints, 1, 1, bs),
    #             *hinted_configs,
    #         ]
    assert not inductor_meta.get("no_x_dim")

    num_warps = 1
    configs = []
    use_configs_filter = config.filter_configs and len(size_hints) != 3
    candidate_blocks = all_candidate_blocks(
        size_hints, use_configs_filter, len(size_hints) == 3
    )

    for blocks in candidate_blocks:
        cfg = _get_config(blocks)
        configs.append(Config(cfg, num_warps=num_warps, num_stages=3))
    # end Modify by CAMBRICON

    if not configs:
        raise NotImplementedError(f"size_hints: {size_hints}")

    configs = _maybe_filter_configs_for_tma_restrictions(inductor_meta, configs)
    if return_configs:
        return configs

    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.POINTWISE,
        filename=filename,
    )


@gorilla.patch(torch._inductor.runtime.triton_heuristics)
def _reduction_configs(
    *,
    size_hints: dict[str, int],
    inductor_meta: dict[str, Any],
    triton_meta: dict[str, Any],
    num_dynamic=0,
) -> list[Config]:
    # Modify by Cambricon
    # reduction_hint = inductor_meta.get("reduction_hint")

    # # Convert reductions to 1D, to simplify heuristics.
    # rnumel = get_total_reduction_numel(size_hints)

    # # Is max autotune enabled
    # max_autotune_enabled = inductor_meta.get("max_autotune") or inductor_meta.get(
    #     "max_autotune_pointwise"
    # )

    # register_intensive = False
    # MAX_R0_BLOCK = 2048
    # loads_and_red = inductor_meta.get("num_load", 0) + inductor_meta.get(
    #     "num_reduction", 0
    # )
    # if size_hints["x"] >= 1024 and loads_and_red >= 10:
    #     # A heuristics to reduce R0_BLOCK if a kernel potentially need many registers.
    #     # Consider load and reduction since load need move data into registers and
    #     # reduction needs an accumulator.
    #     #
    #     # The magic numbers are a bit arbitrary.
    #     #
    #     # We cannot rely on dynamically scaling down R0_BLOCK later, since sometimes
    #     # triton makes it to use less registers with worse perf. Check:
    #     # https://github.com/pytorch/pytorch/issues/126463
    #     #
    #     # The heuristic is a very simple one since registers can be reused. But
    #     # hopefully it can be a good enough indicator.
    #     MAX_R0_BLOCK = 1024
    #     register_intensive = True

    # if triton_meta.get("native_matmul"):
    #     if len(size_hints) == 3:
    #         return [
    #             make_matmul_triton_config(sizes, num_warps, num_stages)
    #             for sizes, num_warps, num_stages in triton_native_mm_configs
    #         ]
    #     elif len(size_hints) == 4:
    #         return [
    #             make_matmul_triton_config(sizes, num_warps, num_stages)
    #             for sizes, num_warps, num_stages in triton_native_bmm_configs
    #         ]
    #     else:
    #         raise NotImplementedError("native matmul only supports mm/bmm pattern")

    # def make_config(
    #     x,
    #     r,
    #     num_warps=None,
    #     num_stages=1,
    #     register_intensive=False,
    #     dynamic_scale_rblock=True,
    #     waves_per_eu=None,
    # ):
    #     # For 3D case with tiling scores, create an adapted version
    #     if "y" in size_hints:
    #         assert "tiling_scores" in inductor_meta
    #         return adapt_config_for_tiling(
    #             size_hints,
    #             inductor_meta["tiling_scores"],
    #             x,
    #             r,
    #             num_warps=num_warps,
    #             num_stages=num_stages,
    #             register_intensive=register_intensive,
    #             waves_per_eu=waves_per_eu,
    #         )
    #     else:
    #         # For other cases, use the original function
    #         return triton_config_reduction(
    #             size_hints,
    #             x,
    #             r,
    #             num_warps=num_warps,
    #             num_stages=num_stages,
    #             register_intensive=register_intensive,
    #             waves_per_eu=waves_per_eu,
    #             dynamic_scale_rblock=dynamic_scale_rblock,
    #             reduction_hint=reduction_hint,
    #         )

    # def outer_config_opt():
    #     # Default to 64 for vectorized loads
    #     max_x_block, x_block = 256, 64
    #     load_factor = inductor_meta.get("num_load", 0)
    #     x = size_hints["x"]
    #     num_warps = None

    #     # Try to use all SMs with small x
    #     if x <= 1024:
    #         x_block = max(min(x // 128, 8), 2)
    #         outer_r_block = min(rnumel, 64)
    #     # Lower bound x = 1024, 1024 // 16 = 128 around # of SMs
    #     elif x // 4096 <= 8:
    #         x_block = 16
    #         outer_r_block = 512 // x_block
    #     elif num_dynamic > 1:
    #         # Lots of compute with multiple dynamic shape per loop iteration
    #         # Larger RBLOCK minimizes loop iteration
    #         outer_r_block = max(min((rnumel // 64), 64), 8)
    #     elif num_dynamic == 1:
    #         # Dynamic shapes introduce a lot register pressure for indexing
    #         outer_r_block = (
    #             1
    #             if load_factor >= 3
    #             else min(next_power_of_2(max(rnumel, 128) // 128), 8)
    #         )
    #     else:
    #         x_block = max(min(max_x_block, next_power_of_2(x // 4096)), x_block)
    #         if load_factor < 4 or rnumel <= 128:
    #             outer_r_block = 512 // x_block
    #         else:
    #             # Heavier reductions contain a lot more overhead per loop iteration
    #             # We minimize the overhead by enlarging r block
    #             if rnumel >= 2048:
    #                 outer_r_block = 64
    #             else:
    #                 outer_r_block = 32
    #             x_block = min(x_block, 32)
    #             num_warps = 4

    #     # Set register intensive to true by default as we try to maximize tiles with heuristic
    #     return make_config(
    #         x_block,
    #         outer_r_block,
    #         num_warps=num_warps,
    #         register_intensive=register_intensive,
    #     )

    # contiguous_config = make_config(
    #     2 if rnumel <= 2048 else 1,  # 1024 or less is persistent
    #     min(rnumel, MAX_R0_BLOCK),
    #     register_intensive=register_intensive,
    # )
    # tiny_config = make_config(
    #     2 * (256 // rnumel) if rnumel <= 256 else 1,
    #     min(rnumel, MAX_R0_BLOCK),
    #     register_intensive=register_intensive,
    # )

    # outer_config = make_config(64, 8, register_intensive=register_intensive)
    # # TODO (paulzhan): Test heuristic on AMD and internal testing
    # # for correctness
    # if not torch.version.hip:
    #     outer_config = outer_config_opt()

    # configs = []

    # if inductor_meta.get("add_persistent_rblock") and loads_and_red <= 8:
    #     xnumel = max(4096 // rnumel, 1)
    #     c = make_config(
    #         xnumel,
    #         min(rnumel, 32768),
    #         register_intensive=register_intensive,
    #         dynamic_scale_rblock=False,
    #     )
    #     configs.append(c)

    # result_configs = []

    # # For 3d tiling, default to more autotuning initially
    # if "y" in size_hints:
    #     pass
    # elif max_autotune_enabled:
    #     pass  # skip all these cases
    # elif reduction_hint == ReductionHint.INNER:
    #     return configs + [contiguous_config]
    # elif reduction_hint == ReductionHint.OUTER:
    #     return configs + [outer_config]
    # elif reduction_hint == ReductionHint.OUTER_TINY:
    #     return configs + [tiny_config]

    # # We continue here under the following conditions:
    # # - max_autotune_enabled is True
    # # - max_autotune_enabled is False and reduction_hint is NOT one of the above cases
    # result_configs = configs + [
    #     contiguous_config,
    #     outer_config,
    #     tiny_config,
    #     make_config(64, 64),
    #     make_config(8, 512),
    #     # halve the XBLOCK/Rn_BLOCK compared to outer_config
    #     # TODO: this may only be beneficial when each iteration of the reduction
    #     # is quite heavy. E.g. https://gist.github.com/shunting314/189a8ef69f90db9d614a823385147a72
    #     make_config(64, 4, num_warps=8),
    # ]

    # if torch.version.hip:
    #     result_configs.extend(
    #         [
    #             make_config(1024, 8, num_warps=4, num_stages=1, waves_per_eu=2),
    #             make_config(512, 8, num_warps=4, num_stages=1, waves_per_eu=1),
    #         ]
    #     )

    # return result_configs

    from torch_mlu._inductor.runtime.triton_heuristics import (
        all_candidate_blocks,
    )

    assert len(size_hints) == 2
    triton_configs = []
    for blocks in all_candidate_blocks(size_hints, False):
        cfg = _get_config(blocks)
        triton_configs.append(Config(cfg, num_warps=1, num_stages=3))
    return triton_configs
    # end Modify by Cambricon


@gorilla.patch(torch._inductor.runtime.triton_heuristics)
def split_scan(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    """Heuristic for TritonSplitScanKernel"""
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    assert triton_meta is not None
    if len(size_hints) != 2:
        raise NotImplementedError(f"size_hints: {size_hints}")

    # Modify by CAMBRICON: Add a parameter 'triton_meta' to function _reduction_configs.
    # configs = _reduction_configs(
    #    size_hints=size_hints, inductor_meta=inductor_meta, triton_meta=triton_meta
    # )
    from torch_mlu._inductor.runtime.triton_heuristics import _reduction_configs

    configs = _reduction_configs(
        size_hints=size_hints, triton_meta=triton_meta, inductor_meta=inductor_meta
    )
    # end Modify by CAMBRICON

    # Fixup configs to enforce the minimum Rn_BLOCK size
    min_rblock = inductor_meta.get("min_split_scan_rblock", 256)
    for cfg in configs:
        for var in list(cfg.kwargs.keys()):
            if var.startswith("R") and cfg.kwargs[var] < min_rblock:
                cfg.kwargs[var] = min_rblock

    configs = _maybe_filter_configs_for_tma_restrictions(inductor_meta, configs)
    configs = filter_reduction_configs_for_determinism(inductor_meta, configs)
    return cached_autotune(
        size_hints,
        configs=configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.SPLIT_SCAN,
        filename=filename,
    )


@functools.lru_cache(1)
def get_core_num():
    core_num = torch.mlu.get_device_properties(
        torch.mlu.current_device()
    ).multi_processor_count
    try:
        core_num = (
            int(os.environ["TORCHINDUCTOR_SET_CORENUM"])
            if "TORCHINDUCTOR_SET_CORENUM" in os.environ
            else core_num
        )
    except ValueError:
        print("Warning: TORCHINDUCTOR_SET_CORENUM is not a valid number.")
    return core_num


def Grid1D_generate(self, meta: dict[str, int]) -> None:
    # Modify by Cambricon
    from torch_mlu._inductor.runtime.triton_heuristics import get_core_num

    # self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))

    x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
    if isinstance(x_grid, int):
        x_grid = min(x_grid, get_core_num())
    elif self.mode == "python":
        x_grid = f"min({x_grid}, {get_core_num()})"
    else:
        x_grid = f"std::min({x_grid}, {get_core_num()}L)"
    self.x_grid = x_grid
    # end Modify by Cambricon


patch = gorilla.Patch(
    torch._inductor.runtime.triton_heuristics.Grid1D, "generate", Grid1D_generate
)
gorilla.apply(patch)


def Grid2D_generate(self, meta: dict[str, int]) -> None:
    # Modify by Cambricon
    from torch_mlu._inductor.runtime.triton_heuristics import get_core_num

    # self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
    # self.y_grid = self.ceildiv("ynumel", meta.get("YBLOCK"))

    x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
    y_grid = self.ceildiv("ynumel", meta.get("YBLOCK"))
    if isinstance(x_grid, int) and isinstance(y_grid, int):
        total_grid = min(x_grid * y_grid, get_core_num())
    elif self.mode == "python":
        total_grid = f"min({x_grid} * {y_grid}, {get_core_num()})"
    else:
        total_grid = f"std::min({x_grid} * {y_grid}, {get_core_num()}L)"
    self.x_grid = total_grid
    # end Modify by Cambricon


patch = gorilla.Patch(
    torch._inductor.runtime.triton_heuristics.Grid2D, "generate", Grid2D_generate
)
gorilla.apply(patch)


def Grid3D_generate(self, meta: dict[str, int]) -> None:
    # Modify by Cambricon
    from torch_mlu._inductor.runtime.triton_heuristics import get_core_num

    # self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
    # self.y_grid = self.ceildiv("ynumel", meta.get("YBLOCK"))
    # self.z_grid = self.ceildiv("znumel", meta.get("ZBLOCK"))

    x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
    y_grid = self.ceildiv("ynumel", meta.get("YBLOCK"))
    z_grid = self.ceildiv("znumel", meta.get("ZBLOCK"))
    if isinstance(x_grid, int) and isinstance(y_grid, int) and isinstance(z_grid, int):
        total_grid = x_grid * y_grid * z_grid
    elif self.mode == "python":
        total_grid = f"min({x_grid} * {y_grid} * {z_grid}, {get_core_num()})"
    else:
        total_grid = f"std::min({x_grid} * {y_grid} * {z_grid}, {get_core_num()}L)"
    self.x_grid = total_grid
    # end Modify by Cambricon


patch = gorilla.Patch(
    torch._inductor.runtime.triton_heuristics.Grid3D, "generate", Grid3D_generate
)
gorilla.apply(patch)


def Grid2DWithYZOverflow_generate(self, meta: dict[str, int]) -> None:
    from torch_mlu._inductor.runtime.triton_heuristics import get_core_num

    x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
    self.prefix = [
        self.assign_tmp("y_grid_raw_", self.ceildiv("ynumel", meta.get("YBLOCK"))),
        self.assign_tmp("y_grid_div_", self.ceildiv("y_grid_raw_", get_max_y_grid())),
    ]
    y_grid = self.ceildiv("y_grid_raw_", "y_grid_div_")
    z_grid = "y_grid_div_"
    if self.mode == "python":
        total_grid = f"min({x_grid} * {y_grid} * {z_grid}, {get_core_num()})"
    else:
        total_grid = f"std::min({x_grid} * {y_grid} * {z_grid}, {get_core_num()}L)"
    self.x_grid = total_grid


def _persistent_reduction_configs(
    size_hints,
    reduction_hint=False,
    inductor_meta=None,
    triton_meta=None,
):
    from torch_mlu._inductor.runtime.triton_heuristics import (
        get_all_factors_and_candidates,
        triton_config_mlu_reduction,
    )

    xnumel = size_hints["x"]
    rnumel = get_total_reduction_numel(size_hints)

    # Modify by CAMBRICON
    # MAX_PERSISTENT_BLOCK_NUMEL = 4096

    # if triton_meta.get("native_matmul"):
    #     if len(size_hints) == 3:
    #         return [
    #             make_matmul_triton_config(sizes, num_warps, num_stages)
    #             for sizes, num_warps, num_stages in triton_native_persistent_mm_configs
    #         ]
    #     elif len(size_hints) == 4:
    #         return [
    #             make_matmul_triton_config(sizes, num_warps, num_stages)
    #             for sizes, num_warps, num_stages in triton_native_persistent_bmm_configs
    #         ]
    #     else:
    #         raise NotImplementedError("native matmul only supports mm/bmm pattern")

    # max_autotune_enabled = inductor_meta.get("max_autotune") or inductor_meta.get(
    #     "max_autotune_pointwise"
    # )

    # if torch.version.hip:
    #     xblock_vals = [1, 4, 8, 16, 32, 64, 128, 256]
    # else:
    #     xblock_vals = [1, 8, 32, 128]

    # if "y" not in size_hints:
    #     configs = [
    #         triton_config_reduction(
    #             size_hints,
    #             xblock,
    #             rnumel,
    #             register_intensive=True,
    #             reduction_hint=reduction_hint,
    #         )
    #         for xblock in xblock_vals
    #         if xblock == 1
    #         or (rnumel * xblock <= MAX_PERSISTENT_BLOCK_NUMEL and xblock <= xnumel)
    #     ]
    # else:
    #     configs = []
    #     assert "tiling_scores" in inductor_meta
    #     x_y_scores = {dim: inductor_meta["tiling_scores"][dim] for dim in ("x", "y")}
    #     for target_block_size in xblock_vals:
    #         if target_block_size * rnumel > MAX_PERSISTENT_BLOCK_NUMEL:
    #             continue

    #         block_sizes = match_target_block_product(
    #             size_hints, x_y_scores, target_block_size
    #         )
    #         configs.append(
    #             triton_config_tiled_reduction(
    #                 size_hints, block_sizes["x"], block_sizes["y"], rnumel
    #             )
    #         )

    # tiny_configs = [
    #     triton_config_reduction(
    #         size_hints,
    #         2 * (256 // rnumel) if rnumel <= 256 else 1,
    #         rnumel,
    #     )
    # ]
    if len(size_hints) == 2:
        # xr
        max_xblock_val = min(xnumel, 65536 // rnumel)
        exponent = math.floor(math.log2(max_xblock_val))
        max_xblock = int(2**exponent)

        candidates = []
        candidates.append(max_xblock)

        if max_xblock > 2:
            candidates2 = max_xblock // 2
            candidates.append(candidates2)

        if max_xblock > 4:
            candidates4 = max_xblock // 4
            candidates.append(candidates4)

        if max_xblock > 8:
            candidates8 = max_xblock // 8
            candidates.append(candidates8)

        configs = [
            triton_config_mlu_reduction(
                size_hints,
                xblock,
                rnumel,
                num_stages=num_stages,
                register_intensive=True,
            )
            for xblock in candidates
            for num_stages in [1, 3]
        ]
    else:
        # yxr or zyxr
        from itertools import product

        dim_candidates = {}
        candidates = []
        prefixs = list(size_hints.keys())[:-1]
        for prefix in prefixs:
            dim_candidates[prefix] = get_all_factors_and_candidates(size_hints[prefix])
            dim_candidates[prefix] = [
                candidate
                for candidate in dim_candidates[prefix]
                if 65536 % candidate == 0
            ]

        if len(size_hints) == 3:
            for blocks in product(*[dim_candidates[prefix] for prefix in prefixs]):
                if blocks[0] * blocks[1] * rnumel > 65536:
                    continue
                candidates.append(dict(zip(prefixs, blocks)))

            configs = [
                triton_config_mlu_reduction(
                    size_hints,
                    blocks["x"],
                    rnumel,
                    blocks["y"],
                    num_stages=num_stages,
                    register_intensive=True,
                )
                for blocks in candidates
                for num_stages in [1, 3]
            ]
        elif len(size_hints) == 4:
            for blocks in product(*[dim_candidates[prefix] for prefix in prefixs]):
                if blocks[0] * blocks[1] * blocks[2] * rnumel > 65536:
                    continue
                # TODO(miaochen): GENESIS-4205
                if blocks[0] == 1:
                    continue
                candidates.append(dict(zip(prefixs, blocks)))

            configs = [
                triton_config_mlu_reduction(
                    size_hints,
                    blocks["x"],
                    rnumel,
                    blocks["y"],
                    blocks["z"],
                    num_stages=num_stages,
                    register_intensive=True,
                )
                for blocks in candidates
                for num_stages in [1, 3]
            ]
    # end Modify by CAMBRICON

    # Modify by CAMBRICON
    # # defer to more autotuning, initially
    # if "y" in size_hints:
    #     pass
    # # TODO(jansel): we should be able to improve these heuristics
    # elif not max_autotune_enabled:  # Do not filter configs when tuning
    #     if reduction_hint == ReductionHint.INNER and rnumel >= 256:
    #         if rnumel > 1024 or xnumel // 8 < 128 or inductor_meta.get("RSPLIT_SIZE"):
    #             configs = configs[:1]
    #         else:
    #             if not torch.cuda.is_available():
    #                 # TODO(Intel): CUDA uses num_warps = 1 to disable shared memory.
    #                 # We apply different configurations from #168335.
    #                 # We currently let cost model in Triton to decide whether to use shared memory.
    #                 loads_and_stores = inductor_meta.get(
    #                     "num_load", 0
    #                 ) + inductor_meta.get("num_store", 0)
    #                 x_block = 8
    #                 if xnumel // x_block < 128 or loads_and_stores >= 5:
    #                     x_block = 1
    #                 num_warps, min_num_warps, reduction_hint = None, None, None
    #             else:
    #                 x_block = min(1024 // rnumel, 8)
    #                 num_warps, min_num_warps = 1, 1
    #             configs = [
    #                 triton_config_reduction(
    #                     size_hints,
    #                     x_block,
    #                     rnumel,
    #                     register_intensive=True,
    #                     num_warps=num_warps,
    #                     min_num_warps=min_num_warps,
    #                     reduction_hint=reduction_hint,
    #                 )
    #             ]

    #     elif reduction_hint == ReductionHint.OUTER:
    #         configs = configs[-1:]
    #     elif reduction_hint == ReductionHint.OUTER_TINY:
    #         configs = tiny_configs
    # else:
    #     if torch.version.hip:
    #         # If autotune is enabled append tiny configs
    #         for conf in tiny_configs:
    #             if conf not in configs:
    #                 configs.append(conf)
    if reduction_hint == ReductionHint.OUTER_TINY:
        size_hints_temp = copy.copy(size_hints)
        size_hints_temp["x"] = next_power_of_2(size_hints_temp["x"])
        configs = [
            triton_config_reduction(
                size_hints_temp,
                2 * (256 // next_power_of_2(rnumel)) if rnumel <= 256 else 1,
                rnumel,
            )
        ]
    # end Modify by CAMBRICON
    for c in configs:
        # we don't need Rn_BLOCK for persistent reduction
        for prefix in size_hints:
            if prefix_is_reduction(prefix):
                c.kwargs.pop(f"{prefix.upper()}BLOCK")

    return configs


patch = gorilla.Patch(
    torch._inductor.runtime.triton_heuristics,
    "_persistent_reduction_configs",
    _persistent_reduction_configs,
)
gorilla.apply(patch)


MAX_GRID_NUM = 65535


@staticmethod
def from_meta(
    inductor_meta: dict[str, Any],
    cfg: Config | dict[str, int],
    mode: Literal["python", "cpp"] = "python",
) -> GridExpr:
    grid_cls = globals()[inductor_meta["grid_type"]]
    assert issubclass(grid_cls, GridExpr)
    grid = grid_cls(inductor_meta=inductor_meta, mode=mode)
    if isinstance(cfg, Config):
        cfg = config_to_dict(cfg)
    grid.generate(cfg)

    # Modify by Cambricon: x_grid validate
    from torch_mlu._inductor.runtime.triton_heuristics import MAX_GRID_NUM

    # x_grid maybe a expression
    if isinstance(grid.x_grid, int) and cfg["num_warps"] * grid.x_grid > MAX_GRID_NUM:
        raise OutOfResources(
            "x_grid",
            str(MAX_GRID_NUM),
            f"grid_x * num_warps: {grid.x_grid} * {cfg['num_warps']}",
        )
    # end Modify by CAMBRICON
    return grid


patch = gorilla.Patch(
    torch._inductor.runtime.triton_heuristics.GridExpr,
    "from_meta",
    from_meta,
)
gorilla.apply(patch)


def ComboKernelGrid_generate(self, meta: dict[str, int]):
    combo_meta = self.inductor_meta["combo_grid_meta"]
    if combo_meta["default_config"]:
        meta = {**combo_meta["default_config"], **meta}
    no_x_dims = []
    xnumels = []
    ynumels = []

    for num in range(combo_meta["num_kernels"]):
        assert combo_meta[f"xnumel_{num}"] is None or combo_meta[f"xnumel_{num}"] > 0
        no_x_dims.append(combo_meta[f"no_x_dim_{num}"])
        xnumels.append(combo_meta[f"xnumel_{num}"] or f"xnumel_{num}")
        if f"ynumel_{num}" in combo_meta:
            ynumels.append(combo_meta[f"ynumel_{num}"] or f"ynumel_{num}")

    self.x_grid = self.combo_x_grid(xnumels, no_x_dims, meta)
    if combo_meta["min_blocks"]:
        self.x_grid = self.maximum([self.x_grid, combo_meta["min_blocks"]])
    # Modify by CAMBRICON
    if ynumels:
        self.y_grid = 1
        self.z_grid = 1
        # self.prefix.extend(
        #    [
        #        self.assign_tmp(
        #            "y_grid_raw_",
        #            self.ceildiv(self.maximum(ynumels), meta.get("YBLOCK")),
        #        ),
        #        self.assign_tmp(
        #            "y_grid_div_", self.ceildiv("y_grid_raw_", get_max_y_grid())
        #        ),
        #    ]
        # )
        # self.y_grid = self.ceildiv("y_grid_raw_", "y_grid_div_")
        # self.z_grid = "y_grid_div_"
    # end Modify by CAMBRICON


patch = gorilla.Patch(
    torch._inductor.runtime.triton_heuristics.ComboKernelGrid,
    "generate",
    ComboKernelGrid_generate,
)
gorilla.apply(patch)


def SequentialComboKernelGrid_combo_x_grid(
    self,
    xnumels: list[int | str],
    no_x_dims: list[bool],
    meta: dict[str, int],
) -> str | int:
    assert len(xnumels) == len(no_x_dims)
    # Modify by Cambricon
    # return self.summation(
    x_grid = self.summation(
        [
            self.ceildiv(x, 1 if no_x_dim else meta.get("XBLOCK"))
            for x, no_x_dim in zip(xnumels, no_x_dims)
        ]
    )
    if isinstance(x_grid, str):
        x_grid_pad = f"(({x_grid}) + 3) & ~3"
    else:
        x_grid_pad = (x_grid + 3) & ~3
    return x_grid_pad
    # end Modify by Cambricon


patch = gorilla.Patch(
    torch._inductor.runtime.triton_heuristics.SequentialComboKernelGrid,
    "combo_x_grid",
    SequentialComboKernelGrid_combo_x_grid,
)
gorilla.apply(patch)


def triton_config_mlu_reduction(
    size_hints,
    x: int,
    r: int,
    y=None,
    z=None,
    num_stages=1,
    num_warps=None,
    register_intensive=False,
) -> Config:
    """
    Construct a reduction triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """
    from torch._inductor.runtime.triton_heuristics import (
        check_config,
        check_max_block,
        _check_max_grid_x,
        _get_nd_reduction_numels,
        _num_warps,
    )

    # Convert the linear reduction numel into a multi-dimensional block.
    rnumels = _get_nd_reduction_numels(r, size_hints)

    # shrink sizes to size hints
    x = min(x, size_hints["x"])
    if len(size_hints) == 3:
        y = min(y, size_hints["y"])
    elif len(size_hints) == 4:
        y = min(y, size_hints["y"])
        z = min(z, size_hints["z"])

    def total_numel() -> int:
        return conditional_product(x, *rnumels.values())

    target = total_numel()
    if conditional_product(*size_hints.values()) < target:
        target //= 8

    # if we are below original block size, scale up where we can
    while x < size_hints["x"] and total_numel() < target:
        x *= 2
    for prefix in sorted(rnumels):
        while rnumels[prefix] < size_hints[prefix] and total_numel() < target:
            rnumels[prefix] *= 2

    if num_warps is None:
        num_warps = total_numel() // 128
    num_warps = _num_warps(
        num_warps, max_num_warps=16, register_intensive=register_intensive
    )

    x, _num_blocks = _check_max_grid_x(size_hints, x, num_warps)

    for prefix in sorted(rnumels):
        while total_numel() > target:
            if rnumels[prefix] == 1:
                break
            rnumels[prefix] //= 2

    if len(size_hints) == 2:
        cfg = _get_config({"x": x, **rnumels})
        check_max_block(cfg)
        check_config(cfg, xnumel=size_hints["x"])
    elif len(size_hints) == 3:
        cfg = _get_config({"y": y, "x": x, **rnumels})
        check_max_block(cfg)
        check_config(cfg, xnumel=size_hints["x"], ynumel=size_hints["y"])
    elif len(size_hints) == 4:
        cfg = _get_config({"z": z, "y": y, "x": x, **rnumels})
        check_max_block(cfg)
        check_config(
            cfg, xnumel=size_hints["x"], ynumel=size_hints["y"], znumel=size_hints["z"]
        )

    # Modify by CAMBRICON
    # return Config(cfg, num_warps=num_warps, num_stages=num_stages)
    return Config(cfg, num_warps=1, num_stages=num_stages)
