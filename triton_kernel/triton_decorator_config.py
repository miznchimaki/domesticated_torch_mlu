import os
from typing import Optional, Dict, Any, List, Callable
from triton.runtime import fast_libentry, libentry
import triton


class triton_kernel_decorator:
    """
    Triton kernel的装饰器管理器，支持通过环境变量控制是否启用autotune。

    该装饰器封装了triton.jit、triton.heuristics、triton.autotune和fast_libentry，
    允许在开发环境使用autotune进行调优，在生产环境使用预定义的heuristics快速执行。

    环境变量控制:
        TRITON_KERNEL_FORCE_AUTOTUNE: 设置为"1"/"true"/"yes"时强制启用autotune模式
    Examples: fbgemm_kernesl
    """

    FORCE_AUTOTUNE = os.getenv("TRITON_KERNEL_FORCE_AUTOTUNE", "").lower() in (
        "1",
        "true",
        "yes",
    )

    def __init__(
        self,
        autotune_configs=None,
        autotune_key=None,
        prune_configs=None,
        heuristics_autotune=None,  # autotune模式专用的heuristics
        heuristics_direct=None,  # direct模式专用的heuristics
        fast_libentry=True,
    ):
        self.use_autotune = self.FORCE_AUTOTUNE and autotune_configs is not None
        self.autotune_configs = autotune_configs
        self.autotune_key = autotune_key or []
        self.prune_configs = prune_configs
        self.heuristics_autotune = heuristics_autotune or {}
        self.heuristics_direct = heuristics_direct or {}
        self.fast_libentry = fast_libentry

    def __call__(self, func):
        # 1. 应用jit
        decorated = triton.jit(func)

        if self.use_autotune:
            # 模式A: 开启autotune
            # 先应用autotune模式的heuristics
            if self.heuristics_autotune:
                decorated = triton.heuristics(self.heuristics_autotune)(decorated)

            # 再应用autotune
            autotune_kwargs = {
                "configs": self.autotune_configs,
                "key": self.autotune_key,
            }
            if self.prune_configs:
                autotune_kwargs["prune_configs_by"] = self.prune_configs
            decorated = triton.autotune(**autotune_kwargs)(decorated)
        else:
            # 模式B: 不使用autotune
            # 应用direct模式的heuristics
            if self.heuristics_direct:
                decorated = triton.heuristics(self.heuristics_direct)(decorated)
            elif self.heuristics_autotune:
                # fallback到autotune模式的heuristics
                decorated = triton.heuristics(self.heuristics_autotune)(decorated)

        # 2. 应用fast_libentry
        if self.fast_libentry:
            decorated = fast_libentry()(decorated)
        else:
            decorated = libentry()(decorated)

        return decorated
