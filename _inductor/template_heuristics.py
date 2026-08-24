from __future__ import annotations

import torch
from torch._inductor.template_heuristics.registry import register_template_heuristic
from torch._inductor.template_heuristics.triton import (
    BaseConfigHeuristic,
    FlexConfig,
    MMTemplateConfigMixin,
    MMPlusMMTemplateConfigMixin,
    CUDAConfigHeuristic,
)
from torch._inductor.template_heuristics.triton_addmm import AddMMConfigMixin

from torch._inductor import config
from torch._inductor.kernel.bmm import bmm_template
from torch._inductor.kernel.mm import mm_template
from torch._inductor.kernel.mm_plus_mm import mm_plus_mm_template


class MLUConfigHeuristic(BaseConfigHeuristic):
    """
    Child class for MLU device specific gemm/flex attention/conv/ configs.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_flex_attn_fwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]:
        flex_attn_fwd_configs: list[FlexConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_fwd_configs
            flex_attn_fwd_configs += self.flex_attn_fwd_autotune_configs

        if head_dim <= 256:
            if dtype == torch.float32:
                default_config = FlexConfig(64, 64, 3, 4)
            else:
                # Modify by CAMBRICON
                # default_config = FlexConfig(128, 64, 3, 4)
                default_config = FlexConfig(128, 128, 0, 1)
                # end Modify by CAMBRICON
        else:
            if dtype == torch.float32:
                default_config = FlexConfig(32, 16, 3, 4)
            else:
                default_config = FlexConfig(64, 32, 3, 4)

        if default_config not in flex_attn_fwd_configs:
            flex_attn_fwd_configs.append(default_config)

        return flex_attn_fwd_configs


@register_template_heuristic(mm_template.uid, "mlu")
@register_template_heuristic(bmm_template.uid, "mlu")
class MLUMMTemplateConfigHeuristic(MMTemplateConfigMixin, CUDAConfigHeuristic):
    """Standard MM template heuristic for MLU"""

    pass


@register_template_heuristic(mm_template.uid, "mlu", op_name="addmm")
@register_template_heuristic(bmm_template.uid, "mlu", op_name="baddbmm")
class MLUAddMMTemplateConfigHeuristic(AddMMConfigMixin, MLUMMTemplateConfigHeuristic):
    """Standard MM template heuristic for MLU"""

    pass


@register_template_heuristic(
    mm_plus_mm_template.uid,
    "mlu",
)
class MLUMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, CUDAConfigHeuristic
):
    """MM Plus MM template heuristic for MLU"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use mm_plus_mm_configs
        self.mm_configs = self.mm_plus_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.mm_plus_mm_configs
