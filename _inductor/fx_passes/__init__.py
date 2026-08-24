from torch_mlu._inductor import config

# List of valid pass names that can be skipped via TORCHINDUCTOR_MLU_SKIPPED_FX_PASSES
SKIPPABLE_FX_PASSES = [
    "repeat2expand",
    "normalization",
    "fold_binaryop",
    "fold_cat",
    "fold_expand",
    "fold_reduce",
    "fold_nest_view",
    "fold_matmul_like_view_pointwise_view",
    "fold_where",
    "fold_stack",
    "fold_clone",
    "fold_abs",
    "fold_maximini",
    "fold_neg",
    "fold_logical_not",
    "fold_log",
    "combine_pointwise_src",
    "cat_reshape",
    "fuse_tmo_bmm",
    "make_contiguous_clone",
    "fused_mm",
    "use_tmo_fa",
    "fused_bmm",
    "div_sqrt_replace",
    "aten_div_sqrt_replace",
    "div_exp_replace",
    "combo_matmul_infer",
    "fuse_layernorm_infer",
]

# List of valid pass names that can be enabled via TORCHINDUCTOR_MLU_ENABLED_FX_PASSES.
ENABLEABLE_FX_PASSES = [
    "combo_matmul",   # Deprecated, use combo_matmul_training or combo_matmul_infer instead.
    "fuse_layernorm",   # Deprecated, use fuse_layernorm_training or fuse_layernorm_infer instead.
    "combo_matmul_training",
    "fuse_layernorm_training",
    "fuse_tmo_layernorm",
    "conv_relu_fusion",
    "conv_leaky_relu_fusion",
]


def get_skippable_fx_passes():
    """
    Get the list of valid pass names that can be skipped via TORCHINDUCTOR_MLU_SKIPPED_FX_PASSES.

    Returns:
        list[str]: List of valid pass names.

    Example:
        >>> from torch_mlu._inductor.fx_passes import get_skippable_fx_passes
        >>> passes = get_skippable_fx_passes()
        >>> print(",".join(passes))
    """
    return SKIPPABLE_FX_PASSES.copy()


def get_enableable_fx_passes():
    """
    Get the list of valid pass names that can be enabled via TORCHINDUCTOR_MLU_ENABLED_FX_PASSES.

    Returns:
        list[str]: List of valid pass names.

    Example:
        >>> from torch_mlu._inductor.fx_passes import get_enableable_fx_passes
        >>> passes = get_enableable_fx_passes()
        >>> print(",".join(passes))
    """
    return ENABLEABLE_FX_PASSES.copy()


def _validate_skip_setting():
    for item in config.skipped_fx_passes:
        if item not in SKIPPABLE_FX_PASSES:
            msg = f"""
Invalid skip pass '{item}' in TORCHINDUCTOR_MLU_SKIPPED_FX_PASSES environment variable.
Valid settings: {", ".join(SKIPPABLE_FX_PASSES)}.
"""
            raise ValueError(msg)


def _validate_enable_setting():
    for item in config.enabled_fx_passes:
        if item not in ENABLEABLE_FX_PASSES:
            msg = f"""
Invalid enable pass '{item}' in TORCHINDUCTOR_MLU_ENABLED_FX_PASSES environment variable.
Valid settings: {", ".join(ENABLEABLE_FX_PASSES)}.
"""
            raise ValueError(msg)


_validate_skip_setting()
_validate_enable_setting()
