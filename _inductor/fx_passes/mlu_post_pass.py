import os
from typing import Any, Optional
import functools
import io
import pickle

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

from torch_mlu._inductor import config
from .utils import (
    get_python_source_file_paths,
    mlu_tensor_in_graph,
    is_tmo_avaiable,
    is_tmo_matmul_available,
)
from .make_contiguous_clone import make_contiguous_clone
from .joint_graph_pass import JOINT_GRAPH_PASS_FILES

PRE_GRAD_PASS_FILES = [
    "mlu_pre_grad_patterns.py",
    "mlu_pre_grad_pass.py",
]

POST_GRAD_PASS_FILES = [
    "mlu_post_grad_patterns.py",
    "mlu_post_pass.py",
]


def get_all_file_paths(directory: str):
    yield from get_python_source_file_paths(
        directory,
        include_names=(
            JOINT_GRAPH_PASS_FILES + PRE_GRAD_PASS_FILES + POST_GRAD_PASS_FILES
        ),
    )


# Note: if you need to support skip new pass, you need to update 'valid_passes' in torch_mlu/_inductor/fx_passes/__init__.py
def mlu_post_grad_pass(graph: torch.fx.graph.Graph, is_inference):
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="post_grad_passes",
    )
    gm = graph.owning_module
    gm.meta["is_inference"] = is_inference

    from . import (
        mlu_post_grad_patterns,
        fold_patterns,
        fused_mm,
        replace_sdpa_patterns,
        group_batch_fusion,
        normalization_pass,
        fused_bmm,
        fused_conv_leaky_relu,
    )
    from .triton_fusion import triton_fusion_pass

    if (
        "use_tmo_fa" not in config.skipped_fx_passes
        and is_tmo_avaiable()
        and is_inference
    ):
        replace_sdpa_patterns.replace_sdpa_pass.apply(graph)

    if "normalization" not in config.skipped_fx_passes:
        normalization_pass.normalization_pass.apply(graph)

    for name, pat in fold_patterns.fold_passes.items():
        if name not in config.skipped_fx_passes:
            pat.apply(graph)

    if "fold_clone" not in config.skipped_fx_passes:
        GraphTransformObserver(gm, "mlu_fold_clone").apply_graph_pass(
            fold_patterns.remove_noop_ops
        )

    group_batch_fusion.group_batch_fusion_passes(graph, is_inference)

    # Need be applied after fold stack pass, because some pass like cat_reshape would change op structure that should be hit by fold pass.
    for key, val in mlu_post_grad_patterns.passes.items():
        if key not in config.skipped_fx_passes:
            # This pass may causes precision loss, so only enable when inference. Ref wiki 541671493.
            if key == "div_exp_replace" and not is_inference:
                continue
            val.apply(graph)

    if "fused_mm" not in config.skipped_fx_passes:
        fused_mm.fused_mm_pass.apply(graph)

    if "fused_bmm" not in config.skipped_fx_passes:
        fused_bmm.fused_bmm_pass.apply(graph)

    if (
        "fuse_tmo_addmm" not in config.skipped_fx_passes
        and is_tmo_matmul_available()
        and is_inference
    ):
        mlu_post_grad_patterns.tmo_addmm_pass.apply(graph)

    if "conv_relu_fusion" in config.enabled_fx_passes:
        mlu_post_grad_patterns.conv_relu_fusion_pass.apply(graph)

    if "conv_leaky_relu_fusion" in config.enabled_fx_passes:
        GraphTransformObserver(gm, "mlu_conv_leaky_relu_fusion").apply_graph_pass(
            fused_conv_leaky_relu.conv_leaky_relu_fusion_pass
        )

    if (
        "fuse_tmo_bmm" not in config.skipped_fx_passes
        and is_tmo_avaiable()
        and is_inference
    ):
        mlu_post_grad_patterns.bmm_add_act_pass.apply(graph)

    if (
        "fuse_tmo_layernorm" in config.enabled_fx_passes
        and is_tmo_avaiable()
        and is_inference
    ):
        mlu_post_grad_patterns.tmo_layernorm_pass.apply(graph)

    if "make_contiguous_clone" not in config.skipped_fx_passes:
        GraphTransformObserver(gm, "mlu_make_contiguous_clone").apply_graph_pass(
            make_contiguous_clone
        )

    if config.enable_triton_fusion:
        GraphTransformObserver(gm, "mlu_triton_fusion").apply_graph_pass(
            triton_fusion_pass
        )

    if config.use_ultra_silu and config.use_ultra_math:
        mlu_post_grad_patterns.silu_pass.apply(graph)

    return


class MLUPostPass(CustomGraphPass):
    def __init__(self, prev_custom_pass=None):
        super().__init__()
        self.prev_custom_pass = prev_custom_pass

    def __call__(self, graph: torch.fx.graph.Graph, *, is_inference=True) -> None:
        if self.prev_custom_pass:
            self.prev_custom_pass(graph)

        if not torch.mlu.is_available() or not mlu_tensor_in_graph(graph):
            return

        mlu_post_grad_pass(graph, is_inference)

    def uuid(self) -> Optional[Any]:
        from torch._inductor.codecache import sha256_hash

        prev_uuid_str = ""
        if isinstance(self.prev_custom_pass, CustomGraphPass) and (
            prev_uuid := self.prev_custom_pass.uuid()
        ):
            stream = io.BytesIO()
            pickler = pickle.Pickler(stream)
            pickler.dump(prev_uuid)
            prev_uuid_str = sha256_hash(stream.getvalue())

        current_dir = os.path.dirname(os.path.abspath(__file__))
        return get_hash_for_files(tuple(get_all_file_paths(current_dir)), prev_uuid_str)

    def __repr__(self):
        return f"{__name__}.MLUPostPass()"
