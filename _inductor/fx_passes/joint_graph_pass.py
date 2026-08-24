import os
from typing import Any, Optional
import io
import pickle

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

from torch_mlu._inductor import config
from .utils import get_python_source_file_paths, mlu_tensor_in_graph


JOINT_GRAPH_PASS_FILES = [
    "mlu_joint_graph_patterns.py",
    "joint_graph_pass.py",
    "naive_layernorm_pattern_with_cast.py",
    "naive_layernorm_pattern.py",
]


def get_all_file_paths(directory: str):
    yield from get_python_source_file_paths(
        directory,
        include_names=JOINT_GRAPH_PASS_FILES,
    )


class MLUJointPass(CustomGraphPass):
    def __init__(self, prev_custom_pass=None):
        super().__init__()
        self.prev_custom_pass = prev_custom_pass

    # Note: if you need to support skip new pass, you need to update 'valid_passes' in torch_mlu/_inductor/fx_passes/__init__.py
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        if self.prev_custom_pass:
            self.prev_custom_pass(graph)

        if not torch.mlu.is_available() or not mlu_tensor_in_graph(graph):
            return

        from . import mlu_joint_graph_patterns

        if (
            "fuse_layernorm_training" in config.enabled_fx_passes
            or "fuse_layernorm" in config.enabled_fx_passes
        ):
            mlu_joint_graph_patterns.replace_layernorm_training_pass.apply(graph)

        if "fuse_layernorm_infer" not in config.skipped_fx_passes:
            mlu_joint_graph_patterns.replace_layernorm_infer_pass.apply(graph)

        if "repeat2expand" not in config.skipped_fx_passes:
            mlu_joint_graph_patterns.repeat_gather_pass.apply(graph)

        return

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
        return f"{__name__}.MLUJointPass()"
