import functools
import os
from typing import Iterable, Optional

import torch
from torch.fx.node import map_aggregate
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch._prims_common import is_integer_dtype

from collections import deque, Counter

counter = Counter()

extract_tensors = lambda node: node.meta["val"]


def is_signed_integer_tensor(t):
    return is_integer_dtype(t.dtype) and t.is_signed()


def is_mlu_tensor_node(node):
    if isinstance(node, torch.fx.Node) and isinstance(
        node.meta.get("val", None), torch.Tensor
    ):
        return node.meta["val"].is_mlu
    return False


def mlu_tensor_in_graph(graph):
    return any(
        (
            node.meta["val"].is_mlu
            for node in graph.nodes
            if isinstance(node.meta.get("val"), torch.Tensor)
        )
    )


def extract_meta(objs):
    def extract_tensor_meta(obj):
        if isinstance(obj, torch.Tensor):
            return _extract_tensor_metadata(obj)
        else:
            return obj

    return map_aggregate(objs, extract_tensor_meta)


def get_python_source_file_paths(
    directory: str,
    *,
    include_names: Optional[Iterable[str]] = None,
    exclude_names: Optional[Iterable[str]] = None,
):
    include_names = set(include_names) if include_names is not None else None
    exclude_names = set(exclude_names or ())

    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(
            directory_name
            for directory_name in dirs
            if directory_name != "__pycache__" and not directory_name.startswith(".")
        )
        for file_name in sorted(files):
            if file_name.startswith(".") or not file_name.endswith(".py"):
                continue
            if include_names is not None and file_name not in include_names:
                continue
            if file_name in exclude_names:
                continue
            yield os.path.join(root, file_name)


@functools.lru_cache(None)
def is_tmo_avaiable():
    try:
        import torch_mlu_ops

        return True
    except ImportError:
        return False


@functools.lru_cache(None)
def is_tmo_matmul_available():
    try:
        import torch_mlu_ops

        return hasattr(torch.ops.torch_mlu_ops, "matmul")
    except ImportError:
        return False


is_tmo_matmul_avaiable = is_tmo_matmul_available
