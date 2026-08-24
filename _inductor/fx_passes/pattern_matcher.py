import os
import sys
import importlib
from pathlib import Path
import torch

aten = torch.ops.aten


MLU_SERIALIZED_PATTERN_PATH = Path(__file__).parent / "serialized_patterns"


def mlu_gen_register_replacement(*args, **kwargs) -> None:
    from torch._inductor.pattern_matcher import gen_register_replacement

    assert callable(args[1])
    search_fn = args[1]
    if "PYTORCH_MLU_GEN_PATTERNS" in os.environ:
        # This branch only generate a draft pattern, then modification is also needed to
        # work around native bug, please ref naive_layernorm_pattern_training/inference as examples.
        orig_path = torch._inductor.pattern_matcher.SERIALIZED_PATTERN_PATH
        torch._inductor.pattern_matcher.SERIALIZED_PATTERN_PATH = (
            MLU_SERIALIZED_PATTERN_PATH
        )
        os.environ["PYTORCH_GEN_PATTERNS"] = "1"
        gen_register_replacement(*args, **kwargs)
        os.environ.pop("PYTORCH_GEN_PATTERNS")
        torch._inductor.pattern_matcher.SERIALIZED_PATTERN_PATH = orig_path
    else:
        pattern_name = search_fn.__name__
        # Run with PYTORCH_GEN_PATTERNS=1 before go into this branch
        pattern_mod = importlib.import_module(
            f"torch_mlu._inductor.fx_passes.serialized_patterns.{pattern_name}"
        )
        m = importlib.import_module(f"torch._inductor.fx_passes.serialized_patterns")
        if hasattr(m, pattern_name):
            raise RuntimeError(f"The module of '{pattern_name}' has already exist")
        setattr(m, pattern_name, pattern_mod)
        pattern_module_name = ".".join(
            ["torch._inductor.fx_passes.serialized_patterns", pattern_name]
        )
        sys.modules[pattern_module_name] = pattern_mod
        gen_register_replacement(*args, **kwargs)
        sys.modules.pop(pattern_module_name, None)
        delattr(m, pattern_name)
