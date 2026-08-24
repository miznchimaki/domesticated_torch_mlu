import os
import torch

# Env var switch (read from os.environ) to enable/disable calling repr(val) when
# building TorchDynamo guard debug strings. Disabled it to avoid
# expensive tensor printing side effects (e.g., large profiling traces).
use_guard_repr = os.environ.get("TORCHDYNAMO_MLU_GUARD_REPR", "1") == "1"

# Disable the recursive dict tag optimization for Dynamo guards.
#
# This optimization causes a use-after-free segfault during interpreter shutdown:
# when Python GC clears weakrefs in Py_FinalizeEx, the PyCapsule objects backing
# guard entries are freed before GuardManager's C++ destructor runs, leaving
# that trigger a crash in cleanup_tag_safe_entries(). See: https://github.com/pytorch/pytorch/issues/178224
#
# The upstream fix (https://github.com/pytorch/pytorch/pull/181873) disabled
# this optimization entirely, judging that its repeated segfaults outweigh the
# modest performance benefit. This backport applies the same change for MLU
# on PyTorch 2.10 and 2.11, which does not include the upstream patch.
torch._dynamo.config.use_recursive_dict_tags_for_guards = False
