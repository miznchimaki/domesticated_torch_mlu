import torch
from torch._dynamo.test_minifier_common import MinifierTestBase
from torch._dynamo.trace_rules import _as_posix_path
from ..utils import gorilla


def _gen_test_code(self, run_code, repro_after, repro_level):
    repro_after_line = ""
    if repro_after == "aot_inductor":
        repro_after_line = (
            "torch._inductor.config.aot_inductor.dump_aoti_minifier = True"
        )
    elif repro_after:
        repro_after_line = f"""\
torch._dynamo.config.repro_after = "{repro_after}"
    """
    return f"""\
import torch
# Add by CAMBRICON
import torch_mlu
import torch._dynamo
import torch._inductor
{_as_posix_path(torch._dynamo.config.codegen_config())}
{_as_posix_path(torch._inductor.config.codegen_config())}
{repro_after_line}
torch._dynamo.config.repro_level = {repro_level}
torch._inductor.config.aot_inductor.repro_level = {repro_level}
torch._dynamo.config.debug_dir_root = "{_as_posix_path(self.DEBUG_DIR)}"
{run_code}
"""


patch = gorilla.Patch(
    torch._dynamo.test_minifier_common.MinifierTestBase,
    "_gen_test_code",
    _gen_test_code,
)
gorilla.apply(patch)
