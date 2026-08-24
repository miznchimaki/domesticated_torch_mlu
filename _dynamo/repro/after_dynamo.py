import textwrap
from collections.abc import Sequence
from typing import Any, Optional

import torch
from torch._dynamo.debug_utils import (
    extra_imports,
    generate_config_string,
    generate_env_vars_string,
    InputWriter,
    NNModuleToString,
)
from torch._dynamo.repro import after_dynamo
from torch.fx.experimental.symbolic_shapes import fx_placeholder_targets
from ...mlu._utils import replace_references
from ...utils import gorilla


# NOTE by CAMBRICON: Because extra_import takes effect too late, it will cause config to report an error, so a patch is needed instead of setting extra_import.
@gorilla.patch(torch._dynamo.repro.after_dynamo)
def generate_dynamo_fx_repro_string(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: Optional[str],
    check_accuracy: bool = False,
    *,
    stable_output: bool = False,
    save_dir: Optional[str] = None,
    command: str = "run",
) -> str:
    """
    Generate a repro string for backend-agnostic minified version.
    """

    model_str = NNModuleToString.convert(gm)

    # TODO: Figure out why torch.compile'd hash isn't work on this codepath
    writer = InputWriter(save_dir, stable_hash=True)
    for placeholder, arg in zip(fx_placeholder_targets(gm), args):
        if isinstance(arg, (int, torch.SymInt)):
            writer.symint(placeholder, arg)
        elif isinstance(arg, torch.Tensor):
            # TODO: improve these names with FQN
            writer.tensor(placeholder, arg)
        else:
            raise TypeError(f"arg is neither SymInt/int nor torch.Tensor, {arg}")
    load_args = "\n".join(writer.lines())

    return textwrap.dedent(
        f"""
{generate_env_vars_string(stable_output=stable_output)}
from math import inf
import torch
# Add by CAMBRICON
import torch_mlu
# end Add by CAMBRICON
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

{generate_config_string(stable_output=stable_output)}

{extra_imports}

{model_str}
mod = Repro()

{load_args}

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy={check_accuracy!r}, command={command!r},
        save_dir={save_dir!r}, autocast={torch.is_autocast_enabled()!r}, backend={compiler_name!r})
"""
    )
