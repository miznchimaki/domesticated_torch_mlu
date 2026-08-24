# mypy: allow-untyped-defs
"""Common utilities and functions for flex attention kernels"""

from functools import partial
from pathlib import Path

from torch._inductor.utils import load_template


_FLEX_TEMPLATE_DIR = Path(__file__).parent / "templates"
load_flex_template = partial(load_template, template_dir=_FLEX_TEMPLATE_DIR)
