# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

import torch
import torch._inductor
import operator

from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    CallFunctionVarArgs,
    CallMethod,
    CallMethodVarArgs,
    CallModule,
    CallModuleVarArgs,
    ExclusiveKeywordArg,
    Ignored,
    KeywordArg,
    ListOf,
    MultiOutputPattern,
    PatternExpr,
    RepeatedExpr,
    _TargetArgsExpr,
    _TargetExpr,
    _TargetExprVarArgs,
    Constant,
)

aten = torch.ops.aten
prims = torch.ops.prims

mul_Scalar = CallFunction(aten.mul.Scalar, KeywordArg("query"), KeywordArg("sca"))
expand = CallFunction(torch.ops.aten.expand.default, mul_Scalar, Ignored())
view_default = CallFunction(aten.reshape.default, expand, Ignored())
permute_default = CallFunction(aten.permute.default, KeywordArg("key"), Ignored())
mul_Scalar_1 = CallFunction(aten.mul.Scalar, permute_default, KeywordArg("sca"))
expand_1 = CallFunction(torch.ops.aten.expand.default, mul_Scalar_1, Ignored())
view_default_1 = CallFunction(aten.reshape.default, expand_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.reshape.default, bmm_default, Ignored())
add_Tensor = CallFunction(aten.add.Tensor, view_default_2, KeywordArg("attn_mask"))
_safe_softmax_default = CallFunction(aten._safe_softmax.default, add_Tensor, Ignored())
expand_2 = CallFunction(torch.ops.aten.expand.default, _safe_softmax_default, Ignored())
view_default_3 = CallFunction(aten.reshape.default, expand_2, Ignored())
expand_3 = CallFunction(torch.ops.aten.expand.default, KeywordArg("value"), Ignored())
view_default_4 = CallFunction(aten.reshape.default, expand_3, Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
sdpa_pattern_mlu_inference_1 = CallFunction(
    aten.reshape.default, bmm_default_1, Ignored()
)
