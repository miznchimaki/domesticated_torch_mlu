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

convert_element_type_default = CallFunction(
    prims.convert_element_type.default, KeywordArg("query"), Ignored()
)
mul_Scalar = CallFunction(
    aten.mul.Scalar, convert_element_type_default, KeywordArg("sca")
)
view_default = CallFunction(aten.reshape.default, mul_Scalar, Ignored())
convert_element_type_default_1 = CallFunction(
    prims.convert_element_type.default, KeywordArg("key"), Ignored()
)
permute_default = CallFunction(
    aten.permute.default, convert_element_type_default_1, Ignored()
)
mul_Scalar_1 = CallFunction(aten.mul.Scalar, permute_default, KeywordArg("sca"))
view_default_1 = CallFunction(aten.reshape.default, mul_Scalar_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.reshape.default, bmm_default, Ignored())
full_default = CallFunction(
    aten.full.default,
    [],
    Ignored(),
    dtype=Ignored(),
    layout=torch.strided,
    device=Ignored(),
    pin_memory=False,
)
full_default_1 = CallFunction(
    aten.full.default,
    [],
    Ignored(),
    dtype=Ignored(),
    layout=torch.strided,
    device=Ignored(),
    pin_memory=False,
)
where_self = CallFunction(
    aten.where.self, KeywordArg("attn_mask"), full_default, full_default_1
)
add_Tensor = CallFunction(aten.add.Tensor, view_default_2, where_self)
_safe_softmax_default = CallFunction(aten._safe_softmax.default, add_Tensor, Ignored())
view_default_3 = CallFunction(aten.reshape.default, _safe_softmax_default, Ignored())
convert_element_type_default_2 = CallFunction(
    prims.convert_element_type.default, KeywordArg("value"), Ignored()
)
view_default_4 = CallFunction(
    aten.reshape.default, convert_element_type_default_2, Ignored()
)
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
view_default_5 = CallFunction(aten.reshape.default, bmm_default_1, Ignored())
sdpa_pattern_mlu_inference_2 = CallFunction(
    prims.convert_element_type.default, view_default_5, Ignored(), _users=0
)
