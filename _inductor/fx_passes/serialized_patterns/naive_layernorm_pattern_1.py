# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

import torch
import torch._inductor
import operator

aten = torch.ops.aten
prims = torch.ops.prims

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
)

var_correction = CallFunction(
    aten.var.correction,
    KeywordArg("inputs"),
    Ignored(),
    correction=Ignored(),
    keepdim=True,
)
add_Tensor = CallFunction(aten.add.Tensor, var_correction, KeywordArg("eps"))
rsqrt_default = CallFunction(aten.rsqrt.default, add_Tensor)
mul_Tensor = CallFunction(
    aten.mul.Tensor, rsqrt_default, KeywordArg("weight"), _users=2
)
mul_Tensor_1 = CallFunction(aten.mul.Tensor, KeywordArg("inputs"), mul_Tensor)
mean_dim = CallFunction(aten.mean.dim, KeywordArg("inputs"), Ignored(), True)
mul_Tensor_2 = CallFunction(aten.mul.Tensor, mean_dim, mul_Tensor)
sub_Tensor = CallFunction(aten.sub.Tensor, KeywordArg("bias"), mul_Tensor_2)
naive_layernorm_pattern_inference_2 = CallFunction(
    aten.add.Tensor, mul_Tensor_1, sub_Tensor, _users=0
)
