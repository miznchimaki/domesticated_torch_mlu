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

# Work around duplicated KeywordArg instances definition bug of native autogen in MultiOutputPattern
# by defining them in the beginning, otherwise, you will encounter no anchor fould mismatch.
inputs = KeywordArg("inputs")
weight = KeywordArg("weight")
tangents_1 = KeywordArg("tangents_1")
mean_dim = CallFunction(aten.mean.dim, inputs, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, inputs, mean_dim, _users=2)
var_correction = CallFunction(
    aten.var.correction, inputs, Ignored(), correction=Ignored(), keepdim=True
)
add_Tensor = CallFunction(aten.add.Tensor, var_correction, KeywordArg("eps"))
rsqrt_default = CallFunction(aten.rsqrt.default, add_Tensor, _users=3)
mul_Tensor = CallFunction(aten.mul.Tensor, sub_Tensor, rsqrt_default, _users=2)
mul_Tensor_1 = CallFunction(aten.mul.Tensor, weight, mul_Tensor)
add_Tensor_1 = CallFunction(aten.add.Tensor, mul_Tensor_1, KeywordArg("bias"))
mul_Tensor_2 = CallFunction(aten.mul.Tensor, tangents_1, weight, _users=2)
mul_Tensor_3 = CallFunction(aten.mul.Tensor, mul_Tensor_2, rsqrt_default, _users=2)
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_2, sub_Tensor)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
mul_Scalar = CallFunction(aten.mul.Scalar, sum_dim_IntList, Ignored())
pow_Tensor_Scalar = CallFunction(aten.pow.Tensor_Scalar, rsqrt_default, Ignored())
mul_Tensor_5 = CallFunction(aten.mul.Tensor, mul_Scalar, pow_Tensor_Scalar)
mul_Scalar_1 = CallFunction(aten.mul.Scalar, mul_Tensor_5, Ignored())
mean_dim_1 = CallFunction(aten.mean.dim, inputs, Ignored(), True)
sub_Tensor_1 = CallFunction(aten.sub.Tensor, inputs, mean_dim_1)
mul_Tensor_6 = CallFunction(aten.mul.Tensor, mul_Scalar_1, sub_Tensor_1)
add_Tensor_2 = CallFunction(aten.add.Tensor, mul_Tensor_3, mul_Tensor_6)
neg_default = CallFunction(aten.neg.default, mul_Tensor_3)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, neg_default, Ignored(), True)
expand_default = CallFunction(aten.expand.default, sum_dim_IntList_1, Ignored())
div_Scalar = CallFunction(aten.div.Scalar, expand_default, Ignored())
add_Tensor_3 = CallFunction(aten.add.Tensor, add_Tensor_2, div_Scalar)
mul_Tensor_7 = CallFunction(aten.mul.Tensor, tangents_1, mul_Tensor)
sum_dim_IntList_2 = CallFunction(aten.sum.dim_IntList, mul_Tensor_7, Ignored(), True)
view_default = CallFunction(aten.view.default, sum_dim_IntList_2, Ignored())
sum_dim_IntList_3 = CallFunction(aten.sum.dim_IntList, tangents_1, Ignored(), True)
view_default_1 = CallFunction(aten.view.default, sum_dim_IntList_3, Ignored())
naive_layernorm_pattern_rsqrt_training = MultiOutputPattern(
    [add_Tensor_1, add_Tensor_3, view_default, view_default_1, None]
)


mean_dim = CallFunction(aten.mean.dim, KeywordArg("inputs"), Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, KeywordArg("inputs"), mean_dim)
var_correction = CallFunction(
    aten.var.correction,
    KeywordArg("inputs"),
    Ignored(),
    correction=Ignored(),
    keepdim=True,
)
add_Tensor = CallFunction(aten.add.Tensor, var_correction, KeywordArg("eps"))
rsqrt_default = CallFunction(aten.rsqrt.default, add_Tensor)
mul_Tensor = CallFunction(aten.mul.Tensor, sub_Tensor, rsqrt_default)
mul_Tensor_1 = CallFunction(aten.mul.Tensor, KeywordArg("weight"), mul_Tensor)
naive_layernorm_pattern_rsqrt_inference = CallFunction(
    aten.add.Tensor, mul_Tensor_1, KeywordArg("bias"), _users=0
)


inputs = KeywordArg("inputs")
weight = KeywordArg("weight")
tangents_1 = KeywordArg("tangents_1")
mean_dim = CallFunction(aten.mean.dim, inputs, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, inputs, mean_dim, _users=2)
var_correction = CallFunction(
    aten.var.correction, inputs, Ignored(), correction=Ignored(), keepdim=True
)
add_Tensor = CallFunction(aten.add.Tensor, var_correction, KeywordArg("eps"))
rsqrt_default = CallFunction(aten.rsqrt.default, add_Tensor, _users=3)
mul_Tensor = CallFunction(aten.mul.Tensor, sub_Tensor, rsqrt_default, _users=2)
mul_Tensor_1 = CallFunction(aten.mul.Tensor, weight, mul_Tensor)
add_Tensor_1 = CallFunction(aten.add.Tensor, mul_Tensor_1, KeywordArg("bias"))
mul_Tensor_2 = CallFunction(aten.mul.Tensor, tangents_1, weight)
convert_element_type_default = CallFunction(
    prims.convert_element_type.default, mul_Tensor_2, Ignored(), _users=2
)
mul_Tensor_3 = CallFunction(
    aten.mul.Tensor, convert_element_type_default, rsqrt_default, _users=2
)
mul_Tensor_4 = CallFunction(aten.mul.Tensor, convert_element_type_default, sub_Tensor)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
mul_Scalar = CallFunction(aten.mul.Scalar, sum_dim_IntList, Ignored())
pow_Tensor_Scalar = CallFunction(aten.pow.Tensor_Scalar, rsqrt_default, Ignored())
mul_Tensor_5 = CallFunction(aten.mul.Tensor, mul_Scalar, pow_Tensor_Scalar)
mul_Scalar_1 = CallFunction(aten.mul.Scalar, mul_Tensor_5, Ignored())
mean_dim_1 = CallFunction(aten.mean.dim, inputs, Ignored(), True)
sub_Tensor_1 = CallFunction(aten.sub.Tensor, inputs, mean_dim_1)
mul_Tensor_6 = CallFunction(aten.mul.Tensor, mul_Scalar_1, sub_Tensor_1)
add_Tensor_2 = CallFunction(aten.add.Tensor, mul_Tensor_3, mul_Tensor_6)
neg_default = CallFunction(aten.neg.default, mul_Tensor_3)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, neg_default, Ignored(), True)
expand_default = CallFunction(aten.expand.default, sum_dim_IntList_1, Ignored())
div_Scalar = CallFunction(aten.div.Scalar, expand_default, Ignored())
add_Tensor_3 = CallFunction(aten.add.Tensor, add_Tensor_2, div_Scalar)
mul_Tensor_7 = CallFunction(aten.mul.Tensor, tangents_1, mul_Tensor)
sum_dim_IntList_2 = CallFunction(aten.sum.dim_IntList, mul_Tensor_7, Ignored(), True)
view_default = CallFunction(aten.view.default, sum_dim_IntList_2, Ignored())
sum_dim_IntList_3 = CallFunction(aten.sum.dim_IntList, tangents_1, Ignored(), True)
view_default_1 = CallFunction(aten.view.default, sum_dim_IntList_3, Ignored())
naive_layernorm_pattern_training_rsqrt_2 = MultiOutputPattern(
    [add_Tensor_1, add_Tensor_3, view_default, view_default_1, None]
)
