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
add_Tensor = CallFunction(aten.add.Tensor, var_correction, KeywordArg("eps"), _users=2)
pow_Tensor_Scalar = CallFunction(
    aten.pow.Tensor_Scalar, add_Tensor, Ignored(), _users=4
)
div_Tensor = CallFunction(aten.div.Tensor, sub_Tensor, pow_Tensor_Scalar, _users=2)
mul_Tensor = CallFunction(aten.mul.Tensor, weight, div_Tensor)
add_Tensor_1 = CallFunction(aten.add.Tensor, mul_Tensor, KeywordArg("bias"))
mul_Tensor_1 = CallFunction(aten.mul.Tensor, tangents_1, weight, _users=2)
div_Tensor_1 = CallFunction(aten.div.Tensor, mul_Tensor_1, pow_Tensor_Scalar, _users=2)
neg_default = CallFunction(aten.neg.default, mul_Tensor_1)
div_Tensor_2 = CallFunction(aten.div.Tensor, sub_Tensor, pow_Tensor_Scalar)
div_Tensor_3 = CallFunction(aten.div.Tensor, div_Tensor_2, pow_Tensor_Scalar)
mul_Tensor_2 = CallFunction(aten.mul.Tensor, neg_default, div_Tensor_3)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, mul_Tensor_2, Ignored(), True)
pow_Tensor_Scalar_1 = CallFunction(aten.pow.Tensor_Scalar, add_Tensor, Ignored())
mul_Scalar = CallFunction(aten.mul.Scalar, pow_Tensor_Scalar_1, Ignored())
mul_Tensor_3 = CallFunction(aten.mul.Tensor, sum_dim_IntList, mul_Scalar)
mul_Scalar_1 = CallFunction(aten.mul.Scalar, mul_Tensor_3, Ignored())
mean_dim_1 = CallFunction(aten.mean.dim, inputs, Ignored(), True)
sub_Tensor_1 = CallFunction(aten.sub.Tensor, inputs, mean_dim_1)
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Scalar_1, sub_Tensor_1)
add_Tensor_2 = CallFunction(aten.add.Tensor, div_Tensor_1, mul_Tensor_4)
neg_default_1 = CallFunction(aten.neg.default, div_Tensor_1)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, neg_default_1, Ignored(), True)
expand_default = CallFunction(aten.expand.default, sum_dim_IntList_1, Ignored())
div_Scalar = CallFunction(aten.div.Scalar, expand_default, Ignored())
add_Tensor_3 = CallFunction(aten.add.Tensor, add_Tensor_2, div_Scalar)
mul_Tensor_5 = CallFunction(aten.mul.Tensor, tangents_1, div_Tensor)
sum_dim_IntList_2 = CallFunction(aten.sum.dim_IntList, mul_Tensor_5, Ignored(), True)
view_default = CallFunction(aten.view.default, sum_dim_IntList_2, Ignored())
sum_dim_IntList_3 = CallFunction(aten.sum.dim_IntList, tangents_1, Ignored(), True)
view_default_1 = CallFunction(aten.view.default, sum_dim_IntList_3, Ignored())
naive_layernorm_pattern_training_1 = MultiOutputPattern(
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
pow_Tensor_Scalar = CallFunction(aten.pow.Tensor_Scalar, add_Tensor, Ignored())
div_Tensor = CallFunction(aten.div.Tensor, sub_Tensor, pow_Tensor_Scalar)
mul_Tensor = CallFunction(aten.mul.Tensor, KeywordArg("weight"), div_Tensor)
naive_layernorm_pattern_inference = CallFunction(
    aten.add.Tensor, mul_Tensor, KeywordArg("bias"), _users=0
)

inputs = KeywordArg("inputs")
weight = KeywordArg("weight")
tangents_1 = KeywordArg("tangents_1")
mean_dim = CallFunction(aten.mean.dim, inputs, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, inputs, mean_dim, _users=2)
var_correction = CallFunction(
    aten.var.correction, inputs, Ignored(), correction=Ignored(), keepdim=True
)
add_Tensor = CallFunction(aten.add.Tensor, var_correction, KeywordArg("eps"), _users=2)
pow_Tensor_Scalar = CallFunction(
    aten.pow.Tensor_Scalar, add_Tensor, Ignored(), _users=4
)
div_Tensor = CallFunction(aten.div.Tensor, sub_Tensor, pow_Tensor_Scalar, _users=2)
mul_Tensor = CallFunction(aten.mul.Tensor, weight, div_Tensor)
add_Tensor_1 = CallFunction(aten.add.Tensor, mul_Tensor, KeywordArg("bias"))
mul_Tensor_1 = CallFunction(aten.mul.Tensor, tangents_1, weight)
convert_element_type_default = CallFunction(
    prims.convert_element_type.default, mul_Tensor_1, Ignored(), _users=2
)
div_Tensor_1 = CallFunction(
    aten.div.Tensor, convert_element_type_default, pow_Tensor_Scalar, _users=2
)
neg_default = CallFunction(aten.neg.default, convert_element_type_default)
div_Tensor_2 = CallFunction(aten.div.Tensor, sub_Tensor, pow_Tensor_Scalar)
div_Tensor_3 = CallFunction(aten.div.Tensor, div_Tensor_2, pow_Tensor_Scalar)
mul_Tensor_2 = CallFunction(aten.mul.Tensor, neg_default, div_Tensor_3)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, mul_Tensor_2, Ignored(), True)
pow_Tensor_Scalar_1 = CallFunction(aten.pow.Tensor_Scalar, add_Tensor, Ignored())
mul_Scalar = CallFunction(aten.mul.Scalar, pow_Tensor_Scalar_1, Ignored())
mul_Tensor_3 = CallFunction(aten.mul.Tensor, sum_dim_IntList, mul_Scalar)
mul_Scalar_1 = CallFunction(aten.mul.Scalar, mul_Tensor_3, Ignored())
mean_dim_1 = CallFunction(aten.mean.dim, inputs, Ignored(), True)
sub_Tensor_1 = CallFunction(aten.sub.Tensor, inputs, mean_dim_1)
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Scalar_1, sub_Tensor_1)
add_Tensor_2 = CallFunction(aten.add.Tensor, div_Tensor_1, mul_Tensor_4)
neg_default_1 = CallFunction(aten.neg.default, div_Tensor_1)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, neg_default_1, Ignored(), True)
expand_default = CallFunction(aten.expand.default, sum_dim_IntList_1, Ignored())
div_Scalar = CallFunction(aten.div.Scalar, expand_default, Ignored())
add_Tensor_3 = CallFunction(aten.add.Tensor, add_Tensor_2, div_Scalar)
mul_Tensor_5 = CallFunction(aten.mul.Tensor, tangents_1, div_Tensor)
sum_dim_IntList_2 = CallFunction(aten.sum.dim_IntList, mul_Tensor_5, Ignored(), True)
view_default = CallFunction(aten.view.default, sum_dim_IntList_2, Ignored())
sum_dim_IntList_3 = CallFunction(aten.sum.dim_IntList, tangents_1, Ignored(), True)
view_default_1 = CallFunction(aten.view.default, sum_dim_IntList_3, Ignored())
naive_layernorm_pattern_training_2 = MultiOutputPattern(
    [add_Tensor_1, add_Tensor_3, view_default, view_default_1, None]
)
