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

convert_element_type_default = CallFunction(
    prims.convert_element_type.default, KeywordArg("weight"), Ignored(), _users=2
)
convert_element_type_default_1 = CallFunction(
    prims.convert_element_type.default, KeywordArg("inputs"), Ignored(), _users=5
)
mean_dim = CallFunction(aten.mean.dim, convert_element_type_default_1, Ignored(), True)
sub_Tensor = CallFunction(
    aten.sub.Tensor, convert_element_type_default_1, mean_dim, _users=2
)
var_correction = CallFunction(
    aten.var.correction,
    convert_element_type_default_1,
    Ignored(),
    correction=Ignored(),
    keepdim=True,
)
add_Tensor = CallFunction(aten.add.Tensor, var_correction, KeywordArg("eps"))
rsqrt_default = CallFunction(aten.rsqrt.default, add_Tensor, _users=3)
mul_Tensor = CallFunction(aten.mul.Tensor, sub_Tensor, rsqrt_default, _users=2)
mul_Tensor_1 = CallFunction(aten.mul.Tensor, convert_element_type_default, mul_Tensor)
convert_element_type_default_2 = CallFunction(
    prims.convert_element_type.default, KeywordArg("bias"), Ignored()
)
add_Tensor_1 = CallFunction(
    aten.add.Tensor, mul_Tensor_1, convert_element_type_default_2
)
convert_element_type_default_3 = CallFunction(
    prims.convert_element_type.default, add_Tensor_1, Ignored()
)
convert_element_type_default_4 = CallFunction(
    prims.convert_element_type.default, KeywordArg("tangents_1"), Ignored(), _users=3
)
mul_Tensor_2 = CallFunction(
    aten.mul.Tensor,
    convert_element_type_default_4,
    convert_element_type_default,
    _users=2,
)
mul_Tensor_3 = CallFunction(aten.mul.Tensor, mul_Tensor_2, rsqrt_default, _users=2)
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_2, sub_Tensor)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
mul_Scalar = CallFunction(aten.mul.Scalar, sum_dim_IntList, Ignored())
pow_Tensor_Scalar = CallFunction(aten.pow.Tensor_Scalar, rsqrt_default, Ignored())
mul_Tensor_5 = CallFunction(aten.mul.Tensor, mul_Scalar, pow_Tensor_Scalar)
mul_Scalar_1 = CallFunction(aten.mul.Scalar, mul_Tensor_5, Ignored())
mean_dim_1 = CallFunction(
    aten.mean.dim, convert_element_type_default_1, Ignored(), True
)
sub_Tensor_1 = CallFunction(aten.sub.Tensor, convert_element_type_default_1, mean_dim_1)
mul_Tensor_6 = CallFunction(aten.mul.Tensor, mul_Scalar_1, sub_Tensor_1)
add_Tensor_2 = CallFunction(aten.add.Tensor, mul_Tensor_3, mul_Tensor_6)
neg_default = CallFunction(aten.neg.default, mul_Tensor_3)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, neg_default, Ignored(), True)
expand_default = CallFunction(aten.expand.default, sum_dim_IntList_1, Ignored())
div_Scalar = CallFunction(aten.div.Scalar, expand_default, Ignored())
add_Tensor_3 = CallFunction(aten.add.Tensor, add_Tensor_2, div_Scalar)
convert_element_type_default_5 = CallFunction(
    prims.convert_element_type.default, add_Tensor_3, Ignored()
)
mul_Tensor_7 = CallFunction(aten.mul.Tensor, convert_element_type_default_4, mul_Tensor)
sum_dim_IntList_2 = CallFunction(aten.sum.dim_IntList, mul_Tensor_7, Ignored(), True)
view_default = CallFunction(aten.view.default, sum_dim_IntList_2, Ignored())
convert_element_type_default_6 = CallFunction(
    prims.convert_element_type.default, view_default, Ignored()
)
sum_dim_IntList_3 = CallFunction(
    aten.sum.dim_IntList, convert_element_type_default_4, Ignored(), True
)
view_default_1 = CallFunction(aten.view.default, sum_dim_IntList_3, Ignored())
convert_element_type_default_7 = CallFunction(
    prims.convert_element_type.default, view_default_1, Ignored()
)
naive_layernorm_pattern_with_cast_rsqrt_training = MultiOutputPattern(
    [
        convert_element_type_default_3,
        convert_element_type_default_5,
        convert_element_type_default_6,
        convert_element_type_default_7,
        None,
    ]
)


convert_element_type_default = CallFunction(
    prims.convert_element_type.default, KeywordArg("weight"), Ignored()
)
convert_element_type_default_1 = CallFunction(
    prims.convert_element_type.default, KeywordArg("inputs"), Ignored(), _users=3
)
mean_dim = CallFunction(aten.mean.dim, convert_element_type_default_1, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default_1, mean_dim)
var_correction = CallFunction(
    aten.var.correction,
    convert_element_type_default_1,
    Ignored(),
    correction=Ignored(),
    keepdim=True,
)
add_Tensor = CallFunction(aten.add.Tensor, var_correction, KeywordArg("eps"))
rsqrt_default = CallFunction(aten.rsqrt.default, add_Tensor)
mul_Tensor = CallFunction(aten.mul.Tensor, sub_Tensor, rsqrt_default)
mul_Tensor_1 = CallFunction(aten.mul.Tensor, convert_element_type_default, mul_Tensor)
convert_element_type_default_2 = CallFunction(
    prims.convert_element_type.default, KeywordArg("bias"), Ignored()
)
add_Tensor_1 = CallFunction(
    aten.add.Tensor, mul_Tensor_1, convert_element_type_default_2
)
naive_layernorm_pattern_with_cast_rsqrt_inference = CallFunction(
    prims.convert_element_type.default, add_Tensor_1, Ignored(), _users=0
)
