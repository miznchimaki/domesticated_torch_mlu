import collections
from typing import Any, Optional
import operator

import torch
from torch._inductor.fx_passes.group_batch_fusion import (
    GroupBatchFusionBase,
    graph_search_options,
    find_independent_subset_greedy,
)
from torch._inductor.fx_passes.post_grad import reorder_for_locality
from torch._inductor.pattern_matcher import (
    Ignored,
    MULTIPLE,
    KeywordArg,
    CallFunction,
    CallFunctionVarArgs,
    _transfer_meta,
    stable_topological_sort,
)
from torch._inductor.fx_passes.split_cat import GetItem
from torch.fx.experimental.symbolic_shapes import (
    statically_known_true,
    sym_eq,
    free_unbacked_symbols,
)
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch._prims_common import is_contiguous_or_false

from torch_mlu._inductor import config
from .utils import extract_meta, extract_tensors, counter

aten = torch.ops.aten


def _is_trans_tensor(node):
    if statically_known_true(node.meta["val"].stride(0) == 1) and statically_known_true(
        node.meta["val"].stride(1) == node.meta["val"].shape[0]
    ):
        return True
    return False


def _is_trans_or_contiguous_tensor(node):
    if (
        node.meta["val"].dim() == 2 and _is_trans_tensor(node)
    ) or is_contiguous_or_false(node.meta["val"]):
        return True
    return False


def _is_fuse_activation(node):
    if len(node.users) != 1:
        return False
    user = next(iter(node.users))
    return user.op == "call_function" and user.target in [
        aten.gelu.default,
        aten.relu.default,
        aten.sigmoid.default,
    ]


class MLUGroupBatchFusionBase(GroupBatchFusionBase):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.is_hit = False
        self.name = name


class ComboMatmul(MLUGroupBatchFusionBase):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.target_ops = [aten.mm.default, aten.addmm.default]

    def _matmul_node_can_be_fused(self, node: torch.fx.Node):
        a, b = node.args[0], node.args[1]
        if not a.meta["val"].is_mlu:
            return False

        if not statically_known_true(node.meta["val"].numel() > 0):
            return False

        # Temporarily skip unbacked m/n shape，because currently aten.slice is not unbacked-safe and grouped_gemm do not support unbacked n.
        if len(free_unbacked_symbols(node.meta["val"].size())) > 0:
            return False

        if (a.meta["val"].dtype != b.meta["val"].dtype) or not torch.is_floating_point(
            a.meta["val"]
        ):
            return False

        # MLU grouped_gemm requires contiguous for all inputs
        if not all([_is_trans_or_contiguous_tensor(t) for t in [a, b]]):
            return False

        return True

    def _addmatmul_node_can_be_fused(self, node: torch.fx.Node):
        a, b, c = node.args
        if not a.meta["val"].is_mlu:
            return False

        if not statically_known_true(node.meta["val"].numel() > 0):
            return False

        # Temporarily skip unbacked m/n shape，because currently aten.slice is not unbacked-safe and grouped_gemm do not support unbacked n.
        if len(free_unbacked_symbols(node.meta["val"].size())) > 0:
            return False

        if a.meta["val"].dim() != 1 and not statically_known_true(
            sym_eq(a.meta["val"].shape, node.meta["val"].shape)
        ):
            return False

        # MLU grouped_gemm requires contiguous for all inputs
        if not all(
            [_is_trans_or_contiguous_tensor(t) for t in [b, c]]
        ) or not is_contiguous_or_false(a.meta["val"]):
            return False

        if (
            (a.meta["val"].dtype == b.meta["val"].dtype == c.meta["val"].dtype)
            and torch.is_floating_point(a.meta["val"])
            and (node.kwargs.get("beta", 1) == 1 and node.kwargs.get("alpha", 1) == 1)
        ):
            return True

        return False

    def _gen_group_key(self, input_a, input_b, add_input):
        if self.name == "mlu_combo_matmul_same_kn":
            return (
                self.name,
                str(input_b.meta["val"].shape),
                None if add_input is None else str(add_input.meta["val"].shape),
                _is_trans_tensor(input_a),
                _is_trans_tensor(input_b),
                str(input_a.meta["val"].dtype),
            )
        else:
            trans_b = _is_trans_tensor(input_b)
            k, n = input_b.meta["val"].shape[0], input_b.meta["val"].shape[1]
            if trans_b:
                k, n = n, k
            if isinstance(k, int) and isinstance(n, int):
                if self.name == "mlu_combo_matmul_mini" and k <= 128 and n <= 128:
                    return (
                        self.name,
                        None if add_input is None else str(add_input.meta["val"].dim()),
                        _is_trans_tensor(input_a),
                        trans_b,
                        str(input_a.meta["val"].dtype),
                    )
                elif self.name == "mlu_combo_matmul_group_kn":
                    return (
                        self.name,
                        None if add_input is None else str(add_input.meta["val"].dim()),
                        str((k + 1) // 256),
                        str((n + 1) // 512),
                        _is_trans_tensor(input_a),
                        trans_b,
                        str(input_a.meta["val"].dtype),
                    )
        return None

    def match(self, node: torch.fx.Node) -> Optional[tuple[str, bool]]:
        group_key = None
        # Other matmul related ops like aten.matmul and aten.linear would be decomposed to these ops.
        if CallFunctionVarArgs(aten.mm.default).match(
            node
        ) and self._matmul_node_can_be_fused(node):
            input_a = node.args[0]
            input_b = node.args[1]
            group_key = self._gen_group_key(input_a, input_b, None)
        elif CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmatmul_node_can_be_fused(node):
            add_input = node.args[0]
            input_a = node.args[1]
            input_b = node.args[2]
            group_key = self._gen_group_key(input_a, input_b, add_input)

        if group_key and _is_fuse_activation(node):
            act_key = str(node.target._opname)
            act_node = next(iter(node.users))
            if act_node.target is aten.gelu.default and act_node.kwargs.get(
                "approximate", None
            ):
                act_key += act_node.kwargs["approximate"]
            group_key += (act_key,)

        return group_key

    def fuse(self, graph: torch.fx.Graph, subset: list[torch.fx.Node]):
        contiguous_memo = {}

        def make_trans_contiguous(graph, x_node):
            if x_node in contiguous_memo:
                return contiguous_memo[x_node]
            with graph.inserting_after(x_node):
                permute_fake_tensor = aten.permute.default(x_node.meta["val"], [1, 0])
                permute_tensor_meta = extract_meta(permute_fake_tensor)
                new_x = graph.call_function(
                    aten.permute.default,
                    args=(x_node, [1, 0]),
                )
                _transfer_meta(new_x.meta, x_node, "mlu_combo_matmul")
                new_x.meta.update(
                    {"val": permute_fake_tensor, "tensor_meta": permute_tensor_meta}
                )
            contiguous_memo[x_node] = new_x
            return new_x

        group_inputs = []
        group_weights = []
        group_biases = []
        group_nodes = []
        trans_i = trans_w = None
        for node in subset:
            if node.target in [aten.addmm.default]:
                bias, input, weight = node.args
            else:
                input, weight = node.args
                bias = None

            if trans_i is None:
                trans_i = _is_trans_tensor(input)
            if trans_i:
                input = make_trans_contiguous(graph, input)

            if trans_w is None:
                trans_w = _is_trans_tensor(weight)
            if trans_w:
                weight = make_trans_contiguous(graph, weight)

            group_nodes.append(node)
            group_inputs.append(input)
            group_weights.append(weight)
            group_biases.append(bias)

        group_len = len(subset)

        if group_biases[0] is None:
            group_biases = None

        beta = group_cs = None
        if group_biases:
            if len(group_biases[0].meta["val"].shape) != 1:
                group_cs = group_biases
                group_biases = None
                beta = [1.0] * group_len
        first_node = sorted(subset)[0]
        with graph.inserting_before(first_node):
            fused_matmul_tensor = torch.ops.torch_mlu.grouped_gemm.default(
                torch.fx.map_arg(group_inputs, extract_tensors),  # a
                torch.fx.map_arg(group_weights, extract_tensors),  # b
                torch.fx.map_arg(group_cs, extract_tensors)
                if group_cs
                else group_cs,  # c
                torch.fx.map_arg(group_biases, extract_tensors)
                if group_biases
                else group_biases,  # bias
                None,  # alpha
                beta,  # beta
                trans_i,  # trans_a
                trans_w,  # trans_b
            )
            fused_matmul_tensor_meta = extract_meta(fused_matmul_tensor)
            fused_matmul = graph.call_function(
                torch.ops.torch_mlu.grouped_gemm.default,
                args=(
                    group_inputs,  # a
                    group_weights,  # b
                    group_cs,  # c
                    group_biases,  # bias
                    None,  # alpha
                    beta,  # beta
                    trans_i,  # trans_a
                    trans_w,  # trans_b
                ),
            )
            _transfer_meta(fused_matmul.meta, first_node, "mlu_combo_matmul")
            fused_matmul.meta.update(
                {"val": fused_matmul_tensor, "tensor_meta": fused_matmul_tensor_meta}
            )

        act_out = fused_matmul
        fuse_act = _is_fuse_activation(first_node)
        if fuse_act:
            act_kwargs = {}
            act_node = next(iter(first_node.users))
            if act_node.target is aten.gelu.default:
                act_kwargs["approximate"] = act_node.kwargs.get("approximate", "none")
            with graph.inserting_after(fused_matmul):
                act_out = graph.call_function(
                    act_node.target, args=(fused_matmul,), kwargs=act_kwargs
                )
                act_out.meta.update(fused_matmul.meta)

        delete_nodes = []
        slice_beg = 0
        has_different_n = True if act_out.meta["val"].dim() == 1 else False
        for i, original_mm in enumerate(group_nodes):
            if has_different_n:
                slice_end = slice_beg + original_mm.meta["val"].numel()
            # Because this grouped_gemm op requires output of (sum(M_i), N) shape, so for compatibility slice at the first dim.
            else:
                slice_end = slice_beg + original_mm.meta["val"].shape[0]
            assert (
                slice_end <= act_out.meta["val"].numel()
            ), f"Slice end {slice_end} exceeds output size {act_out.meta['val'].numel()}"

            with graph.inserting_after(act_out):
                slice_out = graph.call_function(
                    aten.slice.Tensor, args=(act_out, 0, slice_beg, slice_end)
                )
                if has_different_n:
                    slice_tensor = aten.slice.Tensor(
                        act_out.meta["val"], 0, slice_beg, slice_end
                    )
                    slice_tensor_meta = extract_meta(slice_tensor)
                    _transfer_meta(slice_out.meta, act_out, "mlu_combo_matmul")
                    slice_out.meta.update(
                        {"val": slice_tensor, "tensor_meta": slice_tensor_meta}
                    )
                    new_out = graph.call_function(
                        aten.reshape.default, (slice_out, original_mm.meta["val"].shape)
                    )
                else:
                    new_out = slice_out

            if fuse_act:
                old_out = next(iter(original_mm.users))
            else:
                old_out = original_mm
            old_out.replace_all_uses_with(new_out)
            new_out.meta.update(old_out.meta)
            delete_nodes.append(old_out)
            if fuse_act:
                delete_nodes.append(original_mm)
            slice_beg = slice_end
        for node in delete_nodes:
            graph.erase_node(node)
        self.is_hit = True


#                    /-----> mul -----> o1
#  input ---> unbind ------> mul -----> o2
#                    \-----> mul -----> o3
# --->
#                               /-----> o1
#  input ---> mul -----> unbind ------> o2
#                               \-----> o3
unbind_getitem_pattern = GetItem(
    CallFunction(aten.unbind.int, KeywordArg("unbind_input"), 0, _users=MULTIPLE),
    Ignored(),
    _users=MULTIPLE,
)

add_lhs = CallFunction(
    aten.add.Tensor,
    KeywordArg("lhs_arg"),
    unbind_getitem_pattern,
    alpha=KeywordArg("alpha"),
)
add_rhs = CallFunction(
    aten.add.Tensor,
    unbind_getitem_pattern,
    KeywordArg("rhs_arg"),
    alpha=KeywordArg("alpha"),
)
mul_lhs = CallFunction(
    aten.mul.Tensor,
    KeywordArg("lhs_arg"),
    unbind_getitem_pattern,
)
mul_rhs = CallFunction(
    aten.mul.Tensor,
    unbind_getitem_pattern,
    KeywordArg("rhs_arg"),
)
where_lhs = CallFunction(
    aten.where.self,
    KeywordArg("condition"),
    KeywordArg("lhs_arg"),
    unbind_getitem_pattern,
)
where_rhs = CallFunction(
    aten.where.self,
    KeywordArg("condition"),
    unbind_getitem_pattern,
    KeywordArg("rhs_arg"),
)


class CombinePointwiseSrc(MLUGroupBatchFusionBase):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.pats = (
            [add_lhs, mul_lhs, where_lhs]
            if "lhs" in name
            else [add_rhs, mul_rhs, where_rhs]
        )
        self.target_ops = [aten.add.Tensor, aten.mul.Tensor, aten.where.self]

    def _pointwise_can_be_fused(self, match):
        output_node = match.output_node()
        unbind_node = match.kwargs["unbind_input"]
        if unbind_node.meta["val"].dim() <= output_node.meta["val"].dim():
            return False
        return True

    def match(self, node: torch.fx.Node) -> Optional[tuple[str, bool]]:
        if (
            isinstance(node.meta.get("val", None), torch.Tensor)
            and not node.meta["val"].is_mlu
        ):
            return None
        for pat in self.pats:
            match = pat.match(node)
            if match and self._pointwise_can_be_fused(match):
                kwargs = match.kwargs
                lhs_arg = (
                    kwargs["lhs_arg"] if "lhs_arg" in kwargs else kwargs["unbind_input"]
                )
                rhs_arg = (
                    kwargs["rhs_arg"] if "rhs_arg" in kwargs else kwargs["unbind_input"]
                )
                alpha = kwargs.get("alpha", None)
                condition = kwargs.get("condition", None)
                op_name = node.target.name()
                return (
                    "mlu_combine_pointwise_src_pass",
                    op_name,
                    lhs_arg,
                    rhs_arg,
                    alpha,
                    condition,
                )
        return None

    def fuse(self, graph: torch.fx.Graph, subset: list[torch.fx.Node]):
        first_node = sorted(subset)[0]
        for pat in self.pats:
            match = pat.match(first_node)
            if match:
                break
        assert match, f"{match}, invalid candidate fuse node {first_node}!"
        match_kwargs = match.kwargs
        unbind_node = match_kwargs["unbind_input"]
        lhs_arg = match_kwargs["lhs_arg"] if "lhs_arg" in match_kwargs else unbind_node
        rhs_arg = match_kwargs["rhs_arg"] if "rhs_arg" in match_kwargs else unbind_node
        alpha = match_kwargs.get("alpha", None)
        condition = match_kwargs.get("condition", None)
        args = (
            (lhs_arg, rhs_arg) if condition is None else (condition, lhs_arg, rhs_arg)
        )
        kwargs = None if alpha is None else {"alpha": alpha}
        delete_nodes = []
        with graph.inserting_before(first_node):
            new_pointwise_out = first_node.target(
                *torch.fx.map_arg(args, extract_tensors),
                **({} if kwargs is None else kwargs),
            )
            new_pointwise_tensor_meta = extract_meta(new_pointwise_out)
            new_pointwise_node = graph.call_function(first_node.target, args, kwargs)
            _transfer_meta(
                new_pointwise_node.meta, first_node, "mlu_combine_pointwise_src"
            )
            new_pointwise_node.meta.update(
                {"val": new_pointwise_out, "tensor_meta": new_pointwise_tensor_meta}
            )
            new_unbind_tensor = aten.unbind.int(new_pointwise_out, 0)
            new_unbind_tensor_meta = extract_meta(new_unbind_tensor)
            new_unbind_node = graph.call_function(
                aten.unbind.int, (new_pointwise_node, 0)
            )
            _transfer_meta(
                new_unbind_node.meta, first_node, "mlu_combine_pointwise_src"
            )
            new_unbind_node.meta.update(
                {"val": new_unbind_tensor, "tensor_meta": new_unbind_tensor_meta}
            )
            for node in subset:
                getitem_node = node.args[-1 if "lhs_arg" in match_kwargs else -2]
                assert (
                    getitem_node.target is operator.getitem
                ), "Old getitem node acquiration fail!"
                new_getitem_node = graph.call_function(
                    operator.getitem, (new_unbind_node, getitem_node.args[1])
                )
                node.replace_all_uses_with(new_getitem_node)
                new_getitem_node.meta.update(node.meta)
                delete_nodes.append(node)
                if len(getitem_node.users) == 1:
                    delete_nodes.append(getitem_node)
            for node in delete_nodes:
                graph.erase_node(node)
            if len(unbind_node.users) == 0:
                graph.erase_node(unbind_node)
        counter["mlu_combine_pointwise_src"] += 1
        self.is_hit = True


# Directly find target ops to prepare for candidate nodes generation,
# solve the problem that native 'get_fusion_candidates' stop search for
# other different key candidates when they are inputs of a candidate node
def mlu_apply_group_batch_fusion(graph: torch.fx.GraphModule, rule):
    stable_topological_sort(graph)  # type: ignore[arg-type]

    for target_op in rule.target_ops:
        candidate_dict: collections.defaultdict[
            Any, list[torch.fx.Node]
        ] = collections.defaultdict(list)
        for node in reversed(graph.find_nodes(op="call_function", target=target_op)):
            key = rule.match(node)
            if key is not None:
                candidate_nodes = candidate_dict[key]
                if node not in candidate_nodes:
                    candidate_nodes.append(node)

        for key, candidate_nodes in candidate_dict.items():
            if len(candidate_nodes) < rule.graph_search_options["min_fuse_set_size"]:
                continue

            for subset in find_independent_subset_greedy(
                candidate_nodes, rule.graph_search_options
            ):
                rule.fuse(graph, subset)


def group_batch_fusion_passes(graph, is_inference: bool):
    fusions = []
    if (is_inference and "combo_matmul_infer" not in config.skipped_fx_passes) or (
        not is_inference
        and (
            "combo_matmul_training" in config.enabled_fx_passes
            or "combo_matmul" in config.enabled_fx_passes
        )
    ):
        _option = graph_search_options.copy()
        _option["min_fuse_set_size"] = config.min_combine_mm_width
        # For better CNNL perf, apply mini, same_kn and group_kn strategies in order.
        fusions.append(
            ComboMatmul("mlu_combo_matmul_mini", graph_search_options=_option)
        )
        fusions.append(
            ComboMatmul("mlu_combo_matmul_same_kn", graph_search_options=_option)
        )
        fusions.append(
            ComboMatmul("mlu_combo_matmul_group_kn", graph_search_options=_option)
        )
    if is_inference and "combine_pointwise_src" not in config.skipped_fx_passes:
        _option = graph_search_options.copy()
        _option["min_fuse_set_size"] = config.min_combine_poi_width
        # Define two (lhs & rhs) rules to handle the case that both handsides args of a node can match.
        fusions.append(
            CombinePointwiseSrc(
                "mlu_lhs_combine_pointwise_src", graph_search_options=_option
            )
        )
        fusions.append(
            CombinePointwiseSrc(
                "mlu_rhs_combine_pointwise_src", graph_search_options=_option
            )
        )
    for rule in fusions:
        with GraphTransformObserver(
            graph.owning_module,
            rule.name,
        ):
            mlu_apply_group_batch_fusion(graph, rule)

    if len(fusions) > 0 and fusions[-1].is_hit:
        stable_topological_sort(graph)

    need_reorder = False
    for rule in fusions:
        # combo_matmul may break memory locality, so do native reorder_for_locality pass again.
        if rule.name.startswith("mlu_combo_matmul") and rule.is_hit:
            need_reorder = True
            break
    if need_reorder:
        reorder_for_locality(graph)
