import functools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import sympy
from sympy import Expr, Integer
from typing_extensions import override

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.symbolic_shapes import (
    compute_unbacked_bindings,
    rebind_unbacked,
)
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor import config, dependencies
from torch._inductor.codegen.common import BackendFeature, index_prevent_reordering
from torch._inductor.runtime.hints import ReductionHint
from torch._inductor.dependencies import extract_input_node_reduction_ranges
from torch._inductor.ops_handler import ReductionType
from torch._inductor.utils import (
    is_gpu,
    sympy_product,
)
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.ir import (
    Buffer,
    ComputedBuffer,
    FlexibleLayout,
    IRNode,
    _IntLike,
    MutationOutput,
    MultiTemplateBuffer,
    ChoiceCaller,
    NoneLayout,
    Reduction,
    TensorBox,
    TritonTemplateBuffer,
    TritonTemplateCallerBase,
    fuse_reindexing,
    LoopBody,
    as_storage_and_layout,
    get_device_type,
    log,
    ir_node_to_tensor,
    is_storage_and_layout,
    FallbackKernel,
    _P,
    BaseView,
    ExternKernel,
    GeneratorState,
    Layout,
    TorchBindObject,
)
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.virtualized import OpsValue, V
from torch.utils._ordered_set import OrderedSet
from ..utils import gorilla


def simplify_and_reorder(
    self,
    extra_indexing_constraints: Optional[tuple[dict[Any, Any], list[Any]]] = None,
    recompute_sizes_body_func: Optional[Callable[..., Any]] = None,
) -> tuple[tuple[list[Expr], list[Expr]], Optional[LoopBody]]:
    """
    This is a main place where we do loop transformations in a
    backend-agnostic way.

    Here we:
        1) Remove any 1 dimensions
        2) Fuse contiguous dimensions together
        3) Reorder dimensions based on stride orders

    Optional argument extra_indexing_constraints can be used to append additional
    indexing expressions to existing ones derived from buffer's body. This can be useful
    to fuse scheduler nodes with compatible ranges, e.g. (s0*s1*...,) and (s0, s1, s2, ...)
    on CPU by preventing indexing simplifications and obtaining index/reduce ranges for
    the scheduler node compatible with other nodes.
    Optional argument recompute_sizes_body_func can be used to recompute sizes and body
    on the default body. This can be useful to append additional loop transformations.
    """
    (
        (index_size, reduce_size),
        body,
        (index_vars, reduce_vars),
    ) = self.get_default_sizes_body()

    if recompute_sizes_body_func:
        (
            (index_size, reduce_size),
            body,
            (index_vars, reduce_vars),
        ) = recompute_sizes_body_func(
            (index_size, reduce_size), body, (index_vars, reduce_vars)
        )

    index_formulas = [*body.indexing_exprs.values()]
    if extra_indexing_constraints is not None:
        assert (
            isinstance(extra_indexing_constraints, tuple)
            and len(extra_indexing_constraints) == 2
        )
        extra_indexing_ranges, extra_indexing_expr = extra_indexing_constraints
        assert isinstance(extra_indexing_ranges, dict), type(extra_indexing_ranges)
        assert isinstance(extra_indexing_expr, list), type(extra_indexing_expr)
        assert all(isinstance(f, Expr) for f in extra_indexing_expr)

        expected_var_ranges = body.var_ranges
        assert expected_var_ranges == extra_indexing_ranges, (
            expected_var_ranges,
            extra_indexing_ranges,
        )
        # remove already existing expressions
        extra_indexing_expr = [
            e for e in extra_indexing_expr if e not in index_formulas
        ]
        index_formulas += extra_indexing_expr

    memory_addrs = [*body.get_write_exprs()]
    if not V.graph.has_feature(self, BackendFeature.PREFER_STORE_LOOP_ORDER):
        memory_addrs.extend(body.get_read_exprs())

    def simplify_and_reorder(
        x_vars: Sequence[sympy.Symbol],
        support_vars: Sequence[sympy.Symbol],
        sizes: Sequence[int],
        simplify_loops: bool,
    ) -> tuple[
        list[int],
        Callable[[Sequence[int]], Sequence[int]],
        Callable[[Sequence[int]], Sequence[int]],
    ]:
        newsizes, reindex0, reindex1 = self._apply_loop_reordering(
            x_vars, support_vars, sizes, memory_addrs
        )
        # When using native matmul, the codegen assumes the following loop order,
        # regardless of the stride of A and B:
        #
        #   for z -> y -> x -> r:  C[z, y, x] += A[z, y, r] * B[z, r, x]
        # or
        #   for z -> x -> y -> r:  C[z, y, x] += A[z, y, r] * B[z, r, x]
        #
        # The critical point is the position of the "z" (batch) axis in bmm.
        # It is fine to swap the y and x axes (e.g., (z, y, x, r) or (z, x, y, r)),
        # but reordering the z axis (e.g., (y, x, z, r)) breaks codegen.
        #
        # Therefore, if loop reordering changes the "z" location in bmm,
        # it should be reverted to the default.
        # This may not always produce the optimal loop order when strides
        # do not align with the default assumption.
        #
        # TODO: Consider extending tl.dot codegen to support arbitrary loop orders.
        if self.get_reduction_type() == "dot" and len(sizes) == 3:
            order = list(range(len(sizes)))  # default order

            # if z axis is not the outermost, use the default reorder.
            if reindex0(order)[0] != 0:
                newsizes = [sizes[i] for i in order]
                reindex0 = same_reorder(order)
                reindex1 = inverse_reorder(order)

        # for NHWC: reindex0([0,1,2,3]) = [0,2,3,1], reindex1([0,1,2,3]) = [0,3,2,1]
        x_vars = reindex0(x_vars)

        if simplify_loops:
            newsizes, reindex2, _prune = V.graph.sizevars._simplify_loops(
                x_vars,
                newsizes,
                index_prevent_reordering(index_formulas, x_vars, newsizes),
            )
            reindex = fuse_reindexing(reindex1, reindex2)
        else:
            reindex = reindex1
        return newsizes, reindex, reindex1

    support_vars = index_vars + reduce_vars
    should_merge_loops = (
        not is_gpu(get_device_type(self)) or not config.loop_ordering_after_fusion
    )
    iter_ranges, iter_reindex, _ = simplify_and_reorder(
        index_vars,
        support_vars,
        index_size,
        should_merge_loops,
    )

    # Like iteration dimensions, we may also want to delay merging reduction dimensions.
    # E.g., if we reduce a tensor [M, N, K] for its M and N dimensions followed by a pointwise
    # kernel, merging M and N dimension too early makes it hard to decide what loop order
    # we should pick for the piontwise kernel so that it is fusible with the reduction.
    reduce_ranges, reduce_reindex, _ = simplify_and_reorder(
        reduce_vars, support_vars, reduce_size, should_merge_loops
    )

    # retrace the loop body with simplification and reordering applied
    (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
        # Modify by CAMBRICON
        # change the prefix from 'z' to 'w'.
        iter_ranges,
        reduce_ranges,
        # prefix="p",
        prefix="w",
        # end Modify by CAMBRICON
    )
    body = LoopBody(
        body,
        [iter_reindex(iter_vars), reduce_reindex(reduce_vars)],
        var_ranges,
        iter_vars,
        reduce_vars,
    )
    return (iter_ranges, reduce_ranges), body


patch = gorilla.Patch(
    torch._inductor.ir.ComputedBuffer, "simplify_and_reorder", simplify_and_reorder
)
gorilla.apply(patch)


def __init__(  # type: ignore[no-untyped-def]
    self,
    layout,
    inputs,
    make_kernel_render,
    mutated_inputs: Optional[Iterable[IRNode]] = None,
    allowed_prologue_inps: Optional[OrderedSet[str]] = None,
) -> None:
    """
    NOTE:[TritonTemplates with multiple outputs]
    We want the ability for TritonTemplates to output multiple tensors. Triton
    kernels have no notion of outputs and this is done by creating tensors that
    are then mutated by the kernel. Currently our STORE_OUTPUT codegen doesn't
    support creating multinode outputs for triton templates.
    We work around this by creating an extra input buffer during the lowering
    and we mark them as mutated inputs.
    """
    super(TritonTemplateBuffer, self).__init__(layout, inputs, make_kernel_render)
    self.mutated_inputs = mutated_inputs
    self.outputs: list[Buffer] = [self]
    if mutated_inputs is not None:
        # Ensure that the mutated inputs are only allowed for certain nodes
        # Modify by CAMBRICON: append aten.native_group_norm to allowed_set
        allowed_set = (
            torch.ops.higher_order.flex_attention,
            torch.ops.higher_order.flex_attention_backward,
            torch.ops.aten.native_group_norm.default,
        )
        current_node = V.graph.current_node.target
        assert current_node in allowed_set or (
            "triton_fusion_tuned" in str(current_node)
        ), f"Mutated inputs are only allowed for {allowed_set} or triton_fusion_tuned functions but got {current_node}"
        device = self.inputs[0].get_device()
        self.outputs += [
            MutationOutput(NoneLayout(device=device), buf, self)
            for buf in mutated_inputs
        ]

    self.allowed_prologue_inps = (
        allowed_prologue_inps if allowed_prologue_inps else OrderedSet()
    )

    self.subgraph_inps: Optional[list[Optional[Union[IRNode, sympy.Expr]]]] = None
    self.subgraph_outs: Optional[list[Optional[IRNode]]] = None


torch._inductor.ir.TritonTemplateBuffer.__init__ = __init__


def __init__(
    self,
    layout: Layout,
    inputs: Sequence[IRNode],
    choice_timings_fn: Callable[[Optional[int]], dict[ChoiceCaller, float]],
    unfiltered_choices: list[ChoiceCaller],
    allowed_prologue_inps: OrderedSet[str],
) -> None:
    # Add by CAMBRICON
    mutated_inputs = []
    for c in unfiltered_choices:
        if not hasattr(c, "mutated_inputs"):
            continue
        if not c.mutated_inputs:
            continue
        for mi in c.mutated_inputs:
            if mi not in mutated_inputs:
                mutated_inputs.append(mi)
    # end Add by CAMBRICON
    # Modify by CAMBRICON
    # super().__init__(
    #     layout=layout,
    #     inputs=inputs,
    #     make_kernel_render=None,
    #     allowed_prologue_inps=allowed_prologue_inps,
    # )
    super(MultiTemplateBuffer, self).__init__(
        layout=layout,
        inputs=inputs,
        make_kernel_render=None,
        # make_kernel_render=None,
        mutated_inputs=mutated_inputs or None,
        allowed_prologue_inps=allowed_prologue_inps,
    )
    # end Modify by CAMBRICON
    self._choice_timings_fn = choice_timings_fn
    self._choice_timings: dict[Optional[int], dict[ChoiceCaller, float]] = {}
    self._choices: list[ChoiceCaller] = unfiltered_choices
    self.original_inputs = inputs
    self._output_plannable = all(
        isinstance(choice, TritonTemplateCallerBase)
        or (
            isinstance(choice, torch._inductor.select_algorithm.ExternKernelCaller)
            and choice.has_out_variant
        )
        for choice in unfiltered_choices
    )
    self._make_kernel_renders: dict[Optional[int], Any] = {}


patch = gorilla.Patch(
    torch._inductor.ir.MultiTemplateBuffer,
    "__init__",
    __init__,
    settings=gorilla.Settings(use_replace_references=True),
)
gorilla.apply(patch)


@staticmethod
def num_splits(
    device: torch.device,
    dst_dtype: torch.dtype,
    src_dtype: torch.dtype,
    inner_fn: Callable[_P, OpsValue],
    ranges: Sequence[_IntLike],
    reduction_ranges: Sequence[_IntLike],
    reduction_type: Union[ReductionType, Literal["scan"]],
    reduction_numel: Expr,
    input_node: Optional[IRNode] = None,
) -> tuple[ReductionHint, _IntLike]:
    reduction_numel_hint = V.graph.sizevars.symbolic_hint(reduction_numel)
    numel_hint = V.graph.sizevars.symbolic_hint(sympy_product(ranges))

    should_split = reduction_type == "scan" or (
        not V.graph.has_feature(device, BackendFeature.REDUCE_TO_SINGLE_ELEMENT)
        and reduction_type
        not in (
            "argmax",
            "argmin",
        )
        and config.split_reductions
    )
    if not (_is_static(reduction_numel_hint) and _is_static(numel_hint)):
        # We don't support unbacked symints
        return ReductionHint.DEFAULT, 1

    if reduction_type == "dot":
        # Don't split when doing native matmul
        return ReductionHint.DEFAULT, 1

    # Modify by CAMBRICON
    # FIXME(need to be deleted): TODO(miaochen) workaround
    if not should_split:
        return ReductionHint.DEFAULT, 1

    # end Modify by CAMBRICON

    props = DeviceProperties.create(device)
    num_sm = props.multi_processor_count
    min_elements_per_thread = 32
    if should_split:
        inner_reduction_splits: Callable[[int, int], int] = functools.partial(
            V.choices.reduction_split_factor, device, inner_reduction=True
        )
        outer_reduction_splits: Callable[[int, int], int] = functools.partial(
            V.choices.reduction_split_factor, device, inner_reduction=False
        )
    else:

        def inner_reduction_splits(
            reduction_numel_hint: int,
            numel_hint: int,
        ) -> int:
            return 1

        outer_reduction_splits = inner_reduction_splits

    # easy cases
    if numel_hint == 1:
        split = inner_reduction_splits(reduction_numel_hint, numel_hint)
        if split == 1:
            # No need to split.
            return ReductionHint.INNER, split
        if input_node is not None and isinstance(input_node, TensorBox):
            with patch.object(FlexibleLayout, "allow_indexing", True):
                (
                    new_ranges,
                    new_reduction_ranges,
                ) = extract_input_node_reduction_ranges(input_node)
            if new_ranges is not None and new_reduction_ranges is not None:
                extracted_numel_hint = V.graph.sizevars.symbolic_hint(
                    sympy_product(new_ranges + new_reduction_ranges)
                )
                if reduction_numel_hint == extracted_numel_hint:
                    log.debug(
                        "Use previous IRNode's range and reduction_ranges instead of split. "
                        "current ranges: %s, current reduction ranges: %s, current split: %d, "
                        "new ranges: %s, new reduction ranges: %s",
                        ranges,
                        reduction_ranges,
                        split,
                        new_ranges,
                        new_reduction_ranges,
                    )
                    # If the input_node or its dependent nodes are also Reduction nodes,
                    # use reduction_sizes of this node or its dependent nodes directly.
                    return ReductionHint.INNER, -1
        return ReductionHint.INNER, split
    # Modify by CAMBRICON
    # if reduction_numel_hint <= min_elements_per_thread or numel_hint >= num_sm * 2 * 32:
    if reduction_numel_hint <= 8192:
        return ReductionHint.DEFAULT, 1
    # end Modify by CAMBRICON

    r = Reduction(
        device=device,
        dtype=dst_dtype,
        inner_fn=inner_fn,
        ranges=ranges,
        reduction_ranges=reduction_ranges,
        reduction_type=reduction_type if reduction_type != "scan" else "sum",
        src_dtype=src_dtype,
        reduction_hint=ReductionHint.DEFAULT,
    )

    def get_read_indices(r: Reduction) -> tuple[Sequence[Expr], bool]:
        device = r.get_device()
        assert device is not None
        cb = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=device,
                dtype=r.get_dtype(),
                size=r.get_size(),
            ),
            data=r,
        )
        read_writes = cb.get_read_writes()
        # try finding the full size producer
        # TODO this will fail for something like ((1, N) * (N, 1)).sum()
        # this would also possibly be wrong for producers with the different contiguity but we hope those cases are rare
        assert read_writes.range_vars is not None
        range_vars = [
            r
            for r in read_writes.range_vars
            if isinstance(r, Expr) and not isinstance(r, sympy.Number)
        ]
        indices = []
        changed = False
        for md in sorted(read_writes.reads, key=lambda x: x.name):
            if all(r in md.index.free_symbols for r in range_vars):
                indices.append(md.index)
                if md.name in V.graph.name_to_buffer:
                    buf = V.graph.name_to_buffer[md.name]
                    original_stride = getattr(buf.layout, "stride", None)
                    buf.decide_layout()
                    if getattr(buf.layout, "stride", None) != original_stride:
                        changed = True
        return indices, changed

    indices, changed = get_read_indices(r)
    if changed:
        indices, _ = get_read_indices(r)

    if len(indices) == 0:
        # TODO determine splits when all inputs are broadcast
        return ReductionHint.DEFAULT, 1

    (_, reduction_vars), ranges1 = dependencies.index_vars_squeeze(
        r.get_size(), r.get_reduction_size()
    )
    num_outer = 0
    num_inner = 0
    for i in indices:
        j = V.graph.sizevars.simplify_with_ranges(i, ranges1)
        strides = V.graph.sizevars.stride_hints(j, reduction_vars, list(ranges1.keys()))
        # A 0 stride does not make a reduction contiguous.
        # This can happen when the reduction ranges contains a 1.
        outer = all(s == 0 or s > 1 for s in strides)
        if outer:
            num_outer += 1
        else:
            num_inner += 1
    if num_inner > num_outer:
        return ReductionHint.INNER, inner_reduction_splits(
            reduction_numel_hint, numel_hint
        )
    else:
        return ReductionHint.OUTER, outer_reduction_splits(
            reduction_numel_hint, numel_hint
        )


patch = gorilla.Patch(torch._inductor.ir.Reduction, "num_splits", num_splits)
gorilla.apply(patch)


@override
def codegen(self, wrapper: PythonWrapperCodegen) -> None:
    """Overrides the parent member.
    See https://github.com/pytorch/pytorch/issues/151692"""
    kernel = self.op_overload
    assert kernel is not None
    # Modify by Cambricon
    # if kernel.namespace == "aten":
    if kernel.namespace == "aten" or kernel.namespace == "torch_mlu":
        # end Modify by Cambricon
        # Aten Fallback Ops
        assert isinstance(kernel, torch._ops.OpOverload), type(kernel)
        # Modify by Cambricon: update inductor_fallback_ops.
        if V.graph.cpp_wrapper:
            if wrapper.device == "mlu":
                from torch_mlu._inductor.codegen.aoti.fallback_ops import (
                    inductor_fallback_ops,
                )
            else:
                from torchgen.aoti.fallback_ops import inductor_fallback_ops
            # end Modify by Cambricon

            # Modify by Cambricon: FallbackKernel's device is cpu and wrapper_code's device isn't cpu,
            # kernel will use runtime dispatch.
            # if str(kernel) not in inductor_fallback_ops:
            if str(kernel) not in inductor_fallback_ops or (
                self.get_device().type == "cpu" and V.graph.wrapper_code.device != "cpu"
            ):
                # end Modify by Cambricon
                # C shim v2 is torchgen-ed, which should cover all aten ops.
                # If you do hit a missed op, please update fallback_ops.py.
                log.warning(
                    "%s is missing a c-shim implementation, using proxy executor as fallback",
                    kernel,
                )
                self.use_runtime_dispatch = True
    elif kernel.namespace == "_quantized":
        # Internal Quantized Fallback Ops
        assert isinstance(kernel, torch._ops.OpOverload), type(kernel)
    elif V.graph.cpp_wrapper:
        # Add by CAMBRICON
        # custom op of third-party library support FallbackKernel.
        import torch_mlu

        if len(torch_mlu._inductor.config.aot_inductor.custom_ops_to_c_shims) > 0:
            torch_mlu._inductor.config._warn_custom_op_config()

        custom_ops_to_c_shims = {}
        custom_ops_to_c_shims.update(
            torch_mlu._inductor.config.aot_inductor.custom_ops_to_c_shims
        )
        custom_ops_to_c_shims.update(
            torch._inductor.config.aot_inductor.custom_ops_to_c_shims
        )
        # end Add by CAMBRICON
        # For non-aten OpOverload, i.e. custom ops
        # If the op is in custom_ops_to_c_shims, generate direct function call
        # Modify by CAMBRICON
        # self.use_runtime_dispatch = kernel not in config.aot_inductor.custom_ops_to_c_shims
        self.use_runtime_dispatch = (
            "torch.ops." + str(kernel) not in custom_ops_to_c_shims
        )
        # end Modify by CAMBRICON

    # Handle the special case where a complex number is input to a C-shim kernel for
    # a scalar input.  The torchgen'ed shim API will use type "double", which is
    # incompatible with complex numbers, forcing a fallback to runtime dispatch.
    if (
        V.graph.cpp_wrapper
        and isinstance(kernel, torch._ops.OpOverload)
        and not self.use_runtime_dispatch
    ):
        # Modify by Cambricon
        # def is_number(t: torch.JitType) -> bool:
        def is_number(t) -> bool:
            if isinstance(t, torch.OptionalType):
                return is_number(t.getElementType())
            return isinstance(t, torch.NumberType)

        # end Modify by Cambricon

        # Using unflatten_args is a bit of a hack, but all the complex arguments we
        # care about are in self.constant_args, and calling unflatten_args puts them
        # in the correct order without triggering codegen.
        args, kwargs = self.unflatten_args(self.inputs, self.constant_args)
        # Append kwarg values to args.  ordered_kwargs_for_cpp_kernel is guaranteed
        # to be set, since this is an OpOverload kernel.
        args_iter = itertools.chain(
            args,
            (
                self.get_kwargs_value(k, **kwargs)
                for k in self.ordered_kwargs_for_cpp_kernel
            ),
        )
        self.use_runtime_dispatch = any(
            isinstance(v, complex) and is_number(a.real_type)
            for v, a in zip(args_iter, kernel._schema.arguments)
        )

    self.codegen_comment(wrapper)
    if self.use_runtime_dispatch:
        exported_args = self.export_extern_kernel_node()
        assert self.python_kernel_name is not None
        assert self.op_overload is not None

        wrapper.generate_fallback_kernel_with_runtime_lookup(
            self.get_name(),
            self.python_kernel_name,
            lambda: [*self.codegen_args(), *self.codegen_kwargs()],
            self.op_overload,
            exported_args,
            # NOTE: [special handling of all_reduce_coalesced_'s return value]
            self.outputs if self.outputs else self.mutation_outputs,
        )
    else:
        wrapper.generate_fallback_kernel(self)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)
            self.codegen_alignment_asserts(wrapper)
            self.codegen_memory_tracking(wrapper)

    self.codegen_unbacked_symbol_defs(wrapper)


patch = gorilla.Patch(torch._inductor.ir.FallbackKernel, "codegen", codegen)
gorilla.apply(patch)


@classmethod
def process_kernel(  # type: ignore[no-untyped-def]
    cls, kernel, *args, **kwargs
) -> tuple[
    Any,
    list[Any],
    list[Any],
    Callable[[Any, Any], Any],
    Optional[dict[sympy.Symbol, pytree.KeyPath]],
]:
    binded_args = {"args": args, "kwargs": kwargs}

    args_flat, args_spec = pytree.tree_flatten(binded_args)

    is_arg_tensor = []
    # tensor_args can be either tensor or torchbind objects
    tensor_args = []
    non_tensor_args: list[Any] = []
    for arg in args_flat:
        is_arg_tensor.append(
            isinstance(arg, IRNode) and not isinstance(arg, GeneratorState)
        )
        if is_arg_tensor[-1]:
            tensor_args.append(arg)
        else:
            if isinstance(arg, sympy.Expr):
                arg = V.graph.sizevars.shape_env.create_symintnode(arg, hint=None)
            non_tensor_args.append(arg)

    def unflatten_args(new_tensor_args, new_non_tensor_args):  # type: ignore[no-untyped-def]
        result = []
        it_tensors = iter(new_tensor_args)
        it_non_tensors = iter(new_non_tensor_args)
        for is_tensor in is_arg_tensor:
            if is_tensor:
                result.append(next(it_tensors))
            else:
                result.append(next(it_non_tensors))
        r = pytree.tree_unflatten(result, args_spec)
        return r.get("args", []), r.get("kwargs", {})

    # Modified by Cambricon: add require_input_contiguous op list
    require_input_contiguous_ops = [
        "torch_mlu_ops::batch_matmul",
    ]
    if kernel._name in require_input_contiguous_ops:
        tensor_args = [ExternKernel.require_contiguous(x) for x in tensor_args]
    # end Modify by Cambricon
    tensor_args = [cls.realize_input(x) for x in tensor_args]

    # freeze layout otherwise our output stride calculation might
    # become incorrect
    for x in tensor_args:
        if is_storage_and_layout(x):
            as_storage_and_layout(x, freeze=True)

    # Rerun fake tensor propagation, because Inductor may have changed the
    # strides of inputs and we need to determine accurately what the
    # output stride will be.
    example_args: list[Union[torch.Tensor, torch._C.ScriptObject, torch.Generator]] = []

    # We need to retain the constant values of fake tensors that we originally
    # propagated the graph with, because for some operators running without a
    # constant would trigger an error / DataDependentException
    for x in tensor_args:
        # if x is a view of a constant, we need to realize the view
        # (we can't pass the constant into the kernel directly)
        if not isinstance(x, BaseView) and x.get_name() in V.graph.constants:
            example_args.append(V.graph.constants[x.get_name()])
        elif (
            not isinstance(x, BaseView) and x.get_name() in V.graph.torchbind_constants
        ):
            example_args.append(V.graph.torchbind_constants[x.get_name()])
        elif isinstance(x, TorchBindObject):
            example_args.append(x.get_real_obj())
        elif isinstance(x, torch._inductor.ir.GeneratorState):
            device_index = x.device.index
            assert x.device.type == "cuda" and device_index is not None
            example_args.append(
                torch.cuda.default_generators[device_index].clone_state()
            )
        else:
            example_args.append(ir_node_to_tensor(x, guard_shape=True))

    new_args, new_kwargs = unflatten_args(example_args, non_tensor_args)
    # Modify by CAMBRICON
    # example_output = kernel(*new_args, **new_kwargs)
    from torch._dispatch.python import enable_python_dispatcher

    # When new_args is a FakeTensor with device='meta', the corresponding meta function
    # (registered in torch/_meta_registrations.py) should be invoked.
    # If python_dispatcher is disabled, certain operators may be dispatched to AutogradPrivateUse1,
    # triggering the native operator implementation. This can result in a mismatch between
    # the statically inferred shape and the shape produced during actual execution.

    with enable_python_dispatcher():
        example_output = kernel(*new_args, **new_kwargs)
    # end Modify by CAMBRICON

    unbacked_bindings: Optional[dict[sympy.Symbol, pytree.KeyPath]] = None
    if shape_env := V.fake_mode.shape_env:
        rebind_unbacked(shape_env, V.current_node, example_output)
        unbacked_bindings = compute_unbacked_bindings(
            shape_env, example_output, V.current_node.meta.get("val")
        )

    example_out_li = (
        [example_output]
        if not isinstance(example_output, (list, tuple))
        else example_output
    )
    for t in example_out_li:
        if isinstance(t, torch.Tensor) and t.is_sparse:
            msg = (
                "sparsity not handled. Please file issue for sparse inference weights."
            )
            if stack_trace := V.graph.current_node.meta.get("stack_trace", None):
                msg = f"{msg} Found from : \n {stack_trace}"
            V.graph.disable_cudagraphs_reason = msg

    return (
        example_output,
        tensor_args,
        non_tensor_args,
        unflatten_args,
        unbacked_bindings,
    )


patch = gorilla.Patch(torch._inductor.ir.ExternKernel, "process_kernel", process_kernel)
gorilla.apply(patch)


def is_triton(x: Union[IRNode, torch.device, None, str]) -> bool:
    device = get_device_type(x)
    # Special case cpu and cuda as using the method below
    # to determine if the scheduler is a triton scheduler subclass
    # requires instantiating a scheduler for them
    # Modify by CAMBRICON
    import torch_mlu._inductor.config as mlu_config

    if device == "mlu":
        if getattr(mlu_config, f"{device}_backend") == "triton":
            return True
        return False
        # end Modify by CAMBRICON

    if device in ["cpu", "cuda"]:
        if getattr(config, f"{device}_backend") == "triton":
            return True
        return False
    if (
        device is None
        or (device_scheduling := get_scheduling_for_device(device)) is None
    ):
        return False
    from .codegen.triton import TritonScheduling

    assert isinstance(device_scheduling, type), type(device_scheduling)
    return issubclass(device_scheduling, TritonScheduling)


patch = gorilla.Patch(torch._inductor.ir, "is_triton", is_triton)
gorilla.apply(patch)
