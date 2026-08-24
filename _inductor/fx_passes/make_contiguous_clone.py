from collections import deque
import operator
from typing import (
    Callable,
    Optional,
)
from torch.fx.node import map_aggregate
from torch.fx.passes.shape_prop import _extract_tensor_metadata
import torch

from .utils import extract_meta, is_tmo_avaiable, is_tmo_matmul_available

aten = torch.ops.aten


def is_contiguous(node: torch.fx.Node):
    val = node.meta.get("val")
    if isinstance(val, torch._subclasses.fake_tensor.FakeTensor):
        # skip symbolic size such as torch.Size([u0, 512])
        if val._has_symbolic_sizes_strides:
            return True
        else:
            return val.is_contiguous()
    return True


def extract_inputs(val):
    if isinstance(val, (list, tuple)):
        return val.__class__([extract_inputs(x) for x in val])
    elif isinstance(val, dict):
        return {k: extract_inputs(v) for k, v in val.items()}
    elif isinstance(val, torch.fx.Node):
        return val.meta["val"]
    else:
        return val


def is_same_stride(a, b):
    assert isinstance(a, type(b)), f"must be same type, but {type(a)} and {type(b)}"
    if isinstance(a, (list, tuple, torch.fx.immutable_collections.immutable_list)):
        return all(is_same_stride(ai, bi) for ai, bi in zip(a, b))
    elif isinstance(a, (dict, torch.fx.immutable_collections.immutable_dict)):
        return all(is_same_stride(v, b[k]) for k, v in a.items())
    elif isinstance(a, torch._subclasses.fake_tensor.FakeTensor):
        return a.stride() == b.stride()
    else:
        return False


def extract_meta(objs):
    def extract_tensor_meta(obj):
        if isinstance(obj, torch.Tensor):
            return _extract_tensor_metadata(obj)
        else:
            return obj

    return map_aggregate(objs, extract_tensor_meta)


def propagate_stride(node):
    # after invoke replace_input_with(node, clone_node)
    node_chain = deque([node])
    while node_chain:
        user = node_chain.popleft()
        if user.op == "output":
            continue

        fake_args = extract_inputs(user.args)
        fake_kwargs = extract_inputs(user.kwargs)
        fake_out = user.target(*fake_args, **fake_kwargs)
        if user.meta.get("val", None) is None and len(user.users) == 0:
            continue
        fake_out = (
            type(user.meta["val"])(fake_out)
            if not isinstance(fake_out, torch._subclasses.fake_tensor.FakeTensor)
            else fake_out
        )

        if is_same_stride(fake_out, user.meta["val"]):
            continue
        else:
            tensor_meta = extract_meta(fake_out)
            user.meta["val"] = fake_out
            user.meta["tensor_meta"] = tensor_meta
            node_chain.extend(user.users)


def is_pointwise_use(
    use, is_pointwise_fn: Optional[Callable[[torch._ops.OpOverload], bool]] = None
) -> bool:
    """
    Do all uses of this op have torch.Tag.pointwise or return True for optional `is_pointwise_fn`

    Uses in views ops will follow the views uses
    """
    from torch._inductor.utils import is_view

    # Modified by Cambricon: also considered output as view ops
    if use.op == "output":
        return True

    if not use.op == "call_function":
        return False

    if not (
        isinstance(use.target, torch._ops.OpOverload) or use.target is operator.getitem
    ):
        return False

    if use.target is operator.getitem or is_view(use.target):
        return all(is_pointwise_use(u, is_pointwise_fn) for u in use.users)

    return torch.Tag.pointwise in use.target.tags or (
        is_pointwise_fn is not None and is_pointwise_fn(use.target)
    )


def is_input_output_stride_consistent(node: torch.fx.Node) -> bool:
    """
    Determine whether the input and output tensors of a node have consistent strides.

    Logic:
        - If inputs include broadcasting (dim mismatch, stride difference), return False.
        - If output stride is inconsistent with dominant input stride, return False.
        - Small or scalar inputs are ignored in layout checks.

    Returns:
        True if all strides are consistent or scalar, else False.
    """
    from torch._prims_common import NumberType

    if not isinstance(node, torch.fx.Node):
        return True

    if "val" in node.meta and not isinstance(
        node.meta["val"], torch._subclasses.fake_tensor.FakeTensor
    ):
        return True

    input_strides = []

    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            t = arg.meta["val"]
            if not isinstance(t, torch._subclasses.fake_tensor.FakeTensor):
                return True
            input_strides.append(arg.meta["val"].stride())
        elif isinstance(arg, NumberType):
            input_strides.append(tuple())  # Scalars have no stride
        else:
            return True  # Unknown type; assume consistent

    if not input_strides:
        return True

    # Use the input with the most dimensions as the reference.
    # This helps handle partial broadcasting cases where some inputs have fewer dims.
    reference_stride = max(input_strides, key=len)
    output_stride = node.meta["val"].stride()

    # Ops that the input dims is different from the output dims
    # Small tensors (<=2D), where layout mismatches have minimal impact
    if len(reference_stride) != len(output_stride) or len(reference_stride) <= 2:
        return True

    # skip when input maybe broadcast
    for stride in input_strides:
        if stride and len(stride) != len(reference_stride):
            return True

    # If there are at least two inputs with mismatched strides of the same rank,
    # will add aten.clone after current node.
    for stride in input_strides:
        if not stride:
            continue
        if stride != reference_stride:  # and min(stride) != 1:
            return False

    # check output stride with input stride
    if reference_stride != output_stride:
        return False

    return True  # All strides match: layout consistent


def make_contiguous_clone(graph: torch.fx.Graph):
    """
    This pass inserts `aten.clone` nodes in the FX graph when a non-contiguous
    tensor is used in pointwise operations and the input/output strides are inconsistent.

    Purpose:
        Ensure that inputs to pointwise operations are laid out contiguously in memory.
        This avoids implicit transposes or inefficient memory access in the generated
        Triton kernels, thereby improving performance.

    Conditions:
        - The tensor is not already contiguous.
        - The operation is a pointwise op (or chain of view -> pointwise).
        - The operation is not a fallback op, clone op, or triton wrapper.
        - Input and output strides differ (layout mismatch).
        - Output node is not directly affected (to avoid unnecessary clones).
    """
    # TODO,
    # use torch._inductor.virtualized.V.set_real_inputs to record example_inputs
    # use V.get_real_inputs get example.inputs
    # use FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(*example_inputs)
    #     to get output of each node
    # update output_node info like FakeTensorProp.run_node
    from torch._higher_order_ops.triton_kernel_wrap import (
        triton_kernel_wrapper_functional,
    )
    from torch._inductor.lowering import fallbacks

    has_insert_clone = False

    skip_ops = [
        aten.clone,
        aten.clone.default,
        triton_kernel_wrapper_functional,
        aten.bmm.default,
        aten.mm.default,
        aten.addmm.default,
        aten.addbmm.default,
        torch.ops.torch_mlu.grouped_gemm.default,
        torch.ops.torch_mlu.fused_mm.default,
        torch.ops.torch_mlu.fused_bmm.default,
        torch.ops.torch_mlu.fused_convolution.default,
    ]
    if is_tmo_avaiable():
        skip_ops.append(torch.ops.torch_mlu_ops.batch_matmul.default)
    if is_tmo_matmul_available():
        skip_ops.append(torch.ops.torch_mlu_ops.matmul.default)

    for node in graph.nodes:
        # Skip aten.expand, which are expected to be fused
        if is_contiguous(node) or (node.target in {aten.expand, aten.expand.default}):
            continue

        users_to_update = []
        for user in node.users:
            if (
                user.target not in fallbacks
                and user.target not in skip_ops
                and user.op != "output"
                # and not is_contiguous(user)
                and not is_input_output_stride_consistent(user)
            ):
                users_to_update.append(user)

        if users_to_update:
            has_insert_clone = True

            # Insert a contiguous clone after the current node
            with graph.inserting_after(node):
                clone_fake_tensor = aten.clone.default(
                    node.meta["val"], memory_format=torch.contiguous_format
                )
                clone_tensor_meta = extract_meta(clone_fake_tensor)
                clone_node = graph.call_function(
                    aten.clone.default,
                    (node,),
                    {"memory_format": torch.contiguous_format},
                )
                # Add tensor info for new node
                clone_node.meta = {
                    "val": clone_fake_tensor,
                    "tensor_meta": clone_tensor_meta,
                }

            for user in users_to_update:
                user.replace_input_with(node, clone_node)
                propagate_stride(user)

    # update output of graph
    output = list(graph.nodes)[-1]
    if has_insert_clone and output.op == "output":
        out_meta_val = output.meta["val"]
        new_meta_val = [
            arg.meta["val"] if isinstance(arg, torch.fx.Node) else arg
            for arg in output.args[0]
        ]
        output.meta["val"] = tuple(new_meta_val)
        for idx in range(len(output.meta["original_output_strides"])):
            if output.meta["original_output_strides"][idx] is not None:
                output.meta["original_output_strides"][idx] = output.meta["val"][
                    idx
                ].stride()
