from typing import Any, Callable
import torch
from torch._dynamo.utils import lazy_format_graph_code, set_locals_to_steal
from torch._dynamo.external_utils import (
    FakeCompiledAutogradEngine,
)
from torch._dynamo.compiled_autograd import (
    _disable,
    _graph_placeholders,
    snapshot_cudagraph_enabled,
)
from torch._logging import getArtifactLogger, trace_structured
from torch.fx import GraphModule

from ..utils import gorilla


compiled_autograd_log = getArtifactLogger(__name__, "compiled_autograd")
verbose_log = getArtifactLogger(__name__, "compiled_autograd_verbose")


# Note: [Compiled autograd and cudagraphs]
# Eager autograd backward implements scalars as 0-dim tensors, see DivBackward0::other_.
# When compiled autograd traces those nodes, it lifts the scalar tensors, resulting in a graph
# with some cpu 0-dim tensor inputs. To prevent the entire graph from skipping cudagraph, we move the
# scalars tensors to cuda. This works because ATen/prims ops will accept cuda 0-dim tensors too.
@gorilla.patch(torch._dynamo.compiled_autograd.AutogradCompilerInstance)
def move_graph_nodes_to_cuda(self, graph: torch.fx.Graph) -> list[int]:
    to_move: dict[int, torch.fx.Node] = {}
    has_cuda_inputs = False
    nodes = list(graph.nodes)
    assert nodes[0].target == "inputs"
    inputs = nodes[0]
    inputs_users = list(inputs.users.keys())
    # input access nodes should immediately follow placeholder nodes
    first_getitem_idx = len(_graph_placeholders)
    assert nodes[first_getitem_idx] == inputs_users[0]
    last_getitem_idx = first_getitem_idx + len(inputs_users) - 1
    assert nodes[last_getitem_idx] == inputs_users[-1]
    # getitem nodes on inputs
    for i, node in enumerate(inputs_users):
        # Modify by CAMBRICON
        # if not has_cuda_inputs and node.meta["val"].device.type == "cuda":
        if not has_cuda_inputs and node.meta["val"].device.type == "mlu":
            # end Modify by CAMBRICON
            has_cuda_inputs = True
            continue
        is_cpu = node.meta["val"].device.type == "cpu"
        is_scalar = len(node.meta["val"].size()) == 0
        if is_cpu and is_scalar:
            node_users = list(node.users.keys())
            # We can only move the cpu scalar if it is not exposed to user code.
            if all(
                (
                    isinstance(user.target, torch._ops.OpOverload)
                    and user.target.namespace in ("prims", "aten")
                )
                or (isinstance(user.target, Op) and not user.target.is_custom_function)
                for user in node_users
            ):
                # all users are prims/aten, can move safely
                to_move[i] = node
    # only move cpu scalars to cuda if there were cuda activations in this graph,
    # this is to handle the case where cudagraphs is enabled on a cpu-only graph
    if has_cuda_inputs:
        for node in to_move.values():
            # Modify by CAMBRICON
            # verbose_log.debug("Moving node %s from cpu to cuda", node)
            # node.meta["val"] = node.meta["val"].cuda()
            verbose_log.debug("Moving node %s from cpu to mlu", node)
            node.meta["val"] = node.meta["val"].mlu()
            # end Modify by CAMBRICON
        # return runtime indices we need to move to cuda
        return list(to_move.keys())
    return []


@gorilla.patch(torch._dynamo.compiled_autograd.AutogradCompilerInstance)
def end_capture(self, outputs: Any) -> tuple[Callable[..., Any], Any]:
    self.fx_tracer.create_proxy(
        "call_function",
        FakeCompiledAutogradEngine._exec_final_callbacks_stub,
        (),
        {},
    )
    self.stack.close()
    self.fx_tracer.create_node(
        "output",
        "output",
        (self.fx_tracer.create_arg(self.to_proxy(outputs)),),
        {},
    )
    runtime_inputs_to_move: list[int] = []
    if snapshot_cudagraph_enabled():
        runtime_inputs_to_move = self.move_graph_nodes_to_cuda(self.fx_tracer.graph)
    # We traced using dummy tensors. Delete all the metadata of the dummy tensors.
    # It's probably better to refactor this class to use a different tracer
    # than the make_fx tracer, but that is a larger change.
    for node in self.fx_tracer.graph.nodes:
        for field in ["tensor_meta", "example_value", "val"]:
            if field in node.meta:
                del node.meta[field]
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "compiled_autograd_graph_pre_reordering",
            "encoding": "string",
        },
        payload_fn=lambda: GraphModule(
            self.fx_tracer.root,
            self.fx_tracer.graph,
            f"CompiledAutograd{self.id}PreReordering",
        ).print_readable(print_output=False),
    )
    self.delay_unpack_hook_nodes()
    self.reorder_tensor_pre_hook_nodes()
    self.reorder_pre_hook_nodes_to_schedule_asap()
    self.reorder_accumulate_grad_nodes()
    self.reorder_pre_hook_nodes_to_mimic_eager()
    self.reorder_post_acc_grad_hook_nodes()
    self.reorder_post_hook_nodes()
    # TODO(yf225): work around: remove dead codes like `sym_size` and `sym_numel` which are not used downstream. e.g.
    # ```
    # sym_numel_default = torch.ops.aten.sym_numel.default(sum_109);  sum_109 = None
    # eq_115 = 16 == sym_numel_default;  sym_numel_default = eq_115 = None
    # sym_size_int_39 = torch.ops.aten.sym_size.int(getitem_112, 1);  getitem_112 = None
    # eq_116 = 16 == sym_size_int_39;  eq_116 = None
    # eq_117 = 16 == sym_size_int_39;  sym_size_int_39 = eq_117 = None
    # ```
    # Proper fix is Richard's Python compiled autograd effort which will avoid calling make_fx and
    # should prevent these ops from going into the CA graph.
    self.dce()
    if self.nan_checker:
        self.nan_checker.prep_with_graph(self.fx_tracer.graph)

    # keep only sizes that are actually used in the graph
    used_sizes_idx = self.remove_unused_sizes()

    graph = self.create_graph_module(f"CompiledAutograd{self.id}")
    set_locals_to_steal(graph, ["inputs"])
    lazy_graph_code = lazy_format_graph_code(
        "Compiled autograd graph",
        graph,
        include_device=True,
        include_stride=True,
        colored=True,
    )
    compiled_autograd_log.info("%s", lazy_graph_code)
    verbose_log.debug("%s", lazy_graph_code)
    trace_structured(
        "compiled_autograd_graph",
        payload_fn=lambda: graph.print_readable(print_output=False),
    )

    def runtime_wrapper(
        compiled_fn: Callable[..., Any],
        inputs: Any,
        sizes: Any,
        scalars: Any,
        hooks: Any,
        packed_inputs: Any,
    ) -> tuple[Any, Any]:
        global in_compiled_autograd_region
        try:
            in_compiled_autograd_region = True
            if self.nan_checker:
                self.nan_checker.prep_with_inputs(inputs)
            filtered_sizes = []
            for idx, integer in enumerate(sizes):
                if idx in used_sizes_idx:
                    # can't create negative size
                    if integer > 0:
                        filtered_sizes.append(torch.empty(0, integer))
                        torch._dynamo.maybe_mark_dynamic(filtered_sizes[-1], 1)
                    else:
                        filtered_sizes.append(integer)
            for i in runtime_inputs_to_move:
                # Modify by CAMBRICON
                # inputs[i] = inputs[i].pin_memory().cuda(non_blocking=True)
                inputs[i] = inputs[i].pin_memory().mlu(non_blocking=True)
                # end Modify by CAMBRICON
            with _disable(), make_compile_context(self.id):
                out = compiled_fn(inputs, filtered_sizes, scalars, hooks, packed_inputs)
                if self.nan_checker:
                    self.nan_checker.check(out)
                return out
        finally:
            in_compiled_autograd_region = False

    get_chromium_event_logger().log_event_end(
        "compiled_autograd",
        time.time_ns(),
        {"graph_id": self.id},
        self.start_time_ns,
        log_pt2_compile_event=True,
    )
    self.compile_context.__exit__(None, None, None)
    return runtime_wrapper, self.compiler_fn(graph)
