import torch
from ...utils import gorilla


@gorilla.patch(torch.fx.passes.graph_transform_observer.GraphTransformObserver)
def __enter__(self):
    if not self.active:
        return self
    self.gm._register_create_node_hook(self._node_creation_hook)
    self.gm._register_erase_node_hook(self._node_erase_hook)
    self.gm._register_replace_node_hook(self._node_replace_hook)
    self.gm._register_deepcopy_hook(self._deepcopy_hook)

    self.erased_nodes.clear()
    self.created_nodes.clear()
    self.name_to_node.clear()
    self.copied_gms.clear()

    for node in self.gm.graph.nodes:
        self.name_to_node[node.name] = node

    # Add by CAMBRICON
    if self.log_url is not None:
        self.orig_gm_code = self.gm.print_readable(
            print_output=False, include_stride=True, include_device=True
        )
    # end Add by CAMBRICON
    return self


@gorilla.patch(torch.fx.passes.graph_transform_observer.GraphTransformObserver)
def __exit__(self, type, value, tb):
    if not self.active:
        return
    for gm in self.copied_gms + [self.gm]:
        gm._unregister_create_node_hook(self._node_creation_hook)
        gm._unregister_erase_node_hook(self._node_erase_hook)
        gm._unregister_replace_node_hook(self._node_replace_hook)
        gm._unregister_deepcopy_hook(self._deepcopy_hook)

    if self.log_url is None:
        return

    if len(self.created_nodes) > 0 or len(self.erased_nodes) > 0:
        for e in self.input_dot_graph.get_node_list():
            if e.get_name() in self.erased_nodes:
                e.obj_dict["attributes"]["fillcolor"] = "yellow"
            else:
                e.obj_dict["attributes"]["fillcolor"] = "grey"
        if self.log_url is None:
            raise AssertionError("log_url is not set")
        self.input_dot_graph.write(
            os.path.join(
                self.log_url,
                # Modify by CAMBRICON
                # f"pass_{GraphTransformObserver.__pass_count}_{self.passname}_input_graph.dot",
                f"pass_{GraphTransformObserver.get_current_pass_count()}_{self.passname}_input_graph.dot",
                # end Modify by CAMBRICON
            )
        )

        output_dot_graph = FxGraphDrawer(
            self.gm,
            self.passname,
            ignore_getattr=True,
            ignore_parameters_and_buffers=True,
        ).get_dot_graph()
        for e in output_dot_graph.get_node_list():
            if e.get_name() in self.created_nodes:
                e.obj_dict["attributes"]["fillcolor"] = "yellow"
            else:
                e.obj_dict["attributes"]["fillcolor"] = "grey"
        output_dot_graph.write(
            os.path.join(
                self.log_url,
                # Modify by CAMBRICON
                # f"pass_{GraphTransformObserver.__pass_count}_{self.passname}_output_graph.dot",
                f"pass_{GraphTransformObserver.get_current_pass_count()}_{self.passname}_output_graph.dot",
                # end Modify by CAMBRICON
            )
        )
        # Add by CAMBRICON
        # dump GraphModule before pass apply
        if hasattr(self, "orig_gm_code") and self.orig_gm_code is not None:
            with open(
                os.path.join(
                    self.log_url,
                    f"pass_{GraphTransformObserver.get_current_pass_count()}_{self.passname}_input_graph.py",
                ),
                "w",
            ) as fd:
                fd.write(self.orig_gm_code)
        # dump GraphModule after pass apply
        with open(
            os.path.join(
                self.log_url,
                f"pass_{GraphTransformObserver.get_current_pass_count()}_{self.passname}_output_graph.py",
            ),
            "w",
        ) as fd:
            fd.write(
                gm.print_readable(
                    print_output=False, include_stride=True, include_device=True
                )
            )
        # end Add by CAMBRICON
