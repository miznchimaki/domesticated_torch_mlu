# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
from typing import Dict, List, Optional, Any

from .common import utils

logger = utils.get_logger()


class ETNode:
    """
    Represents a node in the Execution Trace.

    Each node corresponds to an operator/tensor operation in PyTorch's execution trace.
    """

    def __init__(
        self,
        node_id: int,
        name: str,
        rf_id: int,
        ctrl_deps: int = 0,
        inputs: Optional[Dict[str, List]] = None,
        outputs: Optional[Dict[str, List]] = None,
        attrs: Optional[List[Dict[str, Any]]] = None,
    ):
        self.id = node_id
        self.name = name
        self.rf_id = rf_id
        self.ctrl_deps = ctrl_deps
        self.inputs = inputs or {}
        self.outputs = outputs or {}

        self._op_schema = ""
        for attr in attrs or []:
            if "name" in attr and attr.get("name") == "op_schema":
                self._op_schema = attr.get("value")
                break

    @property
    def input_shapes(self) -> List:
        """Get input shapes."""
        return self.inputs.get("shapes", [])

    @property
    def input_types(self) -> List:
        """Get input types."""
        return self.inputs.get("types", [])

    @property
    def input_strides(self) -> List:
        """Get input strides."""
        return self.inputs.get("strides", [])

    @property
    def input_values(self) -> List:
        """Get input values."""
        return self.inputs.get("values", [])

    @property
    def output_shapes(self) -> List:
        """Get output shapes."""
        return self.outputs.get("shapes", [])

    @property
    def output_strides(self) -> List:
        """Get output strides."""
        return self.outputs.get("strides", [])

    @property
    def output_types(self) -> List:
        """Get output types."""
        return self.outputs.get("types", [])

    @property
    def output_values(self) -> List:
        """Get output values."""
        return self.outputs.get("values", [])

    @property
    def op_schema(self) -> str:
        """Get operator schema."""
        return self._op_schema

    def __repr__(self):
        return f"ETNode(id={self.id}, name='{self.name}', rf_id={self.rf_id})"


class ExecutionTraceParser:
    """
    Parser for torch.profiler.ExecutionTraceObserver() output.

    Parses the execution trace JSON file and provides:
    - Access to all nodes
    - Mapping from rf_id to nodes
    - Query methods for finding nodes by various criteria
    """

    def __init__(self, et_trace_jsons: List[Dict[str, Any]]):
        """
        Initialize the parser.

        Args:
            et_trace_jsons: execution trace jsons content
        """
        self.et_trace_jsons = et_trace_jsons
        self._nodes: List[ETNode] = []
        self._rf_id_to_nodes: Dict[int, ETNode] = {}
        self._node_id_to_node: Dict[int, ETNode] = {}

    def _parse_nodes(self):
        """Parse all nodes from the execution trace JSON."""
        if not self.et_trace_jsons:
            logger.warning("No jsons found in execution trace")
            return

        self._nodes.clear()
        self._rf_id_to_nodes.clear()
        self._node_id_to_node.clear()

        for et_trace_json in self.et_trace_jsons:
            if "nodes" not in et_trace_json:
                logger.warning("No nodes found in execution trace json")
                continue
            for node_data in et_trace_json["nodes"]:
                node = self._create_node(node_data)
                self._nodes.append(node)
                self._node_id_to_node[node.id] = node

                # Map rf_id to node (only if rf_id is valid and not 0)
                if node.rf_id > 0:
                    if node.rf_id in self._rf_id_to_nodes:
                        logger.debug(
                            f"Duplicate rf_id {node.rf_id} found for nodes "
                            f"{self._rf_id_to_nodes[node.rf_id].id} and {node.id}"
                        )
                    self._rf_id_to_nodes[node.rf_id] = node

    def _create_node(self, node_data: Dict[str, Any]) -> ETNode:
        """
        Create an ETNode from raw JSON data.

        Args:
            node_data: Raw node data from execution trace JSON

        Returns:
            ETNode instance
        """
        node_id = node_data.get("id", 0)
        name = node_data.get("name", "")

        # Extract rf_id from attrs
        rf_id = 0
        attrs = node_data.get("attrs", [])
        for attr in attrs:
            if attr.get("name") == "rf_id":
                rf_id = attr.get("value", 0)
                break

        return ETNode(
            node_id=node_id,
            name=name,
            rf_id=rf_id,
            ctrl_deps=node_data.get("ctrl_deps", 0),
            inputs=node_data.get("inputs", {}),
            outputs=node_data.get("outputs", {}),
            attrs=attrs,
        )

    def get_all_nodes(self) -> List[ETNode]:
        """
        Get all nodes in the execution trace.

        Returns:
            List of all ETNode instances
        """
        return self._nodes[:]

    def get_node_by_rf_id(self, rf_id: int) -> Optional[ETNode]:
        """
        Get a node by its rf_id (record function ID).

        Args:
            rf_id: The record function ID to look up

        Returns:
            ETNode if found, None otherwise
        """
        return self._rf_id_to_nodes.get(rf_id)

    def get_node_by_id(self, node_id: int) -> Optional[ETNode]:
        """
        Get a node by its node ID.

        Args:
            node_id: The node ID to look up

        Returns:
            ETNode if found, None otherwise
        """
        return self._node_id_to_node.get(node_id)

    def get_nodes_by_name(self, name: str) -> List[ETNode]:
        """
        Get all nodes with the given name.

        Args:
            name: The operator name to search for

        Returns:
            List of ETNode instances with matching name
        """
        return [node for node in self._nodes if node.name == name]

    def get_nodes_by_name_pattern(self, pattern: str) -> List[ETNode]:
        """
        Get all nodes whose name matches the given pattern.

        Args:
            pattern: Pattern string (simple substring match)

        Returns:
            List of ETNode instances with matching name pattern
        """
        return [node for node in self._nodes if pattern in node.name]

    def get_rf_id_mapping(self) -> Dict[int, ETNode]:
        """
        Get the complete rf_id to node mapping.

        Returns:
            Dictionary mapping rf_id to ETNode
        """
        return self._rf_id_to_nodes.copy()

    def get_node_id_mapping(self) -> Dict[int, ETNode]:
        """
        Get the complete node ID to node mapping.

        Returns:
            Dictionary mapping node_id to ETNode
        """
        return self._node_id_to_node.copy()

    def get_node_count(self) -> int:
        """
        Get the total number of nodes in the execution trace.

        Returns:
            Number of nodes
        """
        return len(self._nodes)

    def get_tensor_info(self, node: ETNode) -> Dict[str, Any]:
        """
        Get tensor information for a node.

        Args:
            node: The ETNode to analyze

        Returns:
            Dictionary containing tensor information
        """
        return {
            "input_shapes": node.inputs.get("shapes", []),
            "input_types": node.inputs.get("types", []),
            "output_shapes": node.outputs.get("shapes", []),
            "output_types": node.outputs.get("types", []),
        }


def split_triton_node_inputs_outputs(node: ETNode, input_num: int):
    """
    Split output tensors from the inputs dictionary.

    In the execution trace of a Triton kernel, the argument order is typically:
        [input tensors] + [output tensors] + [input scalar args]

    This function separates the output tensors from the `inputs` dictionary
    and moves them into the `outputs` dictionary. After the split:
        - `inputs` will contain input tensors and scalar arguments.
        - `outputs` will contain only output tensors.

    Args:
        node (ETNode): triton ETNode.
        input_num (int): Number of input tensors at the beginning of `inputs`.

    Returns:
        None: The function modifies `inputs` and `outputs` of node in place.
    """

    tensor_indices = [
        i
        for i, t in enumerate(node.inputs["types"])
        if isinstance(t, str) and "Tensor" in t
    ]

    output_num = len(tensor_indices) - input_num

    output_start = input_num
    output_end = input_num + output_num

    for k, arr in node.inputs.items():
        if not isinstance(arr, list):
            continue

        node.outputs[k] = arr[output_start:output_end]

        node.inputs[k] = arr[:input_num] + arr[output_end:]
