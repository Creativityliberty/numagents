"""
Serialization and deserialization utilities for flows and nodes.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import json
from typing import Any, Dict, List, Optional, Type
import importlib
import inspect

from num_agents.core import Node, Flow, SharedStore, ConditionalNode
from num_agents.exceptions import SerializationError, DeserializationError
from num_agents.logging_config import get_logger

logger = get_logger(__name__)


class FlowSerializer:
    """
    Serializer for Flow objects.

    Converts Flow instances to JSON-compatible dictionaries and vice versa.
    """

    @staticmethod
    def serialize_node(node: Node) -> Dict[str, Any]:
        """
        Serialize a node to a dictionary.

        Args:
            node: The node to serialize

        Returns:
            Dictionary representation of the node

        Raises:
            SerializationError: If serialization fails
        """
        try:
            node_data = {
                "id": node.id,
                "name": node.name,
                "class": node.__class__.__name__,
                "module": node.__class__.__module__,
                "next_node_ids": [n.id for n in node.get_next_nodes()],
                "config": {
                    "enable_logging": getattr(node, "_enable_logging", False),
                    "retry_count": getattr(node, "retry_count", 0),
                    "timeout": getattr(node, "timeout", None),
                },
            }

            # Handle ConditionalNode specially
            if isinstance(node, ConditionalNode):
                node_data["conditional"] = {
                    "true_node_id": node.true_node.id if node.true_node else None,
                    "false_node_id": node.false_node.id if node.false_node else None,
                }

            return node_data

        except Exception as e:
            raise SerializationError(
                f"Failed to serialize node '{node.name}'",
                details={"node_id": node.id, "error": str(e)},
            ) from e

    @staticmethod
    def serialize_flow(flow: Flow) -> Dict[str, Any]:
        """
        Serialize a flow to a dictionary.

        Args:
            flow: The flow to serialize

        Returns:
            Dictionary representation of the flow

        Raises:
            SerializationError: If serialization fails
        """
        try:
            # Create node lookup
            node_lookup = {node.id: node for node in flow.nodes}

            # Serialize all nodes
            nodes_data = [
                FlowSerializer.serialize_node(node) for node in flow.nodes
            ]

            flow_data = {
                "name": flow.name,
                "start_node_id": flow._start_node.id if flow._start_node else None,
                "config": {
                    "enable_logging": flow._enable_logging,
                    "fail_fast": flow.fail_fast,
                },
                "nodes": nodes_data,
                "execution_history": flow.execution_history,
            }

            return flow_data

        except Exception as e:
            raise SerializationError(
                f"Failed to serialize flow '{flow.name}'",
                details={"error": str(e)},
            ) from e

    @staticmethod
    def flow_to_json(flow: Flow, indent: Optional[int] = 2) -> str:
        """
        Serialize a flow to JSON string.

        Args:
            flow: The flow to serialize
            indent: Optional indentation for pretty printing

        Returns:
            JSON string representation of the flow
        """
        flow_data = FlowSerializer.serialize_flow(flow)
        return json.dumps(flow_data, indent=indent)

    @staticmethod
    def flow_to_file(flow: Flow, file_path: str, indent: Optional[int] = 2) -> None:
        """
        Serialize a flow to a JSON file.

        Args:
            flow: The flow to serialize
            file_path: Path to the output file
            indent: Optional indentation for pretty printing
        """
        json_str = FlowSerializer.flow_to_json(flow, indent)
        with open(file_path, "w") as f:
            f.write(json_str)

        logger.info(f"Flow '{flow.name}' serialized to {file_path}")


class FlowDeserializer:
    """
    Deserializer for Flow objects.

    Converts JSON-compatible dictionaries back to Flow instances.
    """

    def __init__(self, node_registry: Optional[Dict[str, Type[Node]]] = None) -> None:
        """
        Initialize the deserializer.

        Args:
            node_registry: Optional mapping of class names to Node classes
                          If not provided, will attempt to import from modules
        """
        self.node_registry = node_registry or {}

    def _get_node_class(self, class_name: str, module_name: str) -> Type[Node]:
        """
        Get a node class from the registry or by importing.

        Args:
            class_name: Name of the node class
            module_name: Module containing the class

        Returns:
            The node class

        Raises:
            DeserializationError: If the class cannot be found
        """
        # Check registry first
        if class_name in self.node_registry:
            return self.node_registry[class_name]

        # Try to import the module and get the class
        try:
            module = importlib.import_module(module_name)
            node_class = getattr(module, class_name)

            if not issubclass(node_class, Node):
                raise DeserializationError(
                    f"Class '{class_name}' is not a subclass of Node"
                )

            return node_class

        except Exception as e:
            raise DeserializationError(
                f"Failed to import node class '{class_name}' from '{module_name}'",
                details={"error": str(e)},
            ) from e

    def deserialize_flow(self, flow_data: Dict[str, Any]) -> Flow:
        """
        Deserialize a flow from a dictionary.

        Args:
            flow_data: Dictionary representation of the flow

        Returns:
            The deserialized Flow instance

        Raises:
            DeserializationError: If deserialization fails
        """
        try:
            # Create flow with config
            config = flow_data.get("config", {})
            flow = Flow(
                name=flow_data["name"],
                enable_logging=config.get("enable_logging", False),
                fail_fast=config.get("fail_fast", True),
            )

            # Deserialize nodes (basic structure first, without connections)
            nodes_data = flow_data["nodes"]
            node_instances: Dict[str, Node] = {}

            # Create all node instances
            for node_data in nodes_data:
                node_class = self._get_node_class(
                    node_data["class"], node_data["module"]
                )

                # Create node instance with basic config
                # Note: This creates a basic node - subclasses might need special handling
                node_config = node_data.get("config", {})

                # For now, create nodes without calling their __init__
                # We'll restore their state from the serialized data
                node = node_class.__new__(node_class)
                node.id = node_data["id"]
                node.name = node_data["name"]
                node._next_nodes = []
                node._enable_logging = node_config.get("enable_logging", False)
                node._logger = get_logger(__name__) if node._enable_logging else None
                node.retry_count = node_config.get("retry_count", 0)
                node.timeout = node_config.get("timeout", None)
                node._before_exec_hooks = []
                node._after_exec_hooks = []
                node._on_error_hooks = []

                node_instances[node.id] = node
                flow.nodes.append(node)

            # Reconnect nodes
            for node_data in nodes_data:
                node = node_instances[node_data["id"]]
                for next_id in node_data["next_node_ids"]:
                    if next_id in node_instances:
                        node.add_next(node_instances[next_id])

            # Set start node
            start_node_id = flow_data.get("start_node_id")
            if start_node_id and start_node_id in node_instances:
                flow._start_node = node_instances[start_node_id]

            # Restore execution history
            flow.execution_history = flow_data.get("execution_history", [])

            return flow

        except Exception as e:
            raise DeserializationError(
                f"Failed to deserialize flow",
                details={"error": str(e)},
            ) from e

    def flow_from_json(self, json_str: str) -> Flow:
        """
        Deserialize a flow from a JSON string.

        Args:
            json_str: JSON string representation of the flow

        Returns:
            The deserialized Flow instance
        """
        flow_data = json.loads(json_str)
        return self.deserialize_flow(flow_data)

    def flow_from_file(self, file_path: str) -> Flow:
        """
        Deserialize a flow from a JSON file.

        Args:
            file_path: Path to the input file

        Returns:
            The deserialized Flow instance
        """
        with open(file_path, "r") as f:
            json_str = f.read()

        logger.info(f"Loading flow from {file_path}")
        return self.flow_from_json(json_str)
