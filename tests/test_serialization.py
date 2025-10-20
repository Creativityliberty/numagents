"""
Tests for flow serialization and deserialization.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import json
import pytest
import tempfile
import os
from typing import Dict, Any

from num_agents.core import Node, Flow, SharedStore
from num_agents.serialization import FlowSerializer, FlowDeserializer
from num_agents.exceptions import SerializationError, DeserializationError


class SimpleTestNode(Node):
    """Simple test node for serialization tests."""

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        value = shared.get("counter", 0)
        shared.set("counter", value + 1)
        return {"counter": value + 1}


class TestFlowSerializer:
    """Test FlowSerializer."""

    def test_serialize_node(self) -> None:
        """Test node serialization."""
        node = SimpleTestNode(name="TestNode", enable_logging=True, retry_count=3)
        node_data = FlowSerializer.serialize_node(node)

        assert node_data["name"] == "TestNode"
        assert node_data["class"] == "SimpleTestNode"
        assert node_data["config"]["enable_logging"] is True
        assert node_data["config"]["retry_count"] == 3
        assert "id" in node_data

    def test_serialize_flow(self) -> None:
        """Test flow serialization."""
        node1 = SimpleTestNode(name="Node1")
        node2 = SimpleTestNode(name="Node2")

        flow = Flow(nodes=[node1, node2], name="TestFlow", enable_logging=True)

        flow_data = FlowSerializer.serialize_flow(flow)

        assert flow_data["name"] == "TestFlow"
        assert flow_data["config"]["enable_logging"] is True
        assert len(flow_data["nodes"]) == 2
        assert flow_data["start_node_id"] == node1.id

    def test_flow_to_json(self) -> None:
        """Test flow to JSON string."""
        node = SimpleTestNode(name="TestNode")
        flow = Flow(nodes=[node], name="TestFlow")

        json_str = FlowSerializer.flow_to_json(flow)

        # Verify it's valid JSON
        data = json.loads(json_str)
        assert data["name"] == "TestFlow"

    def test_flow_to_file(self) -> None:
        """Test flow to file."""
        node = SimpleTestNode(name="TestNode")
        flow = Flow(nodes=[node], name="TestFlow")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            FlowSerializer.flow_to_file(flow, temp_path)

            # Verify file was created and contains valid JSON
            with open(temp_path, "r") as f:
                data = json.load(f)
                assert data["name"] == "TestFlow"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestFlowDeserializer:
    """Test FlowDeserializer."""

    def test_deserialize_flow(self) -> None:
        """Test flow deserialization."""
        # Create and serialize a flow
        node1 = SimpleTestNode(name="Node1")
        node2 = SimpleTestNode(name="Node2")
        original_flow = Flow(nodes=[node1, node2], name="TestFlow")

        flow_data = FlowSerializer.serialize_flow(original_flow)

        # Deserialize
        deserializer = FlowDeserializer(
            node_registry={"SimpleTestNode": SimpleTestNode}
        )
        deserialized_flow = deserializer.deserialize_flow(flow_data)

        # Verify structure
        assert deserialized_flow.name == "TestFlow"
        assert len(deserialized_flow.nodes) == 2
        assert deserialized_flow.nodes[0].name == "Node1"
        assert deserialized_flow.nodes[1].name == "Node2"

    def test_flow_from_json(self) -> None:
        """Test flow from JSON string."""
        # Create and serialize a flow
        node = SimpleTestNode(name="TestNode")
        original_flow = Flow(nodes=[node], name="TestFlow")

        json_str = FlowSerializer.flow_to_json(original_flow)

        # Deserialize
        deserializer = FlowDeserializer(
            node_registry={"SimpleTestNode": SimpleTestNode}
        )
        deserialized_flow = deserializer.flow_from_json(json_str)

        assert deserialized_flow.name == "TestFlow"
        assert len(deserialized_flow.nodes) == 1

    def test_flow_from_file(self) -> None:
        """Test flow from file."""
        # Create and serialize a flow
        node = SimpleTestNode(name="TestNode")
        original_flow = Flow(nodes=[node], name="TestFlow")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            FlowSerializer.flow_to_file(original_flow, temp_path)

            # Deserialize
            deserializer = FlowDeserializer(
                node_registry={"SimpleTestNode": SimpleTestNode}
            )
            deserialized_flow = deserializer.flow_from_file(temp_path)

            assert deserialized_flow.name == "TestFlow"
            assert len(deserialized_flow.nodes) == 1
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_roundtrip(self) -> None:
        """Test full serialization/deserialization roundtrip."""
        # Create a flow with multiple nodes
        node1 = SimpleTestNode(name="Node1", enable_logging=True, retry_count=2)
        node2 = SimpleTestNode(name="Node2")
        node3 = SimpleTestNode(name="Node3")

        original_flow = Flow(
            nodes=[node1, node2, node3],
            name="RoundtripFlow",
            enable_logging=True,
            fail_fast=False,
        )

        # Serialize
        flow_data = FlowSerializer.serialize_flow(original_flow)

        # Deserialize
        deserializer = FlowDeserializer(
            node_registry={"SimpleTestNode": SimpleTestNode}
        )
        deserialized_flow = deserializer.deserialize_flow(flow_data)

        # Verify properties
        assert deserialized_flow.name == original_flow.name
        assert len(deserialized_flow.nodes) == len(original_flow.nodes)
        assert deserialized_flow._enable_logging == original_flow._enable_logging
        assert deserialized_flow.fail_fast == original_flow.fail_fast

        # Verify nodes
        for i, node in enumerate(deserialized_flow.nodes):
            original_node = original_flow.nodes[i]
            assert node.name == original_node.name
            assert node.retry_count == original_node.retry_count
