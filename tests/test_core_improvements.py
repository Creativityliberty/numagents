"""
Tests for core improvements: error handling, logging, hooks, and conditional nodes.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import pytest
from typing import Dict, Any

from num_agents.core import Node, Flow, SharedStore, ConditionalNode
from num_agents.exceptions import (
    NodeExecutionError,
    FlowConfigurationError,
    FlowExecutionError,
    SharedStoreKeyError,
)


class SimpleNode(Node):
    """Simple test node."""

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        value = shared.get("counter", 0)
        shared.set("counter", value + 1)
        return {"executed": True, "counter": value + 1}


class FailingNode(Node):
    """Node that always fails."""

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        raise ValueError("This node always fails")


class TestSharedStoreImprovements:
    """Test SharedStore improvements."""

    def test_get_required_success(self) -> None:
        """Test get_required with existing key."""
        store = SharedStore()
        store.set("key", "value")
        assert store.get_required("key") == "value"

    def test_get_required_missing_key(self) -> None:
        """Test get_required with missing key raises exception."""
        store = SharedStore()
        with pytest.raises(SharedStoreKeyError) as exc_info:
            store.get_required("missing_key")
        assert "missing_key" in str(exc_info.value)

    def test_update(self) -> None:
        """Test update method."""
        store = SharedStore()
        store.update({"key1": "value1", "key2": "value2"})
        assert store.get("key1") == "value1"
        assert store.get("key2") == "value2"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        store = SharedStore()
        store.set("key1", "value1")
        store.set("key2", "value2")
        data = store.to_dict()
        assert data == {"key1": "value1", "key2": "value2"}
        # Verify it's a copy
        data["key3"] = "value3"
        assert "key3" not in store

    def test_len(self) -> None:
        """Test len() operator."""
        store = SharedStore()
        assert len(store) == 0
        store.set("key1", "value1")
        assert len(store) == 1
        store.set("key2", "value2")
        assert len(store) == 2


class TestNodeImprovements:
    """Test Node improvements."""

    def test_node_with_logging(self) -> None:
        """Test node with logging enabled."""
        node = SimpleNode(name="TestNode", enable_logging=True)
        shared = SharedStore()
        result = node._execute_with_error_handling(shared)
        assert result["executed"] is True

    def test_node_retry(self) -> None:
        """Test node retry on failure."""
        # Create a node that fails first time, succeeds second time
        class RetryableNode(Node):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.attempt_count = 0

            def exec(self, shared: SharedStore) -> Dict[str, Any]:
                self.attempt_count += 1
                if self.attempt_count < 2:
                    raise ValueError("Temporary failure")
                return {"success": True, "attempts": self.attempt_count}

        node = RetryableNode(retry_count=2)
        shared = SharedStore()
        result = node._execute_with_error_handling(shared)
        assert result["success"] is True
        assert result["attempts"] == 2

    def test_node_retry_exhausted(self) -> None:
        """Test node that fails after all retries."""
        node = FailingNode(retry_count=2)
        shared = SharedStore()
        with pytest.raises(NodeExecutionError) as exc_info:
            node._execute_with_error_handling(shared)
        assert "This node always fails" in str(exc_info.value)

    def test_node_before_hook(self) -> None:
        """Test before execution hook."""
        hook_called = []

        def before_hook(shared: SharedStore) -> None:
            hook_called.append(True)
            shared.set("before_hook_executed", True)

        node = SimpleNode()
        node.add_before_hook(before_hook)
        shared = SharedStore()
        node._execute_with_error_handling(shared)

        assert len(hook_called) == 1
        assert shared.get("before_hook_executed") is True

    def test_node_after_hook(self) -> None:
        """Test after execution hook."""
        hook_results = []

        def after_hook(shared: SharedStore, result: Dict[str, Any]) -> None:
            hook_results.append(result)

        node = SimpleNode()
        node.add_after_hook(after_hook)
        shared = SharedStore()
        result = node._execute_with_error_handling(shared)

        assert len(hook_results) == 1
        assert hook_results[0] == result

    def test_node_error_hook(self) -> None:
        """Test error hook."""
        hook_errors = []

        def error_hook(shared: SharedStore, error: Exception) -> None:
            hook_errors.append(error)

        node = FailingNode(retry_count=0)
        node.add_error_hook(error_hook)
        shared = SharedStore()

        with pytest.raises(NodeExecutionError):
            node._execute_with_error_handling(shared)

        assert len(hook_errors) == 1
        assert isinstance(hook_errors[0], ValueError)


class TestFlowImprovements:
    """Test Flow improvements."""

    def test_flow_with_name(self) -> None:
        """Test flow with custom name."""
        flow = Flow(name="TestFlow")
        assert flow.name == "TestFlow"

    def test_flow_validation_no_start(self) -> None:
        """Test flow validation fails with no start node."""
        flow = Flow()
        with pytest.raises(FlowConfigurationError) as exc_info:
            flow.validate()
        assert "No start node" in str(exc_info.value)

    def test_flow_cycle_detection(self) -> None:
        """Test cycle detection in flow."""
        node1 = SimpleNode(name="Node1")
        node2 = SimpleNode(name="Node2")
        node3 = SimpleNode(name="Node3")

        # Create a cycle: node1 -> node2 -> node3 -> node1
        node1.add_next(node2)
        node2.add_next(node3)
        node3.add_next(node1)

        flow = Flow()
        flow.add_node(node1)
        flow.add_node(node2)
        flow.add_node(node3)
        flow.set_start(node1)

        with pytest.raises(FlowConfigurationError) as exc_info:
            flow.validate()
        assert "Cycle detected" in str(exc_info.value)

    def test_flow_initial_data(self) -> None:
        """Test flow with initial data."""
        node = SimpleNode()
        flow = Flow()
        flow.add_node(node)
        flow.set_start(node)

        initial_data = {"counter": 10}
        results = flow.execute(initial_data=initial_data)

        assert flow.shared.get("counter") == 11

    def test_flow_fail_fast(self) -> None:
        """Test flow with fail_fast=True."""
        node1 = SimpleNode(name="Node1")
        node2 = FailingNode(name="Node2")
        node3 = SimpleNode(name="Node3")

        flow = Flow(nodes=[node1, node2, node3], fail_fast=True)

        with pytest.raises(FlowExecutionError) as exc_info:
            flow.execute()
        assert "Node2" in str(exc_info.value)

    def test_flow_no_fail_fast(self) -> None:
        """Test flow with fail_fast=False."""
        node1 = SimpleNode(name="Node1")
        node2 = FailingNode(name="Node2")

        flow = Flow(fail_fast=False)
        flow.add_node(node1)
        flow.add_node(node2)
        flow.set_start(node1)

        results = flow.execute()

        # Node1 should succeed
        assert "Node1" in results
        # Errors should be recorded
        assert "_errors" in results
        assert "Node2" in results["_errors"]

    def test_flow_execution_metadata(self) -> None:
        """Test flow execution metadata tracking."""
        node = SimpleNode()
        flow = Flow()
        flow.add_node(node)
        flow.set_start(node)

        flow.execute()

        assert flow.last_execution_time is not None
        assert len(flow.execution_history) == 1
        assert flow.execution_history[0]["success"] is True
        assert flow.execution_history[0]["nodes_executed"] == 1

    def test_flow_before_hook(self) -> None:
        """Test flow before execution hook."""
        hook_called = []

        def before_hook(shared: SharedStore) -> None:
            hook_called.append(True)
            shared.set("flow_start_time", 123456)

        node = SimpleNode()
        flow = Flow()
        flow.add_node(node)
        flow.set_start(node)
        flow.add_before_hook(before_hook)

        flow.execute()

        assert len(hook_called) == 1
        assert flow.shared.get("flow_start_time") == 123456

    def test_flow_after_hook(self) -> None:
        """Test flow after execution hook."""
        hook_results = []

        def after_hook(shared: SharedStore, results: Dict[str, Any]) -> None:
            hook_results.append(results)

        node = SimpleNode()
        flow = Flow()
        flow.add_node(node)
        flow.set_start(node)
        flow.add_after_hook(after_hook)

        results = flow.execute()

        assert len(hook_results) == 1
        assert hook_results[0] == results


class TestConditionalNode:
    """Test ConditionalNode."""

    def test_conditional_node_true_path(self) -> None:
        """Test conditional node taking true path."""
        true_node = SimpleNode(name="TrueNode")
        false_node = SimpleNode(name="FalseNode")

        def condition(shared: SharedStore) -> bool:
            return shared.get("value", 0) > 5

        cond_node = ConditionalNode(
            condition=condition,
            true_node=true_node,
            false_node=false_node,
            name="CondNode",
        )

        shared = SharedStore()
        shared.set("value", 10)

        result = cond_node.exec(shared)

        assert result["condition_result"] is True
        assert result["path"] == "true"
        assert shared.get("CondNode_condition_result") is True

    def test_conditional_node_false_path(self) -> None:
        """Test conditional node taking false path."""
        true_node = SimpleNode(name="TrueNode")
        false_node = SimpleNode(name="FalseNode")

        def condition(shared: SharedStore) -> bool:
            return shared.get("value", 0) > 5

        cond_node = ConditionalNode(
            condition=condition,
            true_node=true_node,
            false_node=false_node,
            name="CondNode",
        )

        shared = SharedStore()
        shared.set("value", 3)

        result = cond_node.exec(shared)

        assert result["condition_result"] is False
        assert result["path"] == "false"

    def test_conditional_node_in_flow(self) -> None:
        """Test conditional node in a flow."""

        class SetValueNode(Node):
            def exec(self, shared: SharedStore) -> Dict[str, Any]:
                shared.set("value", 10)
                return {"set": True}

        class TruePathNode(Node):
            def exec(self, shared: SharedStore) -> Dict[str, Any]:
                shared.set("path_taken", "true")
                return {"path": "true"}

        class FalsePathNode(Node):
            def exec(self, shared: SharedStore) -> Dict[str, Any]:
                shared.set("path_taken", "false")
                return {"path": "false"}

        set_node = SetValueNode(name="SetValue")
        true_node = TruePathNode(name="TruePath")
        false_node = FalsePathNode(name="FalsePath")

        def condition(shared: SharedStore) -> bool:
            return shared.get("value", 0) > 5

        cond_node = ConditionalNode(
            condition=condition,
            true_node=true_node,
            false_node=false_node,
            name="Condition",
        )

        flow = Flow()
        flow.add_node(set_node)
        flow.add_node(cond_node)
        flow.add_transition(set_node, cond_node)
        flow.set_start(set_node)

        results = flow.execute()

        # Value should be set to 10, so true path should be taken
        # But note: ConditionalNode returns both paths in get_next_nodes()
        # This is a limitation of the current implementation
        assert flow.shared.get("value") == 10
        assert "Condition" in results
        assert results["Condition"]["condition_result"] is True
