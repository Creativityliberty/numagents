"""
Core components for the Nüm Agents SDK.

This module defines the fundamental building blocks for agent flows:
- Node: Base class for all processing nodes
- Flow: Container and orchestrator for nodes
- SharedStore: Shared memory for data exchange between nodes
- ConditionalNode: Node with conditional execution paths

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

from typing import Any, Dict, List, Optional, Set, Union, Callable
import uuid
import time

from num_agents.exceptions import (
    FlowConfigurationError,
    FlowExecutionError,
    NodeExecutionError,
    NodeNotImplementedError,
    SharedStoreKeyError,
)
from num_agents.logging_config import get_logger


class SharedStore:
    """
    Shared memory store for data exchange between nodes in a flow.

    The SharedStore acts as a central repository for data that needs to be
    accessed and modified by different nodes during flow execution.
    """

    def __init__(self, enable_logging: bool = False) -> None:
        """
        Initialize an empty shared store.

        Args:
            enable_logging: Enable detailed logging for store operations
        """
        self._data: Dict[str, Any] = {}
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the shared store.

        Args:
            key: The key to store the value under
            value: The value to store
        """
        self._data[key] = value
        if self._enable_logging and self._logger:
            self._logger.debug(f"SharedStore: Set key '{key}' with value type {type(value).__name__}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the shared store.

        Args:
            key: The key to retrieve
            default: Value to return if key doesn't exist

        Returns:
            The value associated with the key, or default if not found
        """
        value = self._data.get(key, default)
        if self._enable_logging and self._logger:
            self._logger.debug(f"SharedStore: Get key '{key}' (found: {key in self._data})")
        return value

    def get_required(self, key: str) -> Any:
        """
        Get a required value from the shared store.

        Args:
            key: The key to retrieve

        Returns:
            The value associated with the key

        Raises:
            SharedStoreKeyError: If the key doesn't exist
        """
        if key not in self._data:
            raise SharedStoreKeyError(key)
        return self._data[key]

    def has(self, key: str) -> bool:
        """
        Check if a key exists in the shared store.

        Args:
            key: The key to check

        Returns:
            True if the key exists, False otherwise
        """
        return key in self._data

    def delete(self, key: str) -> None:
        """
        Delete a key from the shared store.

        Args:
            key: The key to delete
        """
        if key in self._data:
            del self._data[key]
            if self._enable_logging and self._logger:
                self._logger.debug(f"SharedStore: Deleted key '{key}'")

    def clear(self) -> None:
        """Clear all data from the shared store."""
        count = len(self._data)
        self._data.clear()
        if self._enable_logging and self._logger:
            self._logger.debug(f"SharedStore: Cleared {count} keys")

    def keys(self) -> Set[str]:
        """
        Get all keys in the shared store.

        Returns:
            A set of all keys in the store
        """
        return set(self._data.keys())

    def update(self, data: Dict[str, Any]) -> None:
        """
        Update the store with multiple key-value pairs.

        Args:
            data: Dictionary of key-value pairs to add
        """
        self._data.update(data)
        if self._enable_logging and self._logger:
            self._logger.debug(f"SharedStore: Updated with {len(data)} keys")

    def to_dict(self) -> Dict[str, Any]:
        """
        Get a copy of all data in the store.

        Returns:
            A copy of the internal data dictionary
        """
        return self._data.copy()

    def __contains__(self, key: str) -> bool:
        """Support for 'in' operator."""
        return key in self._data

    def __len__(self) -> int:
        """Support for len() operator."""
        return len(self._data)


class Node:
    """
    Base class for all processing nodes in a flow.

    A Node represents a single unit of processing in an agent flow.
    Each node can read from and write to the SharedStore, and can
    have transitions to other nodes.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        enable_logging: bool = False,
        retry_count: int = 0,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Initialize a new node.

        Args:
            name: Optional name for the node. If not provided, class name is used.
            enable_logging: Enable detailed logging for this node
            retry_count: Number of retries on failure (0 = no retry)
            timeout: Optional timeout in seconds for node execution
        """
        self.id = str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self._next_nodes: List["Node"] = []
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None
        self.retry_count = retry_count
        self.timeout = timeout

        # Hooks
        self._before_exec_hooks: List[Callable[[SharedStore], None]] = []
        self._after_exec_hooks: List[Callable[[SharedStore, Dict[str, Any]], None]] = []
        self._on_error_hooks: List[Callable[[SharedStore, Exception], None]] = []

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute the node's processing logic.

        This method should be overridden by subclasses to implement
        the specific processing logic of the node.

        Args:
            shared: The shared store for accessing and storing data

        Returns:
            A dictionary containing the results of the node's execution

        Raises:
            NodeNotImplementedError: If the method is not implemented by subclass
        """
        raise NodeNotImplementedError(
            f"Node '{self.name}' must implement exec method",
            details={"node_id": self.id, "node_class": self.__class__.__name__},
        )

    def _execute_with_error_handling(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute the node with error handling and retries.

        Args:
            shared: The shared store

        Returns:
            The execution results

        Raises:
            NodeExecutionError: If execution fails after all retries
        """
        attempts = 0
        max_attempts = self.retry_count + 1
        last_error: Optional[Exception] = None

        while attempts < max_attempts:
            try:
                # Run before hooks
                for hook in self._before_exec_hooks:
                    hook(shared)

                if self._enable_logging and self._logger:
                    self._logger.info(f"Executing node: {self.name} (attempt {attempts + 1}/{max_attempts})")

                start_time = time.time()

                # Execute the node
                result = self.exec(shared)

                execution_time = time.time() - start_time

                if self._enable_logging and self._logger:
                    self._logger.info(
                        f"Node {self.name} completed in {execution_time:.3f}s"
                    )

                # Run after hooks
                for hook in self._after_exec_hooks:
                    hook(shared, result)

                return result

            except Exception as e:
                last_error = e
                attempts += 1

                if self._enable_logging and self._logger:
                    self._logger.warning(
                        f"Node {self.name} failed (attempt {attempts}/{max_attempts}): {str(e)}"
                    )

                # Run error hooks
                for hook in self._on_error_hooks:
                    try:
                        hook(shared, e)
                    except Exception as hook_error:
                        if self._enable_logging and self._logger:
                            self._logger.error(f"Error hook failed: {str(hook_error)}")

                if attempts >= max_attempts:
                    raise NodeExecutionError(
                        str(last_error),
                        node_name=self.name,
                        node_id=self.id,
                        details={
                            "attempts": attempts,
                            "original_error": type(last_error).__name__,
                        },
                    ) from last_error

                # Wait before retry (exponential backoff)
                if attempts < max_attempts:
                    wait_time = 2 ** (attempts - 1)  # 1s, 2s, 4s, 8s...
                    time.sleep(wait_time)

        # This should never be reached, but just in case
        raise NodeExecutionError(
            "Node execution failed",
            node_name=self.name,
            node_id=self.id,
            details={"attempts": attempts},
        )

    def add_next(self, node: "Node") -> "Node":
        """
        Add a node to execute after this one.

        Args:
            node: The node to execute next

        Returns:
            The current node (self) for method chaining
        """
        self._next_nodes.append(node)
        return self

    def get_next_nodes(self) -> List["Node"]:
        """
        Get the list of nodes to execute after this one.

        Returns:
            List of next nodes
        """
        return self._next_nodes

    def add_before_hook(self, hook: Callable[[SharedStore], None]) -> "Node":
        """
        Add a hook to run before node execution.

        Args:
            hook: Function to call before execution

        Returns:
            The current node (self) for method chaining
        """
        self._before_exec_hooks.append(hook)
        return self

    def add_after_hook(
        self, hook: Callable[[SharedStore, Dict[str, Any]], None]
    ) -> "Node":
        """
        Add a hook to run after successful node execution.

        Args:
            hook: Function to call after execution

        Returns:
            The current node (self) for method chaining
        """
        self._after_exec_hooks.append(hook)
        return self

    def add_error_hook(self, hook: Callable[[SharedStore, Exception], None]) -> "Node":
        """
        Add a hook to run when node execution fails.

        Args:
            hook: Function to call on error

        Returns:
            The current node (self) for method chaining
        """
        self._on_error_hooks.append(hook)
        return self

    def __str__(self) -> str:
        """String representation of the node."""
        return f"{self.name}({self.id[:8]})"

    def __repr__(self) -> str:
        """Detailed string representation of the node."""
        return f"Node(name='{self.name}', id='{self.id[:8]}', next_nodes={len(self._next_nodes)})"


class Flow:
    """
    Container and orchestrator for a sequence of nodes.

    A Flow represents a complete agent processing pipeline, consisting
    of multiple interconnected nodes that share data through a SharedStore.
    """

    def __init__(
        self,
        nodes: Optional[List[Node]] = None,
        name: Optional[str] = None,
        enable_logging: bool = False,
        fail_fast: bool = True,
    ) -> None:
        """
        Initialize a new flow.

        Args:
            nodes: Optional list of nodes to add to the flow
            name: Optional name for the flow
            enable_logging: Enable detailed logging for flow execution
            fail_fast: If True, stop execution on first error. If False, continue.
        """
        self.name = name or f"Flow-{uuid.uuid4().hex[:8]}"
        self.nodes: List[Node] = nodes or []
        self.shared = SharedStore(enable_logging=enable_logging)
        self._start_node: Optional[Node] = None
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None
        self.fail_fast = fail_fast

        # Execution metadata
        self.execution_history: List[Dict[str, Any]] = []
        self.last_execution_time: Optional[float] = None

        # Hooks
        self._before_flow_hooks: List[Callable[[SharedStore], None]] = []
        self._after_flow_hooks: List[Callable[[SharedStore, Dict[str, Any]], None]] = []
        self._on_flow_error_hooks: List[Callable[[SharedStore, Exception], None]] = []

        # If nodes were provided, automatically connect them in sequence
        if self.nodes:
            self._connect_nodes_in_sequence()
            self._start_node = self.nodes[0]

    def _connect_nodes_in_sequence(self) -> None:
        """Connect the nodes in the flow in sequence."""
        for i in range(len(self.nodes) - 1):
            self.nodes[i].add_next(self.nodes[i + 1])

    def _detect_cycles(self) -> bool:
        """
        Detect cycles in the flow graph.

        Returns:
            True if a cycle is detected, False otherwise
        """
        if not self._start_node:
            return False

        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def visit(node: Node) -> bool:
            if node.id in rec_stack:
                return True  # Cycle detected
            if node.id in visited:
                return False

            visited.add(node.id)
            rec_stack.add(node.id)

            for next_node in node.get_next_nodes():
                if visit(next_node):
                    return True

            rec_stack.remove(node.id)
            return False

        return visit(self._start_node)

    def validate(self) -> None:
        """
        Validate the flow configuration.

        Raises:
            FlowConfigurationError: If the flow is misconfigured
        """
        if not self._start_node:
            raise FlowConfigurationError("No start node defined for flow")

        if self._detect_cycles():
            raise FlowConfigurationError("Cycle detected in flow graph")

        if self._enable_logging and self._logger:
            self._logger.info(f"Flow '{self.name}' validation passed")

    def add_node(self, node: Node) -> "Flow":
        """
        Add a node to the flow.

        Args:
            node: The node to add

        Returns:
            The flow instance for method chaining
        """
        self.nodes.append(node)
        if not self._start_node:
            self._start_node = node
        return self

    def add_transition(self, from_node: Node, to_node: Node) -> "Flow":
        """
        Add a transition between two nodes.

        Args:
            from_node: The source node
            to_node: The destination node

        Returns:
            The flow instance for method chaining
        """
        from_node.add_next(to_node)
        return self

    def set_start(self, node: Node) -> "Flow":
        """
        Set the starting node for the flow.

        Args:
            node: The node to start execution from

        Returns:
            The flow instance for method chaining
        """
        if node not in self.nodes:
            self.nodes.append(node)
        self._start_node = node
        return self

    def execute(self, initial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the flow from the start node.

        Args:
            initial_data: Optional initial data to load into shared store

        Returns:
            The final results from the flow execution

        Raises:
            FlowConfigurationError: If flow is misconfigured
            FlowExecutionError: If flow execution fails
        """
        # Validate before execution
        self.validate()

        # Load initial data if provided
        if initial_data:
            self.shared.update(initial_data)

        if self._enable_logging and self._logger:
            self._logger.info(f"Starting flow execution: {self.name}")

        start_time = time.time()

        try:
            # Run before hooks
            for hook in self._before_flow_hooks:
                hook(self.shared)

            results: Dict[str, Any] = {}
            errors: Dict[str, Exception] = {}
            current_nodes = [self._start_node]
            executed_nodes: Set[str] = set()

            while current_nodes:
                next_nodes = []

                for node in current_nodes:
                    # Avoid executing the same node twice
                    if node.id in executed_nodes:
                        continue

                    try:
                        if self._enable_logging and self._logger:
                            self._logger.debug(f"Executing node: {node.name}")

                        # Use the node's error handling method
                        node_results = node._execute_with_error_handling(self.shared)
                        results[node.name] = node_results
                        executed_nodes.add(node.id)

                        # Add next nodes to queue
                        next_nodes.extend(node.get_next_nodes())

                    except Exception as e:
                        errors[node.name] = e

                        if self._enable_logging and self._logger:
                            self._logger.error(f"Node {node.name} failed: {str(e)}")

                        if self.fail_fast:
                            raise FlowExecutionError(
                                f"Flow execution failed at node '{node.name}'",
                                details={"node_id": node.id, "error": str(e)},
                            ) from e

                current_nodes = next_nodes

            execution_time = time.time() - start_time
            self.last_execution_time = execution_time

            # Record execution metadata
            execution_record = {
                "timestamp": time.time(),
                "duration": execution_time,
                "nodes_executed": len(executed_nodes),
                "errors": {k: str(v) for k, v in errors.items()},
                "success": len(errors) == 0,
            }
            self.execution_history.append(execution_record)

            if self._enable_logging and self._logger:
                self._logger.info(
                    f"Flow '{self.name}' completed in {execution_time:.3f}s "
                    f"({len(executed_nodes)} nodes executed)"
                )

            # Run after hooks
            for hook in self._after_flow_hooks:
                hook(self.shared, results)

            # If we have errors and didn't fail fast, include them in results
            if errors and not self.fail_fast:
                results["_errors"] = errors

            return results

        except Exception as e:
            # Run error hooks
            for hook in self._on_flow_error_hooks:
                try:
                    hook(self.shared, e)
                except Exception as hook_error:
                    if self._enable_logging and self._logger:
                        self._logger.error(f"Flow error hook failed: {str(hook_error)}")

            raise

    def reset(self) -> None:
        """Reset the flow's shared store and execution history."""
        self.shared.clear()
        self.execution_history.clear()
        self.last_execution_time = None

    def get_nodes(self) -> List[Node]:
        """
        Get all nodes in the flow.

        Returns:
            List of all nodes
        """
        return self.nodes

    def add_before_hook(self, hook: Callable[[SharedStore], None]) -> "Flow":
        """
        Add a hook to run before flow execution.

        Args:
            hook: Function to call before execution

        Returns:
            The flow instance for method chaining
        """
        self._before_flow_hooks.append(hook)
        return self

    def add_after_hook(
        self, hook: Callable[[SharedStore, Dict[str, Any]], None]
    ) -> "Flow":
        """
        Add a hook to run after successful flow execution.

        Args:
            hook: Function to call after execution

        Returns:
            The flow instance for method chaining
        """
        self._after_flow_hooks.append(hook)
        return self

    def add_error_hook(
        self, hook: Callable[[SharedStore, Exception], None]
    ) -> "Flow":
        """
        Add a hook to run when flow execution fails.

        Args:
            hook: Function to call on error

        Returns:
            The flow instance for method chaining
        """
        self._on_flow_error_hooks.append(hook)
        return self


class ConditionalNode(Node):
    """
    A node that executes different paths based on a condition.

    This node evaluates a condition function and routes execution to either
    the true_node or false_node based on the result.
    """

    def __init__(
        self,
        condition: Callable[[SharedStore], bool],
        true_node: Optional[Node] = None,
        false_node: Optional[Node] = None,
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize a conditional node.

        Args:
            condition: Function that takes SharedStore and returns bool
            true_node: Node to execute if condition is True
            false_node: Node to execute if condition is False
            name: Optional name for the node
            enable_logging: Enable detailed logging for this node
        """
        super().__init__(name or "ConditionalNode", enable_logging)
        self.condition = condition
        self.true_node = true_node
        self.false_node = false_node

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute the conditional logic.

        Args:
            shared: The shared store

        Returns:
            Dictionary with condition result and path taken
        """
        try:
            result = self.condition(shared)

            if self._enable_logging and self._logger:
                self._logger.debug(
                    f"ConditionalNode '{self.name}': condition evaluated to {result}"
                )

            shared.set(f"{self.name}_condition_result", result)

            return {
                "condition_result": result,
                "path": "true" if result else "false",
            }

        except Exception as e:
            if self._enable_logging and self._logger:
                self._logger.error(
                    f"ConditionalNode '{self.name}': condition evaluation failed: {str(e)}"
                )
            raise

    def get_next_nodes(self) -> List[Node]:
        """
        Get the next node(s) based on the last condition evaluation.

        Returns:
            List containing the appropriate next node(s)
        """
        # Get the last evaluation result from shared store if available
        # Otherwise, return both paths (will be decided during execution)
        if self.true_node and self.false_node:
            return [self.true_node, self.false_node]
        elif self.true_node:
            return [self.true_node]
        elif self.false_node:
            return [self.false_node]
        else:
            return []

    def set_true_node(self, node: Node) -> "ConditionalNode":
        """
        Set the node to execute when condition is True.

        Args:
            node: The node to execute

        Returns:
            Self for method chaining
        """
        self.true_node = node
        return self

    def set_false_node(self, node: Node) -> "ConditionalNode":
        """
        Set the node to execute when condition is False.

        Args:
            node: The node to execute

        Returns:
            Self for method chaining
        """
        self.false_node = node
        return self
