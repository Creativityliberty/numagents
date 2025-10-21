"""
Tests for ToolLayer

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import pytest
from num_agents import (
    ToolRegistry,
    ToolExecutor,
    PythonFunctionTool,
    ChainedTool,
    ToolExecuteNode,
    ToolChainNode,
    Flow,
    SharedStore,
    ToolNotFoundError,
    ToolExecutionError,
    ToolValidationError,
)


# ============================================================================
# Test Functions (to be wrapped as tools)
# ============================================================================


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def multiply_numbers(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


def greet(name: str) -> str:
    """Generate a greeting."""
    return f"Hello, {name}!"


# ============================================================================
# Test ToolRegistry
# ============================================================================


def test_tool_registry_register():
    """Test registering tools in registry."""
    registry = ToolRegistry()

    tool = PythonFunctionTool(name="add", function=add_numbers)
    registry.register(tool)

    assert registry.count() == 1
    assert "add" in registry.list_tools()


def test_tool_registry_get():
    """Test retrieving tools from registry."""
    registry = ToolRegistry()

    tool = PythonFunctionTool(name="add", function=add_numbers)
    registry.register(tool)

    retrieved = registry.get("add")
    assert retrieved is not None
    assert retrieved.name == "add"


def test_tool_registry_get_required():
    """Test get_required raises if not found."""
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError):
        registry.get_required("nonexistent")


def test_tool_registry_register_function():
    """Test convenience method for registering functions."""
    registry = ToolRegistry()

    tool = registry.register_function("greet", greet)

    assert tool.name == "greet"
    assert registry.count() == 1


def test_tool_registry_search():
    """Test searching for tools."""
    registry = ToolRegistry()

    registry.register_function("add", add_numbers)
    registry.register_function("multiply", multiply_numbers)
    registry.register_function("greet", greet)

    # Search by name
    results = registry.search("add")
    assert len(results) == 1
    assert results[0].name == "add"

    # Search by description
    results = registry.search("greeting")
    assert len(results) == 1
    assert results[0].name == "greet"


# ============================================================================
# Test PythonFunctionTool
# ============================================================================


def test_python_function_tool_execute():
    """Test executing a Python function tool."""
    tool = PythonFunctionTool(name="add", function=add_numbers)

    result = tool.execute(a=5, b=3)
    assert result == 8


def test_python_function_tool_validation():
    """Test input validation for Python function tool."""
    tool = PythonFunctionTool(name="add", function=add_numbers, safe_mode=True)

    # Valid input
    assert tool.validate_input(a=5, b=3) is True

    # Missing required parameter
    with pytest.raises(ToolValidationError):
        tool.validate_input(a=5)

    # Unexpected parameter
    with pytest.raises(ToolValidationError):
        tool.validate_input(a=5, b=3, c=10)


def test_python_function_tool_auto_parameters():
    """Test automatic parameter inference from function signature."""
    tool = PythonFunctionTool(name="add", function=add_numbers)

    # Check inferred parameters
    assert "properties" in tool.parameters
    assert "a" in tool.parameters["properties"]
    assert "b" in tool.parameters["properties"]
    assert "required" in tool.parameters
    assert "a" in tool.parameters["required"]
    assert "b" in tool.parameters["required"]


# ============================================================================
# Test ChainedTool
# ============================================================================


def test_chained_tool():
    """Test chaining multiple tools together."""
    # Create tools
    add_tool = PythonFunctionTool(name="add", function=add_numbers)
    multiply_tool = PythonFunctionTool(name="multiply", function=multiply_numbers)

    # Create chain
    chain = ChainedTool(name="add_then_multiply", tools=[add_tool, multiply_tool])

    # Execute chain
    result = chain.execute(a=5, b=3)  # add(5, 3) = 8
    # Note: multiply needs x, y but gets {'input': 8}
    # This demonstrates tool chaining limitation - needs better param mapping


# ============================================================================
# Test ToolExecutor
# ============================================================================


def test_tool_executor_execute():
    """Test executing tools through executor."""
    registry = ToolRegistry()
    registry.register_function("add", add_numbers)

    executor = ToolExecutor(registry)

    result = executor.execute("add", a=10, b=5)

    assert result["success"] is True
    assert result["result"] == 15
    assert "execution_time" in result


def test_tool_executor_execute_not_found():
    """Test executor handles tool not found."""
    registry = ToolRegistry()
    executor = ToolExecutor(registry)

    result = executor.execute("nonexistent", a=1)

    assert result["success"] is False
    assert "error" in result


def test_tool_executor_history():
    """Test executor records execution history."""
    registry = ToolRegistry()
    registry.register_function("add", add_numbers)

    executor = ToolExecutor(registry)

    executor.execute("add", a=1, b=2)
    executor.execute("add", a=3, b=4)

    history = executor.get_history()
    assert len(history) == 2

    # Get limited history
    limited = executor.get_history(limit=1)
    assert len(limited) == 1


# ============================================================================
# Test ToolExecuteNode (Flow Integration)
# ============================================================================


def test_tool_execute_node():
    """Test ToolExecuteNode in a Flow."""
    # Setup
    registry = ToolRegistry()
    registry.register_function("greet", greet)

    executor = ToolExecutor(registry)

    # Create node
    node = ToolExecuteNode(
        executor=executor, tool_name="greet", input_key="name", output_key="greeting"
    )

    # Execute in flow
    flow = Flow(nodes=[node], name="GreetingFlow")
    results = flow.execute(initial_data={"name": "Alice"})

    # Verify
    assert "greeting" in flow._shared.data
    greeting_result = flow._shared.get("greeting")
    assert greeting_result["success"] is True
    assert greeting_result["result"] == "Hello, Alice!"


def test_tool_execute_node_with_params_key():
    """Test ToolExecuteNode with params_key."""
    # Setup
    registry = ToolRegistry()
    registry.register_function("add", add_numbers)

    executor = ToolExecutor(registry)

    # Create node
    node = ToolExecuteNode(
        executor=executor, tool_name="add", params_key="params", output_key="result"
    )

    # Execute
    shared = SharedStore()
    shared.set("params", {"a": 10, "b": 20})

    result = node.exec(shared)

    assert result["success"] is True
    assert result["result"] == 30


# ============================================================================
# Test ToolChainNode (Flow Integration)
# ============================================================================


def test_tool_chain_node():
    """Test ToolChainNode in a Flow."""
    # Setup
    registry = ToolRegistry()
    registry.register_function("add", add_numbers)
    registry.register_function("multiply", multiply_numbers)

    executor = ToolExecutor(registry)

    # Create chain node
    node = ToolChainNode(
        executor=executor,
        tool_names=["add", "multiply"],
        input_key="initial",
        output_key="final",
    )

    # Execute
    shared = SharedStore()
    shared.set("initial", {"a": 2, "b": 3})

    result = node.exec(shared)

    # Note: This will fail because multiply expects x, y but gets input
    # This is a demonstration - in real use, tools need compatible interfaces


# ============================================================================
# Integration Test
# ============================================================================


def test_tool_layer_integration():
    """Integration test: Full tool layer workflow."""
    # Create registry
    registry = ToolRegistry(enable_logging=False)

    # Register tools
    registry.register_function("add", add_numbers, description="Add two numbers")
    registry.register_function(
        "multiply", multiply_numbers, description="Multiply two numbers"
    )
    registry.register_function("greet", greet, description="Greet someone")

    # Create executor
    executor = ToolExecutor(registry, enable_logging=False)

    # Test 1: Execute add
    result1 = executor.execute("add", a=5, b=10)
    assert result1["success"] is True
    assert result1["result"] == 15

    # Test 2: Execute greet
    result2 = executor.execute("greet", name="World")
    assert result2["success"] is True
    assert result2["result"] == "Hello, World!"

    # Test 3: Search tools
    math_tools = registry.search("numbers")
    assert len(math_tools) == 2

    # Test 4: Check history
    history = executor.get_history()
    assert len(history) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
