"""
ToolLayer - Tool and Action Management for AI Agents

This module provides function calling, tool registry, and execution capabilities
for AI agents to interact with the real world.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import inspect
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Union
import uuid
import time

from num_agents.core import Node, SharedStore
from num_agents.exceptions import NumAgentsException
from num_agents.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class ToolLayerException(NumAgentsException):
    """Base exception for ToolLayer errors."""

    pass


class ToolExecutionError(ToolLayerException):
    """Exception raised when tool execution fails."""

    pass


class ToolRegistrationError(ToolLayerException):
    """Exception raised when tool registration fails."""

    pass


class ToolNotFoundError(ToolLayerException):
    """Exception raised when requested tool is not found."""

    pass


class ToolValidationError(ToolLayerException):
    """Exception raised when tool input validation fails."""

    pass


# ============================================================================
# Tool Abstract Base Class
# ============================================================================


class Tool(ABC):
    """
    Abstract base class for tools.

    A Tool is a capability that an agent can use to perform actions
    in the real world (e.g., call APIs, execute functions, query databases).
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a tool.

        Args:
            name: Unique tool name (e.g., "search_web", "send_email")
            description: Human-readable description of what the tool does
            parameters: JSON schema of expected parameters
            metadata: Optional metadata (tags, version, author, etc.)
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.metadata = metadata or {}
        self.created_at = time.time()

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Tool execution result

        Raises:
            ToolExecutionError: If execution fails
        """
        pass

    @abstractmethod
    def validate_input(self, **kwargs: Any) -> bool:
        """
        Validate input parameters before execution.

        Args:
            **kwargs: Parameters to validate

        Returns:
            True if valid

        Raises:
            ToolValidationError: If validation fails
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


# ============================================================================
# Concrete Tool Implementations
# ============================================================================


class PythonFunctionTool(Tool):
    """
    Tool that wraps a Python function for agent use.

    This is the most common tool type - wrapping existing Python functions.
    """

    def __init__(
        self,
        name: str,
        function: Callable,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        safe_mode: bool = True,
    ) -> None:
        """
        Initialize Python function tool.

        Args:
            name: Tool name
            function: Python callable to execute
            description: Tool description (auto-generated from docstring if not provided)
            parameters: Parameter schema (auto-generated from signature if not provided)
            metadata: Optional metadata
            safe_mode: Enable safety checks (parameter validation, etc.)
        """
        self.function = function
        self.safe_mode = safe_mode

        # Auto-generate description from docstring
        if description is None:
            description = function.__doc__ or f"Execute {function.__name__}"

        # Auto-generate parameters from function signature
        if parameters is None:
            parameters = self._infer_parameters()

        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            metadata=metadata,
        )

    def _infer_parameters(self) -> Dict[str, Any]:
        """Infer parameter schema from function signature."""
        sig = inspect.signature(self.function)
        schema = {"type": "object", "properties": {}, "required": []}

        for param_name, param in sig.parameters.items():
            param_info = {"type": "string"}  # Default type

            # Infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                annotation = param.annotation
                if annotation == int:
                    param_info["type"] = "integer"
                elif annotation == float:
                    param_info["type"] = "number"
                elif annotation == bool:
                    param_info["type"] = "boolean"
                elif annotation == list:
                    param_info["type"] = "array"
                elif annotation == dict:
                    param_info["type"] = "object"

            schema["properties"][param_name] = param_info

            # Mark as required if no default value
            if param.default == inspect.Parameter.empty:
                schema["required"].append(param_name)

        return schema

    def validate_input(self, **kwargs: Any) -> bool:
        """Validate input parameters against function signature."""
        if not self.safe_mode:
            return True

        # Check required parameters
        required = self.parameters.get("required", [])
        for param in required:
            if param not in kwargs:
                raise ToolValidationError(
                    f"Missing required parameter: {param} for tool {self.name}"
                )

        # Check for unexpected parameters
        expected = set(self.parameters.get("properties", {}).keys())
        provided = set(kwargs.keys())
        unexpected = provided - expected

        if unexpected:
            raise ToolValidationError(
                f"Unexpected parameters: {unexpected} for tool {self.name}"
            )

        return True

    def execute(self, **kwargs: Any) -> Any:
        """Execute the wrapped Python function."""
        try:
            # Validate input
            self.validate_input(**kwargs)

            # Execute function
            result = self.function(**kwargs)

            return result

        except ToolValidationError:
            raise
        except Exception as e:
            raise ToolExecutionError(
                f"Tool {self.name} execution failed: {str(e)}"
            ) from e


class PythonCodeTool(Tool):
    """
    Tool that executes arbitrary Python code (sandboxed).

    WARNING: This can be dangerous! Only use with trusted code.
    For production, use proper sandboxing (RestrictedPython, containers, etc.)
    """

    def __init__(
        self,
        name: str,
        code: str,
        description: str,
        allowed_imports: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize Python code tool.

        Args:
            name: Tool name
            code: Python code to execute (must define an 'execute' function)
            description: Tool description
            allowed_imports: Set of allowed import statements for safety
            metadata: Optional metadata
        """
        self.code = code
        self.allowed_imports = allowed_imports or {"json", "math", "re"}
        self._compiled_code = None

        super().__init__(
            name=name, description=description, parameters={}, metadata=metadata
        )

        # Compile code on initialization
        self._compile_code()

    def _compile_code(self) -> None:
        """Compile the Python code."""
        try:
            # Basic safety: check for dangerous imports
            lines = self.code.split("\n")
            for line in lines:
                if line.strip().startswith("import ") or " import " in line:
                    module = line.split("import")[1].strip().split()[0]
                    if module not in self.allowed_imports:
                        raise ToolValidationError(
                            f"Import '{module}' not allowed for tool {self.name}"
                        )

            # Compile code
            self._compiled_code = compile(self.code, f"<tool:{self.name}>", "exec")

        except SyntaxError as e:
            raise ToolValidationError(f"Invalid Python code in tool {self.name}: {e}")

    def validate_input(self, **kwargs: Any) -> bool:
        """Validate that code has been compiled."""
        if self._compiled_code is None:
            raise ToolValidationError(f"Tool {self.name} code not compiled")
        return True

    def execute(self, **kwargs: Any) -> Any:
        """Execute the Python code."""
        try:
            self.validate_input(**kwargs)

            # Create restricted execution namespace
            namespace = {
                "__builtins__": {
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                    "print": print,
                }
            }

            # Add input parameters to namespace
            namespace.update(kwargs)

            # Execute code
            exec(self._compiled_code, namespace)

            # Code must define an 'execute' function or return 'result'
            if "execute" in namespace:
                return namespace["execute"](**kwargs)
            elif "result" in namespace:
                return namespace["result"]
            else:
                raise ToolExecutionError(
                    f"Tool {self.name} code must define 'execute' function or 'result' variable"
                )

        except Exception as e:
            raise ToolExecutionError(
                f"Tool {self.name} execution failed: {str(e)}"
            ) from e


class ChainedTool(Tool):
    """
    Tool that chains multiple tools together.

    Output of one tool becomes input to the next.
    """

    def __init__(
        self,
        name: str,
        tools: List[Tool],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize chained tool.

        Args:
            name: Tool name
            tools: List of tools to chain (executed in order)
            description: Tool description
            metadata: Optional metadata
        """
        self.tools = tools

        if description is None:
            tool_names = " -> ".join(t.name for t in tools)
            description = f"Chain of tools: {tool_names}"

        super().__init__(
            name=name, description=description, parameters={}, metadata=metadata
        )

    def validate_input(self, **kwargs: Any) -> bool:
        """Validate input for first tool in chain."""
        if not self.tools:
            raise ToolValidationError(f"No tools in chain for {self.name}")
        return self.tools[0].validate_input(**kwargs)

    def execute(self, **kwargs: Any) -> Any:
        """Execute tools in sequence."""
        try:
            result = kwargs

            for i, tool in enumerate(self.tools):
                logger.debug(f"Chain {self.name}: Executing tool {i + 1}/{len(self.tools)}: {tool.name}")

                # Execute tool
                if isinstance(result, dict):
                    result = tool.execute(**result)
                else:
                    result = tool.execute(input=result)

            return result

        except Exception as e:
            raise ToolExecutionError(
                f"Chained tool {self.name} failed: {str(e)}"
            ) from e


# ============================================================================
# Tool Registry
# ============================================================================


class ToolRegistry:
    """
    Registry for managing available tools.

    Central catalog of tools that agents can discover and use.
    """

    def __init__(self, enable_logging: bool = False) -> None:
        """
        Initialize tool registry.

        Args:
            enable_logging: Enable detailed logging
        """
        self._tools: Dict[str, Tool] = {}
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None

    def register(self, tool: Tool, overwrite: bool = False) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool to register
            overwrite: Allow overwriting existing tool with same name

        Raises:
            ToolRegistrationError: If tool name already exists and overwrite=False
        """
        if tool.name in self._tools and not overwrite:
            raise ToolRegistrationError(
                f"Tool '{tool.name}' already registered. Use overwrite=True to replace."
            )

        self._tools[tool.name] = tool

        if self._enable_logging and self._logger:
            self._logger.info(f"Registered tool: {tool.name}")

    def register_function(
        self,
        name: str,
        function: Callable,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> PythonFunctionTool:
        """
        Convenience method to register a Python function as a tool.

        Args:
            name: Tool name
            function: Python callable
            description: Tool description
            **kwargs: Additional arguments for PythonFunctionTool

        Returns:
            Created PythonFunctionTool instance
        """
        tool = PythonFunctionTool(
            name=name, function=function, description=description, **kwargs
        )
        self.register(tool)
        return tool

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.

        Args:
            name: Tool name

        Returns:
            True if unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            if self._enable_logging and self._logger:
                self._logger.info(f"Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def get_required(self, name: str) -> Tool:
        """
        Get a tool by name (raises if not found).

        Args:
            name: Tool name

        Returns:
            Tool instance

        Raises:
            ToolNotFoundError: If tool not found
        """
        tool = self.get(name)
        if tool is None:
            raise ToolNotFoundError(f"Tool '{name}' not found in registry")
        return tool

    def list_tools(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def count(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export registry to dictionary."""
        return {
            "tools": {name: tool.to_dict() for name, tool in self._tools.items()},
            "count": len(self._tools),
        }

    def search(self, query: str) -> List[Tool]:
        """
        Search for tools by name or description.

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching tools
        """
        query_lower = query.lower()
        matches = []

        for tool in self._tools.values():
            if (
                query_lower in tool.name.lower()
                or query_lower in tool.description.lower()
            ):
                matches.append(tool)

        return matches


# ============================================================================
# Tool Executor
# ============================================================================


class ToolExecutor:
    """
    Executes tools with safety checks, logging, and error handling.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        enable_logging: bool = False,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Initialize tool executor.

        Args:
            registry: ToolRegistry to execute tools from
            enable_logging: Enable detailed logging
            timeout: Optional execution timeout in seconds (not implemented yet)
        """
        self.registry = registry
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None
        self.timeout = timeout
        self._execution_history: List[Dict[str, Any]] = []

    def execute(
        self, tool_name: str, record_history: bool = True, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of tool to execute
            record_history: Record execution in history
            **kwargs: Tool parameters

        Returns:
            Dictionary with execution results:
                - success: bool
                - result: Any (if successful)
                - error: str (if failed)
                - execution_time: float (seconds)
                - tool_name: str

        Raises:
            ToolNotFoundError: If tool not found
        """
        start_time = time.time()

        try:
            # Get tool
            tool = self.registry.get_required(tool_name)

            if self._enable_logging and self._logger:
                self._logger.info(
                    f"Executing tool: {tool_name} with params: {list(kwargs.keys())}"
                )

            # Execute
            result = tool.execute(**kwargs)
            execution_time = time.time() - start_time

            execution_result = {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "tool_name": tool_name,
                "timestamp": time.time(),
            }

            if self._enable_logging and self._logger:
                self._logger.info(
                    f"Tool {tool_name} executed successfully in {execution_time:.3f}s"
                )

            # Record history
            if record_history:
                self._execution_history.append(
                    {**execution_result, "parameters": kwargs}
                )

            return execution_result

        except (ToolNotFoundError, ToolExecutionError, ToolValidationError) as e:
            execution_time = time.time() - start_time

            execution_result = {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "tool_name": tool_name,
                "timestamp": time.time(),
            }

            if self._enable_logging and self._logger:
                self._logger.error(f"Tool {tool_name} failed: {e}")

            # Record history
            if record_history:
                self._execution_history.append(
                    {**execution_result, "parameters": kwargs}
                )

            return execution_result

    def execute_chain(
        self, tool_names: List[str], initial_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute multiple tools in sequence.

        Args:
            tool_names: List of tool names to execute in order
            initial_input: Initial parameters for first tool

        Returns:
            Final execution result
        """
        result = initial_input

        for tool_name in tool_names:
            execution_result = self.execute(tool_name, **result)

            if not execution_result["success"]:
                return execution_result

            # Use result as input for next tool
            result = {"input": execution_result["result"]}

        return execution_result

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get execution history.

        Args:
            limit: Optional limit on number of entries to return (most recent)

        Returns:
            List of execution records
        """
        if limit:
            return self._execution_history[-limit:]
        return self._execution_history.copy()

    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()


# ============================================================================
# Nodes for Flow Integration
# ============================================================================


class ToolExecuteNode(Node):
    """
    Node that executes a single tool.

    Reads parameters from SharedStore, executes tool, and stores result.
    """

    def __init__(
        self,
        executor: ToolExecutor,
        tool_name: str,
        input_key: str = "tool_input",
        output_key: str = "tool_result",
        params_key: Optional[str] = None,
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize ToolExecuteNode.

        Args:
            executor: ToolExecutor instance
            tool_name: Name of tool to execute
            input_key: Key in SharedStore to read input from (if params_key not set)
            output_key: Key in SharedStore to write result to
            params_key: Optional key to read full parameters dict from
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(name or f"ToolExecute_{tool_name}", enable_logging=enable_logging)
        self.executor = executor
        self.tool_name = tool_name
        self.input_key = input_key
        self.output_key = output_key
        self.params_key = params_key

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute tool.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with execution results
        """
        # Get parameters
        if self.params_key:
            params = shared.get_required(self.params_key)
        else:
            input_value = shared.get_required(self.input_key)
            params = {"input": input_value} if not isinstance(input_value, dict) else input_value

        # Execute tool
        result = self.executor.execute(self.tool_name, **params)

        # Store result
        shared.set(self.output_key, result)

        return result


class ToolChainNode(Node):
    """
    Node that executes multiple tools in sequence.

    Each tool's output becomes the next tool's input.
    """

    def __init__(
        self,
        executor: ToolExecutor,
        tool_names: List[str],
        input_key: str = "chain_input",
        output_key: str = "chain_result",
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize ToolChainNode.

        Args:
            executor: ToolExecutor instance
            tool_names: List of tool names to execute in sequence
            input_key: Key in SharedStore to read initial input from
            output_key: Key in SharedStore to write final result to
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(
            name or f"ToolChain_{len(tool_names)}", enable_logging=enable_logging
        )
        self.executor = executor
        self.tool_names = tool_names
        self.input_key = input_key
        self.output_key = output_key

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute tool chain.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with execution results
        """
        # Get initial input
        initial_input = shared.get_required(self.input_key)

        # Ensure it's a dict
        if not isinstance(initial_input, dict):
            initial_input = {"input": initial_input}

        # Execute chain
        result = self.executor.execute_chain(self.tool_names, initial_input)

        # Store result
        shared.set(self.output_key, result)

        return result
