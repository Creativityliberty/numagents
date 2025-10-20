"""
Nüm Agents SDK - A dynamic agent orchestration framework

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

from num_agents.core import Node, Flow, SharedStore, ConditionalNode
from num_agents.exceptions import (
    NumAgentsException,
    FlowException,
    FlowExecutionError,
    FlowConfigurationError,
    NodeException,
    NodeExecutionError,
    NodeNotImplementedError,
    SharedStoreException,
    SharedStoreKeyError,
    ValidationException,
    SpecificationError,
    CatalogError,
    SerializationException,
    SerializationError,
    DeserializationError,
)
from num_agents.logging_config import get_logger, configure_logging, set_log_level
from num_agents.serialization import FlowSerializer, FlowDeserializer

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Node",
    "Flow",
    "SharedStore",
    "ConditionalNode",
    # Exceptions
    "NumAgentsException",
    "FlowException",
    "FlowExecutionError",
    "FlowConfigurationError",
    "NodeException",
    "NodeExecutionError",
    "NodeNotImplementedError",
    "SharedStoreException",
    "SharedStoreKeyError",
    "ValidationException",
    "SpecificationError",
    "CatalogError",
    "SerializationException",
    "SerializationError",
    "DeserializationError",
    # Logging
    "get_logger",
    "configure_logging",
    "set_log_level",
    # Serialization
    "FlowSerializer",
    "FlowDeserializer",
    # Version
    "__version__",
]
