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

# Optional KnowledgeLayer imports (requires openai for OpenAIEmbeddingProvider)
try:
    from num_agents.modules.knowledge_layer import (
        KnowledgeStore,
        Memory,
        MemoryStoreNode,
        MemoryRecallNode,
        EmbeddingProvider,
        SimpleHashEmbeddingProvider,
        OpenAIEmbeddingProvider,
        KnowledgeLayerException,
        EmbeddingProviderError,
        VectorStoreError,
    )

    _KNOWLEDGE_LAYER_AVAILABLE = True
except ImportError:
    _KNOWLEDGE_LAYER_AVAILABLE = False

# ToolLayer imports
try:
    from num_agents.modules.tool_layer import (
        Tool,
        PythonFunctionTool,
        PythonCodeTool,
        ChainedTool,
        ToolRegistry,
        ToolExecutor,
        ToolExecuteNode,
        ToolChainNode,
        ToolLayerException,
        ToolExecutionError,
        ToolRegistrationError,
        ToolNotFoundError,
        ToolValidationError,
    )

    _TOOL_LAYER_AVAILABLE = True
except ImportError:
    _TOOL_LAYER_AVAILABLE = False

# StateLayer imports
try:
    from num_agents.modules.state_layer import (
        State,
        StateTransition,
        Checkpoint,
        StateMachine,
        StateManager,
        CheckpointManager,
        StateBackend,
        InMemoryBackend,
        FileBackend,
        PickleBackend,
        StateTransitionNode,
        CheckpointNode,
        StateLayerException,
        StateTransitionError,
        CheckpointError,
        StateValidationError,
        StatePersistenceError,
    )

    _STATE_LAYER_AVAILABLE = True
except ImportError:
    _STATE_LAYER_AVAILABLE = False

# SecurityLayer imports
try:
    from num_agents.modules.security_layer import (
        Authenticator,
        TokenAuthenticator,
        APIKeyAuthenticator,
        Sanitizer,
        RegexSanitizer,
        HTMLSanitizer,
        SecretsProvider,
        EnvironmentSecretsProvider,
        FileSecretsProvider,
        AuditLogger,
        RateLimiter,
        SecurityManager,
        AuthenticationNode,
        SanitizationNode,
        AuditNode,
        SecurityLayerException,
        AuthenticationError,
        AuthorizationError,
        SanitizationError,
        SecretsError,
        RateLimitError,
        ContentFilterError,
    )

    _SECURITY_LAYER_AVAILABLE = True
except ImportError:
    _SECURITY_LAYER_AVAILABLE = False

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

# Add KnowledgeLayer to __all__ if available
if _KNOWLEDGE_LAYER_AVAILABLE:
    __all__.extend(
        [
            # KnowledgeLayer
            "KnowledgeStore",
            "Memory",
            "MemoryStoreNode",
            "MemoryRecallNode",
            "EmbeddingProvider",
            "SimpleHashEmbeddingProvider",
            "OpenAIEmbeddingProvider",
            "KnowledgeLayerException",
            "EmbeddingProviderError",
            "VectorStoreError",
        ]
    )

# Add ToolLayer to __all__ if available
if _TOOL_LAYER_AVAILABLE:
    __all__.extend(
        [
            # ToolLayer
            "Tool",
            "PythonFunctionTool",
            "PythonCodeTool",
            "ChainedTool",
            "ToolRegistry",
            "ToolExecutor",
            "ToolExecuteNode",
            "ToolChainNode",
            "ToolLayerException",
            "ToolExecutionError",
            "ToolRegistrationError",
            "ToolNotFoundError",
            "ToolValidationError",
        ]
    )

# Add StateLayer to __all__ if available
if _STATE_LAYER_AVAILABLE:
    __all__.extend(
        [
            # StateLayer
            "State",
            "StateTransition",
            "Checkpoint",
            "StateMachine",
            "StateManager",
            "CheckpointManager",
            "StateBackend",
            "InMemoryBackend",
            "FileBackend",
            "PickleBackend",
            "StateTransitionNode",
            "CheckpointNode",
            "StateLayerException",
            "StateTransitionError",
            "CheckpointError",
            "StateValidationError",
            "StatePersistenceError",
        ]
    )

# Add SecurityLayer to __all__ if available
if _SECURITY_LAYER_AVAILABLE:
    __all__.extend(
        [
            # SecurityLayer
            "Authenticator",
            "TokenAuthenticator",
            "APIKeyAuthenticator",
            "Sanitizer",
            "RegexSanitizer",
            "HTMLSanitizer",
            "SecretsProvider",
            "EnvironmentSecretsProvider",
            "FileSecretsProvider",
            "AuditLogger",
            "RateLimiter",
            "SecurityManager",
            "AuthenticationNode",
            "SanitizationNode",
            "AuditNode",
            "SecurityLayerException",
            "AuthenticationError",
            "AuthorizationError",
            "SanitizationError",
            "SecretsError",
            "RateLimitError",
            "ContentFilterError",
        ]
    )
