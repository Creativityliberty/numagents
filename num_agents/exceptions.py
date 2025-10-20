"""
Custom exceptions for the Nüm Agents SDK.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

from typing import Any, Optional


class NumAgentsException(Exception):
    """Base exception for all Nüm Agents SDK exceptions."""

    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        """
        Initialize the exception.

        Args:
            message: The error message
            details: Optional additional details about the error
        """
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class FlowException(NumAgentsException):
    """Exception raised during flow execution."""

    pass


class FlowExecutionError(FlowException):
    """Exception raised when a flow fails to execute."""

    pass


class FlowConfigurationError(FlowException):
    """Exception raised when a flow is misconfigured."""

    pass


class NodeException(NumAgentsException):
    """Exception raised during node execution."""

    pass


class NodeExecutionError(NodeException):
    """Exception raised when a node fails to execute."""

    def __init__(
        self, message: str, node_name: str, node_id: str, details: Optional[Any] = None
    ) -> None:
        """
        Initialize the node execution error.

        Args:
            message: The error message
            node_name: Name of the node that failed
            node_id: ID of the node that failed
            details: Optional additional details about the error
        """
        super().__init__(message, details)
        self.node_name = node_name
        self.node_id = node_id

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        base_msg = f"Node '{self.node_name}' ({self.node_id[:8]}) failed: {self.message}"
        if self.details:
            return f"{base_msg} | Details: {self.details}"
        return base_msg


class NodeNotImplementedError(NodeException):
    """Exception raised when a node's exec method is not implemented."""

    pass


class SharedStoreException(NumAgentsException):
    """Exception raised during shared store operations."""

    pass


class SharedStoreKeyError(SharedStoreException):
    """Exception raised when a required key is missing from the shared store."""

    def __init__(self, key: str, message: Optional[str] = None) -> None:
        """
        Initialize the shared store key error.

        Args:
            key: The missing key
            message: Optional custom error message
        """
        if message is None:
            message = f"Required key '{key}' not found in shared store"
        super().__init__(message)
        self.key = key


class ValidationException(NumAgentsException):
    """Exception raised during validation."""

    pass


class SpecificationError(ValidationException):
    """Exception raised when an agent specification is invalid."""

    pass


class CatalogError(ValidationException):
    """Exception raised when a universe catalog is invalid."""

    pass


class SerializationException(NumAgentsException):
    """Exception raised during serialization/deserialization."""

    pass


class DeserializationError(SerializationException):
    """Exception raised when deserialization fails."""

    pass


class SerializationError(SerializationException):
    """Exception raised when serialization fails."""

    pass
