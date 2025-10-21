"""
SecurityLayer - Security and Safety for AI Agents

This module provides authentication, input sanitization, secrets management,
audit logging, rate limiting, and content filtering for production agents.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import hashlib
import hmac
import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

from num_agents.core import Node, SharedStore
from num_agents.exceptions import NumAgentsException
from num_agents.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class SecurityLayerException(NumAgentsException):
    """Base exception for SecurityLayer errors."""

    pass


class AuthenticationError(SecurityLayerException):
    """Exception raised when authentication fails."""

    pass


class AuthorizationError(SecurityLayerException):
    """Exception raised when authorization fails."""

    pass


class SanitizationError(SecurityLayerException):
    """Exception raised when input sanitization fails."""

    pass


class SecretsError(SecurityLayerException):
    """Exception raised when secrets management fails."""

    pass


class RateLimitError(SecurityLayerException):
    """Exception raised when rate limit is exceeded."""

    pass


class ContentFilterError(SecurityLayerException):
    """Exception raised when content filtering detects violations."""

    pass


# ============================================================================
# Authenticator (Abstract)
# ============================================================================


class Authenticator(ABC):
    """
    Abstract base class for authentication providers.

    Authenticators verify identity of users/agents.
    """

    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Authenticate user/agent.

        Args:
            credentials: Credentials dictionary (token, username/password, etc.)

        Returns:
            Dictionary with authentication result:
                - authenticated: bool
                - user_id: str (if authenticated)
                - roles: List[str] (if applicable)
                - metadata: Dict[str, Any]

        Raises:
            AuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate an authentication token.

        Args:
            token: Token to validate

        Returns:
            Dictionary with validation result

        Raises:
            AuthenticationError: If token is invalid
        """
        pass


# ============================================================================
# Concrete Authenticator Implementations
# ============================================================================


class TokenAuthenticator(Authenticator):
    """
    Simple token-based authentication.

    Uses HMAC-SHA256 for token generation and validation.
    """

    def __init__(self, secret_key: str, token_expiry: int = 3600) -> None:
        """
        Initialize token authenticator.

        Args:
            secret_key: Secret key for HMAC signing
            token_expiry: Token expiry time in seconds (default: 1 hour)
        """
        self.secret_key = secret_key.encode()
        self.token_expiry = token_expiry
        self._active_tokens: Dict[str, Dict[str, Any]] = {}

    def generate_token(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an authentication token.

        Args:
            user_id: User/agent ID
            metadata: Optional metadata to embed in token

        Returns:
            Authentication token
        """
        # Create token payload
        issued_at = int(time.time())
        expires_at = issued_at + self.token_expiry

        payload = {
            "user_id": user_id,
            "issued_at": issued_at,
            "expires_at": expires_at,
            "metadata": metadata or {},
        }

        # Generate token (HMAC of payload)
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            self.secret_key, payload_str.encode(), hashlib.sha256
        ).hexdigest()

        # Combine payload and signature
        token = f"{payload_str}|{signature}"

        # Store active token
        self._active_tokens[signature] = payload

        return token

    def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Authenticate using token.

        Args:
            credentials: Must contain "token" key

        Returns:
            Authentication result
        """
        token = credentials.get("token")
        if not token:
            raise AuthenticationError("Missing token in credentials")

        return self.validate_token(token)

    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate an authentication token.

        Args:
            token: Token to validate

        Returns:
            Validation result with user info
        """
        try:
            # Split token
            payload_str, signature = token.split("|", 1)

            # Verify signature
            expected_signature = hmac.new(
                self.secret_key, payload_str.encode(), hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                raise AuthenticationError("Invalid token signature")

            # Parse payload
            payload = json.loads(payload_str)

            # Check expiry
            if time.time() > payload["expires_at"]:
                raise AuthenticationError("Token expired")

            return {
                "authenticated": True,
                "user_id": payload["user_id"],
                "metadata": payload.get("metadata", {}),
                "expires_at": payload["expires_at"],
            }

        except (ValueError, KeyError) as e:
            raise AuthenticationError(f"Invalid token format: {e}")

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token.

        Args:
            token: Token to revoke

        Returns:
            True if revoked, False if not found
        """
        try:
            _, signature = token.split("|", 1)
            if signature in self._active_tokens:
                del self._active_tokens[signature]
                return True
        except ValueError:
            pass
        return False


class APIKeyAuthenticator(Authenticator):
    """
    API key-based authentication.

    Simple authentication using pre-configured API keys.
    """

    def __init__(self, api_keys: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Initialize API key authenticator.

        Args:
            api_keys: Dictionary mapping API keys to user info
                      Example: {"key123": {"user_id": "user1", "roles": ["admin"]}}
        """
        self._api_keys = api_keys or {}

    def add_key(
        self, api_key: str, user_id: str, roles: Optional[List[str]] = None
    ) -> None:
        """
        Add an API key.

        Args:
            api_key: API key
            user_id: Associated user ID
            roles: Optional list of roles
        """
        self._api_keys[api_key] = {"user_id": user_id, "roles": roles or []}

    def remove_key(self, api_key: str) -> bool:
        """
        Remove an API key.

        Args:
            api_key: API key to remove

        Returns:
            True if removed, False if not found
        """
        if api_key in self._api_keys:
            del self._api_keys[api_key]
            return True
        return False

    def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Authenticate using API key.

        Args:
            credentials: Must contain "api_key" key

        Returns:
            Authentication result
        """
        api_key = credentials.get("api_key")
        if not api_key:
            raise AuthenticationError("Missing api_key in credentials")

        return self.validate_token(api_key)

    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate an API key.

        Args:
            token: API key to validate

        Returns:
            Validation result
        """
        if token not in self._api_keys:
            raise AuthenticationError("Invalid API key")

        user_info = self._api_keys[token]

        return {
            "authenticated": True,
            "user_id": user_info["user_id"],
            "roles": user_info.get("roles", []),
        }


# ============================================================================
# Input Sanitizer (Abstract)
# ============================================================================


class Sanitizer(ABC):
    """
    Abstract base class for input sanitizers.

    Sanitizers clean and validate user inputs to prevent injection attacks.
    """

    @abstractmethod
    def sanitize(self, input_data: str) -> str:
        """
        Sanitize input string.

        Args:
            input_data: Input to sanitize

        Returns:
            Sanitized input

        Raises:
            SanitizationError: If sanitization fails or input is rejected
        """
        pass

    @abstractmethod
    def validate(self, input_data: str) -> bool:
        """
        Validate input without modification.

        Args:
            input_data: Input to validate

        Returns:
            True if valid

        Raises:
            SanitizationError: If input is invalid
        """
        pass


# ============================================================================
# Concrete Sanitizer Implementations
# ============================================================================


class RegexSanitizer(Sanitizer):
    """
    Regex-based input sanitizer.

    Blocks or removes patterns matching dangerous content.
    """

    def __init__(
        self,
        blocked_patterns: Optional[List[str]] = None,
        remove_patterns: Optional[List[str]] = None,
        max_length: Optional[int] = None,
    ) -> None:
        """
        Initialize regex sanitizer.

        Args:
            blocked_patterns: Patterns that cause rejection if found
            remove_patterns: Patterns to remove from input
            max_length: Maximum input length
        """
        self.blocked_patterns = [
            re.compile(p, re.IGNORECASE) for p in (blocked_patterns or [])
        ]
        self.remove_patterns = [
            re.compile(p, re.IGNORECASE) for p in (remove_patterns or [])
        ]
        self.max_length = max_length

        # Default dangerous patterns (SQL injection, command injection, etc.)
        self._add_default_patterns()

    def _add_default_patterns(self) -> None:
        """Add default dangerous patterns."""
        # SQL injection patterns
        sql_patterns = [
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\bDROP\b.*\bTABLE\b)",
            r"(\bDELETE\b.*\bFROM\b)",
            r"(--\s*$)",  # SQL comments
            r"(/\*.*\*/)",  # SQL comments
        ]

        # Command injection patterns
        cmd_patterns = [
            r"(;\s*rm\s+-rf)",
            r"(&&\s*rm\s+)",
            r"(\|\s*rm\s+)",
            r"(`[^`]+`)",  # Backtick execution
            r"(\$\([^)]+\))",  # Command substitution
        ]

        # Add to blocked patterns
        for pattern in sql_patterns + cmd_patterns:
            self.blocked_patterns.append(re.compile(pattern, re.IGNORECASE))

    def sanitize(self, input_data: str) -> str:
        """
        Sanitize input.

        Args:
            input_data: Input to sanitize

        Returns:
            Sanitized input
        """
        # Check length
        if self.max_length and len(input_data) > self.max_length:
            raise SanitizationError(
                f"Input exceeds max length ({self.max_length} chars)"
            )

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(input_data):
                raise SanitizationError(
                    f"Input contains blocked pattern: {pattern.pattern}"
                )

        # Remove patterns
        sanitized = input_data
        for pattern in self.remove_patterns:
            sanitized = pattern.sub("", sanitized)

        return sanitized

    def validate(self, input_data: str) -> bool:
        """
        Validate input without modification.

        Args:
            input_data: Input to validate

        Returns:
            True if valid
        """
        try:
            self.sanitize(input_data)
            return True
        except SanitizationError:
            return False


class HTMLSanitizer(Sanitizer):
    """
    HTML sanitizer to prevent XSS attacks.

    Escapes HTML entities and removes dangerous tags.
    """

    def __init__(
        self, allowed_tags: Optional[Set[str]] = None, escape_all: bool = False
    ) -> None:
        """
        Initialize HTML sanitizer.

        Args:
            allowed_tags: Set of allowed HTML tags (if escape_all=False)
            escape_all: Escape all HTML (default: False)
        """
        self.allowed_tags = allowed_tags or {"b", "i", "u", "em", "strong", "p"}
        self.escape_all = escape_all

    def sanitize(self, input_data: str) -> str:
        """
        Sanitize HTML input.

        Args:
            input_data: HTML input

        Returns:
            Sanitized HTML
        """
        if self.escape_all:
            return self._escape_html(input_data)

        # Remove dangerous tags
        dangerous_tags = {"script", "iframe", "object", "embed", "style"}
        sanitized = input_data

        for tag in dangerous_tags:
            sanitized = re.sub(
                f"<{tag}[^>]*>.*?</{tag}>", "", sanitized, flags=re.IGNORECASE | re.DOTALL
            )
            sanitized = re.sub(f"<{tag}[^>]*/>", "", sanitized, flags=re.IGNORECASE)

        # Remove event handlers
        sanitized = re.sub(r'\s+on\w+="[^"]*"', "", sanitized)
        sanitized = re.sub(r"\s+on\w+='[^']*'", "", sanitized)

        return sanitized

    def validate(self, input_data: str) -> bool:
        """Validate HTML input."""
        sanitized = self.sanitize(input_data)
        return len(sanitized) > 0

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML entities."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )


# ============================================================================
# Secrets Provider (Abstract)
# ============================================================================


class SecretsProvider(ABC):
    """
    Abstract base class for secrets management.

    Providers securely store and retrieve secrets (API keys, passwords, etc.)
    """

    @abstractmethod
    def get_secret(self, key: str) -> str:
        """
        Get a secret value.

        Args:
            key: Secret key

        Returns:
            Secret value

        Raises:
            SecretsError: If secret not found or error occurs
        """
        pass

    @abstractmethod
    def set_secret(self, key: str, value: str) -> None:
        """
        Set a secret value.

        Args:
            key: Secret key
            value: Secret value

        Raises:
            SecretsError: If error occurs
        """
        pass

    @abstractmethod
    def delete_secret(self, key: str) -> bool:
        """
        Delete a secret.

        Args:
            key: Secret key

        Returns:
            True if deleted, False if not found
        """
        pass


# ============================================================================
# Concrete Secrets Provider Implementations
# ============================================================================


class EnvironmentSecretsProvider(SecretsProvider):
    """
    Secrets provider using environment variables.

    Simple provider that reads from os.environ.
    """

    def __init__(self, prefix: str = "NUMAGENTS_SECRET_") -> None:
        """
        Initialize environment secrets provider.

        Args:
            prefix: Prefix for environment variables
        """
        self.prefix = prefix

    def get_secret(self, key: str) -> str:
        """Get secret from environment variable."""
        env_key = f"{self.prefix}{key.upper()}"
        value = os.environ.get(env_key)

        if value is None:
            raise SecretsError(f"Secret '{key}' not found in environment")

        return value

    def set_secret(self, key: str, value: str) -> None:
        """Set secret in environment variable."""
        env_key = f"{self.prefix}{key.upper()}"
        os.environ[env_key] = value

    def delete_secret(self, key: str) -> bool:
        """Delete secret from environment."""
        env_key = f"{self.prefix}{key.upper()}"
        if env_key in os.environ:
            del os.environ[env_key]
            return True
        return False


class FileSecretsProvider(SecretsProvider):
    """
    Secrets provider using encrypted JSON file.

    WARNING: This is a simple implementation. For production,
    use proper secrets management (AWS Secrets Manager, HashiCorp Vault, etc.)
    """

    def __init__(self, filepath: Union[str, Path], encryption_key: Optional[str] = None) -> None:
        """
        Initialize file secrets provider.

        Args:
            filepath: Path to secrets file
            encryption_key: Optional encryption key (not implemented yet)
        """
        self.filepath = Path(filepath)
        self.encryption_key = encryption_key
        self._secrets: Dict[str, str] = {}

        # Load existing secrets
        if self.filepath.exists():
            self._load()

    def _load(self) -> None:
        """Load secrets from file."""
        try:
            with open(self.filepath, "r") as f:
                self._secrets = json.load(f)
        except Exception as e:
            raise SecretsError(f"Failed to load secrets: {e}") from e

    def _save(self) -> None:
        """Save secrets to file."""
        try:
            # Ensure parent directory exists
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

            # Set restrictive permissions (owner read/write only)
            with open(self.filepath, "w") as f:
                json.dump(self._secrets, f, indent=2)

            # Make file readable only by owner
            os.chmod(self.filepath, 0o600)

        except Exception as e:
            raise SecretsError(f"Failed to save secrets: {e}") from e

    def get_secret(self, key: str) -> str:
        """Get secret from file."""
        if key not in self._secrets:
            raise SecretsError(f"Secret '{key}' not found")
        return self._secrets[key]

    def set_secret(self, key: str, value: str) -> None:
        """Set secret in file."""
        self._secrets[key] = value
        self._save()

    def delete_secret(self, key: str) -> bool:
        """Delete secret from file."""
        if key in self._secrets:
            del self._secrets[key]
            self._save()
            return True
        return False


# ============================================================================
# Audit Logger
# ============================================================================


class AuditLogger:
    """
    Audit logger for tracking security-relevant events.

    Records who did what, when, and from where.
    """

    def __init__(
        self,
        filepath: Optional[Union[str, Path]] = None,
        console_logging: bool = True,
    ) -> None:
        """
        Initialize audit logger.

        Args:
            filepath: Optional file to write audit logs to
            console_logging: Also log to console
        """
        self.filepath = Path(filepath) if filepath else None
        self.console_logging = console_logging
        self._logger = get_logger(__name__)
        self._audit_log: List[Dict[str, Any]] = []

        # Ensure log directory exists
        if self.filepath:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        result: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an audit event.

        Args:
            event_type: Type of event (auth, access, modification, etc.)
            user_id: ID of user/agent performing action
            action: Action performed
            resource: Resource affected
            result: Result of action (success, failure, etc.)
            metadata: Additional metadata
        """
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "event_type": event_type,
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "result": result,
            "metadata": metadata or {},
        }

        # Add to in-memory log
        self._audit_log.append(event)

        # Log to console
        if self.console_logging:
            self._logger.info(
                f"AUDIT: {event_type} | user={user_id} | action={action} | resource={resource} | result={result}"
            )

        # Log to file
        if self.filepath:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(event) + "\n")

    def get_logs(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get audit logs with optional filtering.

        Args:
            user_id: Filter by user ID
            event_type: Filter by event type
            limit: Maximum number of logs to return (most recent)

        Returns:
            List of audit log entries
        """
        logs = self._audit_log

        # Filter by user_id
        if user_id:
            logs = [log for log in logs if log.get("user_id") == user_id]

        # Filter by event_type
        if event_type:
            logs = [log for log in logs if log.get("event_type") == event_type]

        # Apply limit
        if limit:
            logs = logs[-limit:]

        return logs


# ============================================================================
# Rate Limiter
# ============================================================================


class RateLimiter:
    """
    Rate limiter to prevent abuse.

    Tracks requests per user/IP and enforces limits.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed per window
            window_seconds: Time window in seconds
            enable_logging: Enable detailed logging
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None

    def check_limit(self, identifier: str) -> bool:
        """
        Check if request is allowed.

        Args:
            identifier: Unique identifier (user_id, IP address, etc.)

        Returns:
            True if allowed, False if rate limit exceeded

        Raises:
            RateLimitError: If rate limit exceeded
        """
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds

        # Remove old requests
        self._requests[identifier] = [
            t for t in self._requests[identifier] if t > cutoff_time
        ]

        # Check limit
        if len(self._requests[identifier]) >= self.max_requests:
            if self._enable_logging and self._logger:
                self._logger.warning(
                    f"Rate limit exceeded for {identifier}: {len(self._requests[identifier])}/{self.max_requests}"
                )
            raise RateLimitError(
                f"Rate limit exceeded: {self.max_requests} requests per {self.window_seconds}s"
            )

        # Record request
        self._requests[identifier].append(current_time)

        return True

    def get_usage(self, identifier: str) -> Dict[str, Any]:
        """
        Get rate limit usage for identifier.

        Args:
            identifier: Unique identifier

        Returns:
            Usage statistics
        """
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds

        # Count recent requests
        recent_requests = [
            t for t in self._requests.get(identifier, []) if t > cutoff_time
        ]

        return {
            "identifier": identifier,
            "requests_made": len(recent_requests),
            "requests_allowed": self.max_requests,
            "window_seconds": self.window_seconds,
            "remaining": max(0, self.max_requests - len(recent_requests)),
        }

    def reset(self, identifier: str) -> None:
        """Reset rate limit for identifier."""
        if identifier in self._requests:
            del self._requests[identifier]


# ============================================================================
# Security Manager
# ============================================================================


class SecurityManager:
    """
    Central security manager coordinating all security components.

    Combines authentication, sanitization, secrets, audit, and rate limiting.
    """

    def __init__(
        self,
        authenticator: Optional[Authenticator] = None,
        sanitizer: Optional[Sanitizer] = None,
        secrets_provider: Optional[SecretsProvider] = None,
        audit_logger: Optional[AuditLogger] = None,
        rate_limiter: Optional[RateLimiter] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize security manager.

        Args:
            authenticator: Optional authenticator
            sanitizer: Optional sanitizer
            secrets_provider: Optional secrets provider
            audit_logger: Optional audit logger
            rate_limiter: Optional rate limiter
            enable_logging: Enable detailed logging
        """
        self.authenticator = authenticator
        self.sanitizer = sanitizer
        self.secrets_provider = secrets_provider
        self.audit_logger = audit_logger
        self.rate_limiter = rate_limiter
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None

    def authenticate_request(
        self, credentials: Dict[str, Any], identifier: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Authenticate a request.

        Args:
            credentials: Authentication credentials
            identifier: Optional identifier for rate limiting

        Returns:
            Authentication result
        """
        # Check rate limit
        if self.rate_limiter and identifier:
            self.rate_limiter.check_limit(identifier)

        # Authenticate
        if not self.authenticator:
            raise SecurityLayerException("No authenticator configured")

        result = self.authenticator.authenticate(credentials)

        # Audit log
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="authentication",
                user_id=result.get("user_id"),
                action="authenticate",
                result="success" if result.get("authenticated") else "failure",
            )

        return result

    def sanitize_input(self, input_data: str) -> str:
        """
        Sanitize user input.

        Args:
            input_data: Input to sanitize

        Returns:
            Sanitized input
        """
        if not self.sanitizer:
            return input_data

        return self.sanitizer.sanitize(input_data)

    def get_secret(self, key: str, user_id: Optional[str] = None) -> str:
        """
        Get a secret value.

        Args:
            key: Secret key
            user_id: Optional user ID for audit logging

        Returns:
            Secret value
        """
        if not self.secrets_provider:
            raise SecurityLayerException("No secrets provider configured")

        secret = self.secrets_provider.get_secret(key)

        # Audit log
        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="secret_access",
                user_id=user_id,
                action="get_secret",
                resource=key,
                result="success",
            )

        return secret


# ============================================================================
# Nodes for Flow Integration
# ============================================================================


class AuthenticationNode(Node):
    """
    Node that performs authentication.

    Reads credentials from SharedStore and authenticates.
    """

    def __init__(
        self,
        security_manager: SecurityManager,
        credentials_key: str = "credentials",
        identifier_key: Optional[str] = None,
        output_key: str = "auth_result",
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize AuthenticationNode.

        Args:
            security_manager: SecurityManager instance
            credentials_key: Key in SharedStore to read credentials from
            identifier_key: Optional key to read rate limit identifier from
            output_key: Key in SharedStore to write result to
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(name or "Authentication", enable_logging=enable_logging)
        self.security_manager = security_manager
        self.credentials_key = credentials_key
        self.identifier_key = identifier_key
        self.output_key = output_key

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute authentication.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with execution results
        """
        # Get credentials
        credentials = shared.get_required(self.credentials_key)

        # Get optional identifier
        identifier = None
        if self.identifier_key:
            identifier = shared.get(self.identifier_key)

        # Authenticate
        result = self.security_manager.authenticate_request(credentials, identifier)

        # Store result
        shared.set(self.output_key, result)

        return result


class SanitizationNode(Node):
    """
    Node that sanitizes input.

    Reads input from SharedStore, sanitizes it, and stores result.
    """

    def __init__(
        self,
        security_manager: SecurityManager,
        input_key: str = "user_input",
        output_key: str = "sanitized_input",
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize SanitizationNode.

        Args:
            security_manager: SecurityManager instance
            input_key: Key in SharedStore to read input from
            output_key: Key in SharedStore to write sanitized input to
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(name or "Sanitization", enable_logging=enable_logging)
        self.security_manager = security_manager
        self.input_key = input_key
        self.output_key = output_key

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute sanitization.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with execution results
        """
        # Get input
        input_data = shared.get_required(self.input_key)

        # Sanitize
        sanitized = self.security_manager.sanitize_input(input_data)

        # Store result
        shared.set(self.output_key, sanitized)

        return {
            "original_length": len(input_data),
            "sanitized_length": len(sanitized),
            "modified": input_data != sanitized,
        }


class AuditNode(Node):
    """
    Node that logs audit events.

    Records security-relevant events from the flow.
    """

    def __init__(
        self,
        audit_logger: AuditLogger,
        event_type: str,
        user_id_key: Optional[str] = None,
        action_key: Optional[str] = None,
        resource_key: Optional[str] = None,
        result_key: Optional[str] = None,
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize AuditNode.

        Args:
            audit_logger: AuditLogger instance
            event_type: Type of event to log
            user_id_key: Optional key to read user_id from SharedStore
            action_key: Optional key to read action from SharedStore
            resource_key: Optional key to read resource from SharedStore
            result_key: Optional key to read result from SharedStore
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(name or "Audit", enable_logging=enable_logging)
        self.audit_logger = audit_logger
        self.event_type = event_type
        self.user_id_key = user_id_key
        self.action_key = action_key
        self.resource_key = resource_key
        self.result_key = result_key

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Log audit event.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with execution results
        """
        # Get values from SharedStore
        user_id = shared.get(self.user_id_key) if self.user_id_key else None
        action = shared.get(self.action_key) if self.action_key else None
        resource = shared.get(self.resource_key) if self.resource_key else None
        result = shared.get(self.result_key) if self.result_key else None

        # Log event
        self.audit_logger.log_event(
            event_type=self.event_type,
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
        )

        return {
            "event_logged": True,
            "event_type": self.event_type,
            "timestamp": time.time(),
        }
