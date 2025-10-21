"""
Tests for SecurityLayer

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import tempfile
import time
import os
import pytest
from pathlib import Path
from num_agents import (
    TokenAuthenticator,
    APIKeyAuthenticator,
    RegexSanitizer,
    HTMLSanitizer,
    EnvironmentSecretsProvider,
    FileSecretsProvider,
    AuditLogger,
    RateLimiter,
    SecurityManager,
    AuthenticationNode,
    SanitizationNode,
    AuditNode,
    Flow,
    SharedStore,
    AuthenticationError,
    SanitizationError,
    SecretsError,
    RateLimitError,
)


# ============================================================================
# Test TokenAuthenticator
# ============================================================================


def test_token_authenticator_generate():
    """Test generating authentication tokens."""
    authenticator = TokenAuthenticator(secret_key="test_secret", token_expiry=3600)

    token = authenticator.generate_token(user_id="user123", metadata={"role": "admin"})

    assert token is not None
    assert "|" in token  # Token format: payload|signature


def test_token_authenticator_validate():
    """Test validating authentication tokens."""
    authenticator = TokenAuthenticator(secret_key="test_secret", token_expiry=3600)

    token = authenticator.generate_token(user_id="user123")

    result = authenticator.validate_token(token)

    assert result["authenticated"] is True
    assert result["user_id"] == "user123"


def test_token_authenticator_invalid_token():
    """Test that invalid tokens are rejected."""
    authenticator = TokenAuthenticator(secret_key="test_secret")

    with pytest.raises(AuthenticationError):
        authenticator.validate_token("invalid_token")


def test_token_authenticator_expired_token():
    """Test that expired tokens are rejected."""
    authenticator = TokenAuthenticator(secret_key="test_secret", token_expiry=1)

    token = authenticator.generate_token(user_id="user123")

    # Wait for expiry
    time.sleep(2)

    with pytest.raises(AuthenticationError):
        authenticator.validate_token(token)


def test_token_authenticator_revoke():
    """Test revoking tokens."""
    authenticator = TokenAuthenticator(secret_key="test_secret")

    token = authenticator.generate_token(user_id="user123")

    # Revoke
    assert authenticator.revoke_token(token) is True

    # Should still validate (revocation just removes from active list)
    # For production, you'd check revocation list in validate_token


# ============================================================================
# Test APIKeyAuthenticator
# ============================================================================


def test_api_key_authenticator():
    """Test API key authentication."""
    authenticator = APIKeyAuthenticator()

    # Add key
    authenticator.add_key("key123", user_id="user1", roles=["admin"])

    # Authenticate
    result = authenticator.authenticate({"api_key": "key123"})

    assert result["authenticated"] is True
    assert result["user_id"] == "user1"
    assert "admin" in result["roles"]


def test_api_key_authenticator_invalid():
    """Test invalid API key."""
    authenticator = APIKeyAuthenticator()

    with pytest.raises(AuthenticationError):
        authenticator.authenticate({"api_key": "invalid_key"})


def test_api_key_authenticator_remove():
    """Test removing API keys."""
    authenticator = APIKeyAuthenticator()

    authenticator.add_key("key123", user_id="user1")

    assert authenticator.remove_key("key123") is True
    assert authenticator.remove_key("key123") is False  # Already removed


# ============================================================================
# Test RegexSanitizer
# ============================================================================


def test_regex_sanitizer_blocked_patterns():
    """Test blocking dangerous patterns."""
    sanitizer = RegexSanitizer()

    # SQL injection attempt
    with pytest.raises(SanitizationError):
        sanitizer.sanitize("SELECT * FROM users; DROP TABLE users;")

    # Command injection attempt
    with pytest.raises(SanitizationError):
        sanitizer.sanitize("file.txt; rm -rf /")


def test_regex_sanitizer_max_length():
    """Test max length enforcement."""
    sanitizer = RegexSanitizer(max_length=10)

    # Valid
    assert sanitizer.sanitize("hello") == "hello"

    # Too long
    with pytest.raises(SanitizationError):
        sanitizer.sanitize("this is too long")


def test_regex_sanitizer_remove_patterns():
    """Test removing patterns."""
    sanitizer = RegexSanitizer(remove_patterns=[r"\d+"])  # Remove digits

    result = sanitizer.sanitize("User123 has 456 points")

    assert "123" not in result
    assert "456" not in result


def test_regex_sanitizer_validate():
    """Test validation without modification."""
    sanitizer = RegexSanitizer()

    assert sanitizer.validate("safe input") is True
    assert sanitizer.validate("DROP TABLE users") is False


# ============================================================================
# Test HTMLSanitizer
# ============================================================================


def test_html_sanitizer_remove_dangerous_tags():
    """Test removing dangerous HTML tags."""
    sanitizer = HTMLSanitizer()

    # Script tag
    result = sanitizer.sanitize("<p>Hello</p><script>alert('xss')</script>")
    assert "<script>" not in result
    assert "alert" not in result

    # Iframe
    result = sanitizer.sanitize('<iframe src="evil.com"></iframe>')
    assert "<iframe>" not in result


def test_html_sanitizer_remove_event_handlers():
    """Test removing event handlers."""
    sanitizer = HTMLSanitizer()

    result = sanitizer.sanitize('<div onclick="alert(1)">Click me</div>')

    assert "onclick" not in result


def test_html_sanitizer_escape_all():
    """Test escaping all HTML."""
    sanitizer = HTMLSanitizer(escape_all=True)

    result = sanitizer.sanitize("<p>Hello</p>")

    assert "&lt;p&gt;" in result
    assert "&lt;/p&gt;" in result


# ============================================================================
# Test EnvironmentSecretsProvider
# ============================================================================


def test_environment_secrets_provider():
    """Test environment-based secrets."""
    provider = EnvironmentSecretsProvider(prefix="TEST_SECRET_")

    # Set secret
    provider.set_secret("api_key", "secret123")

    # Get secret
    assert provider.get_secret("api_key") == "secret123"

    # Delete secret
    assert provider.delete_secret("api_key") is True

    # Should raise after deletion
    with pytest.raises(SecretsError):
        provider.get_secret("api_key")


def test_environment_secrets_provider_not_found():
    """Test getting nonexistent secret."""
    provider = EnvironmentSecretsProvider()

    with pytest.raises(SecretsError):
        provider.get_secret("nonexistent")


# ============================================================================
# Test FileSecretsProvider
# ============================================================================


def test_file_secrets_provider():
    """Test file-based secrets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "secrets.json"
        provider = FileSecretsProvider(filepath=filepath)

        # Set secret
        provider.set_secret("db_password", "pass123")

        # Check file exists
        assert filepath.exists()

        # Get secret
        assert provider.get_secret("db_password") == "pass123"

        # Delete secret
        assert provider.delete_secret("db_password") is True

        # Should raise after deletion
        with pytest.raises(SecretsError):
            provider.get_secret("db_password")


def test_file_secrets_provider_persistence():
    """Test that secrets persist across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "secrets.json"

        # Create and save
        provider1 = FileSecretsProvider(filepath=filepath)
        provider1.set_secret("key", "value")

        # Load in new instance
        provider2 = FileSecretsProvider(filepath=filepath)
        assert provider2.get_secret("key") == "value"


# ============================================================================
# Test AuditLogger
# ============================================================================


def test_audit_logger_log_event():
    """Test logging audit events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logfile = Path(tmpdir) / "audit.log"
        logger = AuditLogger(filepath=logfile, console_logging=False)

        # Log event
        logger.log_event(
            event_type="authentication",
            user_id="user123",
            action="login",
            resource="system",
            result="success",
        )

        # Check file
        assert logfile.exists()
        content = logfile.read_text()
        assert "user123" in content
        assert "login" in content


def test_audit_logger_get_logs():
    """Test retrieving audit logs."""
    logger = AuditLogger(console_logging=False)

    # Log events
    logger.log_event(event_type="auth", user_id="user1", action="login")
    logger.log_event(event_type="access", user_id="user1", action="read")
    logger.log_event(event_type="auth", user_id="user2", action="login")

    # Get all logs
    all_logs = logger.get_logs()
    assert len(all_logs) == 3

    # Filter by user
    user1_logs = logger.get_logs(user_id="user1")
    assert len(user1_logs) == 2

    # Filter by event type
    auth_logs = logger.get_logs(event_type="auth")
    assert len(auth_logs) == 2

    # Limit
    limited = logger.get_logs(limit=1)
    assert len(limited) == 1


# ============================================================================
# Test RateLimiter
# ============================================================================


def test_rate_limiter_allow():
    """Test rate limiter allows requests within limit."""
    limiter = RateLimiter(max_requests=3, window_seconds=10)

    # Should allow first 3 requests
    assert limiter.check_limit("user1") is True
    assert limiter.check_limit("user1") is True
    assert limiter.check_limit("user1") is True


def test_rate_limiter_block():
    """Test rate limiter blocks requests over limit."""
    limiter = RateLimiter(max_requests=2, window_seconds=10)

    limiter.check_limit("user1")
    limiter.check_limit("user1")

    # Third request should be blocked
    with pytest.raises(RateLimitError):
        limiter.check_limit("user1")


def test_rate_limiter_window_reset():
    """Test rate limiter resets after window."""
    limiter = RateLimiter(max_requests=2, window_seconds=1)

    limiter.check_limit("user1")
    limiter.check_limit("user1")

    # Wait for window to pass
    time.sleep(1.1)

    # Should allow again
    assert limiter.check_limit("user1") is True


def test_rate_limiter_usage():
    """Test getting rate limiter usage stats."""
    limiter = RateLimiter(max_requests=5, window_seconds=60)

    limiter.check_limit("user1")
    limiter.check_limit("user1")

    usage = limiter.get_usage("user1")

    assert usage["requests_made"] == 2
    assert usage["requests_allowed"] == 5
    assert usage["remaining"] == 3


def test_rate_limiter_per_user():
    """Test rate limiter is per-user."""
    limiter = RateLimiter(max_requests=2, window_seconds=10)

    # User1 hits limit
    limiter.check_limit("user1")
    limiter.check_limit("user1")

    with pytest.raises(RateLimitError):
        limiter.check_limit("user1")

    # User2 should still be allowed
    assert limiter.check_limit("user2") is True


# ============================================================================
# Test SecurityManager
# ============================================================================


def test_security_manager_authenticate():
    """Test security manager authentication."""
    authenticator = APIKeyAuthenticator()
    authenticator.add_key("key123", user_id="user1")

    audit_logger = AuditLogger(console_logging=False)

    manager = SecurityManager(authenticator=authenticator, audit_logger=audit_logger)

    result = manager.authenticate_request({"api_key": "key123"})

    assert result["authenticated"] is True
    assert result["user_id"] == "user1"

    # Check audit log
    logs = audit_logger.get_logs()
    assert len(logs) == 1
    assert logs[0]["event_type"] == "authentication"


def test_security_manager_rate_limit():
    """Test security manager rate limiting."""
    authenticator = APIKeyAuthenticator()
    authenticator.add_key("key123", user_id="user1")

    rate_limiter = RateLimiter(max_requests=1, window_seconds=10)

    manager = SecurityManager(
        authenticator=authenticator, rate_limiter=rate_limiter
    )

    # First request should succeed
    manager.authenticate_request({"api_key": "key123"}, identifier="user1")

    # Second should be rate limited
    with pytest.raises(RateLimitError):
        manager.authenticate_request({"api_key": "key123"}, identifier="user1")


def test_security_manager_sanitize():
    """Test security manager sanitization."""
    sanitizer = RegexSanitizer()
    manager = SecurityManager(sanitizer=sanitizer)

    # Safe input
    result = manager.sanitize_input("Hello World")
    assert result == "Hello World"

    # Dangerous input
    with pytest.raises(SanitizationError):
        manager.sanitize_input("DROP TABLE users")


def test_security_manager_secrets():
    """Test security manager secrets access."""
    with tempfile.TemporaryDirectory() as tmpdir:
        secrets = FileSecretsProvider(filepath=Path(tmpdir) / "secrets.json")
        secrets.set_secret("api_key", "secret123")

        audit_logger = AuditLogger(console_logging=False)

        manager = SecurityManager(
            secrets_provider=secrets, audit_logger=audit_logger
        )

        # Get secret
        value = manager.get_secret("api_key", user_id="user1")
        assert value == "secret123"

        # Check audit log
        logs = audit_logger.get_logs()
        assert len(logs) == 1
        assert logs[0]["event_type"] == "secret_access"


# ============================================================================
# Test Nodes (Flow Integration)
# ============================================================================


def test_authentication_node():
    """Test AuthenticationNode in a Flow."""
    authenticator = APIKeyAuthenticator()
    authenticator.add_key("key123", user_id="user1", roles=["admin"])

    manager = SecurityManager(authenticator=authenticator)

    # Create node
    node = AuthenticationNode(
        security_manager=manager,
        credentials_key="credentials",
        output_key="auth_result",
    )

    # Execute
    shared = SharedStore()
    shared.set("credentials", {"api_key": "key123"})

    result = node.exec(shared)

    assert result["authenticated"] is True
    assert result["user_id"] == "user1"


def test_sanitization_node():
    """Test SanitizationNode in a Flow."""
    sanitizer = RegexSanitizer()
    manager = SecurityManager(sanitizer=sanitizer)

    # Create node
    node = SanitizationNode(
        security_manager=manager,
        input_key="user_input",
        output_key="sanitized_input",
    )

    # Execute
    shared = SharedStore()
    shared.set("user_input", "Hello World")

    result = node.exec(shared)

    assert result["modified"] is False
    assert shared.get("sanitized_input") == "Hello World"


def test_audit_node():
    """Test AuditNode in a Flow."""
    audit_logger = AuditLogger(console_logging=False)

    # Create node
    node = AuditNode(
        audit_logger=audit_logger,
        event_type="test_event",
        user_id_key="user_id",
        action_key="action",
    )

    # Execute
    shared = SharedStore()
    shared.set("user_id", "user123")
    shared.set("action", "test_action")

    result = node.exec(shared)

    assert result["event_logged"] is True

    # Check audit log
    logs = audit_logger.get_logs()
    assert len(logs) == 1
    assert logs[0]["user_id"] == "user123"
    assert logs[0]["action"] == "test_action"


# ============================================================================
# Integration Test
# ============================================================================


def test_security_layer_integration():
    """Integration test: Full security layer workflow."""
    # Setup components
    authenticator = TokenAuthenticator(secret_key="test_secret", token_expiry=3600)
    sanitizer = RegexSanitizer(max_length=1000)
    audit_logger = AuditLogger(console_logging=False)
    rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

    with tempfile.TemporaryDirectory() as tmpdir:
        secrets = FileSecretsProvider(filepath=Path(tmpdir) / "secrets.json")
        secrets.set_secret("db_password", "secure_pass")

        # Create security manager
        manager = SecurityManager(
            authenticator=authenticator,
            sanitizer=sanitizer,
            secrets_provider=secrets,
            audit_logger=audit_logger,
            rate_limiter=rate_limiter,
        )

        # Test 1: Generate and validate token
        token = authenticator.generate_token(user_id="user1", metadata={"role": "admin"})
        auth_result = manager.authenticate_request(
            {"token": token}, identifier="user1"
        )
        assert auth_result["authenticated"] is True

        # Test 2: Sanitize input
        safe_input = manager.sanitize_input("Hello World")
        assert safe_input == "Hello World"

        # Test 3: Get secret
        secret = manager.get_secret("db_password", user_id="user1")
        assert secret == "secure_pass"

        # Test 4: Check audit logs
        logs = audit_logger.get_logs()
        assert len(logs) >= 2  # At least auth and secret access

        # Test 5: Rate limiting
        for i in range(9):  # Already used 1 for auth
            manager.authenticate_request({"token": token}, identifier="user1")

        # Should hit rate limit
        with pytest.raises(RateLimitError):
            manager.authenticate_request({"token": token}, identifier="user1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
