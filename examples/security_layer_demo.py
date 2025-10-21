"""
SecurityLayer Demo - Demonstrating Security capabilities

This example shows how to:
1. Authenticate users with tokens and API keys
2. Sanitize user inputs to prevent attacks
3. Manage secrets securely
4. Log security events
5. Implement rate limiting
6. Integrate security with Flows

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import tempfile
import time
from pathlib import Path
from num_agents import (
    TokenAuthenticator,
    APIKeyAuthenticator,
    RegexSanitizer,
    HTMLSanitizer,
    FileSecretsProvider,
    AuditLogger,
    RateLimiter,
    SecurityManager,
    AuthenticationNode,
    SanitizationNode,
    AuditNode,
    Flow,
    Node,
    SharedStore,
)


# ============================================================================
# Example 1: Authentication with Tokens
# ============================================================================


def token_authentication_demo():
    """Demonstrate token-based authentication."""
    print("=" * 60)
    print("Example 1: Token Authentication")
    print("=" * 60)

    # Create authenticator
    authenticator = TokenAuthenticator(
        secret_key="my_secret_key_12345", token_expiry=3600  # 1 hour
    )

    # Generate token for a user
    print("\n🔑 Generating authentication token...")
    token = authenticator.generate_token(
        user_id="alice@example.com", metadata={"role": "admin", "department": "engineering"}
    )
    print(f"Token generated (first 50 chars): {token[:50]}...")

    # Validate token
    print("\n✅ Validating token...")
    result = authenticator.validate_token(token)
    print(f"  Authenticated: {result['authenticated']}")
    print(f"  User ID: {result['user_id']}")
    print(f"  Metadata: {result['metadata']}")

    # Attempt to use invalid token
    print("\n❌ Attempting invalid token...")
    try:
        authenticator.validate_token("invalid_token_12345")
    except Exception as e:
        print(f"  Rejected: {type(e).__name__}")


# ============================================================================
# Example 2: API Key Authentication
# ============================================================================


def api_key_authentication_demo():
    """Demonstrate API key authentication."""
    print("\n" + "=" * 60)
    print("Example 2: API Key Authentication")
    print("=" * 60)

    # Create authenticator
    authenticator = APIKeyAuthenticator()

    # Add API keys for different users
    print("\n📝 Registering API keys...")
    authenticator.add_key("sk_live_abc123", user_id="service_a", roles=["read", "write"])
    authenticator.add_key("sk_live_xyz789", user_id="service_b", roles=["read"])

    # Authenticate with API key
    print("\n🔓 Authenticating service_a...")
    result = authenticator.authenticate({"api_key": "sk_live_abc123"})
    print(f"  User: {result['user_id']}")
    print(f"  Roles: {result['roles']}")

    # Authenticate service_b
    print("\n🔓 Authenticating service_b...")
    result = authenticator.authenticate({"api_key": "sk_live_xyz789"})
    print(f"  User: {result['user_id']}")
    print(f"  Roles: {result['roles']}")


# ============================================================================
# Example 3: Input Sanitization
# ============================================================================


def input_sanitization_demo():
    """Demonstrate input sanitization to prevent attacks."""
    print("\n" + "=" * 60)
    print("Example 3: Input Sanitization")
    print("=" * 60)

    # Create sanitizer
    sanitizer = RegexSanitizer(max_length=1000)

    # Test safe input
    print("\n✅ Testing safe input...")
    safe_input = "Hello, how can I help you today?"
    result = sanitizer.sanitize(safe_input)
    print(f"  Input: '{safe_input}'")
    print(f"  Output: '{result}'")
    print(f"  Status: PASSED")

    # Test SQL injection attempt
    print("\n🚨 Testing SQL injection attempt...")
    sql_injection = "SELECT * FROM users; DROP TABLE users;"
    try:
        sanitizer.sanitize(sql_injection)
        print("  Status: FAILED (should have been blocked)")
    except Exception as e:
        print(f"  Status: BLOCKED ({type(e).__name__})")

    # Test command injection attempt
    print("\n🚨 Testing command injection attempt...")
    cmd_injection = "file.txt; rm -rf /"
    try:
        sanitizer.sanitize(cmd_injection)
        print("  Status: FAILED (should have been blocked)")
    except Exception as e:
        print(f"  Status: BLOCKED ({type(e).__name__})")

    # HTML sanitizer
    print("\n🧹 HTML Sanitization...")
    html_sanitizer = HTMLSanitizer()

    xss_attempt = '<p>Hello</p><script>alert("XSS")</script>'
    cleaned = html_sanitizer.sanitize(xss_attempt)
    print(f"  Input: {xss_attempt}")
    print(f"  Output: {cleaned}")
    print(f"  Script removed: {'<script>' not in cleaned}")


# ============================================================================
# Example 4: Secrets Management
# ============================================================================


def secrets_management_demo():
    """Demonstrate secure secrets management."""
    print("\n" + "=" * 60)
    print("Example 4: Secrets Management")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create secrets provider
        secrets_file = Path(tmpdir) / "secrets.json"
        provider = FileSecretsProvider(filepath=secrets_file)

        # Store secrets
        print("\n🔐 Storing secrets...")
        provider.set_secret("database_password", "super_secret_db_pass")
        provider.set_secret("api_key", "sk_live_abc123xyz")
        provider.set_secret("encryption_key", "aes256_key_here")
        print("  ✅ Secrets stored securely")

        # Retrieve secrets
        print("\n🔑 Retrieving secrets...")
        db_pass = provider.get_secret("database_password")
        print(f"  Database password: {db_pass[:5]}***")

        api_key = provider.get_secret("api_key")
        print(f"  API key: {api_key[:10]}***")

        # Delete secret
        print("\n🗑️  Deleting secret...")
        provider.delete_secret("encryption_key")
        print("  ✅ Secret deleted")

        # Try to access deleted secret
        try:
            provider.get_secret("encryption_key")
        except Exception as e:
            print(f"  ✅ Cannot access deleted secret ({type(e).__name__})")


# ============================================================================
# Example 5: Audit Logging
# ============================================================================


def audit_logging_demo():
    """Demonstrate security audit logging."""
    print("\n" + "=" * 60)
    print("Example 5: Audit Logging")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create audit logger
        audit_file = Path(tmpdir) / "security_audit.log"
        logger = AuditLogger(filepath=audit_file, console_logging=False)

        # Log various security events
        print("\n📝 Logging security events...")

        logger.log_event(
            event_type="authentication",
            user_id="alice@example.com",
            action="login",
            result="success",
            metadata={"ip": "192.168.1.100"},
        )

        logger.log_event(
            event_type="authorization",
            user_id="alice@example.com",
            action="access_resource",
            resource="/api/admin/users",
            result="allowed",
        )

        logger.log_event(
            event_type="secret_access",
            user_id="service_a",
            action="get_secret",
            resource="database_password",
            result="success",
        )

        logger.log_event(
            event_type="authentication",
            user_id="unknown",
            action="login",
            result="failure",
            metadata={"reason": "invalid_credentials"},
        )

        print(f"  ✅ {len(logger.get_logs())} events logged")

        # Query audit logs
        print("\n🔍 Querying audit logs...")

        # Get all auth events
        auth_events = logger.get_logs(event_type="authentication")
        print(f"  Authentication events: {len(auth_events)}")

        # Get events for specific user
        alice_events = logger.get_logs(user_id="alice@example.com")
        print(f"  Events for alice@example.com: {len(alice_events)}")

        # Get recent events
        recent = logger.get_logs(limit=2)
        print(f"  Most recent events: {len(recent)}")

        # Show log file
        print(f"\n📄 Audit log file: {audit_file}")
        print(f"  File size: {audit_file.stat().st_size} bytes")


# ============================================================================
# Example 6: Rate Limiting
# ============================================================================


def rate_limiting_demo():
    """Demonstrate rate limiting to prevent abuse."""
    print("\n" + "=" * 60)
    print("Example 6: Rate Limiting")
    print("=" * 60)

    # Create rate limiter: 3 requests per 5 seconds
    limiter = RateLimiter(max_requests=3, window_seconds=5)

    user_id = "user@example.com"

    print(f"\n⏱️  Rate limit: 3 requests per 5 seconds")

    # Make requests
    print(f"\n📊 Making requests for {user_id}...")

    for i in range(1, 6):
        try:
            limiter.check_limit(user_id)
            usage = limiter.get_usage(user_id)
            print(
                f"  Request {i}: ✅ ALLOWED "
                f"({usage['requests_made']}/{usage['requests_allowed']})"
            )
        except Exception as e:
            usage = limiter.get_usage(user_id)
            print(
                f"  Request {i}: ❌ BLOCKED "
                f"({usage['requests_made']}/{usage['requests_allowed']}) "
                f"- {type(e).__name__}"
            )

    # Wait and try again
    print("\n⏳ Waiting 5 seconds for window to reset...")
    time.sleep(5.1)

    try:
        limiter.check_limit(user_id)
        print("  Request 6: ✅ ALLOWED (window reset)")
    except Exception as e:
        print(f"  Request 6: ❌ BLOCKED - {type(e).__name__}")


# ============================================================================
# Example 7: Integrated Security Manager
# ============================================================================


def security_manager_demo():
    """Demonstrate integrated security management."""
    print("\n" + "=" * 60)
    print("Example 7: Integrated Security Manager")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup components
        authenticator = APIKeyAuthenticator()
        authenticator.add_key("key_abc", user_id="user1", roles=["admin"])

        sanitizer = RegexSanitizer(max_length=500)

        secrets = FileSecretsProvider(filepath=Path(tmpdir) / "secrets.json")
        secrets.set_secret("db_pass", "secret123")

        audit_logger = AuditLogger(console_logging=False)

        rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

        # Create security manager
        manager = SecurityManager(
            authenticator=authenticator,
            sanitizer=sanitizer,
            secrets_provider=secrets,
            audit_logger=audit_logger,
            rate_limiter=rate_limiter,
        )

        print("\n🛡️  Security Manager configured with:")
        print("  ✅ Authentication")
        print("  ✅ Input sanitization")
        print("  ✅ Secrets management")
        print("  ✅ Audit logging")
        print("  ✅ Rate limiting")

        # Use security manager
        print("\n🔐 Authenticating request...")
        auth_result = manager.authenticate_request(
            credentials={"api_key": "key_abc"}, identifier="user1"
        )
        print(f"  User: {auth_result['user_id']}")

        print("\n🧹 Sanitizing user input...")
        user_input = "Please process this data"
        sanitized = manager.sanitize_input(user_input)
        print(f"  Input: '{user_input}'")
        print(f"  Sanitized: '{sanitized}'")

        print("\n🔑 Accessing secret...")
        secret = manager.get_secret("db_pass", user_id="user1")
        print(f"  Secret retrieved: {secret[:3]}***")

        print("\n📋 Audit log summary:")
        logs = audit_logger.get_logs()
        print(f"  Total events: {len(logs)}")
        for log in logs:
            print(f"  - {log['event_type']}: {log['action']} ({log['result']})")


# ============================================================================
# Example 8: Flow Integration
# ============================================================================


class UserRequestNode(Node):
    """Node that simulates a user request."""

    def exec(self, shared: SharedStore) -> dict:
        # Simulate user request
        shared.set("credentials", {"api_key": "secure_key_123"})
        shared.set("user_input", "Show me user data for ID 42")
        shared.set("user_id", "alice@example.com")
        print("\n📥 User request received")
        return {"request_received": True}


class ProcessNode(Node):
    """Node that processes the request."""

    def exec(self, shared: SharedStore) -> dict:
        sanitized_input = shared.get("sanitized_input")
        auth_result = shared.get("auth_result")

        print(f"\n⚙️  Processing request...")
        print(f"  User: {auth_result['user_id']}")
        print(f"  Input: {sanitized_input}")

        shared.set("result", {"status": "success", "data": [{"id": 42, "name": "Bob"}]})
        return {"processed": True}


def flow_integration_demo():
    """Demonstrate integrating security with Flows."""
    print("\n" + "=" * 60)
    print("Example 8: Secure Flow Integration")
    print("=" * 60)

    # Setup security
    authenticator = APIKeyAuthenticator()
    authenticator.add_key("secure_key_123", user_id="alice@example.com", roles=["admin"])

    sanitizer = RegexSanitizer()
    audit_logger = AuditLogger(console_logging=False)

    manager = SecurityManager(
        authenticator=authenticator, sanitizer=sanitizer, audit_logger=audit_logger
    )

    # Build secure flow
    flow = Flow(name="SecureAPIFlow")

    # Nodes
    request_node = UserRequestNode(name="UserRequest")
    auth_node = AuthenticationNode(
        manager, credentials_key="credentials", output_key="auth_result", name="Authentication"
    )
    sanitize_node = SanitizationNode(
        manager, input_key="user_input", output_key="sanitized_input", name="Sanitization"
    )
    audit_node = AuditNode(
        audit_logger,
        event_type="api_access",
        user_id_key="user_id",
        action_key="action",
        name="AuditLog",
    )
    process_node = ProcessNode(name="ProcessRequest")

    # Add to flow
    flow.add_node(request_node)
    flow.add_node(auth_node)
    flow.add_node(sanitize_node)
    flow.add_node(audit_node)
    flow.add_node(process_node)

    flow.set_start(request_node)

    # Execute
    print("\n🚀 Executing secure flow...")
    flow._shared.set("action", "get_user_data")  # For audit
    results = flow.execute()

    print("\n✅ Flow completed successfully")
    print(f"📊 Audit events logged: {len(audit_logger.get_logs())}")


# ============================================================================
# Main Demo
# ============================================================================


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║       SecurityLayer Demo - Agent Security Features        ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Run examples
    token_authentication_demo()
    api_key_authentication_demo()
    input_sanitization_demo()
    secrets_management_demo()
    audit_logging_demo()
    rate_limiting_demo()
    security_manager_demo()
    flow_integration_demo()

    print("\n" + "=" * 60)
    print("✨ Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Multiple authentication methods (tokens, API keys)")
    print("  2. Input sanitization prevents injection attacks")
    print("  3. Secure secrets management")
    print("  4. Comprehensive audit logging")
    print("  5. Rate limiting prevents abuse")
    print("  6. Integrated security manager")
    print("  7. Seamless Flow integration")
    print("\n")
    print("🛡️  Production-Ready Security Features!")
    print("\n")


if __name__ == "__main__":
    main()
