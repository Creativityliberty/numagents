"""
Resilience Module - Retry, Circuit Breaker, and Rate Limiting

This module provides resilience patterns for building fault-tolerant agents:
- Retry with exponential backoff
- Circuit breaker pattern
- Rate limiting (token bucket, sliding window)
- Timeout handling
- Fallback mechanisms

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Type, Union

from num_agents.core import Node, SharedStore
from num_agents.exceptions import NumAgentsException
from num_agents.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class ResilienceException(NumAgentsException):
    """Base exception for resilience errors."""

    pass


class RetryExhausted(ResilienceException):
    """Raised when all retry attempts are exhausted."""

    pass


class CircuitBreakerOpen(ResilienceException):
    """Raised when circuit breaker is open."""

    pass


class RateLimitExceeded(ResilienceException):
    """Raised when rate limit is exceeded."""

    pass


class TimeoutException(ResilienceException):
    """Raised when operation times out."""

    pass


# ============================================================================
# Retry Pattern
# ============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    exceptions: tuple = (Exception,)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        import random

        delay = min(
            self.initial_delay * (self.exponential_base ** (attempt - 1)),
            self.max_delay
        )

        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay


class RetryPolicy:
    """Implements retry logic with exponential backoff."""

    def __init__(self, config: Optional[RetryConfig] = None) -> None:
        """
        Initialize retry policy.

        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
        self._logger = get_logger(__name__)

    def execute(
        self,
        func: Callable,
        *args,
        on_retry: Optional[Callable[[Exception, int], None]] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            args: Positional arguments
            on_retry: Optional callback on retry
            kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            RetryExhausted: When all attempts fail
        """
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return func(*args, **kwargs)

            except self.config.exceptions as e:
                last_exception = e

                if attempt == self.config.max_attempts:
                    raise RetryExhausted(
                        f"All {self.config.max_attempts} attempts failed"
                    ) from e

                delay = self.config.get_delay(attempt)

                self._logger.warning(
                    f"Attempt {attempt}/{self.config.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                if on_retry:
                    on_retry(e, attempt)

                time.sleep(delay)

        raise RetryExhausted(
            f"All {self.config.max_attempts} attempts failed"
        ) from last_exception

    def __call__(self, func: Callable) -> Callable:
        """Decorator for retry."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper


# ============================================================================
# Circuit Breaker Pattern
# ============================================================================


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    exceptions: tuple = (Exception,)


class CircuitBreaker:
    """Implements circuit breaker pattern."""

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = Lock()
        self._logger = get_logger(__name__)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._update_state()
            return self._state

    def _update_state(self) -> None:
        """Update circuit state based on conditions."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and (
                time.time() - self._last_failure_time >= self.config.timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                self._logger.info("Circuit breaker moved to HALF_OPEN")

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: When circuit is open
        """
        with self._lock:
            self._update_state()

            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpen("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.config.exceptions as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle successful execution."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1

                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._logger.info("Circuit breaker moved to CLOSED")

            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._logger.warning("Circuit breaker moved to OPEN (from HALF_OPEN)")

            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.config.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._logger.warning(
                    f"Circuit breaker moved to OPEN "
                    f"({self._failure_count} failures)"
                )

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._logger.info("Circuit breaker reset to CLOSED")

    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper


# ============================================================================
# Rate Limiter
# ============================================================================


class RateLimiter:
    """Base class for rate limiters."""

    def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False otherwise
        """
        raise NotImplementedError

    def __call__(self, func: Callable) -> Callable:
        """Decorator for rate limiting."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.acquire():
                raise RateLimitExceeded("Rate limit exceeded")
            return func(*args, **kwargs)
        return wrapper


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter."""

    def __init__(
        self,
        rate: float,
        capacity: int,
        initial_tokens: Optional[int] = None
    ) -> None:
        """
        Initialize token bucket rate limiter.

        Args:
            rate: Token refill rate (tokens per second)
            capacity: Maximum bucket capacity
            initial_tokens: Initial number of tokens
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = initial_tokens if initial_tokens is not None else capacity
        self._last_refill = time.time()
        self._lock = Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill

        new_tokens = elapsed * self.rate
        self._tokens = min(self.capacity, self._tokens + new_tokens)
        self._last_refill = now

    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from bucket."""
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            return False

    def get_available_tokens(self) -> float:
        """Get number of available tokens."""
        with self._lock:
            self._refill()
            return self._tokens


class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiter."""

    def __init__(self, max_requests: int, window_size: float) -> None:
        """
        Initialize sliding window rate limiter.

        Args:
            max_requests: Maximum requests in window
            window_size: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_size = window_size
        self._requests: List[float] = []
        self._lock = Lock()

    def _clean_old_requests(self, now: float) -> None:
        """Remove requests outside the window."""
        cutoff = now - self.window_size
        self._requests = [r for r in self._requests if r > cutoff]

    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens (record request)."""
        with self._lock:
            now = time.time()
            self._clean_old_requests(now)

            if len(self._requests) + tokens <= self.max_requests:
                for _ in range(tokens):
                    self._requests.append(now)
                return True

            return False

    def get_request_count(self) -> int:
        """Get current request count in window."""
        with self._lock:
            self._clean_old_requests(time.time())
            return len(self._requests)


# ============================================================================
# Timeout Handler
# ============================================================================


@contextmanager
def timeout(seconds: float):
    """
    Context manager for timeout handling.

    Args:
        seconds: Timeout in seconds

    Raises:
        TimeoutException: When operation times out
    """
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds}s")

    # Set signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ============================================================================
# Resilient Node
# ============================================================================


class ResilientNode(Node):
    """Node with built-in resilience patterns."""

    def __init__(
        self,
        name: Optional[str] = None,
        enable_logging: bool = True,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
        timeout_seconds: Optional[float] = None,
        fallback: Optional[Callable[[Exception], Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize resilient node.

        Args:
            name: Node name
            enable_logging: Enable logging
            retry_config: Retry configuration
            circuit_breaker_config: Circuit breaker configuration
            rate_limiter: Rate limiter instance
            timeout_seconds: Timeout in seconds
            fallback: Fallback function on failure
        """
        super().__init__(name, enable_logging)

        self.retry_policy = RetryPolicy(retry_config) if retry_config else None
        self.circuit_breaker = (
            CircuitBreaker(circuit_breaker_config) if circuit_breaker_config else None
        )
        self.rate_limiter = rate_limiter
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute node with resilience patterns.

        Args:
            shared: Shared store

        Returns:
            Execution result
        """
        def execute_with_resilience():
            # Rate limiting
            if self.rate_limiter and not self.rate_limiter.acquire():
                raise RateLimitExceeded(f"Rate limit exceeded for {self.name}")

            # Circuit breaker
            def execute_core():
                # Timeout
                if self.timeout_seconds:
                    import signal

                    def timeout_handler(signum, frame):
                        raise TimeoutException(
                            f"Node {self.name} timed out after {self.timeout_seconds}s"
                        )

                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(self.timeout_seconds))

                    try:
                        return self.execute(shared)
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                else:
                    return self.execute(shared)

            if self.circuit_breaker:
                return self.circuit_breaker.execute(execute_core)
            else:
                return execute_core()

        try:
            # Retry
            if self.retry_policy:
                return self.retry_policy.execute(execute_with_resilience)
            else:
                return execute_with_resilience()

        except Exception as e:
            # Fallback
            if self.fallback:
                logger.warning(f"Node {self.name} failed, using fallback: {e}")
                return self.fallback(e)
            raise

    def execute(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Override this method in subclasses.

        Args:
            shared: Shared store

        Returns:
            Execution result
        """
        raise NotImplementedError("Subclasses must implement execute()")


# ============================================================================
# Helper Functions
# ============================================================================


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for adding retry to functions.

    Args:
        max_attempts: Maximum retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Exceptions to retry on

    Returns:
        Decorated function
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exceptions=exceptions
    )
    return RetryPolicy(config)


def with_circuit_breaker(
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: float = 60.0,
    exceptions: tuple = (Exception,)
) -> CircuitBreaker:
    """
    Decorator for adding circuit breaker to functions.

    Args:
        failure_threshold: Number of failures to open circuit
        success_threshold: Number of successes to close circuit
        timeout: Timeout before trying half-open
        exceptions: Exceptions to track

    Returns:
        Circuit breaker decorator
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        timeout=timeout,
        exceptions=exceptions
    )
    return CircuitBreaker(config)


def with_rate_limit(rate: float, capacity: int) -> TokenBucketRateLimiter:
    """
    Decorator for adding rate limiting to functions.

    Args:
        rate: Token refill rate (per second)
        capacity: Bucket capacity

    Returns:
        Rate limiter decorator
    """
    return TokenBucketRateLimiter(rate=rate, capacity=capacity)
