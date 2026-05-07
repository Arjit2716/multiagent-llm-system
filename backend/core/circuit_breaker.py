"""
Circuit Breaker and Retry Logic for LLM calls.
Implements the circuit breaker pattern with exponential backoff.
"""
import asyncio
import time
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from functools import wraps

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing - rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Implements the circuit breaker pattern for LLM API calls.
    Prevents cascade failures by stopping requests when error rate is high.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = None,
        timeout: int = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold or settings.CIRCUIT_BREAKER_THRESHOLD
        self.timeout = timeout or settings.CIRCUIT_BREAKER_TIMEOUT
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._success_count = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.timeout:
                logger.info("circuit_breaker_half_open", name=self.name)
                self._state = CircuitState.HALF_OPEN
        return self._state

    def record_success(self) -> None:
        self._failure_count = 0
        self._success_count += 1
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            logger.info("circuit_breaker_closed", name=self.name)

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "circuit_breaker_open",
                name=self.name,
                failures=self._failure_count,
            )

    def can_execute(self) -> bool:
        return self.state != CircuitState.OPEN

    def get_metrics(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
        }


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is in OPEN state."""
    pass


async def retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int = None,
    base_delay: float = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    **kwargs,
) -> Any:
    """
    Async retry with exponential backoff and circuit breaker integration.
    Implements jitter to avoid thundering herd.
    """
    max_retries = max_retries or settings.MAX_RETRIES
    base_delay = base_delay or settings.RETRY_DELAY

    if circuit_breaker and not circuit_breaker.can_execute():
        raise CircuitBreakerOpen(f"Circuit breaker '{circuit_breaker.name}' is OPEN")

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            if circuit_breaker:
                circuit_breaker.record_success()
            return result
        except CircuitBreakerOpen:
            raise
        except Exception as e:
            last_exception = e
            if circuit_breaker:
                circuit_breaker.record_failure()

            if attempt == max_retries:
                logger.error(
                    "max_retries_exceeded",
                    func=func.__name__,
                    attempt=attempt,
                    error=str(e),
                )
                raise

            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt)
            jitter = delay * 0.1 * (2 * asyncio.get_event_loop().time() % 1 - 1)
            wait_time = delay + jitter
            logger.warning(
                "retry_attempt",
                func=func.__name__,
                attempt=attempt + 1,
                wait_seconds=round(wait_time, 2),
                error=str(e),
            )
            await asyncio.sleep(wait_time)

    raise last_exception


# Global circuit breakers registry
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> CircuitBreaker:
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name)
    return _circuit_breakers[name]


def get_all_circuit_breaker_metrics() -> list[dict]:
    return [cb.get_metrics() for cb in _circuit_breakers.values()]
