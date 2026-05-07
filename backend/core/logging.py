"""
Structured logging with correlation IDs and JSON output.
Integrates with OpenTelemetry for distributed tracing.
"""
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional
import structlog
from structlog.types import FilteringBoundLogger

# Context variable to store correlation ID across async tasks
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    return correlation_id_var.get() or str(uuid.uuid4())


def set_correlation_id(cid: Optional[str] = None) -> str:
    cid = cid or str(uuid.uuid4())
    correlation_id_var.set(cid)
    return cid


def add_correlation_id(logger: Any, method: str, event_dict: Dict) -> Dict:
    """Structlog processor: inject correlation ID into every log record."""
    event_dict["correlation_id"] = get_correlation_id()
    return event_dict


def configure_logging(log_level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog with JSON rendering and correlation tracking."""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        add_correlation_id,
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Silence noisy libraries
    for lib in ["httpx", "httpcore", "asyncio"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> FilteringBoundLogger:
    return structlog.get_logger(name)
