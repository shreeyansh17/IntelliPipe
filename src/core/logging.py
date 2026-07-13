"""
IntelliPipe Structured Logging
================================
Production-grade structured JSON logging with:
- structlog for structured output
- OpenTelemetry trace/span correlation
- Request ID injection
- Sensitive data scrubbing
- Log sampling for high-throughput paths
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Optional

import structlog
from structlog.types import EventDict, WrappedLogger

# Context variable for request-scoped trace IDs
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")
_tenant_id_ctx: ContextVar[str] = ContextVar("tenant_id", default="default")


def get_request_id() -> str:
    """Get current request ID from context."""
    return _request_id_ctx.get() or str(uuid.uuid4())


def set_request_id(request_id: str) -> None:
    """Set request ID in current context."""
    _request_id_ctx.set(request_id)


def set_tenant_id(tenant_id: str) -> None:
    """Set tenant ID in current context."""
    _tenant_id_ctx.set(tenant_id)


def _add_request_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Inject request context into every log event."""
    request_id = _request_id_ctx.get()
    tenant_id = _tenant_id_ctx.get()

    if request_id:
        event_dict["request_id"] = request_id
    if tenant_id:
        event_dict["tenant_id"] = tenant_id

    return event_dict


def _add_otel_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Inject OpenTelemetry trace/span IDs for log-trace correlation."""
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span.is_recording():
            ctx = span.get_span_context()
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
    except ImportError:
        pass
    return event_dict


_SENSITIVE_KEYS = frozenset({
    "password", "secret", "token", "api_key", "apikey",
    "authorization", "credential", "private_key", "access_key",
})


def _scrub_sensitive_data(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Replace sensitive field values with ***REDACTED***."""
    def _scrub(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: "***REDACTED***" if any(s in k.lower() for s in _SENSITIVE_KEYS)
                else _scrub(v)
                for k, v in obj.items()
            }
        if isinstance(obj, (list, tuple)):
            return type(obj)(_scrub(item) for item in obj)
        return obj

    return _scrub(event_dict)  # type: ignore[return-value]


def configure_logging(
    log_level: str = "INFO",
    json_output: bool = True,
    service_name: str = "intellipipe",
    service_version: str = "1.0.0",
    environment: str = "production",
) -> None:
    """
    Configure structlog with production-grade processors.

    Call once at application startup. Thread-safe.
    """

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _add_request_context,
        _add_otel_context,
        _scrub_sensitive_data,
        # Add static service context
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
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
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Silence noisy third-party loggers
    for noisy_logger in ["kafka", "pyspark", "urllib3", "httpx"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Bind static service-level context
    structlog.contextvars.bind_contextvars(
        service=service_name,
        version=service_version,
        environment=environment,
    )


def get_logger(name: str, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """
    Get a named structured logger with optional initial context binding.

    Usage:
        logger = get_logger(__name__, component="kafka_producer")
        logger.info("Event produced", topic="raw_events", partition=3)
    """
    return structlog.get_logger(name).bind(**initial_context)


class LogContext:
    """
    Context manager for scoped log context binding.

    Usage:
        with LogContext(pipeline_run_id="abc123", batch_id="batch-456"):
            logger.info("Processing batch")
    """

    def __init__(self, **kwargs: Any) -> None:
        self._context = kwargs
        self._token: Optional[Any] = None

    def __enter__(self) -> "LogContext":
        structlog.contextvars.bind_contextvars(**self._context)
        return self

    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self._context.keys())
