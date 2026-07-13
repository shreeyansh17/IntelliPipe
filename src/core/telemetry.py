"""
IntelliPipe Observability — OpenTelemetry + Prometheus
=======================================================
Centralised telemetry bootstrap:
- OTLP trace exporter (Jaeger / Tempo compatible)
- Prometheus metrics registry with IntelliPipe-specific counters/histograms
- Decorator helpers for tracing functions
- FastAPI middleware integration hooks
"""

from __future__ import annotations

import time
import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional, TypeVar, cast

from opentelemetry import metrics as otel_metrics
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

from src.core.logging import get_logger

logger = get_logger(__name__, component="telemetry")

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Prometheus metrics registry
# ---------------------------------------------------------------------------

REGISTRY = CollectorRegistry()

# Pipeline metrics
EVENTS_PRODUCED_TOTAL = Counter(
    "intellipipe_events_produced_total",
    "Total Kafka events produced",
    ["topic", "tenant_id"],
    registry=REGISTRY,
)

EVENTS_CONSUMED_TOTAL = Counter(
    "intellipipe_events_consumed_total",
    "Total Kafka events consumed",
    ["topic", "tenant_id", "status"],
    registry=REGISTRY,
)

SCHEMA_DRIFT_DETECTED_TOTAL = Counter(
    "intellipipe_schema_drift_detected_total",
    "Total schema drift events detected",
    ["table", "tenant_id", "drift_type"],
    registry=REGISTRY,
)

# Anomaly detection metrics
ANOMALY_SCORE_HISTOGRAM = Histogram(
    "intellipipe_anomaly_score",
    "Distribution of ensemble anomaly scores",
    ["table", "model_type"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=REGISTRY,
)

ANOMALIES_DETECTED_TOTAL = Counter(
    "intellipipe_anomalies_detected_total",
    "Total anomalies detected",
    ["table", "anomaly_type", "severity"],
    registry=REGISTRY,
)

# Data quality metrics
DQ_SCORE_GAUGE = Gauge(
    "intellipipe_dq_score",
    "Current data quality score (0-100)",
    ["table", "dimension", "tenant_id"],
    registry=REGISTRY,
)

DQ_CHECKS_TOTAL = Counter(
    "intellipipe_dq_checks_total",
    "Total DQ checks executed",
    ["check_type", "table", "result"],
    registry=REGISTRY,
)

DQ_VIOLATIONS_TOTAL = Counter(
    "intellipipe_dq_violations_total",
    "Total DQ rule violations",
    ["rule_name", "table", "severity"],
    registry=REGISTRY,
)

# LLM agent metrics
LLM_API_CALLS_TOTAL = Counter(
    "intellipipe_llm_api_calls_total",
    "Total LLM API calls made",
    ["model", "operation", "status"],
    registry=REGISTRY,
)

LLM_LATENCY_HISTOGRAM = Histogram(
    "intellipipe_llm_latency_seconds",
    "LLM API call latency",
    ["model", "operation"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

LLM_TOKEN_USAGE_TOTAL = Counter(
    "intellipipe_llm_token_usage_total",
    "Total LLM tokens consumed",
    ["model", "token_type"],
    registry=REGISTRY,
)

# Incident metrics
INCIDENTS_CREATED_TOTAL = Counter(
    "intellipipe_incidents_created_total",
    "Total incidents created",
    ["severity", "source"],
    registry=REGISTRY,
)

INCIDENTS_RESOLVED_TOTAL = Counter(
    "intellipipe_incidents_resolved_total",
    "Total incidents resolved",
    ["severity", "resolution_type"],
    registry=REGISTRY,
)

INCIDENT_RESOLUTION_TIME = Histogram(
    "intellipipe_incident_resolution_seconds",
    "Time to resolve incidents",
    ["severity"],
    buckets=[60, 300, 900, 1800, 3600, 7200, 86400],
    registry=REGISTRY,
)

# API metrics
API_REQUEST_DURATION = Histogram(
    "intellipipe_api_request_duration_seconds",
    "API request duration",
    ["method", "endpoint", "status_code"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY,
)

API_REQUESTS_IN_FLIGHT = Gauge(
    "intellipipe_api_requests_in_flight",
    "Current in-flight API requests",
    ["endpoint"],
    registry=REGISTRY,
)

# RAG metrics
RAG_QUERIES_TOTAL = Counter(
    "intellipipe_rag_queries_total",
    "Total RAG system queries",
    ["query_type", "status"],
    registry=REGISTRY,
)

RAG_RETRIEVAL_LATENCY = Histogram(
    "intellipipe_rag_retrieval_latency_seconds",
    "RAG vector retrieval latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    registry=REGISTRY,
)

# SLA metrics
SLA_VIOLATIONS_TOTAL = Counter(
    "intellipipe_sla_violations_total",
    "Total SLA/freshness violations",
    ["table", "sla_type"],
    registry=REGISTRY,
)

PIPELINE_BATCH_DURATION = Histogram(
    "intellipipe_pipeline_batch_duration_seconds",
    "Duration of pipeline batch processing",
    ["stage"],
    buckets=[1, 5, 10, 30, 60, 120, 300],
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# OpenTelemetry bootstrap
# ---------------------------------------------------------------------------


def setup_telemetry(
    service_name: str,
    service_version: str,
    environment: str,
    otlp_endpoint: str,
    prometheus_port: int = 8000,
    enable_tracing: bool = True,
    enable_metrics: bool = True,
) -> None:
    """
    Bootstrap OpenTelemetry tracing + metrics.
    Call once at application startup before any requests are handled.
    """
    resource = Resource.create(
        {
            ResourceAttributes.SERVICE_NAME: service_name,
            ResourceAttributes.SERVICE_VERSION: service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: environment,
        }
    )

    if enable_tracing:
        tracer_provider = TracerProvider(resource=resource)
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        trace.set_tracer_provider(tracer_provider)
        logger.info("OpenTelemetry tracing enabled", endpoint=otlp_endpoint)

    if enable_metrics:
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True),
            export_interval_millis=15000,
        )
        meter_provider = MeterProvider(
            resource=resource, metric_readers=[metric_reader]
        )
        otel_metrics.set_meter_provider(meter_provider)
        logger.info("OpenTelemetry metrics enabled")

    # Auto-instrument common libraries
    HTTPXClientInstrumentor().instrument()
    SQLAlchemyInstrumentor().instrument()

    # Start Prometheus scrape endpoint
    start_http_server(prometheus_port, registry=REGISTRY)
    logger.info("Prometheus metrics server started", port=prometheus_port)


def instrument_fastapi(app: Any) -> None:
    """Attach OpenTelemetry auto-instrumentation to a FastAPI app."""
    FastAPIInstrumentor.instrument_app(app)


def get_tracer(name: str) -> trace.Tracer:
    """Get a named tracer. Use module __name__ as the tracer name."""
    return trace.get_tracer(name)


# ---------------------------------------------------------------------------
# Decorator helpers
# ---------------------------------------------------------------------------


def traced(
    operation_name: Optional[str] = None,
    attributes: Optional[Dict[str, str]] = None,
) -> Callable[[F], F]:
    """
    Decorator to trace a function with OpenTelemetry.

    Usage:
        @traced("process_batch", attributes={"component": "spark"})
        async def process_batch(self, batch_id: str) -> None:
            ...
    """

    def decorator(func: F) -> F:
        tracer = get_tracer(func.__module__)
        span_name = operation_name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_attribute("success", False)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_attribute("success", False)
                    raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, sync_wrapper)

    return decorator


@contextmanager
def timed_operation(
    metric: Histogram, labels: Dict[str, str]
) -> Generator[None, None, None]:
    """
    Context manager to record operation duration in a Prometheus Histogram.

    Usage:
        with timed_operation(PIPELINE_BATCH_DURATION, {"stage": "validation"}):
            run_validation()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        metric.labels(**labels).observe(duration)
