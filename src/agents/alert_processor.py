"""
IntelliPipe Alert Processor Worker
=====================================
Long-running worker that:
1. Continuously polls the Redis alert queue
2. Deduplicates alerts within a time window
3. Triggers the LangChain orchestration agent for each unique alert
4. Implements circuit breaker for LLM API failures
5. Writes all outcomes to PostgreSQL
6. Emits Prometheus metrics for alert throughput and processing latency
"""

from __future__ import annotations

import asyncio
import json
import signal
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.agents.langchain_agent import IntelliPipeOrchestrationAgent
from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger, LogContext
from src.core.telemetry import (
    INCIDENTS_CREATED_TOTAL,
    timed_operation,
    PIPELINE_BATCH_DURATION,
)
from src.db.repositories.incident_repo import (
    DocumentChunkRepository,
    IncidentMemoryRepository,
    IncidentRepository,
)
from src.db.models import AnomalyType, SeverityLevel
from src.integrations.github_client import GitHubPRClient
from src.integrations.notifications import JiraTicketClient, SlackIncidentNotifier

logger = get_logger(__name__, component="alert_processor")
settings = get_settings()

# ---------------------------------------------------------------------------
# Circuit breaker for LLM API
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Simple circuit breaker to protect against LLM API outages.
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing recovery)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_seconds: int = 60,
    ) -> None:
        self._failures = 0
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout_seconds
        self._last_failure_time: Optional[float] = None
        self._state = "CLOSED"

    @property
    def is_open(self) -> bool:
        if self._state == "OPEN":
            if time.time() - (self._last_failure_time or 0) > self._reset_timeout:
                self._state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return False
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._state = "CLOSED"

    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self._threshold:
            if self._state != "OPEN":
                logger.warning(
                    "Circuit breaker OPEN — LLM API failing",
                    failures=self._failures,
                )
            self._state = "OPEN"


# ---------------------------------------------------------------------------
# Alert deduplication cache
# ---------------------------------------------------------------------------

class AlertDeduplicator:
    """
    Prevents duplicate processing of the same alert within a time window.
    Uses a simple in-memory set with expiry tracking.
    """

    def __init__(self, window_minutes: int = 30) -> None:
        self._seen: Dict[str, float] = {}
        self._window = window_minutes * 60

    def is_duplicate(self, alert: Dict[str, Any]) -> bool:
        key = self._make_key(alert)
        now = time.time()
        if key in self._seen:
            if now - self._seen[key] < self._window:
                return True
            del self._seen[key]
        self._seen[key] = now
        return False

    def _make_key(self, alert: Dict[str, Any]) -> str:
        return f"{alert.get('alert_type')}:{alert.get('table_name')}:{alert.get('tenant_id')}"

    def cleanup_expired(self) -> None:
        now = time.time()
        expired = [k for k, t in self._seen.items() if now - t > self._window]
        for k in expired:
            del self._seen[k]


# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------

ALERT_TYPE_BASE_SEVERITY = {
    "schema_drift": SeverityLevel.HIGH,
    "dq_score_degradation": SeverityLevel.HIGH,
    "null_spike": SeverityLevel.MEDIUM,
    "statistical": SeverityLevel.MEDIUM,
    "freshness": SeverityLevel.HIGH,
    "duplicate": SeverityLevel.LOW,
    "referential": SeverityLevel.MEDIUM,
}

ANOMALY_TYPE_MAP = {
    "schema_drift": AnomalyType.SCHEMA_DRIFT,
    "null_spike": AnomalyType.NULL_SPIKE,
    "statistical": AnomalyType.STATISTICAL,
    "dq_score_degradation": AnomalyType.STATISTICAL,
    "freshness": AnomalyType.FRESHNESS,
    "duplicate": AnomalyType.DUPLICATE,
    "referential": AnomalyType.REFERENTIAL,
}


def compute_severity(alert: Dict[str, Any]) -> SeverityLevel:
    """
    Compute incident severity from alert signals.
    DQ score < 60 escalates to CRITICAL regardless of alert type.
    """
    alert_severity = alert.get("severity", "medium").lower()

    # Override from alert payload if present and valid
    for level in SeverityLevel:
        if alert_severity == level.value:
            # Escalate if DQ score critically low
            dq_score = alert.get("dq_score") or alert.get("scores", {}).get("overall")
            if dq_score is not None and dq_score < 60:
                return SeverityLevel.CRITICAL
            return level

    return ALERT_TYPE_BASE_SEVERITY.get(alert.get("alert_type", ""), SeverityLevel.MEDIUM)


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class AlertProcessorWorker:
    """
    Main alert processing worker.
    Runs as a long-lived async process consuming from Redis.
    """

    POLL_TIMEOUT_SECONDS = 2
    DEDUP_WINDOW_MINUTES = 30
    MAX_CONCURRENT_ALERTS = 3
    CLEANUP_INTERVAL_SECONDS = 300

    def __init__(self) -> None:
        self._settings = settings
        self._redis: Optional[aioredis.Redis] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout_seconds=120)
        self._deduplicator = AlertDeduplicator(window_minutes=self.DEDUP_WINDOW_MINUTES)
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_ALERTS)
        self._shutdown_event = asyncio.Event()
        self._processed_count = 0
        self._error_count = 0
        self._last_cleanup = time.time()

        logger.info(
            "AlertProcessorWorker initialised",
            max_concurrent=self.MAX_CONCURRENT_ALERTS,
            dedup_window_minutes=self.DEDUP_WINDOW_MINUTES,
        )

    async def start(self) -> None:
        """Initialise connections and start processing loop."""
        configure_logging(
            log_level=settings.observability.log_level,
            json_output=settings.is_production,
            service_name="intellipipe-alert-processor",
        )

        # Database connection
        engine = create_async_engine(
            settings.database.url,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_recycle=settings.database.pool_recycle,
            echo=settings.database.echo,
        )
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)

        # Redis connection
        self._redis = await aioredis.from_url(
            settings.redis.url,
            max_connections=20,
            decode_responses=True,
        )

        logger.info("Alert processor started, listening for alerts")
        await self._run_loop()

    async def _run_loop(self) -> None:
        """Main processing loop — blocks until shutdown signal."""
        while not self._shutdown_event.is_set():
            try:
                # Periodic deduplication cache cleanup
                if time.time() - self._last_cleanup > self.CLEANUP_INTERVAL_SECONDS:
                    self._deduplicator.cleanup_expired()
                    self._last_cleanup = time.time()
                    logger.debug(
                        "Alert processor stats",
                        processed=self._processed_count,
                        errors=self._error_count,
                    )

                # Blocking right-pop from alert queue (2s timeout)
                result = await self._redis.brpop(
                    settings.redis.alert_queue_key,
                    timeout=self.POLL_TIMEOUT_SECONDS,
                )
                if result is None:
                    continue

                _, raw_alert = result
                try:
                    alert = json.loads(raw_alert)
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse alert JSON", error=str(e), raw=raw_alert[:200])
                    self._error_count += 1
                    continue

                # Deduplication check
                if self._deduplicator.is_duplicate(alert):
                    logger.debug(
                        "Alert deduplicated",
                        alert_type=alert.get("alert_type"),
                        table=alert.get("table_name"),
                    )
                    continue

                # Circuit breaker check
                if self._circuit_breaker.is_open:
                    logger.warning(
                        "Circuit breaker OPEN — skipping LLM processing, alert queued to DLQ",
                        alert_type=alert.get("alert_type"),
                    )
                    # Re-queue to a degraded queue for manual review
                    await self._redis.lpush("intellipipe:alerts:degraded", raw_alert)
                    continue

                # Process alert concurrently (up to MAX_CONCURRENT_ALERTS)
                asyncio.create_task(self._process_alert_safe(alert))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alert loop error", error=str(e), exc_info=True)
                self._error_count += 1
                await asyncio.sleep(5)

        logger.info("Alert processor shutdown complete", processed=self._processed_count)

    async def _process_alert_safe(self, alert: Dict[str, Any]) -> None:
        """Process a single alert with concurrency control and error handling."""
        async with self._semaphore:
            with LogContext(
                alert_type=alert.get("alert_type", "unknown"),
                table_name=alert.get("table_name", "unknown"),
                tenant_id=alert.get("tenant_id", "default"),
            ):
                start_time = time.perf_counter()
                try:
                    await self._process_alert(alert)
                    self._circuit_breaker.record_success()
                    self._processed_count += 1
                    duration = time.perf_counter() - start_time
                    logger.info(
                        "Alert processed successfully",
                        duration_ms=round(duration * 1000),
                        total_processed=self._processed_count,
                    )
                except Exception as e:
                    self._circuit_breaker.record_failure()
                    self._error_count += 1
                    logger.error(
                        "Alert processing failed",
                        error=str(e),
                        alert_type=alert.get("alert_type"),
                        exc_info=True,
                    )
                    # Push to dead-letter for manual investigation
                    await self._redis.lpush(
                        "intellipipe:alerts:failed",
                        json.dumps({
                            "alert": alert,
                            "error": str(e),
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                        }),
                    )

    async def _process_alert(self, alert: Dict[str, Any]) -> None:
        """
        Full alert → incident lifecycle:
        1. Persist incident to DB
        2. Run LLM orchestration (RCA + fix + notifications)
        3. Update incident with external refs
        """
        async with self._session_factory() as session:
            incident_repo = IncidentRepository(session)
            memory_repo = IncidentMemoryRepository(session)
            doc_repo = DocumentChunkRepository(session)

            # 1. Determine severity and create DB incident record
            severity = compute_severity(alert)
            anomaly_type = ANOMALY_TYPE_MAP.get(
                alert.get("alert_type", "statistical"), AnomalyType.STATISTICAL
            )

            # Find the table record (or use a placeholder UUID)
            import uuid
            table_id = uuid.uuid4()  # In production: lookup via PipelineTable repo

            incident = await incident_repo.create(
                tenant_id=alert.get("tenant_id", "default"),
                table_id=table_id,
                title=f"[{severity.value.upper()}] {alert.get('alert_type')} on {alert.get('table_name')}",
                anomaly_type=anomaly_type,
                severity=severity,
                description=json.dumps(alert, default=str),
                anomaly_score=alert.get("dq_score"),
                affected_columns=alert.get("columns_removed", []) + alert.get("columns_added", []),
                detection_metadata=alert,
            )

            INCIDENTS_CREATED_TOTAL.labels(
                severity=severity.value,
                source=alert.get("alert_type", "unknown"),
            ).inc()

            await session.commit()

            logger.info(
                "Incident created",
                incident_id=str(incident.id),
                severity=severity.value,
                anomaly_type=anomaly_type.value,
            )

        # 2. Run LLM orchestration (outside session to avoid long-held connections)
        github_client = GitHubPRClient()
        jira_client = JiraTicketClient()
        slack_client = SlackIncidentNotifier()

        async with self._session_factory() as session:
            incident_repo = IncidentRepository(session)
            memory_repo = IncidentMemoryRepository(session)
            doc_repo = DocumentChunkRepository(session)

            agent = IntelliPipeOrchestrationAgent(
                github_client=github_client,
                jira_client=jira_client,
                slack_client=slack_client,
                incident_repo=incident_repo,
                memory_repo=memory_repo,
                doc_repo=doc_repo,
            )

            with timed_operation(PIPELINE_BATCH_DURATION, {"stage": "llm_orchestration"}):
                result = await agent.handle_alert(
                    alert=alert,
                    tenant_id=alert.get("tenant_id", "default"),
                )

            # 3. Update incident with LLM results and external refs
            await incident_repo.update_rca(
                incident_id=incident.id,
                root_cause_analysis=result.get("rca", {}).get("root_cause", ""),
                fix_code=result.get("fix", {}).get("dbt_model_sql"),
                llm_model_used=settings.llm.claude_model,
            )

            await incident_repo.update_external_refs(
                incident_id=incident.id,
                github_pr_url=result.get("github_pr_url"),
                github_pr_number=result.get("github_pr_number"),
                jira_ticket_key=result.get("jira_ticket_key"),
                jira_ticket_url=result.get("jira_ticket_url"),
                slack_message_ts=result.get("slack_message_ts"),
            )

            await session.commit()

            logger.info(
                "Incident fully processed",
                incident_id=str(incident.id),
                pr_url=result.get("github_pr_url"),
                jira_key=result.get("jira_ticket_key"),
                processing_ms=result.get("processing_time_ms"),
            )

    def shutdown(self) -> None:
        """Signal graceful shutdown."""
        logger.info("Shutdown signal received")
        self._shutdown_event.set()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    worker = AlertProcessorWorker()

    # Graceful shutdown on SIGTERM / SIGINT
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, worker.shutdown)

    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
