"""
IntelliPipe Integration Tests
================================
Tests the full pipeline with real Redis and PostgreSQL connections
but mocked external services (Claude API, GitHub, Jira, Slack).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
import redis.asyncio as aioredis
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from src.core.config import get_settings
from src.db.models import Base, AnomalyType, SeverityLevel

settings = get_settings()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="session")
async def db_engine():
    """Create test database engine and tables."""
    engine = create_async_engine(settings.database.url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    """Provide a transactional test session that rolls back after each test."""
    async with db_engine.begin() as conn:
        session_factory = async_sessionmaker(bind=conn, expire_on_commit=False)
        async with session_factory() as session:
            yield session
            await session.rollback()


@pytest_asyncio.fixture
async def redis_client():
    """Provide a test Redis client and clean up test keys after each test."""
    client = await aioredis.from_url(settings.redis.url, decode_responses=True)
    yield client
    # Cleanup test keys
    keys = await client.keys("intellipipe:test:*")
    if keys:
        await client.delete(*keys)
    await client.close()


@pytest.fixture
def sample_alert() -> Dict[str, Any]:
    return {
        "alert_type": "null_spike",
        "tenant_id": "test_tenant",
        "table_name": "raw_orders",
        "severity": "high",
        "dq_score": 65.3,
        "scores": {
            "overall": 65.3,
            "completeness": 55.0,
            "validity": 98.0,
        },
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": 999,
    }


# ---------------------------------------------------------------------------
# Redis integration tests
# ---------------------------------------------------------------------------


class TestRedisAlertQueue:
    """Integration tests for Redis alert queue operations."""

    @pytest.mark.asyncio
    async def test_alert_enqueue_and_dequeue(
        self,
        redis_client: aioredis.Redis,
        sample_alert: Dict[str, Any],
    ) -> None:
        """Alert pushed to queue should be retrievable."""
        test_key = "intellipipe:test:alerts"

        await redis_client.lpush(test_key, json.dumps(sample_alert))
        raw = await redis_client.rpop(test_key)

        assert raw is not None
        dequeued = json.loads(raw)
        assert dequeued["alert_type"] == sample_alert["alert_type"]
        assert dequeued["table_name"] == sample_alert["table_name"]

    @pytest.mark.asyncio
    async def test_dq_score_cache_roundtrip(
        self,
        redis_client: aioredis.Redis,
    ) -> None:
        """DQ scores written to Redis should be retrievable and parseable."""
        test_key = "intellipipe:test:dq_scores:test_tenant:raw_orders"
        score_data = {
            "overall": 87.5,
            "completeness": 99.0,
            "table": "raw_orders",
            "tenant_id": "test_tenant",
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

        await redis_client.setex(test_key, 300, json.dumps(score_data))
        raw = await redis_client.get(test_key)

        assert raw is not None
        retrieved = json.loads(raw)
        assert retrieved["overall"] == 87.5
        assert retrieved["completeness"] == 99.0

    @pytest.mark.asyncio
    async def test_alert_queue_fifo_ordering(
        self,
        redis_client: aioredis.Redis,
    ) -> None:
        """Alerts should be processed in FIFO order (LPUSH + BRPOP)."""
        test_key = "intellipipe:test:fifo"

        alerts = [{"id": i, "alert_type": f"type_{i}"} for i in range(5)]
        for alert in alerts:
            await redis_client.lpush(test_key, json.dumps(alert))

        dequeued_ids = []
        for _ in range(5):
            raw = await redis_client.rpop(test_key)
            if raw:
                dequeued_ids.append(json.loads(raw)["id"])

        # LPUSH order reversed, BRPOP pops from right → FIFO
        assert dequeued_ids == list(range(5))


# ---------------------------------------------------------------------------
# Repository integration tests
# ---------------------------------------------------------------------------


class TestIncidentRepository:
    """Integration tests for IncidentRepository with real PostgreSQL."""

    @pytest.mark.asyncio
    async def test_create_and_retrieve_incident(self, db_session: AsyncSession) -> None:
        """Created incident should be retrievable by ID."""
        from src.db.repositories.incident_repo import IncidentRepository

        repo = IncidentRepository(db_session)
        table_id = uuid.uuid4()

        incident = await repo.create(
            tenant_id="test_tenant",
            table_id=table_id,
            title="Test null spike incident",
            anomaly_type=AnomalyType.NULL_SPIKE,
            severity=SeverityLevel.HIGH,
            description="Integration test incident",
            anomaly_score=0.75,
            affected_columns=["customer_email", "shipping_city"],
        )

        assert incident.id is not None
        assert incident.title == "Test null spike incident"
        assert incident.severity == SeverityLevel.HIGH

        retrieved = await repo.get_by_id(incident.id)
        assert retrieved is not None
        assert retrieved.id == incident.id
        assert retrieved.anomaly_type == AnomalyType.NULL_SPIKE

    @pytest.mark.asyncio
    async def test_update_rca(self, db_session: AsyncSession) -> None:
        """RCA update should persist root cause and fix code."""
        from src.db.repositories.incident_repo import IncidentRepository

        repo = IncidentRepository(db_session)
        incident = await repo.create(
            tenant_id="test_tenant",
            table_id=uuid.uuid4(),
            title="Test RCA update",
            anomaly_type=AnomalyType.STATISTICAL,
            severity=SeverityLevel.MEDIUM,
        )

        updated = await repo.update_rca(
            incident_id=incident.id,
            root_cause_analysis="The upstream service stopped populating the email field.",
            fix_code="SELECT coalesce(email, 'unknown@example.com') as email ...",
            llm_model_used="claude-opus-4-5",
        )

        assert updated is not None
        assert updated.root_cause_analysis is not None
        assert "email field" in updated.root_cause_analysis
        assert updated.llm_model_used == "claude-opus-4-5"

    @pytest.mark.asyncio
    async def test_list_open_incidents_with_tenant_isolation(
        self, db_session: AsyncSession
    ) -> None:
        """Open incidents should be scoped to the requesting tenant."""
        from src.db.repositories.incident_repo import IncidentRepository

        repo = IncidentRepository(db_session)

        # Create incidents for two different tenants
        for tenant in ["tenant_a", "tenant_b"]:
            await repo.create(
                tenant_id=tenant,
                table_id=uuid.uuid4(),
                title=f"Incident for {tenant}",
                anomaly_type=AnomalyType.NULL_SPIKE,
                severity=SeverityLevel.LOW,
            )

        tenant_a_incidents, total_a = await repo.list_open("tenant_a")
        tenant_b_incidents, total_b = await repo.list_open("tenant_b")

        # Each tenant should only see their own incidents
        for incident in tenant_a_incidents:
            assert incident.tenant_id == "tenant_a"
        for incident in tenant_b_incidents:
            assert incident.tenant_id == "tenant_b"


# ---------------------------------------------------------------------------
# FastAPI integration tests
# ---------------------------------------------------------------------------


class TestAPIIntegration:
    """Full API integration tests with in-process test client."""

    @pytest_asyncio.fixture
    async def auth_client(self) -> AsyncClient:
        from src.api.main import app

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.keys = AsyncMock(return_value=[])
        mock_redis.get = AsyncMock(return_value=None)
        app.state.redis = mock_redis

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Get auth token
            resp = await client.post(
                "/api/v1/auth/token",
                json={
                    "username": "admin",
                    "password": "intellipipe-admin",
                    "tenant_id": "default",
                },
            )
            assert resp.status_code == 200
            token = resp.json()["access_token"]
            client.headers.update({"Authorization": f"Bearer {token}"})
            yield client

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_200(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")

    @pytest.mark.asyncio
    async def test_dq_scores_empty_when_no_data(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.get("/api/v1/dq/scores")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_incident_list_pagination(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.get("/api/v1/incidents?page=1&page_size=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "incidents" in data
        assert "total" in data
        assert "page" in data
        assert data["page"] == 1
        assert data["page_size"] == 10

    @pytest.mark.asyncio
    async def test_rag_query_returns_structured_response(
        self, auth_client: AsyncClient
    ) -> None:
        resp = await auth_client.post(
            "/api/v1/rag/query",
            json={"question": "What is the stg_raw_orders model?", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert isinstance(data["sources"], list)

    @pytest.mark.asyncio
    async def test_simulate_alert_requires_admin(
        self, auth_client: AsyncClient
    ) -> None:
        """Alert simulation should require admin role."""
        # Current client is admin — should work
        mock_redis = auth_client.app.state.redis
        mock_redis.lpush = AsyncMock()

        resp = await auth_client.post(
            "/api/v1/alerts/simulate",
            params={
                "alert_type": "null_spike",
                "table_name": "raw_orders",
                "severity": "medium",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_viewer_cannot_simulate_alert(self) -> None:
        """Viewer role should receive 403 on admin endpoints."""
        from src.api.main import app

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        app.state.redis = mock_redis

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Login as viewer
            resp = await client.post(
                "/api/v1/auth/token",
                json={
                    "username": "viewer",
                    "password": "intellipipe-view",
                    "tenant_id": "default",
                },
            )
            assert resp.status_code == 200
            viewer_token = resp.json()["access_token"]
            client.headers.update({"Authorization": f"Bearer {viewer_token}"})

            # Attempt admin-only endpoint
            resp = await client.post(
                "/api/v1/alerts/simulate",
                params={"alert_type": "null_spike"},
            )
            assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_invalid_token_rejected(self) -> None:
        from src.api.main import app

        mock_redis = AsyncMock()
        app.state.redis = mock_redis

        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.get(
                "/api/v1/dq/scores",
                headers={"Authorization": "Bearer invalid.jwt.token"},
            )
            assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Alert processor deduplication tests
# ---------------------------------------------------------------------------


class TestAlertDeduplicator:
    """Integration tests for the alert deduplication logic."""

    def test_same_alert_deduplicated(self, sample_alert: Dict[str, Any]) -> None:
        from src.agents.alert_processor import AlertDeduplicator

        dedup = AlertDeduplicator(window_minutes=30)

        assert not dedup.is_duplicate(sample_alert)
        assert dedup.is_duplicate(sample_alert)  # Second call → duplicate

    def test_different_alerts_not_deduplicated(
        self, sample_alert: Dict[str, Any]
    ) -> None:
        from src.agents.alert_processor import AlertDeduplicator

        dedup = AlertDeduplicator(window_minutes=30)
        alert_2 = {
            **sample_alert,
            "alert_type": "schema_drift",
            "table_name": "stg_orders",
        }

        assert not dedup.is_duplicate(sample_alert)
        assert not dedup.is_duplicate(alert_2)  # Different key → not a duplicate

    def test_severity_computation(self, sample_alert: Dict[str, Any]) -> None:
        from src.agents.alert_processor import compute_severity

        high_score_alert = {**sample_alert, "severity": "low", "dq_score": 45.0}
        assert compute_severity(high_score_alert) == SeverityLevel.CRITICAL

        normal_alert = {**sample_alert, "severity": "medium", "dq_score": 82.0}
        assert compute_severity(normal_alert) == SeverityLevel.MEDIUM
