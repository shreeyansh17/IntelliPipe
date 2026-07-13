"""
IntelliPipe Unit Tests
========================
Tests for:
- Anomaly ensemble scoring logic
- Schema drift detection
- DQ metrics computation
- FastAPI endpoint contracts
- Repository pattern
- LLM prompt formatting
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_order_batch() -> pd.DataFrame:
    """Generate a clean batch of order events for testing."""
    return pd.DataFrame([
        {
            "event_id": str(uuid.uuid4()),
            "order_id": f"ORD-{i:06d}",
            "total_amount": 150.0 + i * 5,
            "subtotal": 130.0 + i * 5,
            "tax_amount": 10.0,
            "shipping_amount": 10.0,
            "item_count": 2,
            "avg_item_price": 65.0 + i * 2.5,
            "max_item_quantity": 2,
            "total_quantity": 3,
            "discount_variance": 0.05,
        }
        for i in range(100)
    ])


@pytest.fixture
def anomalous_order_batch() -> pd.DataFrame:
    """Batch with injected anomalies — extreme prices and quantities."""
    base = pd.DataFrame([
        {
            "event_id": str(uuid.uuid4()),
            "order_id": f"ORD-{i:06d}",
            "total_amount": 150.0,
            "subtotal": 130.0,
            "tax_amount": 10.0,
            "shipping_amount": 10.0,
            "item_count": 2,
            "avg_item_price": 65.0,
            "max_item_quantity": 2,
            "total_quantity": 3,
            "discount_variance": 0.05,
        }
        for i in range(98)
    ])
    # Inject 2 anomalous rows
    anomalies = pd.DataFrame([
        {
            "event_id": str(uuid.uuid4()),
            "order_id": "ORD-ANOMALY-01",
            "total_amount": 999999.0,   # Extreme price
            "subtotal": 999000.0,
            "tax_amount": 999.0,
            "shipping_amount": 0.0,
            "item_count": 1,
            "avg_item_price": 999999.0,
            "max_item_quantity": 50000,  # Extreme quantity
            "total_quantity": 50000,
            "discount_variance": 0.0,
        },
        {
            "event_id": str(uuid.uuid4()),
            "order_id": "ORD-ANOMALY-02",
            "total_amount": 0.01,       # Near-zero amount
            "subtotal": 0.01,
            "tax_amount": 0.0,
            "shipping_amount": 0.0,
            "item_count": 1000,         # Suspicious item count
            "avg_item_price": 0.00001,
            "max_item_quantity": 1000,
            "total_quantity": 1000,
            "discount_variance": 0.99,
        },
    ])
    return pd.concat([base, anomalies], ignore_index=True)


@pytest.fixture
def sample_alert() -> Dict[str, Any]:
    return {
        "alert_type": "null_spike",
        "tenant_id": "tenant_alpha",
        "table_name": "raw_orders",
        "severity": "high",
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "dq_score": 62.5,
        "batch_id": 42,
    }


@pytest.fixture
def sample_rca() -> Dict[str, Any]:
    return {
        "root_cause": "Upstream order service deployed v2.3.1 at 14:00 UTC introducing a bug where customer_email is not populated for guest checkouts.",
        "confidence": 0.87,
        "blast_radius": ["stg_orders", "mart_customer_ltv", "mart_daily_revenue"],
        "investigation_steps": [
            "Check order-service deployment logs for v2.3.1 at 14:00 UTC",
            "Verify guest checkout flow in staging environment",
            "Review null rate trend over the past 6 hours",
        ],
        "remediation_approach": "Add null coalesce in stg_raw_orders for customer_email from orders table",
        "estimated_impact": "12,450 orders affected; downstream revenue metrics may undercount by ~8%",
        "escalate_immediately": False,
        "severity_reasoning": "High severity — affects analytics but not operational systems",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIsolationForestDetector:
    """Unit tests for IsolationForestDetector."""

    def test_train_returns_run_id(self, clean_order_batch: pd.DataFrame):
        """Train should log to MLflow and return a run ID string."""
        from src.anomaly.ensemble import IsolationForestDetector

        detector = IsolationForestDetector()
        with patch("src.anomaly.ensemble.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "abc123def456"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            mock_mlflow.set_experiment.return_value.experiment_id = "exp-1"

            run_id = detector.train(clean_order_batch, "test-experiment")

        assert isinstance(run_id, str)

    def test_score_returns_array_in_range(self, clean_order_batch: pd.DataFrame):
        """Scores must be in [0, 1] for all rows."""
        from src.anomaly.ensemble import IsolationForestDetector
        from sklearn.preprocessing import StandardScaler

        detector = IsolationForestDetector()
        # Manually set up model without MLflow
        from sklearn.ensemble import IsolationForest
        from src.anomaly.ensemble import extract_features

        X = extract_features(clean_order_batch)
        detector._scaler = StandardScaler()
        X_scaled = detector._scaler.fit_transform(X)
        detector._model = IsolationForest(contamination=0.05, random_state=42)
        detector._model.fit(X_scaled)

        scores = detector.score(clean_order_batch)

        assert len(scores) == len(clean_order_batch)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_untrained_detector_returns_zeros(self, clean_order_batch: pd.DataFrame):
        """Untrained detector should return zeros rather than crash."""
        from src.anomaly.ensemble import IsolationForestDetector

        detector = IsolationForestDetector()
        scores = detector.score(clean_order_batch)

        assert len(scores) == len(clean_order_batch)
        assert all(s == 0.0 for s in scores)


class TestZScoreDetector:
    """Unit tests for Z-Score fallback detector."""

    def test_anomalies_score_higher(
        self,
        clean_order_batch: pd.DataFrame,
        anomalous_order_batch: pd.DataFrame,
    ):
        """Anomalous rows should receive higher Z-scores than clean rows."""
        from src.anomaly.ensemble import ZScoreDetector

        detector = ZScoreDetector()
        detector.fit(clean_order_batch)

        clean_scores = detector.score(clean_order_batch)
        anomaly_scores = detector.score(anomalous_order_batch.tail(2))

        assert anomaly_scores.max() > clean_scores.max(), (
            f"Expected anomaly scores {anomaly_scores.max():.3f} > "
            f"clean scores {clean_scores.max():.3f}"
        )

    def test_clean_batch_low_scores(self, clean_order_batch: pd.DataFrame):
        """Clean data should yield low Z-scores."""
        from src.anomaly.ensemble import ZScoreDetector

        detector = ZScoreDetector()
        detector.fit(clean_order_batch)
        scores = detector.score(clean_order_batch)

        # 95th percentile of clean scores should be < 0.5
        assert np.percentile(scores, 95) < 0.5

    def test_feature_contribution_format(self, clean_order_batch: pd.DataFrame):
        """max_zscores_per_row should return correct length arrays."""
        from src.anomaly.ensemble import ZScoreDetector

        detector = ZScoreDetector()
        detector.fit(clean_order_batch)
        scores, features = detector.max_zscores_per_row(clean_order_batch.head(10))

        assert len(scores) == 10
        assert len(features) == 10


class TestEnsembleScorer:
    """Unit tests for ensemble anomaly scoring."""

    def _make_scorer_with_mocks(self):
        from src.anomaly.ensemble import (
            AutoencoderDetector,
            EnsembleAnomalyScorer,
            IsolationForestDetector,
            ZScoreDetector,
        )

        if_det = MagicMock(spec=IsolationForestDetector)
        ae_det = MagicMock(spec=AutoencoderDetector)
        z_det = MagicMock(spec=ZScoreDetector)

        if_det._version = "v1"
        ae_det._version = "v1"
        z_det._version = "v1"

        return EnsembleAnomalyScorer(if_det, ae_det, z_det), if_det, ae_det, z_det

    def test_high_score_flagged_as_anomaly(self, clean_order_batch: pd.DataFrame):
        """Rows with ensemble score >= 0.3 should be flagged."""
        scorer, if_det, ae_det, z_det = self._make_scorer_with_mocks()

        n = len(clean_order_batch)
        if_det.score.return_value = np.full(n, 0.9)
        ae_det.score.return_value = np.full(n, 0.9)
        z_det.score.return_value = np.full(n, 0.9)
        z_det.max_zscores_per_row.return_value = (np.full(n, 5.0), ["total_amount"] * n)
        if_det.feature_contributions.return_value = {"total_amount": 0.8}

        results = scorer.score_batch(clean_order_batch)

        assert all(r.is_anomaly for r in results)
        assert all(r.severity in ("critical", "high") for r in results)

    def test_low_score_not_flagged(self, clean_order_batch: pd.DataFrame):
        """Rows with low ensemble score should not be flagged as anomalies."""
        scorer, if_det, ae_det, z_det = self._make_scorer_with_mocks()

        n = len(clean_order_batch)
        if_det.score.return_value = np.zeros(n)
        ae_det.score.return_value = np.zeros(n)
        z_det.score.return_value = np.zeros(n)
        z_det.max_zscores_per_row.return_value = (np.zeros(n), [""] * n)

        results = scorer.score_batch(clean_order_batch)

        assert not any(r.is_anomaly for r in results)

    def test_result_count_matches_input(self, clean_order_batch: pd.DataFrame):
        """Result list length must equal input DataFrame length."""
        scorer, if_det, ae_det, z_det = self._make_scorer_with_mocks()

        n = len(clean_order_batch)
        if_det.score.return_value = np.random.uniform(0, 1, n)
        ae_det.score.return_value = np.random.uniform(0, 1, n)
        z_det.score.return_value = np.random.uniform(0, 1, n)
        z_det.max_zscores_per_row.return_value = (np.random.uniform(0, 5, n), ["total_amount"] * n)
        if_det.feature_contributions.return_value = {}

        results = scorer.score_batch(clean_order_batch)

        assert len(results) == n


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureExtraction:
    """Tests for the extract_features function."""

    def test_output_shape(self, clean_order_batch: pd.DataFrame):
        from src.anomaly.ensemble import extract_features, ORDER_NUMERIC_FEATURES
        features = extract_features(clean_order_batch)
        assert features.shape[0] == len(clean_order_batch)
        assert features.shape[1] == len(ORDER_NUMERIC_FEATURES)

    def test_no_nan_values(self, clean_order_batch: pd.DataFrame):
        from src.anomaly.ensemble import extract_features
        features = extract_features(clean_order_batch)
        assert not features.isnull().any().any(), "Features must not contain NaN"

    def test_handles_missing_columns(self):
        """extract_features must not crash on partial input."""
        from src.anomaly.ensemble import extract_features
        minimal_df = pd.DataFrame([{"total_amount": 100.0}])
        features = extract_features(minimal_df)
        assert features.shape[0] == 1
        assert not features.isnull().any().any()


# ─────────────────────────────────────────────────────────────────────────────
# Kafka Producer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEventGenerator:
    """Tests for synthetic event generation."""

    def test_clean_event_has_required_fields(self):
        from src.pipeline.kafka_producer import generate_order_event
        event = generate_order_event()

        required = ["event_id", "order_id", "tenant_id", "total_amount", "event_timestamp"]
        for field in required:
            assert field in event, f"Missing required field: {field}"

    def test_null_spike_injection(self):
        from src.pipeline.kafka_producer import generate_order_event
        events = [generate_order_event(anomaly_type="null_spike") for _ in range(20)]
        null_count = sum(1 for e in events if e.get("customer_email") is None)
        assert null_count > 10, "null_spike should produce many null customer_email values"

    def test_schema_add_injects_extra_columns(self):
        from src.pipeline.kafka_producer import generate_order_event
        event = generate_order_event(anomaly_type="schema_add")
        assert "promo_code" in event or "loyalty_points" in event, (
            "schema_add should add promo_code or loyalty_points"
        )

    def test_schema_remove_drops_columns(self):
        from src.pipeline.kafka_producer import generate_order_event
        event = generate_order_event(anomaly_type="schema_remove")
        assert "shipping_amount" not in event, "schema_remove should drop shipping_amount"

    def test_outlier_injection_extreme_values(self):
        from src.pipeline.kafka_producer import generate_order_event
        events = [generate_order_event(anomaly_type="outlier") for _ in range(20)]
        extreme = [e for e in events if e.get("total_amount", 0) > 10000]
        assert len(extreme) > 5, "outlier should produce some extreme total_amount values"

    def test_invalid_enum_injection(self):
        from src.pipeline.kafka_producer import generate_order_event
        VALID_STATUSES = {"pending", "confirmed", "shipped", "delivered", "cancelled", "refunded"}
        events = [generate_order_event(anomaly_type="invalid_enum") for _ in range(20)]
        invalid = [e for e in events if e.get("order_status") not in VALID_STATUSES]
        assert len(invalid) > 10, "invalid_enum should produce invalid order statuses"

    def test_event_id_is_uuid_format(self):
        from src.pipeline.kafka_producer import generate_order_event
        event = generate_order_event()
        try:
            uuid.UUID(event["event_id"])
        except ValueError:
            pytest.fail(f"event_id is not a valid UUID: {event['event_id']}")

    def test_tenant_assignment(self):
        from src.pipeline.kafka_producer import generate_order_event
        event = generate_order_event(tenant_id="tenant_alpha")
        assert event["tenant_id"] == "tenant_alpha"

        events = [generate_order_event() for _ in range(50)]
        observed_tenants = {e["tenant_id"] for e in events}
        assert len(observed_tenants) > 1, "Should produce events for multiple tenants"


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIHealth:
    """Tests for health and readiness endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked Redis."""
        from src.api.main import app

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        app.state.redis = mock_redis

        with TestClient(app) as c:
            yield c

    def test_readiness_probe(self, client: TestClient):
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_health_check_structure(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "checks" in data
        assert "timestamp" in data


class TestAPIAuth:
    """Tests for JWT authentication flow."""

    @pytest.fixture
    def client(self):
        from src.api.main import app
        app.state.redis = AsyncMock()
        with TestClient(app) as c:
            yield c

    def test_login_valid_credentials(self, client: TestClient):
        resp = client.post(
            "/api/v1/auth/token",
            json={"username": "admin", "password": "intellipipe-admin", "tenant_id": "default"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_login_invalid_credentials(self, client: TestClient):
        resp = client.post(
            "/api/v1/auth/token",
            json={"username": "admin", "password": "wrong-password", "tenant_id": "default"},
        )
        assert resp.status_code == 401

    def test_protected_endpoint_without_token(self, client: TestClient):
        resp = client.get("/api/v1/dq/scores")
        assert resp.status_code == 403  # HTTPBearer returns 403 when no auth

    def test_protected_endpoint_with_valid_token(self, client: TestClient):
        # Get token
        token_resp = client.post(
            "/api/v1/auth/token",
            json={"username": "admin", "password": "intellipipe-admin", "tenant_id": "default"},
        )
        token = token_resp.json()["access_token"]

        # Use token
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(return_value=[])
        client.app.state.redis = mock_redis

        resp = client.get(
            "/api/v1/dq/scores",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Schema Drift Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaDriftDetection:
    """Tests for the SchemaDriftDetector."""

    def _make_detector(self):
        mock_redis = MagicMock()
        mock_redis.lpush = MagicMock()

        from src.pipeline.spark_consumer import SchemaDriftDetector
        return SchemaDriftDetector(mock_redis, "test_tenant", "raw_orders"), mock_redis

    def test_no_drift_returns_none(self, clean_order_batch: pd.DataFrame):
        """Clean batch matching the schema should return None."""
        from src.pipeline.spark_consumer import EXPECTED_COLUMNS

        detector, _ = self._make_detector()

        # Create a mock Spark-like DataFrame with the expected columns
        class MockDF:
            columns = list(EXPECTED_COLUMNS)

        result = detector.detect_drift(MockDF(), batch_id=1)
        assert result is None

    def test_added_columns_detected(self):
        from src.pipeline.spark_consumer import EXPECTED_COLUMNS

        detector, mock_redis = self._make_detector()

        class MockDF:
            columns = list(EXPECTED_COLUMNS) + ["promo_code", "loyalty_points"]

        result = detector.detect_drift(MockDF(), batch_id=2)

        assert result is not None
        assert "promo_code" in result["columns_added"]
        assert "loyalty_points" in result["columns_added"]
        assert len(result["columns_removed"]) == 0
        mock_redis.lpush.assert_called_once()

    def test_removed_columns_flagged_critical(self):
        from src.pipeline.spark_consumer import EXPECTED_COLUMNS, REQUIRED_COLUMNS

        detector, _ = self._make_detector()
        required_col = next(iter(REQUIRED_COLUMNS))

        class MockDF:
            columns = [c for c in EXPECTED_COLUMNS if c != required_col]

        result = detector.detect_drift(MockDF(), batch_id=3)

        assert result is not None
        assert required_col in result["columns_removed"]
        assert result["severity"] == "critical"


# ─────────────────────────────────────────────────────────────────────────────
# LLM Agent Prompt Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMPrompts:
    """Tests for LLM prompt construction and parsing."""

    def test_jira_description_format(self, sample_alert: Dict, sample_rca: Dict):
        from src.agents.langchain_agent import IntelliPipeOrchestrationAgent
        desc = IntelliPipeOrchestrationAgent._format_jira_description(sample_alert, sample_rca)

        assert "null_spike" in desc
        assert "raw_orders" in desc
        assert "IntelliPipe" in desc
        assert sample_rca["root_cause"][:50] in desc

    def test_rca_fallback_on_json_parse_error(self, sample_alert: Dict):
        """RCA agent should handle non-JSON responses gracefully."""
        import asyncio
        from src.agents.langchain_agent import RootCauseAnalysisAgent

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=(
            "The root cause is a missing null check in the upstream service.",
            {"input_tokens": 100, "output_tokens": 50},
        ))

        agent = RootCauseAnalysisAgent(mock_llm)
        rca = asyncio.get_event_loop().run_until_complete(
            agent.analyse(alert=sample_alert)
        )

        # Should not raise, should return a fallback dict
        assert "root_cause" in rca
        assert "escalate_immediately" in rca
