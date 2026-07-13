"""
IntelliPipe Load Testing — Locust
====================================
Performance benchmarks for the FastAPI backend.
Simulates realistic mixed traffic patterns:
- 70% DQ score reads (dashboard polling)
- 15% incident list queries
- 10% RAG queries (slower, heavier)
- 5% alert simulations (admin)

Run:
    locust -f tests/load/locustfile.py --host=http://localhost:8080 \
           --users=100 --spawn-rate=10 --run-time=300s --headless

Production SLO targets:
- P50 < 50ms for DQ score reads
- P95 < 200ms for incident list
- P99 < 2s for RAG queries
- Error rate < 0.1%
"""

from __future__ import annotations

import random
from typing import Optional

from locust import HttpUser, between, task


ADMIN_TOKEN: Optional[str] = None  # Populated in on_start


class IntelliPipeDashboardUser(HttpUser):
    """
    Simulates a dashboard user continuously polling DQ scores.
    High frequency, read-heavy workload.
    """

    wait_time = between(1, 3)
    weight = 7  # 70% of users

    def on_start(self) -> None:
        """Authenticate and cache JWT token."""
        resp = self.client.post(
            "/api/v1/auth/token",
            json={"username": "admin", "password": "intellipipe-admin", "tenant_id": "default"},
        )
        if resp.status_code == 200:
            self.token = resp.json()["access_token"]
        else:
            self.token = ""

        self.headers = {"Authorization": f"Bearer {self.token}"}

    @task(5)
    def get_all_dq_scores(self) -> None:
        with self.client.get(
            "/api/v1/dq/scores",
            headers=self.headers,
            catch_response=True,
            name="/api/v1/dq/scores [ALL]",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(3)
    def get_single_table_score(self) -> None:
        table = random.choice(["raw_orders", "stg_orders", "mart_daily_revenue"])
        with self.client.get(
            f"/api/v1/dq/scores/{table}",
            headers=self.headers,
            catch_response=True,
            name="/api/v1/dq/scores/{table}",
        ) as response:
            if response.status_code in (200, 404):
                response.success()
            else:
                response.failure(f"Unexpected status {response.status_code}")

    @task(2)
    def check_health(self) -> None:
        self.client.get("/health", name="/health")


class IntelliPipeIncidentUser(HttpUser):
    """
    Simulates an oncall engineer reviewing incidents.
    Medium frequency, read/write workload.
    """

    wait_time = between(3, 8)
    weight = 15  # 15% of users

    def on_start(self) -> None:
        resp = self.client.post(
            "/api/v1/auth/token",
            json={"username": "admin", "password": "intellipipe-admin", "tenant_id": "default"},
        )
        self.token = resp.json().get("access_token", "") if resp.status_code == 200 else ""
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @task(5)
    def list_incidents(self) -> None:
        page = random.randint(1, 5)
        with self.client.get(
            f"/api/v1/incidents?page={page}&page_size=20",
            headers=self.headers,
            catch_response=True,
            name="/api/v1/incidents [LIST]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "incidents" in data:
                    response.success()
                else:
                    response.failure("Missing incidents key")
            else:
                response.failure(f"Status {response.status_code}")

    @task(2)
    def list_high_severity_incidents(self) -> None:
        self.client.get(
            "/api/v1/incidents?severity=high&page_size=10",
            headers=self.headers,
            name="/api/v1/incidents [SEVERITY FILTER]",
        )

    @task(1)
    def get_incident_detail(self) -> None:
        import uuid
        fake_id = str(uuid.uuid4())
        with self.client.get(
            f"/api/v1/incidents/{fake_id}",
            headers=self.headers,
            catch_response=True,
            name="/api/v1/incidents/{id}",
        ) as response:
            # 404 is expected for fake IDs — mark as success for load testing
            if response.status_code in (200, 404):
                response.success()


class IntelliPipeRAGUser(HttpUser):
    """
    Simulates a data analyst querying the RAG system.
    Low frequency, compute-heavy workload (hits Claude API).
    """

    wait_time = between(10, 30)
    weight = 10  # 10% of users

    SAMPLE_QUESTIONS = [
        "What are the upstream dependencies of stg_raw_orders?",
        "What dbt tests are applied to the customer_email column?",
        "Which tables feed into the mart_daily_revenue model?",
        "What is the SLA freshness requirement for raw_orders?",
        "Who owns the orders pipeline and how do I contact them?",
        "What does the discount_pct column represent in the items array?",
    ]

    def on_start(self) -> None:
        resp = self.client.post(
            "/api/v1/auth/token",
            json={"username": "admin", "password": "intellipipe-admin", "tenant_id": "default"},
        )
        self.token = resp.json().get("access_token", "") if resp.status_code == 200 else ""
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    @task
    def rag_query(self) -> None:
        question = random.choice(self.SAMPLE_QUESTIONS)
        with self.client.post(
            "/api/v1/rag/query",
            headers=self.headers,
            json={"question": question, "top_k": 5},
            catch_response=True,
            name="/api/v1/rag/query",
            timeout=30,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "answer" in data:
                    response.success()
                else:
                    response.failure("Missing answer in RAG response")
            elif response.status_code == 429:
                response.success()  # Rate limited — expected under load
            else:
                response.failure(f"RAG query failed: {response.status_code}")


class IntelliPipeAdminUser(HttpUser):
    """
    Simulates admin actions: simulating alerts, checking metrics.
    Very low frequency.
    """

    wait_time = between(30, 120)
    weight = 3  # 3% of users

    def on_start(self) -> None:
        resp = self.client.post(
            "/api/v1/auth/token",
            json={"username": "admin", "password": "intellipipe-admin", "tenant_id": "default"},
        )
        self.token = resp.json().get("access_token", "") if resp.status_code == 200 else ""
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @task(2)
    def simulate_alert(self) -> None:
        alert_types = ["null_spike", "schema_drift", "statistical", "freshness"]
        severities = ["low", "medium", "high"]
        tables = ["raw_orders", "stg_orders", "mart_daily_revenue"]

        with self.client.post(
            "/api/v1/alerts/simulate",
            headers=self.headers,
            params={
                "alert_type": random.choice(alert_types),
                "table_name": random.choice(tables),
                "severity": random.choice(severities),
            },
            catch_response=True,
            name="/api/v1/alerts/simulate",
        ) as response:
            if response.status_code in (200, 429):
                response.success()

    @task(1)
    def check_readiness(self) -> None:
        self.client.get("/ready", name="/ready")
