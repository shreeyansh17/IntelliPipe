"""
IntelliPipe Database Models
============================
SQLAlchemy 2.x ORM models with:
- pgvector extension for embedding storage
- Full audit trails (created_at / updated_at / deleted_at)
- Multi-tenant scoping via tenant_id
- Soft deletes
- JSONB columns for flexible metadata
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ---------------------------------------------------------------------------
# Base model with audit fields
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """SQLAlchemy declarative base with common audit columns."""


class TimestampMixin:
    """Mixin for created_at / updated_at audit columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """Mixin for soft deletes via deleted_at."""

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SeverityLevel(str, PyEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IncidentStatus(str, PyEnum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"


class AnomalyType(str, PyEnum):
    SCHEMA_DRIFT = "schema_drift"
    NULL_SPIKE = "null_spike"
    STATISTICAL = "statistical"
    FRESHNESS = "freshness"
    DUPLICATE = "duplicate"
    REFERENTIAL = "referential"
    DISTRIBUTION = "distribution"


class RemediationStatus(str, PyEnum):
    PENDING = "pending"
    PR_CREATED = "pr_created"
    PR_MERGED = "pr_merged"
    APPLIED = "applied"
    REJECTED = "rejected"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


class PipelineTable(Base, TimestampMixin, SoftDeleteMixin):
    """
    Registry of tables monitored by IntelliPipe.
    One record per tenant-table combination.
    """

    __tablename__ = "pipeline_tables"
    __table_args__ = (
        UniqueConstraint("tenant_id", "schema_name", "table_name"),
        Index("ix_pipeline_tables_tenant", "tenant_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    schema_name: Mapped[str] = mapped_column(String(128), nullable=False)
    table_name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    owner_team: Mapped[str | None] = mapped_column(String(128))
    sla_freshness_minutes: Mapped[int] = mapped_column(Integer, default=30)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    dbt_model_ref: Mapped[str | None] = mapped_column(String(256))
    kafka_topic: Mapped[str | None] = mapped_column(String(256))

    # Relationships
    incidents: Mapped[list[Incident]] = relationship(back_populates="table")
    dq_snapshots: Mapped[list[DQSnapshot]] = relationship(back_populates="table")
    schema_versions: Mapped[list[SchemaVersion]] = relationship(back_populates="table")


class SchemaVersion(Base, TimestampMixin):
    """
    Versioned schema snapshots for drift detection.
    Each record represents the observed schema at a point in time.
    """

    __tablename__ = "schema_versions"
    __table_args__ = (Index("ix_schema_versions_table_id", "table_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    table_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_tables.id"), nullable=False
    )
    version_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    columns: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    partition_info: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    row_count: Mapped[int | None] = mapped_column(BigInteger)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    table: Mapped[PipelineTable] = relationship(back_populates="schema_versions")


class DQSnapshot(Base, TimestampMixin):
    """
    Point-in-time data quality scorecard snapshot.
    Stores dimension scores (completeness, validity, freshness, etc.)
    and the raw GE validation results.
    """

    __tablename__ = "dq_snapshots"
    __table_args__ = (
        Index("ix_dq_snapshots_table_captured", "table_id", "captured_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    table_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_tables.id"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    # Dimension scores (0-100)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    completeness_score: Mapped[float] = mapped_column(Float, default=100.0)
    validity_score: Mapped[float] = mapped_column(Float, default=100.0)
    uniqueness_score: Mapped[float] = mapped_column(Float, default=100.0)
    freshness_score: Mapped[float] = mapped_column(Float, default=100.0)
    consistency_score: Mapped[float] = mapped_column(Float, default=100.0)
    referential_score: Mapped[float] = mapped_column(Float, default=100.0)

    row_count: Mapped[int] = mapped_column(BigInteger, default=0)
    failed_checks: Mapped[int] = mapped_column(Integer, default=0)
    total_checks: Mapped[int] = mapped_column(Integer, default=0)
    ge_results: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict
    )

    table: Mapped[PipelineTable] = relationship(back_populates="dq_snapshots")


class Incident(Base, TimestampMixin, SoftDeleteMixin):
    """
    Data quality / anomaly incident record.
    Full lifecycle from detection → RCA → remediation.
    """

    __tablename__ = "incidents"
    __table_args__ = (
        Index("ix_incidents_tenant_status", "tenant_id", "status"),
        Index("ix_incidents_table_id", "table_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    table_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_tables.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    anomaly_type: Mapped[AnomalyType] = mapped_column(Enum(AnomalyType), nullable=False)
    severity: Mapped[SeverityLevel] = mapped_column(Enum(SeverityLevel), nullable=False)
    status: Mapped[IncidentStatus] = mapped_column(
        Enum(IncidentStatus), default=IncidentStatus.OPEN, nullable=False
    )

    # Anomaly detection context
    anomaly_score: Mapped[float | None] = mapped_column(Float)
    affected_columns: Mapped[list[str]] = mapped_column(JSONB, default=list)
    affected_row_count: Mapped[int | None] = mapped_column(BigInteger)
    detection_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    # LLM-generated content
    root_cause_analysis: Mapped[str | None] = mapped_column(Text)
    fix_code: Mapped[str | None] = mapped_column(Text)
    postmortem: Mapped[str | None] = mapped_column(Text)
    llm_model_used: Mapped[str | None] = mapped_column(String(128))

    # External references
    github_pr_url: Mapped[str | None] = mapped_column(String(512))
    github_pr_number: Mapped[int | None] = mapped_column(Integer)
    jira_ticket_key: Mapped[str | None] = mapped_column(String(64))
    jira_ticket_url: Mapped[str | None] = mapped_column(String(512))
    slack_message_ts: Mapped[str | None] = mapped_column(String(64))

    # Resolution tracking
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_by: Mapped[str | None] = mapped_column(String(256))
    resolution_notes: Mapped[str | None] = mapped_column(Text)

    # Relationships
    table: Mapped[PipelineTable] = relationship(back_populates="incidents")
    remediations: Mapped[list[RemediationAction]] = relationship(
        back_populates="incident"
    )


class RemediationAction(Base, TimestampMixin):
    """
    Auto-remediation actions generated by the LLM agent.
    Tracks PR lifecycle, approval status, and execution outcome.
    """

    __tablename__ = "remediation_actions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    incident_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False, index=True
    )
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[RemediationStatus] = mapped_column(
        Enum(RemediationStatus), default=RemediationStatus.PENDING
    )
    generated_sql: Mapped[str | None] = mapped_column(Text)
    generated_dbt_code: Mapped[str | None] = mapped_column(Text)
    pr_branch: Mapped[str | None] = mapped_column(String(256))
    pr_url: Mapped[str | None] = mapped_column(String(512))
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    approved_by: Mapped[str | None] = mapped_column(String(256))
    execution_log: Mapped[str | None] = mapped_column(Text)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict
    )

    incident: Mapped[Incident] = relationship(back_populates="remediations")


class AnomalyModelRun(Base, TimestampMixin):
    """
    MLflow-linked model run metadata for anomaly detection models.
    Shadow evaluation results stored here for model governance.
    """

    __tablename__ = "anomaly_model_runs"
    __table_args__ = (Index("ix_model_runs_table_type", "table_id", "model_type"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    table_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_tables.id"), nullable=False
    )
    model_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # isolation_forest | autoencoder | zscore
    mlflow_run_id: Mapped[str | None] = mapped_column(String(128))
    mlflow_experiment_id: Mapped[str | None] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    is_production: Mapped[bool] = mapped_column(Boolean, default=False)
    is_shadow: Mapped[bool] = mapped_column(Boolean, default=False)
    training_rows: Mapped[int | None] = mapped_column(BigInteger)
    precision: Mapped[float | None] = mapped_column(Float)
    recall: Mapped[float | None] = mapped_column(Float)
    f1_score: Mapped[float | None] = mapped_column(Float)
    auc_roc: Mapped[float | None] = mapped_column(Float)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    feature_importance: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DocumentChunk(Base, TimestampMixin):
    """
    RAG document chunks with pgvector embeddings.
    Stores dbt model docs, lineage descriptions, and data contracts.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_document_chunks_source_type", "source_type", "source_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # dbt_model | lineage | contract
    source_id: Mapped[str] = mapped_column(String(256), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(512))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1536)
    )  # OpenAI ada-002 dim
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class IncidentMemory(Base, TimestampMixin):
    """
    Long-term vector memory for the LLM agent.
    Enables cross-incident pattern recognition and solution retrieval.
    """

    __tablename__ = "incident_memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    incident_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("incidents.id"), nullable=True
    )
    memory_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # rca | fix | postmortem
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536))
    relevance_score: Mapped[float | None] = mapped_column(Float)
    tags: Mapped[list[str]] = mapped_column(JSONB, default=list)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict
    )


class SLATrend(Base, TimestampMixin):
    """
    Historical SLA metrics for trend forecasting.
    Aggregated daily/hourly for forecasting with ARIMA/Prophet.
    """

    __tablename__ = "sla_trends"
    __table_args__ = (
        UniqueConstraint("table_id", "window_start", "granularity"),
        Index("ix_sla_trends_table_window", "table_id", "window_start"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    table_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_tables.id"), nullable=False
    )
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    granularity: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # hourly | daily
    row_count: Mapped[int | None] = mapped_column(BigInteger)
    avg_dq_score: Mapped[float | None] = mapped_column(Float)
    incident_count: Mapped[int] = mapped_column(Integer, default=0)
    sla_breaches: Mapped[int] = mapped_column(Integer, default=0)
    p95_latency_ms: Mapped[float | None] = mapped_column(Float)
    anomaly_count: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
