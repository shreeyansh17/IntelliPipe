"""
Alembic initial migration — IntelliPipe full schema
====================================================
Creates all tables, indexes, extensions, and pgvector setup.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None

VECTOR_DIM = 1536  # Match embedding model dimension


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")  # For text search

    # ---------- pipeline_tables ----------
    op.create_table(
        "pipeline_tables",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("schema_name", sa.String(128), nullable=False),
        sa.Column("table_name", sa.String(256), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("owner_team", sa.String(128)),
        sa.Column("sla_freshness_minutes", sa.Integer, default=30),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("config", sa.dialects.postgresql.JSONB, default={}),
        sa.Column("dbt_model_ref", sa.String(256)),
        sa.Column("kafka_topic", sa.String(256)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("tenant_id", "schema_name", "table_name", name="uq_pipeline_table"),
    )
    op.create_index("ix_pipeline_tables_tenant", "pipeline_tables", ["tenant_id"])

    # ---------- schema_versions ----------
    op.create_table(
        "schema_versions",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("table_id", sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("pipeline_tables.id"), nullable=False),
        sa.Column("version_hash", sa.String(64), nullable=False),
        sa.Column("columns", sa.dialects.postgresql.JSONB, nullable=False),
        sa.Column("partition_info", sa.dialects.postgresql.JSONB),
        sa.Column("row_count", sa.BigInteger),
        sa.Column("size_bytes", sa.BigInteger),
        sa.Column("captured_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_schema_versions_table_id", "schema_versions", ["table_id"])
    op.create_index("ix_schema_versions_captured", "schema_versions", ["captured_at"])

    # ---------- dq_snapshots ----------
    op.create_table(
        "dq_snapshots",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("table_id", sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("pipeline_tables.id"), nullable=False),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("overall_score", sa.Float, nullable=False),
        sa.Column("completeness_score", sa.Float, default=100.0),
        sa.Column("validity_score", sa.Float, default=100.0),
        sa.Column("uniqueness_score", sa.Float, default=100.0),
        sa.Column("freshness_score", sa.Float, default=100.0),
        sa.Column("consistency_score", sa.Float, default=100.0),
        sa.Column("referential_score", sa.Float, default=100.0),
        sa.Column("row_count", sa.BigInteger, default=0),
        sa.Column("failed_checks", sa.Integer, default=0),
        sa.Column("total_checks", sa.Integer, default=0),
        sa.Column("ge_results", sa.dialects.postgresql.JSONB, default={}),
        sa.Column("metadata", sa.dialects.postgresql.JSONB, default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_dq_snapshots_table_captured", "dq_snapshots", ["table_id", "captured_at"])
    op.create_index("ix_dq_snapshots_tenant", "dq_snapshots", ["tenant_id"])

    # ---------- incidents ----------
    op.create_table(
        "incidents",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("table_id", sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("pipeline_tables.id"), nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("anomaly_type", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("status", sa.String(32), default="open", nullable=False),
        sa.Column("anomaly_score", sa.Float),
        sa.Column("affected_columns", sa.dialects.postgresql.JSONB, default=[]),
        sa.Column("affected_row_count", sa.BigInteger),
        sa.Column("detection_metadata", sa.dialects.postgresql.JSONB, default={}),
        sa.Column("root_cause_analysis", sa.Text),
        sa.Column("fix_code", sa.Text),
        sa.Column("postmortem", sa.Text),
        sa.Column("llm_model_used", sa.String(128)),
        sa.Column("github_pr_url", sa.String(512)),
        sa.Column("github_pr_number", sa.Integer),
        sa.Column("jira_ticket_key", sa.String(64)),
        sa.Column("jira_ticket_url", sa.String(512)),
        sa.Column("slack_message_ts", sa.String(64)),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.Column("resolved_by", sa.String(256)),
        sa.Column("resolution_notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_incidents_tenant_status", "incidents", ["tenant_id", "status"])
    op.create_index("ix_incidents_table_id", "incidents", ["table_id"])
    op.create_index("ix_incidents_created_at", "incidents", ["created_at"])

    # ---------- remediation_actions ----------
    op.create_table(
        "remediation_actions",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("incident_id", sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("incidents.id"), nullable=False),
        sa.Column("action_type", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), default="pending"),
        sa.Column("generated_sql", sa.Text),
        sa.Column("generated_dbt_code", sa.Text),
        sa.Column("pr_branch", sa.String(256)),
        sa.Column("pr_url", sa.String(512)),
        sa.Column("applied_at", sa.DateTime(timezone=True)),
        sa.Column("approved_by", sa.String(256)),
        sa.Column("execution_log", sa.Text),
        sa.Column("metadata", sa.dialects.postgresql.JSONB, default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_remediation_incident_id", "remediation_actions", ["incident_id"])

    # ---------- anomaly_model_runs ----------
    op.create_table(
        "anomaly_model_runs",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("table_id", sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("pipeline_tables.id"), nullable=False),
        sa.Column("model_type", sa.String(64), nullable=False),
        sa.Column("mlflow_run_id", sa.String(128)),
        sa.Column("mlflow_experiment_id", sa.String(64)),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("is_production", sa.Boolean, default=False),
        sa.Column("is_shadow", sa.Boolean, default=False),
        sa.Column("training_rows", sa.BigInteger),
        sa.Column("precision", sa.Float),
        sa.Column("recall", sa.Float),
        sa.Column("f1_score", sa.Float),
        sa.Column("auc_roc", sa.Float),
        sa.Column("parameters", sa.dialects.postgresql.JSONB, default={}),
        sa.Column("feature_importance", sa.dialects.postgresql.JSONB, default={}),
        sa.Column("trained_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_model_runs_table_type", "anomaly_model_runs", ["table_id", "model_type"])

    # ---------- document_chunks (RAG) ----------
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("source_type", sa.String(64), nullable=False),
        sa.Column("source_id", sa.String(256), nullable=False),
        sa.Column("source_url", sa.String(512)),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("chunk_index", sa.Integer, default=0),
        sa.Column("embedding", Vector(VECTOR_DIM)),
        sa.Column("metadata", sa.dialects.postgresql.JSONB, default={}),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_document_chunks_tenant", "document_chunks", ["tenant_id"])
    op.create_index("ix_document_chunks_source", "document_chunks", ["source_type", "source_id"])
    # HNSW vector index for fast ANN search
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    # ---------- incident_memories ----------
    op.create_table(
        "incident_memories",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("incident_id", sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("incidents.id")),
        sa.Column("memory_type", sa.String(64), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("summary", sa.Text),
        sa.Column("embedding", Vector(VECTOR_DIM)),
        sa.Column("relevance_score", sa.Float),
        sa.Column("tags", sa.dialects.postgresql.JSONB, default=[]),
        sa.Column("metadata", sa.dialects.postgresql.JSONB, default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_incident_memories_tenant", "incident_memories", ["tenant_id"])
    op.execute(
        "CREATE INDEX ix_incident_memories_embedding_hnsw ON incident_memories "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    # ---------- sla_trends ----------
    op.create_table(
        "sla_trends",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("table_id", sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("pipeline_tables.id"), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("granularity", sa.String(16), nullable=False),
        sa.Column("row_count", sa.BigInteger),
        sa.Column("avg_dq_score", sa.Float),
        sa.Column("incident_count", sa.Integer, default=0),
        sa.Column("sla_breaches", sa.Integer, default=0),
        sa.Column("p95_latency_ms", sa.Float),
        sa.Column("anomaly_count", sa.Integer, default=0),
        sa.Column("metrics", sa.dialects.postgresql.JSONB, default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("table_id", "window_start", "granularity", name="uq_sla_trend_window"),
    )
    op.create_index("ix_sla_trends_table_window", "sla_trends", ["table_id", "window_start"])


def downgrade() -> None:
    tables = [
        "sla_trends", "incident_memories", "document_chunks",
        "anomaly_model_runs", "remediation_actions", "incidents",
        "dq_snapshots", "schema_versions", "pipeline_tables",
    ]
    for table in tables:
        op.drop_table(table)
    op.execute("DROP EXTENSION IF EXISTS vector")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
