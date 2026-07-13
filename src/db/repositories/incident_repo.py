"""
IntelliPipe Repository Layer
==============================
Repository pattern implementation for clean data-access abstraction.
All DB queries are encapsulated here — service layer never uses SQLAlchemy directly.
Uses async SQLAlchemy 2.x with proper connection pooling.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger
from src.db.models import (
    AnomalyType,
    DQSnapshot,
    DocumentChunk,
    Incident,
    IncidentMemory,
    IncidentStatus,
    SeverityLevel,
)

logger = get_logger(__name__, component="repository")


class BaseRepository:
    """Base class providing session injection."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session


# ---------------------------------------------------------------------------
# Incident Repository
# ---------------------------------------------------------------------------

class IncidentRepository(BaseRepository):
    """All database operations for Incident domain objects."""

    async def create(
        self,
        tenant_id: str,
        table_id: uuid.UUID,
        title: str,
        anomaly_type: AnomalyType,
        severity: SeverityLevel,
        description: Optional[str] = None,
        anomaly_score: Optional[float] = None,
        affected_columns: Optional[List[str]] = None,
        affected_row_count: Optional[int] = None,
        detection_metadata: Optional[Dict[str, Any]] = None,
    ) -> Incident:
        """Create and persist a new incident."""
        incident = Incident(
            tenant_id=tenant_id,
            table_id=table_id,
            title=title,
            description=description,
            anomaly_type=anomaly_type,
            severity=severity,
            anomaly_score=anomaly_score,
            affected_columns=affected_columns or [],
            affected_row_count=affected_row_count,
            detection_metadata=detection_metadata or {},
        )
        self._session.add(incident)
        await self._session.flush()
        logger.info(
            "Incident created",
            incident_id=str(incident.id),
            severity=severity.value,
            anomaly_type=anomaly_type.value,
        )
        return incident

    async def get_by_id(self, incident_id: uuid.UUID) -> Optional[Incident]:
        result = await self._session.execute(
            select(Incident).where(Incident.id == incident_id)
        )
        return result.scalar_one_or_none()

    async def list_open(
        self,
        tenant_id: str,
        limit: int = 50,
        offset: int = 0,
        severity: Optional[SeverityLevel] = None,
    ) -> Tuple[List[Incident], int]:
        """List open incidents with pagination and optional severity filter."""
        conditions = [
            Incident.tenant_id == tenant_id,
            Incident.status.in_([IncidentStatus.OPEN, IncidentStatus.ACKNOWLEDGED]),
        ]
        if severity:
            conditions.append(Incident.severity == severity)

        count_result = await self._session.execute(
            select(func.count(Incident.id)).where(and_(*conditions))
        )
        total = count_result.scalar_one()

        result = await self._session.execute(
            select(Incident)
            .where(and_(*conditions))
            .order_by(desc(Incident.created_at))
            .limit(limit)
            .offset(offset)
        )
        incidents = list(result.scalars().all())
        return incidents, total

    async def update_rca(
        self,
        incident_id: uuid.UUID,
        root_cause_analysis: str,
        fix_code: Optional[str],
        llm_model_used: str,
    ) -> Optional[Incident]:
        """Attach LLM-generated RCA and fix code to an incident."""
        await self._session.execute(
            update(Incident)
            .where(Incident.id == incident_id)
            .values(
                root_cause_analysis=root_cause_analysis,
                fix_code=fix_code,
                llm_model_used=llm_model_used,
                status=IncidentStatus.IN_PROGRESS,
            )
        )
        return await self.get_by_id(incident_id)

    async def update_external_refs(
        self,
        incident_id: uuid.UUID,
        github_pr_url: Optional[str] = None,
        github_pr_number: Optional[int] = None,
        jira_ticket_key: Optional[str] = None,
        jira_ticket_url: Optional[str] = None,
        slack_message_ts: Optional[str] = None,
    ) -> None:
        """Update external system references (GitHub, Jira, Slack)."""
        values: Dict[str, Any] = {}
        if github_pr_url:
            values["github_pr_url"] = github_pr_url
        if github_pr_number:
            values["github_pr_number"] = github_pr_number
        if jira_ticket_key:
            values["jira_ticket_key"] = jira_ticket_key
        if jira_ticket_url:
            values["jira_ticket_url"] = jira_ticket_url
        if slack_message_ts:
            values["slack_message_ts"] = slack_message_ts
        if values:
            await self._session.execute(
                update(Incident).where(Incident.id == incident_id).values(**values)
            )

    async def resolve(
        self,
        incident_id: uuid.UUID,
        resolved_by: str,
        resolution_notes: Optional[str] = None,
    ) -> None:
        """Mark an incident as resolved."""
        await self._session.execute(
            update(Incident)
            .where(Incident.id == incident_id)
            .values(
                status=IncidentStatus.RESOLVED,
                resolved_at=func.now(),
                resolved_by=resolved_by,
                resolution_notes=resolution_notes,
            )
        )

    async def get_recent_by_table(
        self,
        table_id: uuid.UUID,
        hours: int = 24,
        limit: int = 10,
    ) -> List[Incident]:
        """Fetch recent incidents for a specific table (for context in RCA)."""
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        result = await self._session.execute(
            select(Incident)
            .where(
                and_(
                    Incident.table_id == table_id,
                    Incident.created_at >= since,
                )
            )
            .order_by(desc(Incident.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# DQ Snapshot Repository
# ---------------------------------------------------------------------------

class DQSnapshotRepository(BaseRepository):
    """Repository for data quality scorecard snapshots."""

    async def create(
        self,
        table_id: uuid.UUID,
        tenant_id: str,
        overall_score: float,
        completeness_score: float = 100.0,
        validity_score: float = 100.0,
        uniqueness_score: float = 100.0,
        freshness_score: float = 100.0,
        consistency_score: float = 100.0,
        referential_score: float = 100.0,
        row_count: int = 0,
        failed_checks: int = 0,
        total_checks: int = 0,
        ge_results: Optional[Dict[str, Any]] = None,
    ) -> DQSnapshot:
        snapshot = DQSnapshot(
            table_id=table_id,
            tenant_id=tenant_id,
            overall_score=overall_score,
            completeness_score=completeness_score,
            validity_score=validity_score,
            uniqueness_score=uniqueness_score,
            freshness_score=freshness_score,
            consistency_score=consistency_score,
            referential_score=referential_score,
            row_count=row_count,
            failed_checks=failed_checks,
            total_checks=total_checks,
            ge_results=ge_results or {},
        )
        self._session.add(snapshot)
        await self._session.flush()
        return snapshot

    async def get_latest(
        self,
        table_id: uuid.UUID,
        tenant_id: str,
    ) -> Optional[DQSnapshot]:
        result = await self._session.execute(
            select(DQSnapshot)
            .where(
                and_(
                    DQSnapshot.table_id == table_id,
                    DQSnapshot.tenant_id == tenant_id,
                )
            )
            .order_by(desc(DQSnapshot.captured_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_trend(
        self,
        table_id: uuid.UUID,
        tenant_id: str,
        hours: int = 168,  # 7 days
    ) -> List[DQSnapshot]:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        result = await self._session.execute(
            select(DQSnapshot)
            .where(
                and_(
                    DQSnapshot.table_id == table_id,
                    DQSnapshot.tenant_id == tenant_id,
                    DQSnapshot.captured_at >= since,
                )
            )
            .order_by(DQSnapshot.captured_at)
        )
        return list(result.scalars().all())

    async def get_all_latest(self, tenant_id: str) -> List[DQSnapshot]:
        """Get the latest snapshot for every table in the tenant."""
        subq = (
            select(
                DQSnapshot.table_id,
                func.max(DQSnapshot.captured_at).label("max_ts"),
            )
            .where(DQSnapshot.tenant_id == tenant_id)
            .group_by(DQSnapshot.table_id)
            .subquery()
        )
        result = await self._session.execute(
            select(DQSnapshot).join(
                subq,
                and_(
                    DQSnapshot.table_id == subq.c.table_id,
                    DQSnapshot.captured_at == subq.c.max_ts,
                ),
            )
        )
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Document / RAG Repository
# ---------------------------------------------------------------------------

class DocumentChunkRepository(BaseRepository):
    """Repository for RAG document chunks with vector similarity search."""

    async def upsert_chunks(
        self,
        tenant_id: str,
        source_type: str,
        source_id: str,
        chunks: List[Dict[str, Any]],
    ) -> int:
        """
        Upsert document chunks — delete existing for source, re-insert.
        Returns number of chunks stored.
        """
        from sqlalchemy import delete
        await self._session.execute(
            delete(DocumentChunk).where(
                and_(
                    DocumentChunk.tenant_id == tenant_id,
                    DocumentChunk.source_id == source_id,
                )
            )
        )

        chunk_models = [
            DocumentChunk(
                tenant_id=tenant_id,
                source_type=source_type,
                source_id=source_id,
                source_url=chunk.get("source_url"),
                content=chunk["content"],
                chunk_index=i,
                embedding=chunk.get("embedding"),
                metadata=chunk.get("metadata", {}),
            )
            for i, chunk in enumerate(chunks)
        ]
        self._session.add_all(chunk_models)
        await self._session.flush()
        return len(chunk_models)

    async def vector_search(
        self,
        tenant_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        source_type: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Approximate nearest-neighbour search using pgvector cosine similarity.
        Requires a vector index (IVFFlat or HNSW) on the embedding column.
        """
        conditions = [DocumentChunk.tenant_id == tenant_id]
        if source_type:
            conditions.append(DocumentChunk.source_type == source_type)

        # pgvector cosine distance operator: <=>
        result = await self._session.execute(
            select(DocumentChunk)
            .where(and_(*conditions))
            .order_by(
                DocumentChunk.embedding.cosine_distance(query_embedding)
            )
            .limit(top_k)
        )
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Incident Memory Repository (long-term agent memory)
# ---------------------------------------------------------------------------

class IncidentMemoryRepository(BaseRepository):
    """Repository for LLM agent long-term vector memory."""

    async def store(
        self,
        tenant_id: str,
        incident_id: Optional[uuid.UUID],
        memory_type: str,
        content: str,
        embedding: Optional[List[float]],
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IncidentMemory:
        memory = IncidentMemory(
            tenant_id=tenant_id,
            incident_id=incident_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            summary=summary,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._session.add(memory)
        await self._session.flush()
        return memory

    async def search_similar(
        self,
        tenant_id: str,
        query_embedding: List[float],
        memory_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[IncidentMemory]:
        """Find similar past incidents/solutions via vector similarity."""
        conditions = [IncidentMemory.tenant_id == tenant_id]
        if memory_type:
            conditions.append(IncidentMemory.memory_type == memory_type)

        result = await self._session.execute(
            select(IncidentMemory)
            .where(and_(*conditions))
            .order_by(IncidentMemory.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        return list(result.scalars().all())
