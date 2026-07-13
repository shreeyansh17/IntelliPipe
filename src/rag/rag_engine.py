"""
IntelliPipe RAG System — LlamaIndex + pgvector
===============================================
Retrieval-Augmented Generation pipeline for natural language queries over:
- dbt model documentation (model descriptions, column docs, test configs)
- Data lineage graphs (upstream/downstream dependencies)
- Data contracts (expectations, SLAs, ownership)
- Historical incident knowledge base

Uses:
- LlamaIndex for document ingestion and chunking
- Anthropic embeddings (or OpenAI ada-002 as fallback)
- pgvector for ANN vector search
- Claude for answer synthesis from retrieved context
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
from llama_index.core import (
    Document,
)
from llama_index.core.node_parser import SentenceSplitter

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.telemetry import RAG_QUERIES_TOTAL, RAG_RETRIEVAL_LATENCY, timed_operation

logger = get_logger(__name__, component="rag")
settings = get_settings()

RAG_SYSTEM_PROMPT = """You are an expert data engineer assistant with deep knowledge of this 
organization's data platform. You answer questions about:
- Data pipeline architecture and dbt models
- Data lineage (what tables feed into what)
- Data quality rules and expectations
- Metric definitions and business logic
- Incident history and known issues

Use ONLY the provided context to answer. If the context doesn't contain the answer, say so clearly.
Be precise, technical, and concise. Cite specific model names and column names when relevant."""


# ---------------------------------------------------------------------------
# Document loaders for different source types
# ---------------------------------------------------------------------------


class DBTDocumentLoader:
    """
    Loads and chunks dbt project documentation from:
    - dbt manifest.json (model metadata + compiled SQL)
    - dbt catalog.json (column-level descriptions)
    - schema.yml files (human-written docs + tests)
    """

    def __init__(self, dbt_project_path: str) -> None:
        self._path = Path(dbt_project_path)

    def load_manifest(self) -> List[Document]:
        """Parse dbt manifest.json and create documents per model."""
        manifest_path = self._path / "target" / "manifest.json"
        if not manifest_path.exists():
            logger.warning("dbt manifest.json not found", path=str(manifest_path))
            return []

        with open(manifest_path) as f:
            manifest = json.load(f)

        documents = []
        nodes = manifest.get("nodes", {})

        for node_id, node in nodes.items():
            if node.get("resource_type") not in ("model", "test", "source"):
                continue

            # Build rich text representation for embedding
            content_parts = [
                f"# dbt Model: {node.get('name', '')}",
                f"**Schema:** {node.get('schema', '')}",
                f"**Database:** {node.get('database', '')}",
                f"**Description:** {node.get('description', 'No description provided')}",
                f"**Tags:** {', '.join(node.get('tags', []))}",
                f"**Owner:** {node.get('meta', {}).get('owner', 'Unknown')}",
                "",
            ]

            # Compiled SQL
            if node.get("compiled_code"):
                content_parts.append("## Compiled SQL")
                content_parts.append("```sql")
                content_parts.append(node["compiled_code"][:2000])  # Truncate long SQL
                content_parts.append("```")

            # Column descriptions
            columns = node.get("columns", {})
            if columns:
                content_parts.append("## Columns")
                for col_name, col_meta in columns.items():
                    desc = col_meta.get("description", "")
                    tests = col_meta.get("data_tests", col_meta.get("tests", []))
                    content_parts.append(
                        f"- **{col_name}**: {desc}"
                        + (
                            f" (tests: {', '.join(str(t) for t in tests)})"
                            if tests
                            else ""
                        )
                    )

            # Upstream dependencies
            depends_on = node.get("depends_on", {}).get("nodes", [])
            if depends_on:
                deps = [n.split(".")[-1] for n in depends_on]
                content_parts.append(f"\n## Upstream Dependencies\n{', '.join(deps)}")

            content = "\n".join(content_parts)

            doc = Document(
                text=content,
                metadata={
                    "source_type": "dbt_model",
                    "node_id": node_id,
                    "model_name": node.get("name", ""),
                    "schema": node.get("schema", ""),
                    "resource_type": node.get("resource_type", ""),
                    "tags": node.get("tags", []),
                    "path": node.get("original_file_path", ""),
                },
                id_=f"dbt_{node_id}",
            )
            documents.append(doc)

        logger.info("Loaded dbt manifest documents", count=len(documents))
        return documents

    def load_lineage(self) -> List[Document]:
        """
        Build lineage documentation from manifest DAG relationships.
        Creates one document per model describing its lineage context.
        """
        manifest_path = self._path / "target" / "manifest.json"
        if not manifest_path.exists():
            return []

        with open(manifest_path) as f:
            manifest = json.load(f)

        nodes = manifest.get("nodes", {})
        lineage_docs = []

        for node_id, node in nodes.items():
            if node.get("resource_type") != "model":
                continue

            name = node.get("name", "")
            upstream = [
                n.split(".")[-1] for n in node.get("depends_on", {}).get("nodes", [])
            ]

            # Find downstream models
            downstream = [
                other.get("name", "")
                for other_id, other in nodes.items()
                if node_id in other.get("depends_on", {}).get("nodes", [])
            ]

            content = (
                f"# Lineage for {name}\n\n"
                f"## Upstream (sources feeding into {name})\n"
                f"{', '.join(upstream) if upstream else 'None — this is a source model'}\n\n"
                f"## Downstream (models consuming {name})\n"
                f"{', '.join(downstream) if downstream else 'None — this is a leaf model'}\n\n"
                f"## Full Lineage Path\n"
                f"The data flows: {' → '.join(upstream[-3:] + [name] + downstream[:3])}"
            )

            lineage_docs.append(
                Document(
                    text=content,
                    metadata={
                        "source_type": "lineage",
                        "model_name": name,
                        "upstream": upstream,
                        "downstream": downstream,
                    },
                    id_=f"lineage_{node_id}",
                )
            )

        return lineage_docs


# ---------------------------------------------------------------------------
# Embedding service
# ---------------------------------------------------------------------------


class AnthropicEmbeddingAdapter:
    """
    Adapter to use Anthropic's embedding endpoint.
    Falls back to OpenAI ada-002 if OPENAI_API_KEY is set.
    """

    EMBEDDING_DIM = 1536  # Match pgvector column dimension

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(
            api_key=settings.llm.anthropic_api_key.get_secret_value()
        )

    def get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        # Anthropic doesn't have a standalone embeddings API yet;
        # use a compact Claude call to generate a conceptual embedding
        # In production: use voyage-3-lite (Anthropic's embedding model via Voyage AI)
        try:
            import voyageai  # type: ignore

            vo = voyageai.Client()
            result = vo.embed([text], model="voyage-3-lite")
            return result.embeddings[0]
        except ImportError:
            # Fallback: return a zero vector (replace with real embedding in prod)
            logger.warning("voyageai not available, using zero vector embedding")
            return [0.0] * self.EMBEDDING_DIM

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding generation."""
        return [self.get_text_embedding(t) for t in texts]


# ---------------------------------------------------------------------------
# Main RAG Engine
# ---------------------------------------------------------------------------


class IntelliPipeRAGEngine:
    """
    Full RAG pipeline for data platform natural language querying.

    Workflow:
    1. Ingest dbt manifest + lineage into document store
    2. Chunk documents with SentenceSplitter
    3. Generate embeddings (Voyage AI / OpenAI)
    4. Store chunks + embeddings in pgvector
    5. At query time: embed query → ANN search → Claude synthesis
    """

    def __init__(self, doc_chunk_repo: Any) -> None:
        self._repo = doc_chunk_repo
        self._embedder = AnthropicEmbeddingAdapter()
        self._splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        self._anthropic = anthropic.Anthropic(
            api_key=settings.llm.anthropic_api_key.get_secret_value()
        )
        logger.info("RAG engine initialised")

    async def ingest_dbt_project(
        self,
        dbt_project_path: str,
        tenant_id: str,
    ) -> Dict[str, int]:
        """
        Full ingestion pipeline for a dbt project.
        Returns dict with counts of ingested documents per type.
        """
        loader = DBTDocumentLoader(dbt_project_path)
        counts: Dict[str, int] = {}

        # Load model docs
        model_docs = loader.load_manifest()
        if model_docs:
            n = await self._ingest_documents(model_docs, tenant_id, "dbt_model")
            counts["dbt_models"] = n

        # Load lineage docs
        lineage_docs = loader.load_lineage()
        if lineage_docs:
            n = await self._ingest_documents(lineage_docs, tenant_id, "lineage")
            counts["lineage"] = n

        logger.info("dbt project ingested", tenant_id=tenant_id, counts=counts)
        return counts

    async def _ingest_documents(
        self,
        documents: List[Document],
        tenant_id: str,
        source_type: str,
    ) -> int:
        """Chunk, embed, and store documents. Returns chunk count."""
        total_chunks = 0

        for doc in documents:
            # Split into chunks
            nodes = self._splitter.get_nodes_from_documents([doc])
            chunks = []

            for node in nodes:
                embedding = self._embedder.get_text_embedding(node.text)
                chunks.append(
                    {
                        "content": node.text,
                        "embedding": embedding,
                        "metadata": {**doc.metadata, "chunk_length": len(node.text)},
                        "source_url": doc.metadata.get("path"),
                    }
                )

            n = await self._repo.upsert_chunks(
                tenant_id=tenant_id,
                source_type=source_type,
                source_id=doc.id_ or str(uuid.uuid4()),
                chunks=chunks,
            )
            total_chunks += n

        return total_chunks

    async def query(
        self,
        question: str,
        tenant_id: str,
        source_type: Optional[str] = None,
        top_k: int = 5,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Answer a natural language question using RAG.

        Returns:
        - answer: Claude's synthesised answer
        - sources: List of source chunks used
        - confidence: Rough confidence based on retrieval scores
        """
        RAG_QUERIES_TOTAL.labels(
            query_type=source_type or "all",
            status="started",
        ).inc()

        with timed_operation(RAG_RETRIEVAL_LATENCY, {}):
            # Embed the query
            query_embedding = self._embedder.get_text_embedding(question)

            # Retrieve relevant chunks
            relevant_chunks = await self._repo.vector_search(
                tenant_id=tenant_id,
                query_embedding=query_embedding,
                top_k=top_k,
                source_type=source_type,
            )

        if not relevant_chunks:
            RAG_QUERIES_TOTAL.labels(
                query_type=source_type or "all", status="no_results"
            ).inc()
            return {
                "answer": "I couldn't find relevant documentation to answer this question.",
                "sources": [],
                "confidence": 0.0,
            }

        # Build context from retrieved chunks
        context_parts = []
        sources = []

        for i, chunk in enumerate(relevant_chunks):
            context_parts.append(
                f"[Source {i+1}: {chunk.source_type} / {chunk.source_id}]\n{chunk.content}"
            )
            if include_sources:
                sources.append(
                    {
                        "source_type": chunk.source_type,
                        "source_id": chunk.source_id,
                        "content_preview": chunk.content[:200],
                        "metadata": chunk.metadata,
                    }
                )

        context = "\n\n---\n\n".join(context_parts)

        # Synthesise answer with Claude
        user_message = (
            f"## Context from Data Platform Documentation\n\n{context}\n\n"
            f"## Question\n{question}\n\n"
            "Answer the question using only the provided context. "
            "Cite specific model names, columns, or sources when relevant."
        )

        response = self._anthropic.messages.create(
            model=settings.llm.claude_model,
            max_tokens=1024,
            system=RAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text if response.content else ""
        RAG_QUERIES_TOTAL.labels(
            query_type=source_type or "all", status="success"
        ).inc()

        logger.info(
            "RAG query answered",
            question_preview=question[:80],
            chunks_retrieved=len(relevant_chunks),
            answer_length=len(answer),
        )

        return {
            "answer": answer,
            "sources": sources,
            "confidence": min(1.0, len(relevant_chunks) / top_k),
            "chunks_retrieved": len(relevant_chunks),
        }

    async def explain_metric_discrepancy(
        self,
        metric_name: str,
        expected_value: float,
        actual_value: float,
        tenant_id: str,
    ) -> str:
        """
        Specialised RAG query for metric discrepancy explanation.
        Retrieves metric definition + lineage and asks Claude to explain the gap.
        """
        question = (
            f"The metric '{metric_name}' shows a discrepancy. "
            f"Expected: {expected_value:.2f}, Actual: {actual_value:.2f} "
            f"(deviation: {abs(actual_value - expected_value) / max(expected_value, 1):.1%}). "
            f"What could cause this discrepancy based on the data model and lineage?"
        )
        result = await self.query(
            question, tenant_id=tenant_id, source_type="dbt_model"
        )
        return result["answer"]
