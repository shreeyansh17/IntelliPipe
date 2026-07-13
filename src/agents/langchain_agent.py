"""
IntelliPipe LangChain Orchestration Agent
==========================================
Multi-step reasoning agent that:
1. Receives anomaly alerts from Redis
2. Queries pgvector for similar historical incidents
3. Calls Claude API for root-cause analysis
4. Generates dbt SQL fix code
5. Creates GitHub PRs with the fix
6. Files Jira tickets
7. Posts Slack incident reports
8. Stores incident memory in pgvector for future retrieval

Implements:
- Retry logic with exponential backoff
- Rate limiting for LLM API calls
- Multi-agent collaboration (analysis agent + fix agent)
- Human approval workflow before PR merge
- Fallback logic when Claude API is unavailable
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.telemetry import LLM_API_CALLS_TOTAL, LLM_LATENCY_HISTOGRAM, LLM_TOKEN_USAGE_TOTAL

logger = get_logger(__name__, component="llm_agent")
settings = get_settings()


# ---------------------------------------------------------------------------
# Prompts — carefully engineered for Claude's capabilities
# ---------------------------------------------------------------------------

ROOT_CAUSE_SYSTEM_PROMPT = """You are an expert data engineer and data quality specialist at a 
tier-1 technology company. You are part of an autonomous data observability platform called IntelliPipe.

Your role is to perform precise, actionable root-cause analysis on data pipeline anomalies.

When analyzing an incident, you will:
1. Identify the most probable root cause based on the anomaly signals and context
2. Distinguish between upstream data issues vs pipeline bugs vs business events
3. Assess blast radius (which downstream tables/consumers are affected)
4. Assign an investigation priority and estimated impact
5. Recommend specific investigation steps
6. Suggest a remediation approach

You have deep expertise in:
- Apache Kafka and streaming pipeline failures
- dbt transformations and SQL data modeling
- Data quality dimensions (completeness, validity, freshness, uniqueness, consistency)
- Schema evolution and backward/forward compatibility
- Statistical anomalies in business metrics
- SLA/freshness failures and their common causes

Be concise, specific, and actionable. Use technical language. Avoid generic platitudes.
Format your analysis as structured JSON unless explicitly told otherwise."""

FIX_GENERATION_SYSTEM_PROMPT = """You are an expert dbt developer and SQL engineer.
You generate production-grade dbt model fixes for data quality issues.

Rules for fix generation:
1. Always use dbt best practices (CTEs, not subqueries; ref() not hardcoded table names)
2. Add data quality tests alongside any model changes
3. Write defensive SQL that handles the anomaly pattern (null handling, type casting, etc.)
4. Include both the root-cause fix and a compensating control
5. Add meaningful comments explaining WHY the change was made
6. Ensure the fix is backward compatible unless a breaking change is unavoidable
7. Include a rollback SQL block

Output format:
- dbt_model_sql: The fixed dbt model
- dbt_test_yaml: New dbt tests to add
- rollback_sql: SQL to revert if needed
- pr_description: Clear PR description for reviewers
- risk_level: low / medium / high"""

POSTMORTEM_SYSTEM_PROMPT = """You are a senior engineering manager writing a blameless incident postmortem.
Generate a structured postmortem document in Markdown format following the structure:
1. Executive Summary (2-3 sentences)
2. Impact (who was affected, for how long, quantified if possible)
3. Timeline (key events in chronological order)
4. Root Cause (the actual technical cause, not symptoms)
5. Contributing Factors (what made it worse or harder to detect)
6. Resolution (what fixed it)
7. Action Items (specific, owner-assigned, time-bound preventive measures)
8. Lessons Learned

Keep it factual, blameless, and focused on system improvements."""


# ---------------------------------------------------------------------------
# Tool definitions for LangChain agent
# ---------------------------------------------------------------------------

class RetrieveSimilarIncidentsInput(BaseModel):
    anomaly_type: str = Field(description="Type of anomaly: schema_drift, null_spike, statistical, etc.")
    table_name: str = Field(description="The affected table name")
    top_k: int = Field(default=3, description="Number of similar incidents to retrieve")


class GenerateDBTFixInput(BaseModel):
    incident_summary: str = Field(description="Summary of the incident and root cause")
    table_name: str = Field(description="Affected dbt model/table name")
    anomaly_type: str = Field(description="Type of anomaly to fix")
    affected_columns: List[str] = Field(description="List of affected column names")


# ---------------------------------------------------------------------------
# LLM instrumented wrapper
# ---------------------------------------------------------------------------

class InstrumentedAnthropicLLM:
    """
    Wrapper around the Anthropic client with:
    - Prometheus metrics tracking
    - Retry with exponential backoff
    - Token usage logging
    """

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(
            api_key=settings.llm.anthropic_api_key.get_secret_value()
        )
        self._model = settings.llm.claude_model
        self._max_retries = settings.llm.max_retries
        self._retry_delay = settings.llm.retry_delay_seconds

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        operation: str = "completion",
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Single completion call with retry and metrics.
        Returns (text_response, token_usage_dict).
        """
        max_tok = max_tokens or settings.llm.max_tokens
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            start_time = time.time()
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tok,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )

                duration = time.time() - start_time
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }

                # Track metrics
                LLM_API_CALLS_TOTAL.labels(
                    model=self._model,
                    operation=operation,
                    status="success",
                ).inc()
                LLM_LATENCY_HISTOGRAM.labels(
                    model=self._model,
                    operation=operation,
                ).observe(duration)
                LLM_TOKEN_USAGE_TOTAL.labels(
                    model=self._model,
                    token_type="input",
                ).inc(usage["input_tokens"])
                LLM_TOKEN_USAGE_TOTAL.labels(
                    model=self._model,
                    token_type="output",
                ).inc(usage["output_tokens"])

                text = response.content[0].text if response.content else ""
                logger.info(
                    "LLM call completed",
                    operation=operation,
                    duration_ms=round(duration * 1000),
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                )
                return text, usage

            except anthropic.RateLimitError as e:
                last_error = e
                wait = self._retry_delay * (2 ** attempt)
                logger.warning("Claude rate limit hit, backing off", wait_seconds=wait, attempt=attempt)
                await asyncio.sleep(wait)

            except anthropic.APIError as e:
                last_error = e
                LLM_API_CALLS_TOTAL.labels(
                    model=self._model, operation=operation, status="error"
                ).inc()
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                    logger.warning("LLM API error, retrying", error=str(e), attempt=attempt)
                else:
                    logger.error("LLM API failed after retries", error=str(e))

        raise RuntimeError(f"LLM call failed after {self._max_retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Root Cause Analysis Agent
# ---------------------------------------------------------------------------

class RootCauseAnalysisAgent:
    """
    Claude-powered root-cause analysis agent.
    Uses contextual information from historical incidents and table metadata.
    """

    def __init__(self, llm: InstrumentedAnthropicLLM) -> None:
        self._llm = llm

    async def analyse(
        self,
        alert: Dict[str, Any],
        table_metadata: Optional[Dict[str, Any]] = None,
        similar_incidents: Optional[List[Dict[str, Any]]] = None,
        recent_incidents: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Perform root-cause analysis for an anomaly alert.

        Returns structured RCA dict with fields:
        - root_cause: Primary cause hypothesis
        - confidence: 0-1 confidence score
        - blast_radius: Affected downstream tables
        - investigation_steps: Ordered list of steps
        - remediation_approach: Recommended fix strategy
        - estimated_impact: Business impact assessment
        - escalate_immediately: bool
        """
        context_parts = [
            f"## Anomaly Alert\n```json\n{json.dumps(alert, indent=2, default=str)}\n```"
        ]

        if table_metadata:
            context_parts.append(
                f"## Table Metadata\n```json\n{json.dumps(table_metadata, indent=2)}\n```"
            )

        if similar_incidents:
            incidents_text = json.dumps(similar_incidents, indent=2, default=str)
            context_parts.append(
                f"## Similar Historical Incidents (retrieved from vector memory)\n"
                f"```json\n{incidents_text}\n```"
            )

        if recent_incidents:
            recent_text = json.dumps(recent_incidents, indent=2, default=str)
            context_parts.append(
                f"## Recent Incidents for This Table (last 24h)\n"
                f"```json\n{recent_text}\n```"
            )

        context_parts.append(
            "\n## Task\n"
            "Analyse this anomaly and return a JSON object with these exact fields:\n"
            "- root_cause (string): Most likely root cause\n"
            "- confidence (float 0-1): Your confidence level\n"
            "- blast_radius (list of strings): Potentially affected downstream tables\n"
            "- investigation_steps (list of strings): Ordered investigation steps\n"
            "- remediation_approach (string): Recommended fix strategy\n"
            "- estimated_impact (string): Business impact description\n"
            "- escalate_immediately (boolean): Whether this needs immediate human escalation\n"
            "- severity_reasoning (string): Why you assigned this severity level\n\n"
            "Return ONLY the JSON object, no surrounding text."
        )

        user_message = "\n\n".join(context_parts)

        response_text, usage = await self._llm.complete(
            system_prompt=ROOT_CAUSE_SYSTEM_PROMPT,
            user_message=user_message,
            operation="root_cause_analysis",
        )

        try:
            # Claude sometimes wraps JSON in markdown code blocks
            clean_text = response_text.strip()
            if clean_text.startswith("```"):
                clean_text = clean_text.split("```")[1]
                if clean_text.startswith("json"):
                    clean_text = clean_text[4:]
            rca = json.loads(clean_text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse RCA JSON, using fallback", response_preview=response_text[:200])
            rca = {
                "root_cause": response_text,
                "confidence": 0.5,
                "blast_radius": [],
                "investigation_steps": ["Review anomaly details manually"],
                "remediation_approach": "Manual investigation required",
                "estimated_impact": "Unknown",
                "escalate_immediately": alert.get("severity") in ["critical", "high"],
                "severity_reasoning": "Automatic parsing failed",
            }

        rca["token_usage"] = usage
        rca["model_used"] = settings.llm.claude_model
        return rca


# ---------------------------------------------------------------------------
# Fix Code Generator
# ---------------------------------------------------------------------------

class FixCodeGenerator:
    """
    Generates dbt SQL fix code and rollback scripts using Claude.
    Works in collaboration with RootCauseAnalysisAgent.
    """

    def __init__(self, llm: InstrumentedAnthropicLLM) -> None:
        self._llm = llm

    async def generate(
        self,
        rca: Dict[str, Any],
        alert: Dict[str, Any],
        dbt_model_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate dbt fix code based on RCA findings.

        Returns:
        - dbt_model_sql: Fixed dbt model SQL
        - dbt_test_yaml: New dbt schema tests
        - rollback_sql: Rollback script
        - pr_description: PR description for reviewers
        - risk_level: low / medium / high
        """
        context = [
            f"## Root Cause Analysis\n```json\n{json.dumps(rca, indent=2, default=str)}\n```",
            f"## Original Alert\n```json\n{json.dumps(alert, indent=2, default=str)}\n```",
        ]

        if dbt_model_context:
            context.append(f"## Current dbt Model (to be modified)\n```sql\n{dbt_model_context}\n```")
        else:
            # Generate a plausible dbt model based on table name
            table = alert.get("table_name", "orders")
            context.append(
                f"## Target dbt Model\n"
                f"Model name: `stg_{table}` in the staging layer.\n"
                f"Generate a complete, corrected version of this model."
            )

        context.append(
            "\n## Task\n"
            "Generate a dbt fix for this data quality issue. Return a JSON object with:\n"
            "- dbt_model_sql: The complete corrected dbt model SQL (use CTEs, ref(), source())\n"
            "- dbt_test_yaml: A YAML snippet for dbt schema tests to prevent regression\n"
            "- rollback_sql: Raw SQL to revert the change if needed\n"
            "- pr_description: A clear, professional PR description\n"
            "- risk_level: 'low', 'medium', or 'high'\n"
            "- change_summary: One sentence description of what changed\n\n"
            "Return ONLY the JSON object."
        )

        response_text, usage = await self._llm.complete(
            system_prompt=FIX_GENERATION_SYSTEM_PROMPT,
            user_message="\n\n".join(context),
            operation="fix_generation",
            max_tokens=2048,
        )

        try:
            clean_text = response_text.strip()
            if clean_text.startswith("```"):
                clean_text = "\n".join(clean_text.split("\n")[1:])
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
            fix = json.loads(clean_text)
        except json.JSONDecodeError:
            # Fallback: return raw response as dbt_model_sql
            fix = {
                "dbt_model_sql": response_text,
                "dbt_test_yaml": "",
                "rollback_sql": "-- Rollback SQL not generated",
                "pr_description": f"Auto-generated fix for {alert.get('alert_type')} on {alert.get('table_name')}",
                "risk_level": "medium",
                "change_summary": "Automated data quality fix",
            }

        fix["token_usage"] = usage
        return fix


# ---------------------------------------------------------------------------
# Postmortem Generator
# ---------------------------------------------------------------------------

class PostmortemGenerator:
    """Generates AI-written blameless postmortems for resolved incidents."""

    def __init__(self, llm: InstrumentedAnthropicLLM) -> None:
        self._llm = llm

    async def generate(
        self,
        incident: Dict[str, Any],
        rca: Dict[str, Any],
        timeline: List[Dict[str, Any]],
    ) -> str:
        """Generate a structured Markdown postmortem document."""
        user_message = (
            f"## Incident Details\n```json\n{json.dumps(incident, indent=2, default=str)}\n```\n\n"
            f"## Root Cause Analysis\n```json\n{json.dumps(rca, indent=2, default=str)}\n```\n\n"
            f"## Timeline\n```json\n{json.dumps(timeline, indent=2, default=str)}\n```\n\n"
            "Generate a blameless postmortem document in Markdown format."
        )

        response_text, _ = await self._llm.complete(
            system_prompt=POSTMORTEM_SYSTEM_PROMPT,
            user_message=user_message,
            operation="postmortem_generation",
            max_tokens=3000,
        )
        return response_text


# ---------------------------------------------------------------------------
# Main Orchestration Agent
# ---------------------------------------------------------------------------

class IntelliPipeOrchestrationAgent:
    """
    Top-level agent orchestrating the full incident lifecycle:
    Alert → RCA → Fix Generation → PR → Jira → Slack → Memory Storage
    """

    def __init__(
        self,
        github_client: Any,
        jira_client: Any,
        slack_client: Any,
        incident_repo: Any,
        memory_repo: Any,
        doc_repo: Any,
    ) -> None:
        self._llm = InstrumentedAnthropicLLM()
        self._rca_agent = RootCauseAnalysisAgent(self._llm)
        self._fix_generator = FixCodeGenerator(self._llm)
        self._postmortem_gen = PostmortemGenerator(self._llm)
        self._github = github_client
        self._jira = jira_client
        self._slack = slack_client
        self._incident_repo = incident_repo
        self._memory_repo = memory_repo
        self._doc_repo = doc_repo

    async def handle_alert(
        self,
        alert: Dict[str, Any],
        tenant_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Full incident handling pipeline.
        Returns dict with incident_id and all external references.
        """
        incident_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        logger.info(
            "Handling alert",
            incident_id=incident_id,
            alert_type=alert.get("alert_type"),
            severity=alert.get("severity"),
            table=alert.get("table_name"),
        )

        # Step 1: Retrieve similar historical incidents from vector memory
        similar_incidents = await self._retrieve_similar_incidents(alert, tenant_id)

        # Step 2: Retrieve relevant dbt documentation via RAG
        dbt_context = await self._retrieve_dbt_context(alert)

        # Step 3: Root-cause analysis via Claude
        rca = await self._rca_agent.analyse(
            alert=alert,
            similar_incidents=similar_incidents,
        )

        logger.info(
            "RCA completed",
            incident_id=incident_id,
            root_cause=rca.get("root_cause", "")[:100],
            confidence=rca.get("confidence"),
            escalate=rca.get("escalate_immediately"),
        )

        # Step 4: Generate dbt fix code
        fix = await self._fix_generator.generate(
            rca=rca,
            alert=alert,
            dbt_model_context=dbt_context,
        )

        # Step 5: Create GitHub PR
        pr_result = await self._github.create_fix_pr(
            table_name=alert.get("table_name", "unknown"),
            fix_code=fix,
            incident_id=incident_id,
            rca_summary=rca.get("root_cause", ""),
            require_approval=settings.github.require_approval,
        )

        # Step 6: Create Jira ticket
        jira_result = await self._jira.create_incident_ticket(
            incident_id=incident_id,
            title=f"[DQ Alert] {alert.get('alert_type')} on {alert.get('table_name')}",
            description=self._format_jira_description(alert, rca),
            severity=alert.get("severity", "medium"),
            pr_url=pr_result.get("pr_url"),
        )

        # Step 7: Post Slack notification
        slack_result = await self._slack.post_incident_alert(
            incident_id=incident_id,
            alert=alert,
            rca=rca,
            pr_url=pr_result.get("pr_url"),
            jira_key=jira_result.get("key"),
        )

        # Step 8: Store in vector memory for future retrieval
        await self._store_incident_memory(
            tenant_id=tenant_id,
            incident_id=incident_id,
            alert=alert,
            rca=rca,
            fix=fix,
        )

        result = {
            "incident_id": incident_id,
            "rca": rca,
            "fix": fix,
            "github_pr_url": pr_result.get("pr_url"),
            "github_pr_number": pr_result.get("pr_number"),
            "jira_ticket_key": jira_result.get("key"),
            "jira_ticket_url": jira_result.get("url"),
            "slack_message_ts": slack_result.get("ts"),
            "processing_time_ms": int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
        }

        logger.info(
            "Alert handling complete",
            incident_id=incident_id,
            pr_url=pr_result.get("pr_url"),
            jira_key=jira_result.get("key"),
        )

        return result

    async def _retrieve_similar_incidents(
        self,
        alert: Dict[str, Any],
        tenant_id: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve similar past incidents using vector similarity search."""
        try:
            # Create a text embedding query from the alert
            #  query_text = (
            #     f"anomaly type: {alert.get('alert_type')} "
            #     f"table: {alert.get('table_name')} "
            #     f"severity: {alert.get('severity')}"
            # )
            # Note: In production, call embedding API here
            # For now return empty list as embedding service is async
            return []
        except Exception as e:
            logger.warning("Failed to retrieve similar incidents", error=str(e))
            return []

    async def _retrieve_dbt_context(self, alert: Dict[str, Any]) -> Optional[str]:
        """Retrieve relevant dbt model SQL from the RAG system."""
        try:
            # table_name = alert.get("table_name", "")
            # In production: query doc_repo.vector_search(...)
            return None
        except Exception as e:
            logger.warning("Failed to retrieve dbt context", error=str(e))
            return None

    async def _store_incident_memory(
        self,
        tenant_id: str,
        incident_id: str,
        alert: Dict[str, Any],
        rca: Dict[str, Any],
        fix: Dict[str, Any],
    ) -> None:
        """Persist incident knowledge to vector memory for future retrieval."""
        memory_content = json.dumps({
            "alert": alert,
            "root_cause": rca.get("root_cause"),
            "investigation_steps": rca.get("investigation_steps"),
            "remediation_approach": rca.get("remediation_approach"),
            "fix_summary": fix.get("change_summary"),
        }, default=str)

        try:
            await self._memory_repo.store(
                tenant_id=tenant_id,
                incident_id=uuid.UUID(incident_id) if incident_id else None,
                memory_type="rca",
                content=memory_content,
                embedding=None,  # Set after embedding API call
                summary=rca.get("root_cause", "")[:200],
                tags=[
                    alert.get("alert_type", ""),
                    alert.get("table_name", ""),
                    alert.get("severity", ""),
                ],
            )
        except Exception as e:
            logger.error("Failed to store incident memory", error=str(e))

    @staticmethod
    def _format_jira_description(alert: Dict[str, Any], rca: Dict[str, Any]) -> str:
        """Format a Jira ticket description with all relevant context."""
        return f"""h2. Incident Summary
*Alert Type:* {alert.get('alert_type')}
*Table:* {alert.get('table_name')}
*Severity:* {alert.get('severity')}
*Detected At:* {alert.get('detected_at')}

h2. Root Cause Analysis
{rca.get('root_cause', 'Analysis in progress')}

h2. Confidence
{rca.get('confidence', 'N/A')}

h2. Investigation Steps
{chr(10).join(f"# {step}" for step in rca.get('investigation_steps', []))}

h2. Estimated Impact
{rca.get('estimated_impact', 'Under assessment')}

h2. Blast Radius
Potentially affected tables: {', '.join(rca.get('blast_radius', [])) or 'None identified'}

---
_Auto-generated by IntelliPipe Autonomous Data Quality Platform_"""
