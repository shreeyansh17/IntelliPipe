"""
IntelliPipe Slack & Jira Integrations
=======================================
Slack: Rich Block Kit incident notifications with action buttons
Jira: Automated ticket creation with severity routing and SLA tracking
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__, component="integrations")
settings = get_settings()


# ---------------------------------------------------------------------------
# Slack Client
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {
    "critical": "#FF0000",
    "high": "#FF6600",
    "medium": "#FFD700",
    "low": "#36A64F",
    "info": "#439FE0",
}

SEVERITY_EMOJI = {
    "critical": ":rotating_light:",
    "high": ":warning:",
    "medium": ":large_yellow_circle:",
    "low": ":large_green_circle:",
    "info": ":information_source:",
}


class SlackIncidentNotifier:
    """
    Sends rich Block Kit messages to Slack with:
    - Incident summary + severity colour coding
    - Root cause summary
    - GitHub PR and Jira ticket links
    - Action buttons (Acknowledge / Resolve / View Dashboard)
    """

    def __init__(self) -> None:
        self._cfg = settings.slack
        self._client = AsyncWebClient(
            token=self._cfg.bot_token.get_secret_value()
        )

    async def post_incident_alert(
        self,
        incident_id: str,
        alert: Dict[str, Any],
        rca: Dict[str, Any],
        pr_url: Optional[str] = None,
        jira_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Post a rich incident notification to the appropriate Slack channel."""
        severity = alert.get("severity", "medium")
        channel = (
            self._cfg.incident_channel
            if severity in ("critical", "high")
            else self._cfg.low_severity_channel
        )

        blocks = self._build_incident_blocks(
            incident_id=incident_id,
            alert=alert,
            rca=rca,
            pr_url=pr_url,
            jira_key=jira_key,
        )

        try:
            response = await self._client.chat_postMessage(
                channel=channel,
                text=f"{SEVERITY_EMOJI.get(severity, ':bell:')} Data Quality Incident: {alert.get('table_name')}",
                blocks=blocks,
                attachments=[
                    {
                        "color": SEVERITY_COLORS.get(severity, "#888888"),
                        "fallback": f"DQ Incident {incident_id[:8]} on {alert.get('table_name')}",
                    }
                ],
            )
            ts = response.get("ts", "")
            logger.info("Slack alert posted", channel=channel, ts=ts, incident_id=incident_id)
            return {"ts": ts, "channel": channel}

        except SlackApiError as e:
            logger.error("Slack API error", error=str(e), channel=channel)
            return {"ts": "", "channel": channel}

    async def post_resolution_notice(
        self,
        channel_ts: str,
        channel: str,
        incident_id: str,
        resolution_summary: str,
    ) -> None:
        """Thread a resolution message to the original incident alert."""
        try:
            await self._client.chat_postMessage(
                channel=channel,
                thread_ts=channel_ts,
                text=(
                    f":white_check_mark: *Incident `{incident_id[:8]}` Resolved*\n"
                    f"{resolution_summary}"
                ),
            )
        except SlackApiError as e:
            logger.error("Failed to post resolution", error=str(e))

    def _build_incident_blocks(
        self,
        incident_id: str,
        alert: Dict[str, Any],
        rca: Dict[str, Any],
        pr_url: Optional[str],
        jira_key: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Construct Slack Block Kit payload."""
        severity = alert.get("severity", "medium")
        emoji = SEVERITY_EMOJI.get(severity, ":bell:")
        short_id = incident_id[:8]

        root_cause = rca.get("root_cause", "Analysis in progress...")[:300]
        blast_radius = ", ".join(rca.get("blast_radius", [])) or "None identified"

        blocks: List[Dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Data Quality Incident — {alert.get('table_name', 'Unknown Table')}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Incident ID*\n`{short_id}`"},
                    {"type": "mrkdwn", "text": f"*Severity*\n`{severity.upper()}`"},
                    {"type": "mrkdwn", "text": f"*Alert Type*\n`{alert.get('alert_type', 'unknown')}`"},
                    {"type": "mrkdwn", "text": f"*Table*\n`{alert.get('table_name', 'unknown')}`"},
                    {"type": "mrkdwn", "text": f"*Detected*\n`{alert.get('detected_at', 'now')}`"},
                    {"type": "mrkdwn", "text": f"*Tenant*\n`{alert.get('tenant_id', 'default')}`"},
                ],
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*:mag: Root Cause Analysis*\n{root_cause}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Confidence*\n`{rca.get('confidence', 'N/A')}`",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Blast Radius*\n{blast_radius}",
                    },
                ],
            },
        ]

        # Action buttons
        action_elements: List[Dict[str, Any]] = []

        if pr_url:
            action_elements.append({
                "type": "button",
                "text": {"type": "plain_text", "text": ":github: View Fix PR"},
                "url": pr_url,
                "style": "primary",
            })

        if jira_key:
            jira_url = f"{settings.jira.url}/browse/{jira_key}"
            action_elements.append({
                "type": "button",
                "text": {"type": "plain_text", "text": ":jira: Jira Ticket"},
                "url": str(jira_url),
            })

        # Dashboard deep link (placeholder)
        action_elements.append({
            "type": "button",
            "text": {"type": "plain_text", "text": ":bar_chart: Dashboard"},
            "url": f"http://localhost:3000/incidents/{incident_id}",
        })

        if action_elements:
            blocks.append({"type": "actions", "elements": action_elements})

        blocks.append({"type": "divider"})
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"_IntelliPipe Autonomous Data Quality Platform • "
                        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} • "
                        f"Incident `{short_id}`_"
                    ),
                }
            ],
        })

        return blocks


# ---------------------------------------------------------------------------
# Jira Client
# ---------------------------------------------------------------------------

JIRA_SEVERITY_PRIORITY_MAP = {
    "critical": "Highest",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "info": "Lowest",
}


class JiraTicketClient:
    """
    Jira REST API v3 client for automated incident ticket creation.
    Supports:
    - Issue creation with severity-based priority
    - Label assignment
    - Component tagging
    - SLA tracking via custom fields
    - Transition to "In Progress" on creation
    """

    def __init__(self) -> None:
        self._cfg = settings.jira
        auth = (
            self._cfg.email,
            self._cfg.api_token.get_secret_value(),
        )
        self._client = httpx.AsyncClient(
            base_url=str(self._cfg.url),
            auth=auth,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=30.0,
        )

    async def create_incident_ticket(
        self,
        incident_id: str,
        title: str,
        description: str,
        severity: str,
        pr_url: Optional[str] = None,
        affected_tables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a Jira incident ticket and return key + URL."""
        priority = JIRA_SEVERITY_PRIORITY_MAP.get(severity, "Medium")
        labels = [
            "intellipipe",
            "data-quality",
            f"severity-{severity}",
        ]
        if severity in ("critical", "high"):
            labels.append(self._cfg.high_severity_label)

        payload = {
            "fields": {
                "project": {"key": self._cfg.project_key},
                "summary": title,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": description}],
                        }
                    ],
                },
                "issuetype": {"name": self._cfg.incident_issue_type},
                "priority": {"name": priority},
                "labels": labels,
                "customfield_10016": incident_id,  # Custom field: IntelliPipe Incident ID
            }
        }

        try:
            resp = await self._client.post("/rest/api/3/issue", content=json.dumps(payload))
            resp.raise_for_status()
            data = resp.json()
            key = data["key"]
            issue_url = f"{self._cfg.url}/browse/{key}"

            logger.info("Jira ticket created", key=key, priority=priority)

            # Add PR link as remote link if available
            if pr_url:
                await self._add_remote_link(key, pr_url, "Fix PR")

            return {"key": key, "url": issue_url, "id": data["id"]}

        except httpx.HTTPStatusError as e:
            logger.error("Jira API error", status=e.response.status_code, detail=e.response.text[:200])
            return {"key": "", "url": "", "id": ""}
        except Exception as e:
            logger.error("Jira ticket creation failed", error=str(e))
            return {"key": "", "url": "", "id": ""}

    async def _add_remote_link(self, issue_key: str, url: str, title: str) -> None:
        """Attach an external link to a Jira issue."""
        payload = {
            "object": {
                "url": url,
                "title": title,
                "icon": {"url16x16": "https://github.com/favicon.ico"},
            }
        }
        try:
            resp = await self._client.post(
                f"/rest/api/3/issue/{issue_key}/remotelink",
                content=json.dumps(payload),
            )
            if resp.status_code not in (200, 201):
                logger.warning("Failed to add remote link to Jira", key=issue_key)
        except Exception:
            pass

    async def transition_to_in_progress(self, issue_key: str) -> None:
        """Move a ticket to 'In Progress' state."""
        try:
            # Get available transitions
            resp = await self._client.get(f"/rest/api/3/issue/{issue_key}/transitions")
            resp.raise_for_status()
            transitions = resp.json()["transitions"]

            in_progress = next(
                (t for t in transitions if "progress" in t["name"].lower()), None
            )
            if in_progress:
                await self._client.post(
                    f"/rest/api/3/issue/{issue_key}/transitions",
                    content=json.dumps({"transition": {"id": in_progress["id"]}}),
                )
        except Exception as e:
            logger.warning("Failed to transition Jira ticket", key=issue_key, error=str(e))

    async def close(self) -> None:
        await self._client.aclose()
