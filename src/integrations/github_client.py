"""
IntelliPipe GitHub PR Automation
==================================
Automatically creates pull requests with LLM-generated dbt fixes.
Features:
- Branch creation from base branch
- File diff computation and commit creation
- PR creation with rich description
- Reviewer assignment
- Human approval gate
- PR status polling
- Auto-merge on approval (configurable)
"""

from __future__ import annotations

import base64
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__, component="github_client")
settings = get_settings()


class PRCreationResult(BaseModel):
    pr_url: str
    pr_number: int
    branch_name: str
    commit_sha: str
    status: str  # open / draft / merged / closed


class GitHubPRClient:
    """
    GitHub REST API v3 client for automated PR creation.
    Uses personal access token auth (swap for GitHub App in production).
    """

    BASE_URL = "https://api.github.com"

    def __init__(self) -> None:
        self._cfg = settings.github
        self._token = self._cfg.token.get_secret_value()
        self._headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers=self._headers,
            timeout=30.0,
        )

    async def create_fix_pr(
        self,
        table_name: str,
        fix_code: Dict[str, Any],
        incident_id: str,
        rca_summary: str,
        require_approval: bool = True,
    ) -> Dict[str, Any]:
        """
        Full PR creation workflow:
        1. Get base branch SHA
        2. Create feature branch
        3. Commit generated dbt files
        4. Open PR with rich description
        5. Assign reviewers
        6. Return PR metadata
        """
        try:
            org = self._cfg.org
            repo = self._cfg.repo
            base_branch = self._cfg.base_branch

            # Sanitise table name for branch naming
            safe_table = re.sub(r"[^a-z0-9_-]", "-", table_name.lower())
            branch_name = f"intellipipe/dq-fix/{safe_table}/{incident_id[:8]}"

            # 1. Get base branch SHA
            base_sha = await self._get_branch_sha(org, repo, base_branch)

            # 2. Create feature branch
            await self._create_branch(org, repo, branch_name, base_sha)
            logger.info("Branch created", branch=branch_name)

            # 3. Prepare file commits
            files_to_commit = self._prepare_files(table_name, fix_code, incident_id)

            # 4. Create commits for each file
            latest_sha = base_sha
            for file_path, content in files_to_commit.items():
                latest_sha = await self._create_or_update_file(
                    org=org,
                    repo=repo,
                    path=file_path,
                    content=content,
                    branch=branch_name,
                    message=f"feat(dq): auto-fix {table_name} DQ issue [{incident_id[:8]}]",
                )

            # 5. Create PR
            pr_body = self._build_pr_body(
                table_name=table_name,
                incident_id=incident_id,
                rca_summary=rca_summary,
                fix_code=fix_code,
                files_changed=list(files_to_commit.keys()),
                require_approval=require_approval,
            )

            pr_data = await self._create_pull_request(
                org=org,
                repo=repo,
                title=f"[IntelliPipe] Auto-fix: DQ issue on {table_name} [{incident_id[:8]}]",
                body=pr_body,
                head=branch_name,
                base=base_branch,
                draft=require_approval,  # Open as draft if approval needed
            )

            pr_number = pr_data["number"]
            pr_url = pr_data["html_url"]

            # 6. Assign reviewers
            if self._cfg.pr_reviewers:
                await self._request_reviewers(org, repo, pr_number, self._cfg.pr_reviewers)

            # 7. Add labels
            await self._add_labels(org, repo, pr_number, ["intellipipe", "auto-generated", "data-quality"])

            logger.info("PR created", pr_url=pr_url, pr_number=pr_number, branch=branch_name)

            return {
                "pr_url": pr_url,
                "pr_number": pr_number,
                "branch_name": branch_name,
                "commit_sha": latest_sha,
                "status": "draft" if require_approval else "open",
            }

        except httpx.HTTPStatusError as e:
            logger.error("GitHub API error", status=e.response.status_code, detail=e.response.text)
            return {"pr_url": "", "pr_number": 0, "branch_name": "", "commit_sha": "", "status": "failed"}
        except Exception as e:
            logger.error("PR creation failed", error=str(e))
            return {"pr_url": "", "pr_number": 0, "branch_name": "", "commit_sha": "", "status": "failed"}

    def _prepare_files(
        self,
        table_name: str,
        fix_code: Dict[str, Any],
        incident_id: str,
    ) -> Dict[str, str]:
        """Build the dict of file_path → content to commit."""
        safe_table = re.sub(r"[^a-z0-9_]", "_", table_name.lower())
        files = {}

        # dbt model SQL
        if fix_code.get("dbt_model_sql"):
            files[f"models/staging/stg_{safe_table}.sql"] = fix_code["dbt_model_sql"]

        # dbt schema tests YAML
        if fix_code.get("dbt_test_yaml"):
            files[f"models/staging/stg_{safe_table}.yml"] = fix_code["dbt_test_yaml"]

        # Rollback SQL
        if fix_code.get("rollback_sql"):
            files[f"rollbacks/{safe_table}_{incident_id[:8]}_rollback.sql"] = fix_code["rollback_sql"]

        # Incident metadata
        import json
        files[f".intellipipe/incidents/{incident_id[:8]}.json"] = json.dumps(
            {
                "incident_id": incident_id,
                "table_name": table_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "risk_level": fix_code.get("risk_level", "medium"),
                "change_summary": fix_code.get("change_summary", ""),
            },
            indent=2,
        )

        return files

    @staticmethod
    def _build_pr_body(
        table_name: str,
        incident_id: str,
        rca_summary: str,
        fix_code: Dict[str, Any],
        files_changed: List[str],
        require_approval: bool,
    ) -> str:
        """Build a structured PR description markdown."""
        approval_note = (
            "\n> ⚠️ **This PR requires human approval before merge.** "
            "Review the changes carefully before approving.\n"
            if require_approval
            else ""
        )

        files_list = "\n".join(f"- `{f}`" for f in files_changed)

        return f"""## 🤖 IntelliPipe Auto-Generated Fix
{approval_note}
### Incident Summary
| Field | Value |
|-------|-------|
| **Incident ID** | `{incident_id}` |
| **Affected Table** | `{table_name}` |
| **Risk Level** | `{fix_code.get("risk_level", "medium")}` |
| **Generated At** | `{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}` |

### Root Cause
{rca_summary}

### Change Summary
{fix_code.get("change_summary", "Automated data quality remediation")}

### Files Changed
{files_list}

### Rollback Instructions
If this fix causes issues, run the rollback SQL located in `rollbacks/` directory.

### Review Checklist
- [ ] Root cause analysis is accurate
- [ ] Generated SQL logic is correct
- [ ] dbt tests cover the regression scenario
- [ ] Rollback script is tested
- [ ] No sensitive data in the fix

---
*Auto-generated by [IntelliPipe](https://github.com/{settings.github.org}/{settings.github.repo}) 
Autonomous Data Quality Platform — Incident `{incident_id[:8]}`*"""

    async def _get_branch_sha(self, org: str, repo: str, branch: str) -> str:
        resp = await self._client.get(f"/repos/{org}/{repo}/git/ref/heads/{branch}")
        resp.raise_for_status()
        return resp.json()["object"]["sha"]

    async def _create_branch(self, org: str, repo: str, branch: str, sha: str) -> None:
        resp = await self._client.post(
            f"/repos/{org}/{repo}/git/refs",
            json={"ref": f"refs/heads/{branch}", "sha": sha},
        )
        if resp.status_code not in (201, 422):  # 422 = already exists
            resp.raise_for_status()

    async def _create_or_update_file(
        self,
        org: str,
        repo: str,
        path: str,
        content: str,
        branch: str,
        message: str,
    ) -> str:
        """Commit a file, creating or updating it. Returns new commit SHA."""
        encoded = base64.b64encode(content.encode()).decode()
        payload: Dict[str, Any] = {
            "message": message,
            "content": encoded,
            "branch": branch,
        }

        # Check if file already exists (get its SHA for update)
        check = await self._client.get(
            f"/repos/{org}/{repo}/contents/{path}",
            params={"ref": branch},
        )
        if check.status_code == 200:
            payload["sha"] = check.json()["sha"]

        resp = await self._client.put(f"/repos/{org}/{repo}/contents/{path}", json=payload)
        resp.raise_for_status()
        return resp.json()["commit"]["sha"]

    async def _create_pull_request(
        self,
        org: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str,
        draft: bool = True,
    ) -> Dict[str, Any]:
        resp = await self._client.post(
            f"/repos/{org}/{repo}/pulls",
            json={
                "title": title,
                "body": body,
                "head": head,
                "base": base,
                "draft": draft,
                "maintainer_can_modify": True,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def _request_reviewers(
        self, org: str, repo: str, pr_number: int, reviewers: List[str]
    ) -> None:
        resp = await self._client.post(
            f"/repos/{org}/{repo}/pulls/{pr_number}/requested_reviewers",
            json={"reviewers": reviewers},
        )
        if resp.status_code not in (200, 201, 422):
            logger.warning("Failed to add reviewers", status=resp.status_code)

    async def _add_labels(
        self, org: str, repo: str, pr_number: int, labels: List[str]
    ) -> None:
        resp = await self._client.post(
            f"/repos/{org}/{repo}/issues/{pr_number}/labels",
            json={"labels": labels},
        )
        if resp.status_code not in (200, 201):
            logger.warning("Failed to add labels", status=resp.status_code)

    async def get_pr_status(self, pr_number: int) -> Dict[str, Any]:
        """Poll PR status — used by the human approval workflow."""
        resp = await self._client.get(
            f"/repos/{self._cfg.org}/{self._cfg.repo}/pulls/{pr_number}"
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "state": data["state"],
            "merged": data.get("merged", False),
            "mergeable": data.get("mergeable"),
            "review_decision": data.get("review_decision"),
            "approvals": sum(
                1 for r in data.get("requested_reviewers", [])
            ),
        }

    async def close(self) -> None:
        await self._client.aclose()
