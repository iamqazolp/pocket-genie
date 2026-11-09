"""Work coding tools: GitHub repository/PR/issue/CI inspection and agent session inspection."""

import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Callable

from pocket_genie.registry.tools import ToolRegistry

_GITHUB_SLUG_RE = re.compile(r"github\.com[:/](?P<owner>[^/\s]+)/(?P<repo>[^/\s]+?)(?:\.git)?$")


def parse_github_remote_slug(remote_url: str) -> str | None:
    """Extract 'owner/repo' from an HTTPS or SSH GitHub remote URL."""
    m = _GITHUB_SLUG_RE.search(remote_url.strip())
    return f"{m.group('owner')}/{m.group('repo')}" if m else None


def detect_workspace_github_slug(workspace_dir: Path) -> str | None:
    """Detect GitHub repository slug for the given workspace directory."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(workspace_dir), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return parse_github_remote_slug(proc.stdout.strip())
    except Exception:
        pass
    git_cfg = workspace_dir / ".git" / "config"
    if git_cfg.is_file():
        try:
            for line in git_cfg.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("url ="):
                    parsed = parse_github_remote_slug(line.split("=", 1)[1].strip())
                    if parsed:
                        return parsed
        except Exception:
            pass
    return None


def _default_http_get(url: str, headers: dict[str, str]) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "pocket-genie", **headers})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def register_work_coding_tools(
    registry: ToolRegistry,
    workspace_dir: Path,
    agy_root: Path | None = None,
    claude_root: Path | None = None,
    http_get: Callable[[str, dict[str, str]], Any] | None = None,
) -> None:
    """Register GitHub operations and coding agent session inspection tools."""
    getter = http_get or _default_http_get

    def _headers() -> dict[str, str]:
        token = os.environ.get("GITHUB_TOKEN")
        auth = {"Authorization": f"Bearer {token}"} if token else {}
        return {"Accept": "application/vnd.github+json", **auth}

    def _slug(val: str | None) -> str:
        s = val or detect_workspace_github_slug(workspace_dir)
        if not s:
            raise ValueError("GitHub slug not provided and could not be detected from workspace.")
        return s

    @registry.register(
        name="github.repo_overview",
        description="Fetch high-level overview metadata for a GitHub repository.",
        parameters={"type": "object", "properties": {"slug": {"type": "string"}}},
        domain="github",
        mutating=False,
    )
    def repo_overview(slug: str | None = None) -> dict[str, Any]:
        s = _slug(slug)
        d = getter(f"https://api.github.com/repos/{s}", _headers())
        return {
            "full_name": d.get("full_name", s),
            "description": d.get("description"),
            "default_branch": d.get("default_branch", "main"),
            "open_issues_count": d.get("open_issues_count", 0),
            "stargazers_count": d.get("stargazers_count", 0),
            "html_url": d.get("html_url"),
        }

    @registry.register(
        name="github.list_prs",
        description="List pull requests for a repository.",
        parameters={"type": "object", "properties": {"slug": {"type": "string"}, "state": {"type": "string"}, "limit": {"type": "integer"}}},
        domain="github",
        mutating=False,
    )
    def list_prs(slug: str | None = None, state: str = "open", limit: int = 10) -> list[dict[str, Any]]:
        s = _slug(slug)
        items = getter(f"https://api.github.com/repos/{s}/pulls?state={state}&per_page={limit}", _headers())
        return [{
            "number": it.get("number"),
            "title": it.get("title"),
            "state": it.get("state"),
            "user": it.get("user", {}).get("login"),
            "created_at": it.get("created_at"),
            "html_url": it.get("html_url"),
        } for it in items[:limit]]

    @registry.register(
        name="github.list_issues",
        description="List pure issues (excluding pull requests) for a repository.",
        parameters={"type": "object", "properties": {"slug": {"type": "string"}, "state": {"type": "string"}, "limit": {"type": "integer"}}},
        domain="github",
        mutating=False,
    )
    def list_issues(slug: str | None = None, state: str = "open", limit: int = 10) -> list[dict[str, Any]]:
        s = _slug(slug)
        items = getter(f"https://api.github.com/repos/{s}/issues?state={state}&per_page={limit * 2}", _headers())
        return [{
            "number": it.get("number"),
            "title": it.get("title"),
            "state": it.get("state"),
            "user": it.get("user", {}).get("login"),
            "comments": it.get("comments", 0),
            "created_at": it.get("created_at"),
            "html_url": it.get("html_url"),
        } for it in items if "pull_request" not in it][:limit]

    @registry.register(
        name="github.ci_runs",
        description="List recent GitHub Actions CI workflow runs.",
        parameters={"type": "object", "properties": {"slug": {"type": "string"}, "limit": {"type": "integer"}}},
        domain="github",
        mutating=False,
    )
    def ci_runs(slug: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        s = _slug(slug)
        data = getter(f"https://api.github.com/repos/{s}/actions/runs?per_page={limit}", _headers())
        runs = (data.get("workflow_runs", []) if isinstance(data, dict) else [])[:limit]
        return [{
            "id": r.get("id"),
            "name": r.get("name"),
            "status": r.get("status"),
            "conclusion": r.get("conclusion"),
            "branch": r.get("head_branch"),
            "event": r.get("event"),
            "created_at": r.get("created_at"),
            "html_url": r.get("html_url"),
        } for r in runs]

    @registry.register(
        name="github.git_branch_ops",
        description="Perform git branch and commit operations within workspace.",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "create_branch", "switch", "commit"]},
                "branch": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["action"],
        },
        domain="github",
        mutating=True,
        project_scoped=True,
    )
    def git_branch_ops(action: str, branch: str | None = None, message: str | None = None) -> dict[str, Any]:
        def _run(args: list[str]) -> str:
            res = subprocess.run(["git", "-C", str(workspace_dir), *args], capture_output=True, text=True, check=True)
            return res.stdout.strip()

        if action == "list":
            branches = [ln.strip().lstrip("* ") for ln in _run(["branch", "--list"]).splitlines() if ln.strip()]
            return {"current": _run(["branch", "--show-current"]), "branches": branches}
        if action != "commit" and (not branch or not re.match(r"^[a-zA-Z0-9_\-\./]+$", branch)):
            raise ValueError("Valid branch name required for branch operations.")
        if action == "create_branch":
            _run(["checkout", "-b", str(branch)])
            return {"status": "created", "branch": branch}
        if action == "switch":
            _run(["checkout", str(branch)])
            return {"status": "switched", "branch": branch}
        if action == "commit":
            if not message:
                raise ValueError("Commit message is required.")
            _run(["commit", "-am", message])
            return {"status": "committed", "commit": _run(["rev-parse", "--short", "HEAD"]), "message": message}
        raise ValueError(f"Unsupported action: {action}")

    @registry.register(
        name="agent_ops.antigravity_sessions",
        description="Inspect Antigravity session notes, artifacts, and hub headings.",
        parameters={"type": "object", "properties": {"limit": {"type": "integer", "default": 8}}},
        domain="agent_ops",
        mutating=False,
    )
    def antigravity_sessions(limit: int = 8) -> dict[str, Any]:
        sessions: list[dict[str, Any]] = []
        base_dir = agy_root or Path.home() / ".gemini" / "antigravity-cli" / "brain"
        hub = workspace_dir / "PROJECT_HUB.md"
        heading = next((ln.strip() for ln in hub.read_text("utf-8").splitlines() if ln.startswith("#")), None) if hub.is_file() else None

        if base_dir.is_dir():
            dirs = sorted(base_dir.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
            for s_dir in [d for d in dirs if d.is_dir()][:limit]:
                last_input = None
                t_file = s_dir / ".system_generated" / "logs" / "transcript.jsonl"
                if t_file.is_file():
                    try:
                        for line in reversed(t_file.read_text(encoding="utf-8").splitlines()):
                            if line.strip() and json.loads(line).get("type") == "USER_INPUT":
                                last_input = json.loads(line).get("content")
                                break
                    except Exception:
                        pass
                sessions.append({
                    "session_id": s_dir.name,
                    "artifacts": [f.name for f in s_dir.glob("*.md")],
                    "last_user_input": last_input,
                })

        local_notes = workspace_dir / ".local-notes"
        return {
            "project_hub_heading": heading,
            "local_notes_count": len(list(local_notes.glob("*.md"))) if local_notes.is_dir() else 0,
            "sessions": sessions,
        }

    @registry.register(
        name="agent_ops.claude_code_sessions",
        description="Inspect Claude Code project sessions and custom commands.",
        parameters={"type": "object", "properties": {"limit": {"type": "integer", "default": 8}}},
        domain="agent_ops",
        mutating=False,
    )
    def claude_code_sessions(limit: int = 8) -> dict[str, Any]:
        cmd_dir = workspace_dir / ".claude" / "commands"
        c_root = claude_root or Path.home() / ".claude" / "projects"
        sessions: list[dict[str, Any]] = []
        if c_root.is_dir():
            all_files = sorted(c_root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
            sessions = [{"session_file": f.name, "path": str(f)} for f in all_files if f.is_file()][:limit]

        return {
            "has_claude_md": (workspace_dir / "CLAUDE.md").is_file(),
            "commands": [f.stem for f in cmd_dir.glob("*.md")] if cmd_dir.is_dir() else [],
            "sessions": sessions,
        }
