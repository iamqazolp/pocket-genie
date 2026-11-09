"""Tests for work_coding tools."""

import json
from pathlib import Path

from pocket_genie.registry.tools import ToolRegistry
from pocket_genie.tools.work_coding import (
    detect_workspace_github_slug,
    parse_github_remote_slug,
    register_work_coding_tools,
)


def test_parse_github_remote_slug() -> None:
    assert parse_github_remote_slug("git@github.com:owner/repo.git") == "owner/repo"
    assert parse_github_remote_slug("https://github.com/owner/repo.git") == "owner/repo"
    assert parse_github_remote_slug("https://github.com/owner/repo") == "owner/repo"
    assert parse_github_remote_slug("https://gitlab.com/owner/repo.git") is None
    assert parse_github_remote_slug("invalid") is None


def test_detect_workspace_github_slug(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir(parents=True)
    cfg = git_dir / "config"
    cfg.write_text(
        '[remote "origin"]\n    url = git@github.com:acme/pocket-genie.git\n    fetch = +refs/heads/*\n',
        encoding="utf-8",
    )
    assert detect_workspace_github_slug(tmp_path) == "acme/pocket-genie"


def test_github_tools_with_mock_http(tmp_path: Path) -> None:
    registry = ToolRegistry()

    def mock_http_get(url: str, headers: dict[str, str]) -> dict[str, object] | list[dict[str, object]]:
        if "pulls" in url:
            return [{"number": 1, "title": "Add feature", "state": "open", "user": {"login": "alice"}}]
        if "issues" in url:
            return [
                {"number": 2, "title": "PR issue", "pull_request": {}},
                {"number": 3, "title": "Real bug", "state": "open", "user": {"login": "bob"}},
            ]
        if "actions/runs" in url:
            return {
                "workflow_runs": [
                    {"id": 101, "name": "CI", "status": "completed", "conclusion": "success"}
                ]
            }
        return {"full_name": "acme/demo", "default_branch": "main", "stargazers_count": 42}

    register_work_coding_tools(
        registry=registry,
        workspace_dir=tmp_path,
        http_get=mock_http_get,
    )

    overview = registry.execute("github.repo_overview", {"slug": "acme/demo"})
    assert overview["full_name"] == "acme/demo"
    assert overview["stargazers_count"] == 42

    prs = registry.execute("github.list_prs", {"slug": "acme/demo"})
    assert len(prs) == 1
    assert prs[0]["number"] == 1

    issues = registry.execute("github.list_issues", {"slug": "acme/demo"})
    assert len(issues) == 1
    assert issues[0]["number"] == 3

    runs = registry.execute("github.ci_runs", {"slug": "acme/demo"})
    assert len(runs) == 1
    assert runs[0]["id"] == 101


def test_agent_ops_sessions(tmp_path: Path) -> None:
    registry = ToolRegistry()
    agy_root = tmp_path / "brain"
    session_dir = agy_root / "sess-123"
    logs_dir = session_dir / ".system_generated" / "logs"
    logs_dir.mkdir(parents=True)
    (session_dir / "SPEC.md").write_text("# Spec", encoding="utf-8")

    transcript = logs_dir / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "SYSTEM", "content": "init"})
        + "\n"
        + json.dumps({"type": "USER_INPUT", "content": "Build feature X"})
        + "\n",
        encoding="utf-8",
    )

    (tmp_path / "PROJECT_HUB.md").write_text("# Pocket Genie Hub\nDetails", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("Claude rules", encoding="utf-8")

    claude_root = tmp_path / "claude_projects"
    claude_root.mkdir(parents=True)
    (claude_root / "session.jsonl").write_text("{}", encoding="utf-8")

    register_work_coding_tools(
        registry=registry,
        workspace_dir=tmp_path,
        agy_root=agy_root,
        claude_root=claude_root,
        http_get=lambda url, h: {},
    )

    agy_result = registry.execute("agent_ops.antigravity_sessions", {"limit": 5})
    assert agy_result["project_hub_heading"] == "# Pocket Genie Hub"
    assert len(agy_result["sessions"]) == 1
    assert agy_result["sessions"][0]["session_id"] == "sess-123"
    assert agy_result["sessions"][0]["last_user_input"] == "Build feature X"

    claude_result = registry.execute("agent_ops.claude_code_sessions", {"limit": 5})
    assert claude_result["has_claude_md"] is True
    assert len(claude_result["sessions"]) == 1
