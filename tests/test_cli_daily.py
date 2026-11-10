"""Tests for Pocket Genie Daily Briefing and Interactive REPL."""

from pathlib import Path
from typing import Any

from pocket_genie.cli_daily import (
    build_daily_briefing,
    format_briefing_markdown,
    run_vibe_repl,
)
from pocket_genie.server import PocketGenieService


def test_build_daily_briefing_and_formatting(tmp_path: Path) -> None:
    svc = PocketGenieService(workspace_dir=tmp_path)
    brief = build_daily_briefing(svc)

    assert "tier" in brief
    assert "project_id" in brief
    assert "session_id" in brief
    assert "databricks_configured" in brief
    assert "git_status" in brief
    assert "deadlines" in brief
    assert "messages" in brief
    assert "due_flashcards" in brief
    assert "antigravity_sessions" in brief
    assert "claude_code_sessions" in brief

    md = format_briefing_markdown(brief)
    assert "# Pocket Genie Daily Briefing" in md
    assert "**Databricks**" in md
    assert "**Deadlines**" in md
    assert "**Agent Ops**" in md


def test_run_vibe_repl_slash_commands(tmp_path: Path) -> None:
    svc = PocketGenieService(workspace_dir=tmp_path)
    commands = [
        "/help",
        "/brief",
        "/status",
        "/diff",
        "/test",
        "/tier t2",
        "/tier invalid",
        "/reset",
        "/exit",
    ]
    inputs = iter(commands)
    outputs: list[str] = []

    ret = run_vibe_repl(svc, input_fn=lambda _: next(inputs), output_fn=outputs.append)
    assert ret == 0

    joined = "\n".join(outputs)
    assert "Commands:" in joined
    assert "Pocket Genie Daily Briefing" in joined
    assert "Active tier switched to: t2" in joined
    assert "Current tier: t2. Usage: /tier <t1|t2|t3>" in joined
    assert "Session reset successfully." in joined
    assert "Goodbye!" in joined


def test_run_vibe_repl_normal_turn(tmp_path: Path) -> None:
    svc = PocketGenieService(workspace_dir=tmp_path)
    inputs = iter(["hello", "/quit"])
    outputs: list[str] = []

    ret = run_vibe_repl(svc, input_fn=lambda _: next(inputs), output_fn=outputs.append)
    assert ret == 0
    assert len(outputs) >= 2


def test_run_vibe_repl_hitl_approval_flow(tmp_path: Path) -> None:
    svc = PocketGenieService(workspace_dir=tmp_path)
    orig_run_turn = svc.run_turn

    def run_turn_with_hitl(prompt: str, tier: str = "t1") -> dict[str, Any]:
        return orig_run_turn(prompt=prompt, tier=tier, fault="hitl_patch")

    svc.run_turn = run_turn_with_hitl  # type: ignore[assignment]

    inputs = iter(["trigger mutating notification", "y", "/exit"])
    outputs: list[str] = []

    ret = run_vibe_repl(svc, input_fn=lambda _: next(inputs), output_fn=outputs.append)
    assert ret == 0

    joined = "\n".join(outputs)
    assert "[HITL Required]" in joined
    assert "device.send_notification" in joined
    assert "approved" in joined.lower() or "alert" in joined.lower()
