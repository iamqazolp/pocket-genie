"""Daily briefing builder and interactive terminal REPL for Pocket Genie."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from pocket_genie.server import PocketGenieService


def _safe_tool_call(service: PocketGenieService, name: str, **kwargs: Any) -> Any:
    spec = service.tool_registry.get(name)
    if spec is None:
        return None
    try:
        return spec.handler(**kwargs)
    except Exception as exc:
        return {"error": str(exc)}


def build_daily_briefing(service: PocketGenieService) -> dict[str, Any]:
    state = service.get_state_summary()
    cfg_file = Path.home() / ".databrickscfg"
    db_ok = bool(os.environ.get("DATABRICKS_HOST")) or cfg_file.exists()
    due_fc = _safe_tool_call(service, "study.due_flashcards")
    if isinstance(due_fc, str):
        try:
            due_fc = json.loads(due_fc)
        except Exception:
            pass
    return {
        "tier": state.get("active_tier", "t1"),
        "project_id": state.get("project_id", ""),
        "session_id": state.get("session_id", ""),
        "databricks_configured": db_ok,
        "git_status": _safe_tool_call(service, "devops.git_status"),
        "deadlines": _safe_tool_call(service, "personal.list_deadlines") or [],
        "messages": _safe_tool_call(service, "personal.list_messages") or [],
        "due_flashcards": due_fc or [],
        "antigravity_sessions": _safe_tool_call(service, "agent_ops.antigravity_sessions") or {},
        "claude_code_sessions": _safe_tool_call(service, "agent_ops.claude_code_sessions") or {},
    }


def format_briefing_markdown(brief: dict[str, Any]) -> str:
    lines = [
        "# Pocket Genie Daily Briefing",
        (
            f"- **Project**: `{brief.get('project_id', 'unknown')}` | "
            f"**Session**: `{brief.get('session_id', 'none')}` | "
            f"**Tier**: `{brief.get('tier', 't1')}`"
        ),
        f"- **Databricks**: {'Configured' if brief.get('databricks_configured') else 'Not configured'}",
    ]
    if git_st := brief.get("git_status"):
        lines.append(f"- **Git Status**: {str(git_st).splitlines()[0]}")
    deadlines = brief.get("deadlines", [])
    lines.append(f"- **Deadlines**: {len(deadlines) if isinstance(deadlines, list) else 0} active")
    if isinstance(deadlines, list):
        for dl in deadlines[:3]:
            if isinstance(dl, dict):
                lines.append(
                    f"  * [{dl.get('priority', 'normal')}] {dl.get('title', 'Task')} "
                    f"(due: {dl.get('due_at', 'n/a')})"
                )
    messages = brief.get("messages", [])
    lines.append(f"- **Messages**: {len(messages) if isinstance(messages, list) else 0} pending")
    due_fc = brief.get("due_flashcards", [])
    lines.append(f"- **Due Flashcards**: {len(due_fc) if isinstance(due_fc, list) else 0}")
    agy = brief.get("antigravity_sessions", {})
    claude = brief.get("claude_code_sessions", {})
    agy_n = len(agy.get("sessions", [])) if isinstance(agy, dict) else 0
    c_n = len(claude.get("sessions", [])) if isinstance(claude, dict) else 0
    lines.append(f"- **Agent Ops**: {agy_n} Antigravity session(s), {c_n} Claude session(s)")
    return "\n".join(lines)


def _resolve_hitl_call(service: PocketGenieService, call_id: str, approved: bool) -> dict[str, Any]:
    try:
        return service.resolve_hitl(call_id=call_id, approved=approved)
    except TypeError:
        return service.resolve_hitl(request_id=call_id, approved=approved)


def run_vibe_repl(
    service: PocketGenieService,
    initial_tier: str = "t1",
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    current_tier = initial_tier.lower()
    output_fn("Welcome to Pocket Genie Vibe REPL! Type /help for available commands.")

    while True:
        try:
            raw = input_fn(f"genie [{current_tier}]> ")
        except (EOFError, KeyboardInterrupt):
            output_fn("\nExiting Pocket Genie REPL.")
            return 0
        line = raw.strip()
        if not line:
            continue
        if line in ("/exit", "/quit"):
            output_fn("Goodbye!")
            return 0
        if line == "/help":
            output_fn(
                "Commands:\n"
                "  /help              Show this help message\n"
                "  /brief             Generate and show daily briefing dashboard\n"
                "  /status            Display current session & tier summary\n"
                "  /diff              Inspect workspace git diff\n"
                "  /test              Run project test suite\n"
                "  /tier <t1|t2|t3>   Switch execution tier\n"
                "  /reset             Reset current agent session\n"
                "  /exit, /quit       Exit REPL"
            )
            continue
        if line == "/brief":
            output_fn(format_briefing_markdown(build_daily_briefing(service)))
            continue
        if line == "/status":
            summary = service.get_state_summary()
            output_fn(
                f"Session: {summary.get('session_id')} | Tier: {current_tier} | "
                f"Project: {summary.get('project_id')} | Traces: "
                f"{summary.get('observability', {}).get('trace_count', 0)}"
            )
            continue
        if line == "/diff":
            output_fn(str(_safe_tool_call(service, "devops.git_diff") or "No diff available."))
            continue
        if line == "/test":
            output_fn(str(_safe_tool_call(service, "devops.run_pytest") or "Test run completed."))
            continue
        if line.startswith("/tier"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].lower() in ("t1", "t2", "t3"):
                current_tier = parts[1].lower()
                service.state.active_tier = current_tier  # type: ignore[assignment]
                output_fn(f"Active tier switched to: {current_tier}")
            else:
                output_fn(f"Current tier: {current_tier}. Usage: /tier <t1|t2|t3>")
            continue
        if line == "/reset":
            service.reset_session()
            output_fn("Session reset successfully.")
            continue

        res = service.run_turn(prompt=line, tier=current_tier)  # type: ignore[arg-type]
        if reply := (res.get("reply") or res.get("raw_reply")):
            output_fn(reply)

        pending = res.get("pending_hitl") or (
            [res["approval_request"]] if res.get("approval_request") else []
        )
        for item in pending:
            d = item if isinstance(item, dict) else item.to_dict()
            call_id = d.get("request_id") or d.get("call_id") or d.get("id", "")
            tool_name = d.get("tool_name", "tool")
            args = d.get("arguments", {})
            output_fn(f"[HITL Required] Tool '{tool_name}' with args {args}")
            try:
                user_resp = input_fn("Approve execution? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                user_resp = "n"
            hitl_res = _resolve_hitl_call(
                service, call_id=call_id, approved=user_resp in ("y", "yes")
            )
            output_fn(
                hitl_res.get("assistant_reply")
                or f"Action {call_id}: status={hitl_res.get('status', 'unknown')}"
            )
