from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

from pocket_genie.shared import StreamEvent

_PREFIX_TOOL_RE = re.compile(r"^\[([a-zA-Z0-9_.:-]+)\]\s*(.*)$", re.DOTALL)


@dataclasses.dataclass(slots=True)
class TimelineStep:
    step_type: str
    title: str
    detail: str = ""
    specialist: str = "supervisor"
    tier: str = "t1"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def format_human_readable_reply(raw_reply: str) -> str:
    """Transform raw tool observation payloads into clean, readable assistant text."""
    text = raw_reply.strip()
    m = _PREFIX_TOOL_RE.match(text)
    tool_name = m.group(1) if m else ""
    body = m.group(2).strip() if m else text

    if body.startswith("[HITL_PENDING:"):
        return "This action modifies system or device state and is waiting for your approval below."

    try:
        parsed = json.loads(body)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        if "row_count" in parsed and "rows" in parsed:
            rows = parsed.get("rows") or []
            if not rows:
                return "Query returned 0 rows."
            cols = list(rows[0].keys())
            header = " | ".join(cols)
            sep = " | ".join("---" for _ in cols)
            lines = [f"Returned {len(rows)} row(s):", "", header, sep]
            for r in rows:
                lines.append(" | ".join(str(r.get(c, "")) for c in cols))
            return "\n".join(lines)

        if "results" in parsed and "query" in parsed:
            hits = parsed.get("results") or []
            if not hits:
                return f"No matching chunks found for `{parsed['query']}`."
            lines = [f"Found {len(hits)} relevant chunk(s) (alpha={parsed.get('alpha', 0.5)}):"]
            for idx, h in enumerate(hits, 1):
                snippet = " ".join(str(h.get("text", "")).split())
                if len(snippet) > 220:
                    snippet = snippet[:217] + "..."
                lines.append(
                    f"\n{idx}. {h.get('citation', 'doc')} (score={h.get('rerank_score', 0):.2f})\n   {snippet}"
                )
            return "\n".join(lines)

        if "platform" in parsed and "cpu_count" in parsed:
            bat = parsed.get("battery") or {}
            return (
                f"System & Device Health:\n"
                f"• Platform: {parsed.get('platform')} ({parsed.get('machine')})\n"
                f"• CPU Cores: {parsed.get('cpu_count')} (1m load avg: {parsed.get('load_avg_1m')})\n"
                f"• Free Disk Space: {parsed.get('disk_free_gb')} GB\n"
                f"• Battery: {bat.get('percentage', 100)}% ({bat.get('status', 'OK')}, {bat.get('temperature', 0)}°C)"
            )

    if tool_name == "devops.git_status":
        return f"Repository Git Status:\n{body}"
    if tool_name == "devops.git_diff":
        return f"Repository Diff:\n{body}"
    return body or text


def build_timeline_steps(events: list[StreamEvent]) -> list[dict[str, Any]]:
    """Convert internal StreamEvents into clean UI TimelineStep objects for React/Flutter/Web."""
    steps: list[TimelineStep] = []
    for ev in events:
        et = ev.event_type
        d = ev.data
        if et == "subagent_reasoning":
            steps.append(
                TimelineStep(
                    step_type="reasoning",
                    title=f"{d.get('specialist', 'agent')} ({d.get('tier', 't1')})",
                    detail=str(d.get("reasoning", "")),
                    specialist=str(d.get("specialist", "supervisor")),
                    tier=str(d.get("tier", "t1")),
                )
            )
        elif et == "tool_message":
            obs = str(d.get("observation", ""))
            short_obs = obs if len(obs) <= 260 else obs[:257] + "..."
            steps.append(
                TimelineStep(
                    step_type="tool_call",
                    title=f"Executed {d.get('tool', 'tool')}",
                    detail=short_obs,
                    specialist=str(d.get("specialist", "specialist")),
                )
            )
        elif et == "middleware_event":
            mw = str(d.get("middleware", "guardrail"))
            if mw == "fallback":
                steps.append(
                    TimelineStep(
                        step_type="escalation",
                        title=f"Precision escalated: {d.get('from_tier')} -> {d.get('to_tier')}",
                        detail=f"Reason: {d.get('reason', 'fault')}",
                        tier=str(d.get("to_tier", "t2")),
                    )
                )
            elif mw in ("repeat_tool_call_breaker", "sql_table_allowlist", "tool_result_truncation", "human_in_the_loop"):
                steps.append(
                    TimelineStep(
                        step_type="guardrail",
                        title=f"Guardrail active: {mw}",
                        detail=json.dumps({k: v for k, v in d.items() if k != "middleware"}),
                    )
                )
    return [s.to_dict() for s in steps]


def build_openapi_spec() -> dict[str, Any]:
    """Return an OpenAPI 3.1.0 schema so React (openapi-typescript) and Flutter (dio/openapi) clients can auto-generate bindings."""
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Pocket-Genie Headless Agent API",
            "version": "1.0.0",
            "description": "Client-agnostic REST and SSE API for React web apps, Flutter mobile apps, and CLI clients.",
        },
        "paths": {
            "/api/v1/status": {
                "get": {
                    "summary": "Get workspace status, quantization telemetry, skills, specialists, and pending approvals.",
                    "operationId": "getStatus",
                }
            },
            "/api/v1/chat": {
                "post": {
                    "summary": "Send a user message and receive a formatted assistant response, timeline steps, and inline approval cards.",
                    "operationId": "sendChatTurn",
                }
            },
            "/api/v1/chat/stream": {
                "get": {
                    "summary": "Server-Sent Events (SSE) stream for real-time subagent steps and token telemetry.",
                    "operationId": "streamChatTurn",
                }
            },
            "/api/v1/approvals/{request_id}": {
                "post": {
                    "summary": "Approve or reject a pending Human-in-the-Loop action.",
                    "operationId": "resolveApproval",
                }
            },
            "/api/v1/session/reset": {
                "post": {
                    "summary": "Reset the active conversation session and clear history.",
                    "operationId": "resetSession",
                }
            },
        },
    }
