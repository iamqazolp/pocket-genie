from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_sql_table_registry() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "sql_table_registry.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_supervisor_system_prompt(
    session_created_at: str,
    project_id: str | None,
    pinned_memories: list[str],
    skill_catalog: str,
    running_summary: str = "",
) -> str:
    """Render a deterministic, prefix-cacheable system prompt without per-turn dynamic timestamps."""
    memory_block = (
        "\n".join(f"- {m}" for m in pinned_memories)
        if pinned_memories
        else "- None recorded."
    )
    project_scope = project_id or "global-workspace"
    sections = [
        "You are Pocket-Genie Supervisor, an orchestrator for local engineering, codebase retrieval, SQL analytics, and edge device control.",
        f"Session Pinned Timestamp: {session_created_at} (Retrieve live wall-clock time on demand via `get_current_datetime`).",
        f"Active Workspace Scope: {project_scope}",
        "Core Routing Rules:\n"
        "1. Delegate domain-specific work using `create_task_<specialist>_t<tier>` tools rather than guessing file contents or SQL metrics.\n"
        "2. Keep mutating actions isolated and respect Human-in-the-Loop approval gates.\n"
        "3. Load full playbook instructions on demand via `load_skill` when tackling multi-step triage or quantization debugging.",
        f"Pinned User Memories:\n{memory_block}",
        skill_catalog,
    ]
    if running_summary:
        sections.append(f"Compacted Session History Summary:\n{running_summary}")
    return "\n\n".join(sections)
