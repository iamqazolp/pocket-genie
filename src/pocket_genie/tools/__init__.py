from __future__ import annotations

from pathlib import Path
from typing import Any

from pocket_genie.core.memory import SessionMemoryStore
from pocket_genie.registry.skills import SkillRegistry
from pocket_genie.registry.tools import ToolRegistry
from pocket_genie.retrieval.hybrid_store import HybridKnowledgeStore
from pocket_genie.tools.device_termux import register_device_tools
from pocket_genie.tools.devops import register_devops_tools
from pocket_genie.tools.jev_tools import register_jev_tools
from pocket_genie.tools.memory_tools import register_memory_and_kb_tools
from pocket_genie.tools.sql_workspace import WorkspaceSQLStore, register_sql_tools
from pocket_genie.tools.visualization import register_visualization_tools


def register_all_builtin_tools(
    registry: ToolRegistry,
    workspace_dir: Path,
    memory_store: SessionMemoryStore | None = None,
    skill_registry: SkillRegistry | None = None,
    kb_store: HybridKnowledgeStore | None = None,
    sql_store: WorkspaceSQLStore | None = None,
) -> dict[str, Any]:
    mem = memory_store or SessionMemoryStore()
    skills = skill_registry or SkillRegistry()
    kb = kb_store or HybridKnowledgeStore()
    if kb.num_chunks == 0 and workspace_dir.exists():
        kb.index_directory(workspace_dir, max_files=40)

    register_devops_tools(registry, workspace_dir)
    sql = register_sql_tools(registry, sql_store, workspace_dir=workspace_dir)
    dev_state = register_device_tools(registry)
    register_visualization_tools(registry)
    register_memory_and_kb_tools(registry, mem, skills, kb)
    jev_client = register_jev_tools(registry)
    return {
        "memory_store": mem,
        "skill_registry": skills,
        "kb_store": kb,
        "sql_store": sql,
        "device_state": dev_state,
        "jev_client": jev_client,
    }


__all__ = [
    "WorkspaceSQLStore",
    "register_all_builtin_tools",
    "register_device_tools",
    "register_devops_tools",
    "register_jev_tools",
    "register_memory_and_kb_tools",
    "register_sql_tools",
    "register_visualization_tools",
]
