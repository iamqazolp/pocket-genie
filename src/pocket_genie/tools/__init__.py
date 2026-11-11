from __future__ import annotations

from pathlib import Path
from typing import Any

from pocket_genie.core.memory import SessionMemoryStore
from pocket_genie.registry.skills import SkillRegistry
from pocket_genie.registry.tools import ToolRegistry
from pocket_genie.retrieval.hybrid_store import HybridKnowledgeStore
from pocket_genie.tools.databricks_tools import (
    DatabricksWorkspaceClient,
    register_databricks_tools,
)
from pocket_genie.tools.device_termux import register_device_tools
from pocket_genie.tools.devops import register_devops_tools
from pocket_genie.tools.jev_tools import register_jev_tools
from pocket_genie.tools.memory_tools import register_memory_and_kb_tools
from pocket_genie.tools.personal_hub import PersonalHubStore, register_personal_tools
from pocket_genie.tools.research_study import (
    StudyDeckStore,
    register_research_and_study_tools,
)
from pocket_genie.tools.sql_workspace import WorkspaceSQLStore, register_sql_tools
from pocket_genie.tools.visualization import register_visualization_tools
from pocket_genie.tools.work_coding import register_work_coding_tools


def register_all_builtin_tools(
    registry: ToolRegistry,
    workspace_dir: Path,
    memory_store: SessionMemoryStore | None = None,
    skill_registry: SkillRegistry | None = None,
    kb_store: HybridKnowledgeStore | None = None,
    sql_store: WorkspaceSQLStore | None = None,
    personal_store: PersonalHubStore | None = None,
    study_store: StudyDeckStore | None = None,
    databricks_client: DatabricksWorkspaceClient | None = None,
) -> dict[str, Any]:
    mem = memory_store or SessionMemoryStore()
    skills = skill_registry or SkillRegistry()
    kb = kb_store or HybridKnowledgeStore()
    if kb.num_chunks == 0 and workspace_dir.exists():
        if kb.embedder_name.startswith("hash-"):
            kb.index_directory(workspace_dir, max_files=40)
        else:
            kb.index_directory(
                workspace_dir,
                glob_patterns=("*.md", "*.py", "*.yaml"),
                max_files=35,
                chunk_chars=3200,
                overlap_chars=200,
            )

    register_devops_tools(registry, workspace_dir)
    sql = register_sql_tools(registry, sql_store, workspace_dir=workspace_dir)
    dev_state = register_device_tools(registry)
    register_visualization_tools(registry)
    register_memory_and_kb_tools(registry, mem, skills, kb)
    jev_client = register_jev_tools(registry)
    personal = register_personal_tools(
        registry, workspace_dir, store=personal_store, jev_client=jev_client
    )
    register_work_coding_tools(registry, workspace_dir)
    dbx = register_databricks_tools(registry, client=databricks_client)
    study = register_research_and_study_tools(
        registry,
        workspace_dir,
        kb_store=kb,
        study_store=study_store,
        jev_client=jev_client,
    )
    return {
        "memory_store": mem,
        "skill_registry": skills,
        "kb_store": kb,
        "sql_store": sql,
        "device_state": dev_state,
        "jev_client": jev_client,
        "personal_store": personal,
        "databricks_client": dbx,
        "study_store": study,
    }


__all__ = [
    "DatabricksWorkspaceClient",
    "PersonalHubStore",
    "StudyDeckStore",
    "WorkspaceSQLStore",
    "register_all_builtin_tools",
    "register_databricks_tools",
    "register_device_tools",
    "register_devops_tools",
    "register_jev_tools",
    "register_memory_and_kb_tools",
    "register_personal_tools",
    "register_research_and_study_tools",
    "register_sql_tools",
    "register_visualization_tools",
    "register_work_coding_tools",
]
