from __future__ import annotations

import json
from pathlib import Path

from pocket_genie.core import GenieExecutionGraph, GenieState, TieredAgentFactory
from pocket_genie.registry import ToolRegistry
from pocket_genie.shared import ChatMessage, GenieSharedContext
from pocket_genie.tools import register_all_builtin_tools


def test_builtin_tools_registration_and_execution(tmp_path: Path):
    sample_file = tmp_path / "notes.md"
    sample_file.write_text(
        "# Quantization Notes\n\nLayer 0 mlp.down_proj has kurtosis 542.6 under W4A8.\n",
        encoding="utf-8",
    )

    reg = ToolRegistry()
    ctx_objs = register_all_builtin_tools(reg, workspace_dir=tmp_path)

    chunk_text = reg.execute("devops.read_file_chunk", {"path": "notes.md", "start_line": 1, "end_line": 5})
    assert "kurtosis 542.6" in chunk_text

    sql_res = json.loads(
        reg.execute(
            "sql.query_workspace",
            {"query": "SELECT recipe_id, values_exact_rate FROM ablation_results WHERE recipe_id = 'E4s'"},
        )
    )
    assert sql_res["row_count"] == 1
    assert sql_res["rows"][0]["recipe_id"] == "E4s"

    kb_res = json.loads(
        reg.execute("kb.search_workspace", {"query": "kurtosis 542.6", "alpha": 0.2, "top_k": 2})
    )
    assert kb_res["results"]
    assert "notes.md#chunk=0" in kb_res["results"][0]["citation"]

    skill_text = reg.execute("memory.load_skill", {"name": "quant-diagnosis"})
    assert "mlp.down_proj" in skill_text

    dev_res = json.loads(reg.execute("device.system_telemetry", {}))
    assert "cpu_count" in dev_res

    chart_res = json.loads(
        reg.execute(
            "viz.build_chart_spec",
            {"title": "Recipe Exact Match", "labels": ["E0", "E4s"], "values": [70.9, 64.1]},
        )
    )
    assert chart_res["widget"] == "pocket_genie_chart"

    graph = GenieExecutionGraph(
        factory=TieredAgentFactory(),
        tool_registry=reg,
        memory_store=ctx_objs["memory_store"],
        skill_registry=ctx_objs["skill_registry"],
    )
    state = GenieState(
        session_id="sess-tools",
        session_created_at="2026-11-08T16:00:00Z",
        messages=[ChatMessage(role="user", content="query_workspace ablation_results")],
    )
    shared = GenieSharedContext(
        session_id="sess-tools",
        session_created_at="2026-11-08T16:00:00Z",
        workspace_dir=tmp_path,
        project_id="pocket-genie",
        agent_type="direct-specialist",
        target_specialist="sql_agent",
    )
    reply, _ = graph.invoke(state, shared)
    assert "row_count" in reply.content
