from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pocket_genie.core.state import GenieState
from pocket_genie.middlewares.access_control import (
    DYNAMIC_SQL_TABLES,
    SQLTableAllowlistMiddleware,
    clear_dynamic_sql_tables,
)
from pocket_genie.middlewares.base import MiddlewarePipeline, ToolCallContext
from pocket_genie.registry.tools import ToolRegistry
from pocket_genie.shared import GenieSharedContext, StreamWriter, ToolInvocation
from pocket_genie.tools.sql_workspace import WorkspaceSQLStore, register_sql_tools


def _make_tool_ctx(
    reg: ToolRegistry,
    workspace_dir: Path,
    tool_name: str,
    arguments: dict,
    agent_type: str = "scoped-demo",
) -> ToolCallContext:
    state = GenieState(
        session_id="sess-sql-ws",
        session_created_at="2026-11-10T10:00:00Z",
        agent_type=agent_type,  # type: ignore[arg-type]
    )
    shared = GenieSharedContext(
        session_id="sess-sql-ws",
        session_created_at="2026-11-10T10:00:00Z",
        workspace_dir=workspace_dir,
        agent_type=agent_type,  # type: ignore[arg-type]
    )
    return ToolCallContext(
        state=state,
        shared=shared,
        invocation=ToolInvocation(id="call_sql_1", name=tool_name, arguments=arguments),
        spec=reg.get(tool_name),
        specialist="sql_agent",
        writer=StreamWriter(),
    )


def test_autodiscovery_of_csv_and_jsonl_in_workspace(tmp_path: Path) -> None:
    clear_dynamic_sql_tables()
    csv_file = tmp_path / "device_metrics.csv"
    csv_file.write_text(
        "device_id,latency_ms,active\ndev-1,12.5,1\ndev-2,8.0,0\n",
        encoding="utf-8",
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    jsonl_file = data_dir / "2026_events.jsonl"
    jsonl_file.write_text(
        '{"event": "boot", "count": 3}\n{"event": "infer", "count": 15}\n',
        encoding="utf-8",
    )

    reg = ToolRegistry()
    store = register_sql_tools(reg, workspace_dir=tmp_path)

    assert "device_metrics" in store.dynamic_tables
    assert "t_2026_events" in store.dynamic_tables
    assert "device_metrics" in DYNAMIC_SQL_TABLES
    assert "t_2026_events" in DYNAMIC_SQL_TABLES

    desc = json.loads(reg.execute("sql.describe_tables", {}))
    assert "device_metrics" in desc["tables"]
    assert "t_2026_events" in desc["tables"]

    rows_csv = store.execute_select(
        "SELECT device_id, latency_ms FROM device_metrics WHERE latency_ms > 10.0"
    )
    assert len(rows_csv) == 1
    assert rows_csv[0]["device_id"] == "dev-1"
    assert isinstance(rows_csv[0]["latency_ms"], float)

    rows_jsonl = store.execute_select("SELECT SUM(count) AS total_count FROM t_2026_events")
    assert rows_jsonl[0]["total_count"] == 18


def test_load_dataset_csv_jsonl_sqlite_and_allowlist_enforcement(tmp_path: Path) -> None:
    clear_dynamic_sql_tables()
    reg = ToolRegistry()
    store = register_sql_tools(reg, workspace_dir=tmp_path)
    pipeline = MiddlewarePipeline([SQLTableAllowlistMiddleware()])

    # 1. CSV dataset via sql.load_dataset
    csv_path = tmp_path / "bench_runs.csv"
    csv_path.write_text("run_id,tokens_per_sec\nrun_a,44.2\nrun_b,51.8\n", encoding="utf-8")
    load_csv_res = json.loads(
        reg.execute("sql.load_dataset", {"path": "bench_runs.csv", "table_name": "custom_runs"})
    )
    assert load_csv_res["table_name"] == "custom_runs"
    assert load_csv_res["row_count"] == 2
    assert load_csv_res["columns"] == ["run_id", "tokens_per_sec"]

    # 2. JSONL dataset via sql.load_dataset
    jsonl_path = tmp_path / "traces.jsonl"
    jsonl_path.write_text(
        '{"step": 1, "tool": "sql.query_workspace"}\n{"step": 2, "tool": "kb.search_workspace"}\n',
        encoding="utf-8",
    )
    load_jsonl_res = json.loads(reg.execute("sql.load_dataset", {"path": "traces.jsonl"}))
    assert load_jsonl_res["table_name"] == "traces"
    assert load_jsonl_res["row_count"] == 2

    # 3. SQLite dataset via sql.load_dataset
    sqlite_path = tmp_path / "edge_store.sqlite"
    with sqlite3.connect(str(sqlite_path)) as ext_conn:
        ext_conn.execute("CREATE TABLE sensors (sensor_id TEXT, temp_c REAL)")
        ext_conn.executemany(
            "INSERT INTO sensors VALUES (?, ?)",
            [("s1", 36.5), ("s2", 41.0)],
        )
        ext_conn.commit()

    load_sqlite_res = json.loads(reg.execute("sql.load_dataset", {"path": "edge_store.sqlite"}))
    assert load_sqlite_res["table_name"] == "edge_store"
    assert load_sqlite_res["row_count"] == 2

    # 4. Query dynamically loaded tables through SQLTableAllowlistMiddleware
    for query in [
        "SELECT run_id, tokens_per_sec FROM custom_runs WHERE tokens_per_sec > 50",
        "SELECT step, tool FROM traces ORDER BY step",
        "SELECT sensor_id, temp_c FROM edge_store WHERE temp_c > 40",
        "SELECT sensor_id, temp_c FROM sensors WHERE temp_c > 40",
    ]:
        ctx = _make_tool_ctx(reg, tmp_path, "sql.query_workspace", {"query": query})
        raw_out = pipeline.run_tool(ctx, lambda c: reg.execute(c.invocation.name, c.invocation.arguments))
        parsed = json.loads(raw_out)
        assert parsed["row_count"] >= 1

    # 5. Confirm unlisted table `secret_users` is still blocked by SQLTableAllowlistMiddleware
    blocked_ctx = _make_tool_ctx(
        reg,
        tmp_path,
        "sql.query_workspace",
        {"query": "SELECT * FROM secret_users"},
    )
    blocked_out = pipeline.run_tool(
        blocked_ctx,
        lambda c: reg.execute(c.invocation.name, c.invocation.arguments),
    )
    assert "[SQL_ALLOWLIST_ERROR]" in blocked_out
    assert "secret_users" in blocked_out

    # Ensure core tables remain protected from overwrite
    with pytest.raises(ValueError, match="protected core SQL table"):
        store.ingest_file("bench_runs.csv", table_name="ablation_results")


def test_path_traversal_rejected_on_load_dataset(tmp_path: Path) -> None:
    clear_dynamic_sql_tables()
    outside_csv = tmp_path.parent / "outside.csv"
    outside_csv.write_text("secret,value\nkey,123\n", encoding="utf-8")
    try:
        reg = ToolRegistry()
        store = WorkspaceSQLStore(workspace_dir=tmp_path)
        register_sql_tools(reg, store=store, workspace_dir=tmp_path)

        with pytest.raises(ValueError, match="escapes workspace boundary"):
            store.ingest_file("../outside.csv")

        with pytest.raises(ValueError, match="escapes workspace boundary"):
            reg.execute("sql.load_dataset", {"path": "../outside.csv"})
    finally:
        if outside_csv.exists():
            outside_csv.unlink()
