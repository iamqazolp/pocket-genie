from __future__ import annotations

import json
import sqlite3
from typing import Any

from pocket_genie.prompts.supervisor_prompt import load_sql_table_registry
from pocket_genie.registry.tools import ToolRegistry


class WorkspaceSQLStore:
    """In-memory SQLite analytics engine populated with quantization and session telemetry tables."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._bootstrap_tables()

    def _bootstrap_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE ablation_results (
                recipe_id TEXT,
                avg_bits_linear REAL,
                footprint_mb REAL,
                wikitext2_ppl REAL,
                mode TEXT,
                json_valid_rate REAL,
                values_exact_rate REAL
            );
            CREATE TABLE failure_taxonomy (
                recipe_id TEXT,
                mode TEXT,
                scenario TEXT,
                total INTEGER,
                exact INTEGER,
                corrupted_json INTEGER,
                wrong_tool INTEGER,
                invalid_schema INTEGER,
                value_mismatch INTEGER
            );
            CREATE TABLE layer_telemetry (
                layer_idx INTEGER,
                projection TEXT,
                kurtosis_all REAL,
                kurtosis_no_bos REAL,
                max_abs_all REAL,
                max_abs_no_bos REAL,
                outlier_ratio REAL
            );
            CREATE TABLE session_events (
                session_id TEXT,
                turn_idx INTEGER,
                specialist TEXT,
                tier TEXT,
                profile TEXT,
                effective_act_bits REAL,
                kv_saved_mb REAL
            );
            """
        )
        cur.executemany(
            "INSERT INTO ablation_results VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("E0", 32.0, 988.0, 16.07, "unconstrained", 0.991, 0.709),
                ("E1", 8.0, 494.1, 16.42, "constrained", 1.000, 0.698),
                ("E2", 4.0, 315.1, 4347.52, "unconstrained", 0.000, 0.000),
                ("E2a", 4.0, 315.1, 37.71, "constrained", 1.000, 0.665),
                ("E4s", 4.0, 315.1, 70.87, "constrained", 1.000, 0.641),
            ],
        )
        cur.executemany(
            "INSERT INTO failure_taxonomy VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("E0", "unconstrained", "iot", 42, 20, 1, 2, 4, 15),
                ("E2", "unconstrained", "iot", 42, 0, 42, 0, 0, 0),
                ("E4s", "constrained", "iot", 42, 27, 0, 1, 2, 12),
                ("E2a", "constrained", "iot", 42, 29, 0, 1, 1, 11),
            ],
        )
        cur.executemany(
            "INSERT INTO layer_telemetry VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (0, "mlp.down_proj", 542.6, 541.3, 94.2, 93.8, 14.2),
                (6, "mlp.up_proj", 184.1, 19.4, 46.6, 19.9, 5.4),
                (18, "mlp.down_proj", 218.5, 215.9, 68.4, 67.9, 10.8),
                (23, "mlp.down_proj", 312.4, 310.8, 119.6, 118.9, 12.6),
            ],
        )
        self._conn.commit()

    def execute_select(self, query: str) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def register_sql_tools(registry: ToolRegistry, store: WorkspaceSQLStore | None = None) -> WorkspaceSQLStore:
    sql_store = store or WorkspaceSQLStore()

    @registry.register(
        name="sql.query_workspace",
        description="Execute a read-only SQL SELECT query over allowlisted workspace analytics tables.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Read-only SQL SELECT statement over ablation_results, failure_taxonomy, layer_telemetry, or session_events.",
                },
            },
            "required": ["query"],
        },
        domain="sql",
        mutating=False,
    )
    def query_workspace(query: str) -> str:
        rows = sql_store.execute_select(query)
        return json.dumps({"row_count": len(rows), "rows": rows}, indent=2)

    @registry.register(
        name="sql.describe_tables",
        description="List available SQL tables, descriptions, and column names.",
        parameters={"type": "object", "properties": {}},
        domain="sql",
        mutating=False,
    )
    def describe_tables() -> str:
        return json.dumps(load_sql_table_registry(), indent=2)

    return sql_store
