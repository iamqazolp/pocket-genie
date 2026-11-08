from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from pocket_genie.middlewares.access_control import register_dynamic_sql_table
from pocket_genie.prompts.supervisor_prompt import load_sql_table_registry
from pocket_genie.registry.tools import ToolRegistry

_CORE_TABLES = {
    "ablation_results",
    "failure_taxonomy",
    "layer_telemetry",
    "session_events",
}
_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "benchmarks",
}
_MAX_AUTODISCOVER_BYTES = 5 * 1024 * 1024


def _sanitize_identifier(raw: str, prefix: str = "t_") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", raw.strip().lower()).strip("_")
    if not cleaned:
        cleaned = "dataset"
    if cleaned[0].isdigit():
        cleaned = f"{prefix}{cleaned}"
    return cleaned


def _coerce_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return str(value)


class WorkspaceSQLStore:
    """In-memory SQLite analytics engine populated with empirical benchmark artifacts and quantization tables."""

    def __init__(self, workspace_dir: Path | None = None) -> None:
        self._workspace_dir = workspace_dir
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.dynamic_tables: dict[str, dict[str, Any]] = {}
        self._bootstrap_tables()
        self._ingest_latest_eval_artifact()
        if self._workspace_dir is not None:
            self._autodiscover_workspace_datasets()

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

    def _ingest_latest_eval_artifact(self) -> None:
        candidates: list[Path] = []
        if self._workspace_dir is not None:
            candidates.append(self._workspace_dir / "benchmarks" / "latest_eval.json")
        repo_root = Path(__file__).resolve().parents[3]
        candidates.append(repo_root / "benchmarks" / "latest_eval.json")

        artifact_path = next((p for p in candidates if p.exists() and p.is_file()), None)
        if artifact_path is None:
            return
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except Exception:
            return

        cur = self._conn.cursor()
        for r in payload.get("recipes") or []:
            cur.execute(
                "INSERT INTO ablation_results VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(r.get("recipe_id", "eval")),
                    float(r.get("avg_act_bits", 8.0)),
                    float(r.get("footprint_mb", 315.1)),
                    0.0,
                    str(r.get("mode", "constrained")),
                    float(r.get("json_valid_rate", 0.0)),
                    float(r.get("tool_exact_rate", 0.0)),
                ),
            )
            cur.execute(
                "INSERT INTO failure_taxonomy VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(r.get("recipe_id", "eval")),
                    str(r.get("mode", "constrained")),
                    "multi_domain_30",
                    int(r.get("total", 0)),
                    int(r.get("tool_exact", 0)),
                    int(r.get("corrupted_json", 0)),
                    int(r.get("wrong_tool", 0)),
                    int(r.get("invalid_schema", 0)),
                    0,
                ),
            )
        for lyr in payload.get("top_sensitive_layers") or []:
            kurt = float(lyr.get("kurtosis_all", 0.0))
            cur.execute(
                "INSERT INTO layer_telemetry VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    int(lyr.get("layer_idx", 0)),
                    str(lyr.get("projection", "linear")),
                    kurt,
                    kurt,
                    0.0,
                    0.0,
                    float(lyr.get("mse_w4a8", 0.0)),
                ),
            )
        self._conn.commit()

    def _autodiscover_workspace_datasets(self) -> None:
        if self._workspace_dir is None or not self._workspace_dir.exists():
            return
        ws_root = self._workspace_dir.resolve()
        search_dirs: list[Path] = [ws_root]
        for child in sorted(ws_root.iterdir()):
            if child.is_dir() and not child.name.startswith(".") and child.name not in _SKIP_DIRS:
                search_dirs.append(child)
                for grandchild in sorted(child.iterdir()):
                    if (
                        grandchild.is_dir()
                        and not grandchild.name.startswith(".")
                        and grandchild.name not in _SKIP_DIRS
                    ):
                        search_dirs.append(grandchild)

        seen_files: set[Path] = set()
        for folder in search_dirs:
            for item in sorted(folder.iterdir()):
                if not item.is_file() or item.name.startswith("."):
                    continue
                resolved = item.resolve()
                if resolved in seen_files:
                    continue
                seen_files.add(resolved)
                ext = item.suffix.lower()
                if ext not in {".csv", ".jsonl", ".sqlite", ".sqlite3", ".db"}:
                    continue
                try:
                    if item.stat().st_size > _MAX_AUTODISCOVER_BYTES:
                        continue
                    rel_path = resolved.relative_to(ws_root)
                    self.ingest_file(rel_path)
                except Exception:
                    continue

    def _resolve_and_validate_path(self, file_path: str | Path) -> Path:
        raw_path = Path(file_path)
        if self._workspace_dir is not None:
            ws_root = self._workspace_dir.resolve()
            resolved = raw_path.resolve() if raw_path.is_absolute() else (ws_root / raw_path).resolve()
            if not resolved.is_relative_to(ws_root):
                raise ValueError(
                    f"Dataset path '{file_path}' escapes workspace boundary '{ws_root}'."
                )
        else:
            if ".." in raw_path.parts:
                raise ValueError(f"Path traversal is not permitted in dataset path '{file_path}'.")
            resolved = raw_path.resolve()
        if not resolved.exists() or not resolved.is_file():
            raise ValueError(f"Dataset file does not exist: {file_path}")
        return resolved

    def _load_rows_into_table(
        self,
        sanitized_name: str,
        fieldnames: list[str],
        raw_rows: list[dict[str, Any]],
        source_str: str,
    ) -> dict[str, Any]:
        if sanitized_name in _CORE_TABLES:
            raise ValueError(f"Cannot overwrite protected core SQL table: {sanitized_name}")

        col_map: list[tuple[str, str]] = []
        seen_cols: set[str] = set()
        for raw_col in fieldnames:
            base_col = _sanitize_identifier(str(raw_col), prefix="c_")
            col_name = base_col
            suffix = 2
            while col_name in seen_cols:
                col_name = f"{base_col}_{suffix}"
                suffix += 1
            seen_cols.add(col_name)
            col_map.append((str(raw_col), col_name))

        if not col_map:
            raise ValueError(f"Dataset '{source_str}' contains no columns.")

        coerced_columns: dict[str, list[Any]] = {}
        col_types: dict[str, str] = {}
        for raw_col, col_name in col_map:
            parsed = [_coerce_cell(row.get(raw_col)) for row in raw_rows]
            non_null = [v for v in parsed if v is not None]
            if not non_null:
                col_types[col_name] = "TEXT"
                coerced_columns[col_name] = parsed
            elif all(isinstance(v, int) and not isinstance(v, bool) for v in non_null):
                col_types[col_name] = "INTEGER"
                coerced_columns[col_name] = parsed
            elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
                col_types[col_name] = "REAL"
                coerced_columns[col_name] = [float(v) if v is not None else None for v in parsed]
            else:
                col_types[col_name] = "TEXT"
                coerced_columns[col_name] = [str(v) if v is not None else None for v in parsed]

        cur = self._conn.cursor()
        cur.execute(f'DROP TABLE IF EXISTS "{sanitized_name}"')
        col_defs = ", ".join(f'"{col_name}" {col_types[col_name]}' for _, col_name in col_map)
        cur.execute(f'CREATE TABLE "{sanitized_name}" ({col_defs})')
        if raw_rows:
            placeholders = ", ".join("?" for _ in col_map)
            insert_sql = f'INSERT INTO "{sanitized_name}" VALUES ({placeholders})'
            rows_to_insert = [
                tuple(coerced_columns[col_name][idx] for _, col_name in col_map)
                for idx in range(len(raw_rows))
            ]
            cur.executemany(insert_sql, rows_to_insert)
        self._conn.commit()

        columns = [col_name for _, col_name in col_map]
        row_count = len(raw_rows)
        register_dynamic_sql_table(sanitized_name)
        self.dynamic_tables[sanitized_name] = {
            "description": f"Dynamically loaded workspace dataset from {source_str} ({row_count} rows).",
            "columns": columns,
            "row_count": row_count,
            "source_path": source_str,
        }
        return {
            "table_name": sanitized_name,
            "columns": columns,
            "row_count": row_count,
            "source": source_str,
        }

    def ingest_file(self, file_path: str | Path, table_name: str | None = None) -> dict[str, Any]:
        resolved = self._resolve_and_validate_path(file_path)
        sanitized_name = _sanitize_identifier(table_name if table_name else resolved.stem, prefix="t_")
        if sanitized_name in _CORE_TABLES:
            raise ValueError(f"Cannot overwrite protected core SQL table: {sanitized_name}")

        ext = resolved.suffix.lower()
        source_str = str(file_path)

        if ext == ".csv":
            with resolved.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                raw_rows = [dict(r) for r in reader]
            return self._load_rows_into_table(sanitized_name, fieldnames, raw_rows, source_str)

        if ext == ".jsonl":
            raw_rows = []
            fieldnames_list: list[str] = []
            seen_keys: set[str] = set()
            for line in resolved.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                if not isinstance(obj, dict):
                    raise ValueError(f"JSONL line in '{file_path}' is not a JSON object.")
                for k in obj.keys():
                    if k not in seen_keys:
                        seen_keys.add(k)
                        fieldnames_list.append(str(k))
                raw_rows.append(obj)
            return self._load_rows_into_table(sanitized_name, fieldnames_list, raw_rows, source_str)

        if ext == ".json":
            payload = json.loads(resolved.read_text(encoding="utf-8", errors="replace"))
            records: list[dict[str, Any]] | None = None
            if isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
                records = payload
            elif isinstance(payload, dict):
                for val in payload.values():
                    if isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
                        records = val
                        break
            if records is None:
                raise ValueError(f"JSON file '{file_path}' does not contain a tabular list of objects.")
            fieldnames_list = []
            seen_keys = set()
            for obj in records:
                for k in obj.keys():
                    if k not in seen_keys:
                        seen_keys.add(k)
                        fieldnames_list.append(str(k))
            return self._load_rows_into_table(sanitized_name, fieldnames_list, records, source_str)

        if ext in {".sqlite", ".sqlite3", ".db"}:
            with sqlite3.connect(str(resolved)) as ext_conn:
                ext_conn.row_factory = sqlite3.Row
                ext_cur = ext_conn.cursor()
                ext_cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
                src_tables = [str(r["name"]) for r in ext_cur.fetchall()]
                if not src_tables:
                    raise ValueError(f"SQLite file '{file_path}' contains no user tables.")

                primary_src = src_tables[0]
                for st in src_tables:
                    if _sanitize_identifier(st, prefix="t_") == sanitized_name:
                        primary_src = st
                        break

                primary_result: dict[str, Any] | None = None
                for st in src_tables:
                    ext_cur.execute(f'SELECT * FROM "{st}"')
                    fetched = ext_cur.fetchall()
                    cols = [desc[0] for desc in (ext_cur.description or [])]
                    rows_dicts = [dict(r) for r in fetched]
                    st_sanitized = _sanitize_identifier(st, prefix="t_")
                    if st == primary_src:
                        primary_result = self._load_rows_into_table(
                            sanitized_name, cols, rows_dicts, source_str
                        )
                    if st_sanitized != sanitized_name and st_sanitized not in _CORE_TABLES:
                        self._load_rows_into_table(st_sanitized, cols, rows_dicts, source_str)

                assert primary_result is not None
                return primary_result

        raise ValueError(f"Unsupported dataset file extension '{ext}' for '{file_path}'.")

    def describe_all_tables(self) -> dict[str, Any]:
        base = load_sql_table_registry()
        tables = dict(base.get("tables", {}))
        for name, meta in self.dynamic_tables.items():
            tables[name] = dict(meta)
        scoped_demo = list(base.get("scoped_demo_allowlist", []))
        for name in self.dynamic_tables:
            if name not in scoped_demo:
                scoped_demo.append(name)
        return {
            **base,
            "tables": tables,
            "scoped_demo_allowlist": scoped_demo,
            "dynamic_tables": dict(self.dynamic_tables),
        }

    def execute_select(self, query: str) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def register_sql_tools(
    registry: ToolRegistry,
    store: WorkspaceSQLStore | None = None,
    workspace_dir: Path | None = None,
) -> WorkspaceSQLStore:
    if store is not None and store._workspace_dir is None and workspace_dir is not None:
        store._workspace_dir = workspace_dir
    sql_store = store or WorkspaceSQLStore(workspace_dir=workspace_dir)

    @registry.register(
        name="sql.query_workspace",
        description="Execute a read-only SQL SELECT query over allowlisted workspace analytics tables.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Read-only SQL SELECT statement over ablation_results, failure_taxonomy, layer_telemetry, session_events, or dynamically loaded workspace tables.",
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
        return json.dumps(sql_store.describe_all_tables(), indent=2)

    @registry.register(
        name="sql.load_dataset",
        description="Load a workspace dataset (.csv, .jsonl, .json, .sqlite, .db) into the in-memory SQLite store and register it in the SQL allowlist.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to a .csv, .jsonl, .json, .sqlite, or .db file inside the workspace.",
                },
                "table_name": {
                    "type": "string",
                    "description": "Optional custom SQL table name for the loaded dataset.",
                },
            },
            "required": ["path"],
        },
        domain="sql",
        mutating=False,
    )
    def load_dataset(path: str, table_name: str | None = None) -> str:
        summary = sql_store.ingest_file(path, table_name=table_name or None)
        return json.dumps(summary, indent=2)

    return sql_store
