from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pocket_genie.registry.tools import ToolRegistry
from pocket_genie.tools.databricks_tools import (
    DatabricksConfig,
    DatabricksWorkspaceClient,
    register_databricks_tools,
    resolve_databricks_config,
    validate_readonly_databricks_sql,
)


def test_databricks_config_normalization() -> None:
    cfg = DatabricksConfig(
        host="my-workspace.cloud.databricks.com/",
        token="  dapi12345  ",
        warehouse_id="/sql/1.0/warehouses/wh-abc-123",
    )
    assert cfg.host == "https://my-workspace.cloud.databricks.com"
    assert cfg.token == "dapi12345"
    assert cfg.warehouse_id == "wh-abc-123"
    assert cfg.is_configured is True
    assert cfg.profile == "DEFAULT"
    assert cfg.source == "env"


def test_resolve_config_explicit_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_HOST", "https://env-host.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "env-token")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "env-wh")

    cfg = resolve_databricks_config(
        host="https://explicit-host.databricks.com",
        token="explicit-token",
        warehouse_id="explicit-wh",
        profile="production",
    )
    assert cfg is not None
    assert cfg.host == "https://explicit-host.databricks.com"
    assert cfg.token == "explicit-token"
    assert cfg.warehouse_id == "explicit-wh"
    assert cfg.profile == "production"
    assert cfg.source == "explicit"


def test_resolve_config_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_HOST", "https://env-only.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-env-token")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/endpoints/wh-endpoint-456")

    cfg = resolve_databricks_config()
    assert cfg is not None
    assert cfg.host == "https://env-only.databricks.com"
    assert cfg.token == "dapi-env-token"
    assert cfg.warehouse_id == "wh-endpoint-456"
    assert cfg.source == "env"


def test_resolve_config_databrickscfg_ini(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("DATABRICKS_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_HTTP_PATH", raising=False)

    ini_content = """[DEFAULT]
host = https://default-instance.databricks.com
token = dapi-default-token
warehouse_id = wh-default

[analytics]
host = https://analytics-instance.databricks.com
token = dapi-analytics-token
http_path = /sql/1.0/warehouses/wh-analytics
"""
    cfg_file = tmp_path / ".databrickscfg"
    cfg_file.write_text(ini_content, encoding="utf-8")

    cfg_default = resolve_databricks_config(config_path=cfg_file)
    assert cfg_default is not None
    assert cfg_default.host == "https://default-instance.databricks.com"
    assert cfg_default.token == "dapi-default-token"
    assert cfg_default.warehouse_id == "wh-default"
    assert cfg_default.source == "file"

    cfg_analytics = resolve_databricks_config(config_path=cfg_file, profile="analytics")
    assert cfg_analytics is not None
    assert cfg_analytics.host == "https://analytics-instance.databricks.com"
    assert cfg_analytics.token == "dapi-analytics-token"
    assert cfg_analytics.warehouse_id == "wh-analytics"
    assert cfg_analytics.profile == "analytics"
    assert cfg_analytics.source == "file"


def test_resolve_config_unconfigured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("DATABRICKS_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_HTTP_PATH", raising=False)

    non_existent = tmp_path / "absent.cfg"
    assert resolve_databricks_config(config_path=non_existent) is None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "SELECT id, name FROM default.users WHERE active = 1",
        "WITH cte AS (SELECT id FROM users) SELECT * FROM cte",
        "SHOW TABLES IN hive_metastore.default",
        "DESCRIBE TABLE EXTENDED diamonds",
        "DESC diamonds",
        "EXPLAIN SELECT * FROM diamonds",
        "-- leading comment\nSELECT * FROM tbl",
        "/* multi-line comment */ SELECT count(*) FROM tbl",
        "SELECT * FROM audit WHERE status = 'DROP'",
    ],
)
def test_validate_readonly_databricks_sql_allowed(sql: str) -> None:
    cleaned = validate_readonly_databricks_sql(sql)
    assert cleaned.strip() == sql.strip()


@pytest.mark.parametrize(
    "sql,forbidden_word",
    [
        ("DROP TABLE users", "DROP"),
        ("DELETE FROM users WHERE id = 1", "DELETE"),
        ("INSERT INTO users VALUES (1, 'Alice')", "INSERT"),
        ("UPDATE users SET name = 'Bob'", "UPDATE"),
        ("MERGE INTO target USING source ON target.id = source.id", "MERGE"),
        ("ALTER TABLE users ADD COLUMN age INT", "ALTER"),
        ("TRUNCATE TABLE users", "TRUNCATE"),
        ("CREATE TABLE users (id INT)", "CREATE"),
        ("GRANT SELECT ON TABLE users TO alice", "GRANT"),
        ("REVOKE SELECT ON TABLE users FROM alice", "REVOKE"),
        ("COPY INTO my_table FROM 's3://bucket/data'", "COPY"),
        ("SELECT * FROM users; DROP TABLE users", "DROP"),
        ("VACUUM users", "VACUUM"),
    ],
)
def test_validate_readonly_databricks_sql_forbidden(sql: str, forbidden_word: str) -> None:
    with pytest.raises(ValueError):
        validate_readonly_databricks_sql(sql)


def test_validate_readonly_sql_empty() -> None:
    with pytest.raises(ValueError, match="SQL statement cannot be empty"):
        validate_readonly_databricks_sql("   ")
    with pytest.raises(ValueError, match="contains no executable query"):
        validate_readonly_databricks_sql("-- only comments\n")


def test_unconfigured_client_clean_responses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    client = DatabricksWorkspaceClient(config_path=tmp_path / "none.cfg")
    assert client.is_configured is False

    jobs = client.list_jobs()
    assert jobs["configured"] is False
    assert "hint" in jobs

    runs = client.job_runs()
    assert runs["configured"] is False
    assert "hint" in runs

    clusters = client.clusters_status()
    assert clusters["configured"] is False
    assert "hint" in clusters

    sql = client.sql_query("SELECT 1")
    assert sql["configured"] is False
    assert "hint" in sql


def test_client_list_jobs_with_transport() -> None:
    captured: list[tuple[str, str, dict[str, Any] | None, dict[str, str], float]] = []

    def mock_transport(
        method: str,
        url: str,
        body: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        captured.append((method, url, body, headers, timeout))
        return {
            "jobs": [
                {"job_id": 101, "settings": {"name": "nightly_quant_eval"}},
                {"job_id": 102, "settings": {"name": "daytime_training"}},
                {"job_id": 103, "settings": {"name": "quant_ablation_bench"}},
            ],
            "has_more": False,
        }

    cfg = DatabricksConfig(host="https://dbc-test.cloud.databricks.com", token="dapi-test")
    client = DatabricksWorkspaceClient(config=cfg, transport=mock_transport)

    res = client.list_jobs(limit=15, name_filter="quant")
    assert len(captured) == 1
    method, url, body, headers, _timeout = captured[0]
    assert method == "GET"
    assert "https://dbc-test.cloud.databricks.com/api/2.1/jobs/list" in url
    assert "limit=15" in url
    assert "name=quant" in url
    assert headers["Authorization"] == "Bearer dapi-test"

    assert res["configured"] is True
    assert res["total_count"] == 2
    assert [j["job_id"] for j in res["jobs"]] == [101, 103]


def test_client_job_runs_with_failed_runs_count() -> None:
    captured: list[tuple[str, str, dict[str, Any] | None]] = []

    def mock_transport(
        method: str,
        url: str,
        body: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        captured.append((method, url, body))
        return {
            "runs": [
                {"run_id": 201, "job_id": 101, "state": {"result_state": "SUCCESS"}},
                {"run_id": 202, "job_id": 101, "state": {"result_state": "FAILED"}},
                {"run_id": 203, "job_id": 101, "state": {"result_state": "FAILED"}},
                {"run_id": 204, "job_id": 101, "state": {"life_cycle_state": "RUNNING"}},
            ],
            "has_more": False,
        }

    cfg = DatabricksConfig(host="https://dbc-test.cloud.databricks.com", token="dapi-test")
    client = DatabricksWorkspaceClient(config=cfg, transport=mock_transport)

    res = client.job_runs(job_id=101, active_only=True, limit=5)
    assert len(captured) == 1
    method, url, _body = captured[0]
    assert method == "GET"
    assert "job_id=101" in url
    assert "active_only=true" in url
    assert "limit=5" in url

    assert res["failed_runs_count"] == 2
    assert res["total_count"] == 4


def test_client_clusters_status_with_transport() -> None:
    def mock_transport(
        method: str,
        url: str,
        body: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        return {
            "clusters": [
                {"cluster_id": "c1", "cluster_name": "worker-1", "state": "RUNNING"},
                {"cluster_id": "c2", "cluster_name": "worker-2", "state": "TERMINATED"},
                {"cluster_id": "c3", "cluster_name": "worker-3", "state": "PENDING"},
            ]
        }

    cfg = DatabricksConfig(host="https://dbc-test.cloud.databricks.com", token="dapi-test")
    client = DatabricksWorkspaceClient(config=cfg, transport=mock_transport)

    res = client.clusters_status(limit=2)
    assert res["total_count"] == 2
    assert len(res["clusters"]) == 2
    assert res["clusters"][0]["cluster_id"] == "c1"


def test_client_sql_query_zips_manifest_columns_with_data_array() -> None:
    captured: list[tuple[str, str, dict[str, Any] | None]] = []

    def mock_transport(
        method: str,
        url: str,
        body: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        captured.append((method, url, body))
        return {
            "statement_id": "stmt-001",
            "status": {"state": "SUCCEEDED"},
            "manifest": {
                "schema": {
                    "column_count": 3,
                    "columns": [
                        {"name": "recipe_id", "type_text": "STRING"},
                        {"name": "avg_bits", "type_text": "DOUBLE"},
                        {"name": "exact_match", "type_text": "DOUBLE"},
                    ],
                }
            },
            "result": {
                "row_count": 2,
                "data_array": [
                    ["E0_baseline", 16.0, 70.9],
                    ["E4s_int4", 4.0, 64.1],
                ],
            },
        }

    cfg = DatabricksConfig(
        host="https://dbc-test.cloud.databricks.com",
        token="dapi-test",
        warehouse_id="default-wh-123",
    )
    client = DatabricksWorkspaceClient(config=cfg, transport=mock_transport)

    res = client.sql_query(
        statement="SELECT recipe_id, avg_bits, exact_match FROM quant_ablations",
        catalog="hive_metastore",
        schema="default",
        row_limit=25,
    )

    assert len(captured) == 1
    method, url, body = captured[0]
    assert method == "POST"
    assert url == "https://dbc-test.cloud.databricks.com/api/2.0/sql/statements"
    assert body is not None
    assert body["warehouse_id"] == "default-wh-123"
    assert body["catalog"] == "hive_metastore"
    assert body["schema"] == "default"
    assert body["row_limit"] == 25

    assert res["columns"] == ["recipe_id", "avg_bits", "exact_match"]
    assert res["row_count"] == 2
    assert res["rows"] == [
        {"recipe_id": "E0_baseline", "avg_bits": 16.0, "exact_match": 70.9},
        {"recipe_id": "E4s_int4", "avg_bits": 4.0, "exact_match": 64.1},
    ]


def test_client_sql_query_requires_warehouse_id() -> None:
    cfg = DatabricksConfig(host="https://dbc-test.cloud.databricks.com", token="dapi-test")
    client = DatabricksWorkspaceClient(config=cfg, transport=lambda *_: {})
    res = client.sql_query("SELECT 1")
    assert "error" in res
    assert "No warehouse_id provided" in res["error"]


def test_client_sql_query_rejects_mutating_statement() -> None:
    cfg = DatabricksConfig(
        host="https://dbc-test.cloud.databricks.com",
        token="dapi-test",
        warehouse_id="wh-1",
    )
    client = DatabricksWorkspaceClient(config=cfg, transport=lambda *_: {})
    with pytest.raises(ValueError, match="Forbidden mutating SQL keyword detected: DROP"):
        client.sql_query("DROP TABLE important_data")


def test_register_databricks_tools_and_execution() -> None:
    def mock_transport(
        method: str,
        url: str,
        body: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        if "jobs/list" in url:
            return {"jobs": [{"job_id": 1, "name": "quant-job"}]}
        if "jobs/runs/list" in url:
            return {"runs": [{"run_id": 10, "state": {"result_state": "FAILED"}}]}
        if "clusters/list" in url:
            return {"clusters": [{"cluster_id": "cl-1", "state": "RUNNING"}]}
        if "sql/statements" in url:
            return {
                "manifest": {"schema": {"columns": [{"name": "cnt"}]}},
                "result": {"data_array": [[42]]},
            }
        return {}

    cfg = DatabricksConfig(
        host="https://dbc-test.cloud.databricks.com",
        token="dapi-test",
        warehouse_id="wh-test",
    )
    client = DatabricksWorkspaceClient(config=cfg, transport=mock_transport)

    registry = ToolRegistry()
    ret_client = register_databricks_tools(registry, client=client)
    assert ret_client is client

    # Verify specs in registry
    tools = registry.list_tools(domain="databricks")
    names = [t.name for t in tools]
    assert "databricks.list_jobs" in names
    assert "databricks.job_runs" in names
    assert "databricks.clusters_status" in names
    assert "databricks.sql_query" in names

    for t in tools:
        assert t.domain == "databricks"
        assert t.mutating is False

    # Execute tools via registry
    jobs_json = registry.execute("databricks.list_jobs", {"limit": 10})
    jobs_data = json.loads(jobs_json)
    assert jobs_data["jobs"][0]["name"] == "quant-job"

    runs_json = registry.execute("databricks.job_runs", {"job_id": 1, "limit": 5})
    runs_data = json.loads(runs_json)
    assert runs_data["failed_runs_count"] == 1

    clusters_json = registry.execute("databricks.clusters_status", {"limit": 5})
    clusters_data = json.loads(clusters_json)
    assert clusters_data["clusters"][0]["cluster_id"] == "cl-1"

    sql_json = registry.execute("databricks.sql_query", {"statement": "SELECT 42 AS cnt"})
    sql_data = json.loads(sql_json)
    assert sql_data["rows"] == [{"cnt": 42}]


def test_register_databricks_tools_defaults_unconfigured_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    registry = ToolRegistry()
    client = register_databricks_tools(registry)
    assert client.is_configured is False

    raw_json = registry.execute("databricks.list_jobs", {})
    data = json.loads(raw_json)
    assert data["configured"] is False
    assert "hint" in data
