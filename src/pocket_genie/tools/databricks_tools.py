from __future__ import annotations

import configparser
import dataclasses
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from pocket_genie.registry.tools import ToolRegistry

_ALLOWED_SQL_PREFIXES = ("SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN")
_FORBIDDEN_SQL_RE = re.compile(
    r"\b(?:DROP|DELETE|INSERT|UPDATE|MERGE|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|COPY)\b",
    re.IGNORECASE,
)

UNCONFIGURED_HINT = (
    "Databricks credentials not configured. Please supply host and token via explicit args, "
    "DATABRICKS_HOST/DATABRICKS_TOKEN environment variables, or ~/.databrickscfg."
)


def _extract_warehouse_id(raw_path_or_id: str | None) -> str:
    if not raw_path_or_id:
        return ""
    cleaned = raw_path_or_id.strip()
    if "/warehouses/" in cleaned:
        return cleaned.split("/warehouses/")[-1].strip("/")
    if "/endpoints/" in cleaned:
        return cleaned.split("/endpoints/")[-1].strip("/")
    if "/" in cleaned:
        return cleaned.split("/")[-1].strip("/")
    return cleaned


@dataclasses.dataclass(slots=True)
class DatabricksConfig:
    host: str
    token: str
    warehouse_id: str = ""
    profile: str = "DEFAULT"
    source: str = "env"

    def __post_init__(self) -> None:
        if self.host:
            h = self.host.strip()
            if not (h.startswith("https://") or h.startswith("http://")):
                h = f"https://{h}"
            self.host = h.rstrip("/")
        if self.token:
            self.token = self.token.strip()
        if self.warehouse_id:
            self.warehouse_id = _extract_warehouse_id(self.warehouse_id)

    @property
    def is_configured(self) -> bool:
        return bool(self.host and self.token)


def resolve_databricks_config(
    host: str | None = None,
    token: str | None = None,
    warehouse_id: str | None = None,
    config_path: str | Path | None = None,
    profile: str | None = None,
) -> DatabricksConfig | None:
    """Resolve Databricks configuration with precedence:
    1. Explicit arguments
    2. DATABRICKS_HOST / DATABRICKS_TOKEN / DATABRICKS_WAREHOUSE_ID / DATABRICKS_HTTP_PATH
    3. ~/.databrickscfg INI file via configparser.
    """
    if host and token:
        return DatabricksConfig(
            host=host,
            token=token,
            warehouse_id=warehouse_id or "",
            profile=profile or "DEFAULT",
            source="explicit",
        )

    env_host = os.environ.get("DATABRICKS_HOST", "").strip()
    env_token = os.environ.get("DATABRICKS_TOKEN", "").strip()
    env_wh = (
        os.environ.get("DATABRICKS_WAREHOUSE_ID", "").strip()
        or _extract_warehouse_id(os.environ.get("DATABRICKS_HTTP_PATH", "").strip())
    )

    eff_host = host or env_host
    eff_token = token or env_token
    eff_wh = warehouse_id or env_wh

    if eff_host and eff_token:
        return DatabricksConfig(
            host=eff_host,
            token=eff_token,
            warehouse_id=eff_wh,
            profile=profile or "DEFAULT",
            source="explicit" if (host and token) else "env",
        )

    target_path = (
        Path(config_path).expanduser()
        if config_path
        else Path("~/.databrickscfg").expanduser()
    )
    if target_path.exists() and target_path.is_file():
        parser = configparser.ConfigParser()
        try:
            parser.read(target_path, encoding="utf-8")
            prof = profile or os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")

            target_sec_name = None
            if prof in parser._sections or prof.upper() == "DEFAULT":
                target_sec_name = prof
            else:
                for sec in parser.sections():
                    if sec.lower() == prof.lower():
                        target_sec_name = sec
                        break

            local_sec: dict[str, str] = {}
            if target_sec_name and target_sec_name in parser._sections:
                local_sec = dict(parser._sections[target_sec_name])
            elif target_sec_name and target_sec_name.upper() == "DEFAULT":
                local_sec = dict(parser.defaults())

            default_sec: dict[str, str] = dict(parser.defaults())

            cfg_host = (
                local_sec.get("host", "").strip().strip("\"'")
                or default_sec.get("host", "").strip().strip("\"'")
            )
            cfg_token = (
                local_sec.get("token", "").strip().strip("\"'")
                or default_sec.get("token", "").strip().strip("\"'")
            )
            cfg_wh = (
                local_sec.get("warehouse_id", "").strip().strip("\"'")
                or _extract_warehouse_id(local_sec.get("http_path", "").strip().strip("\"'"))
                or default_sec.get("warehouse_id", "").strip().strip("\"'")
                or _extract_warehouse_id(default_sec.get("http_path", "").strip().strip("\"'"))
            )

            final_host = eff_host or cfg_host
            final_token = eff_token or cfg_token
            final_wh = eff_wh or cfg_wh

            if final_host and final_token:
                return DatabricksConfig(
                    host=final_host,
                    token=final_token,
                    warehouse_id=final_wh,
                    profile=prof,
                    source="file",
                )
        except Exception:
            pass

    return None


class DatabricksWorkspaceClient:
    """Zero-dependency Databricks REST client with pluggable transport for testing."""

    def __init__(
        self,
        config: DatabricksConfig | None = None,
        config_path: str | Path | None = None,
        transport: Callable[
            [str, str, dict[str, Any] | None, dict[str, str], float],
            dict[str, Any],
        ]
        | None = None,
    ) -> None:
        self.config: DatabricksConfig | None = (
            config if config is not None else resolve_databricks_config(config_path=config_path)
        )
        self._transport = transport or self._default_transport

    @property
    def is_configured(self) -> bool:
        return bool(self.config and self.config.is_configured)

    def _unconfigured_response(self) -> dict[str, Any]:
        return {
            "configured": False,
            "hint": UNCONFIGURED_HINT,
        }

    def _default_transport(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        data_bytes = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                if not raw.strip():
                    return {}
                return dict(json.loads(raw))
        except urllib.error.HTTPError as exc:
            err_content = exc.read().decode("utf-8", errors="replace")
            try:
                err_json = json.loads(err_content)
            except Exception:
                err_json = {"raw": err_content}
            return {
                "error": f"HTTP {exc.code}: {exc.reason}",
                "status_code": exc.code,
                "details": err_json,
            }
        except urllib.error.URLError as exc:
            return {"error": f"Network error: {exc.reason}"}
        except Exception as exc:
            return {"error": str(exc)}

    def _request(
        self,
        method: str,
        path: str,
        query_params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        if not self.is_configured or self.config is None:
            return self._unconfigured_response()

        url = f"{self.config.host}{path}"
        if query_params:
            filtered = {
                k: v for k, v in query_params.items() if v is not None and v != ""
            }
            if filtered:
                encoded = urllib.parse.urlencode(filtered)
                url = f"{url}?{encoded}"

        headers = {
            "Authorization": f"Bearer {self.config.token}",
            "Content-Type": "application/json",
            "User-Agent": "pocket-genie/0.3.0",
        }
        return self._transport(method, url, body, headers, timeout)

    def list_jobs(self, limit: int = 15, name_filter: str = "") -> dict[str, Any]:
        """Fetch Databricks jobs via GET /api/2.1/jobs/list."""
        if not self.is_configured:
            return self._unconfigured_response()

        params: dict[str, Any] = {"limit": limit}
        if name_filter:
            params["name"] = name_filter

        resp = self._request("GET", "/api/2.1/jobs/list", query_params=params)
        if not isinstance(resp, dict):
            return {"configured": True, "jobs": [], "total_count": 0}
        if "error" in resp:
            return resp

        jobs = resp.get("jobs", [])
        if name_filter:
            nf_lower = name_filter.lower()
            jobs = [
                j
                for j in jobs
                if nf_lower in str((j.get("settings") or {}).get("name", "")).lower()
                or nf_lower in str(j.get("name", "")).lower()
            ]

        out = dict(resp)
        out["jobs"] = jobs
        out["total_count"] = len(jobs)
        out.setdefault("configured", True)
        return out

    def job_runs(
        self,
        job_id: int = 0,
        active_only: bool = False,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Fetch Databricks job runs via GET /api/2.1/jobs/runs/list with failed_runs_count."""
        if not self.is_configured:
            return self._unconfigured_response()

        params: dict[str, Any] = {"limit": limit}
        if job_id:
            params["job_id"] = job_id
        if active_only:
            params["active_only"] = "true"

        resp = self._request("GET", "/api/2.1/jobs/runs/list", query_params=params)
        if not isinstance(resp, dict):
            return {"configured": True, "runs": [], "failed_runs_count": 0}
        if "error" in resp:
            return resp

        runs = resp.get("runs", [])
        failed_count = sum(
            1
            for r in runs
            if str((r.get("state") or {}).get("result_state", "")).upper() == "FAILED"
            or str(r.get("result_state", "")).upper() == "FAILED"
        )

        out = dict(resp)
        out["runs"] = runs
        out["failed_runs_count"] = failed_count
        out["total_count"] = len(runs)
        out.setdefault("configured", True)
        return out

    def clusters_status(self, limit: int = 10) -> dict[str, Any]:
        """Fetch Databricks clusters status via GET /api/2.0/clusters/list."""
        if not self.is_configured:
            return self._unconfigured_response()

        resp = self._request("GET", "/api/2.0/clusters/list")
        if not isinstance(resp, dict):
            return {"configured": True, "clusters": [], "total_count": 0}
        if "error" in resp:
            return resp

        clusters = resp.get("clusters", [])
        if limit > 0:
            clusters = clusters[:limit]

        out = dict(resp)
        out["clusters"] = clusters
        out["total_count"] = len(clusters)
        out.setdefault("configured", True)
        return out


def register_databricks_tools(
    registry: ToolRegistry,
    client: DatabricksWorkspaceClient | None = None,
) -> DatabricksWorkspaceClient:
    """Register Databricks workspace inspection and SQL querying tools into registry."""
    databricks_client = client or DatabricksWorkspaceClient()

    @registry.register(
        name="databricks.list_jobs",
        description="List Databricks jobs in the workspace with optional name filtering.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of jobs to return (default: 15).",
                },
                "name_filter": {
                    "type": "string",
                    "description": "Optional substring filter on job name.",
                },
            },
        },
        domain="databricks",
        mutating=False,
    )
    def list_jobs(limit: int = 15, name_filter: str = "") -> str:
        res = databricks_client.list_jobs(limit=limit, name_filter=name_filter)
        return json.dumps(res, indent=2)

    @registry.register(
        name="databricks.job_runs",
        description="List recent Databricks job runs with failure count summary.",
        parameters={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "integer",
                    "description": "Optional job ID to filter runs for.",
                },
                "active_only": {
                    "type": "boolean",
                    "description": "If True, return only currently active job runs.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of runs to return (default: 10).",
                },
            },
        },
        domain="databricks",
        mutating=False,
    )
    def job_runs(job_id: int = 0, active_only: bool = False, limit: int = 10) -> str:
        res = databricks_client.job_runs(job_id=job_id, active_only=active_only, limit=limit)
        return json.dumps(res, indent=2)

    @registry.register(
        name="databricks.clusters_status",
        description="Inspect Databricks interactive and automated cluster runtime states.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of clusters to inspect (default: 10).",
                },
            },
        },
        domain="databricks",
        mutating=False,
    )
    def clusters_status(limit: int = 10) -> str:
        res = databricks_client.clusters_status(limit=limit)
        return json.dumps(res, indent=2)

    return dbx
