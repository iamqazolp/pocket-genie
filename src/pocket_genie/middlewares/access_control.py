from __future__ import annotations

import re
from typing import Callable

from pocket_genie.middlewares.base import (
    BaseMiddleware,
    ModelCallContext,
    ToolCallContext,
)
from pocket_genie.prompts.supervisor_prompt import load_sql_table_registry

_SQL_TABLE_RE = re.compile(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
_SQL_CTE_RE = re.compile(r"\b(?:WITH|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(", re.IGNORECASE)
_FORBIDDEN_SQL_RE = re.compile(
    r"\b(?:DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|ATTACH|DETACH)\b",
    re.IGNORECASE,
)

DYNAMIC_SQL_TABLES: set[str] = set()


def register_dynamic_sql_table(table_name: str) -> None:
    """Register a dynamically loaded workspace SQL table in the runtime allowlist."""
    cleaned = table_name.strip().lower()
    if cleaned:
        DYNAMIC_SQL_TABLES.add(cleaned)


def clear_dynamic_sql_tables() -> None:
    """Clear all dynamically registered SQL tables."""
    DYNAMIC_SQL_TABLES.clear()


class ProjectScopedToolsMiddleware(BaseMiddleware):
    """Filters out project-scoped or mutating tools when session context lacks project authorization."""

    name = "project_scoped_tools"

    def before_model(self, ctx: ModelCallContext) -> None:
        filtered = []
        removed: list[str] = []
        for spec in ctx.tools:
            if spec.project_scoped and not ctx.shared.project_id:
                removed.append(spec.name)
                continue
            if ctx.shared.agent_type == "scoped-demo" and spec.mutating:
                removed.append(spec.name)
                continue
            filtered.append(spec)
        if removed:
            ctx.tools[:] = filtered
            event = {"middleware": self.name, "filtered_tools": removed}
            ctx.state.middleware_trace.append(event)
            ctx.writer.emit("middleware_event", **event)


class SQLTableAllowlistMiddleware(BaseMiddleware):
    """Enforces read-only SQL semantics and table allowlist boundaries."""

    name = "sql_table_allowlist"

    def __init__(self, allowed_tables: set[str] | None = None) -> None:
        reg = load_sql_table_registry()
        self._all_tables = set(reg.get("tables", {}).keys())
        self._demo_tables = set(reg.get("scoped_demo_allowlist", []))
        self._custom_allowed = allowed_tables

    def wrap_tool(
        self,
        ctx: ToolCallContext,
        call_next: Callable[[ToolCallContext], str],
    ) -> str:
        if not ctx.invocation.name.startswith("sql."):
            return call_next(ctx)

        query = str(ctx.invocation.arguments.get("query", ""))
        if _FORBIDDEN_SQL_RE.search(query):
            event = {
                "middleware": self.name,
                "blocked_query": query,
                "reason": "mutating_sql_forbidden",
            }
            ctx.state.middleware_trace.append(event)
            ctx.writer.emit("middleware_event", **event)
            return "[SQL_ALLOWLIST_ERROR] Only read-only SELECT queries are permitted."

        allowed = (
            self._custom_allowed
            if self._custom_allowed is not None
            else (
                (
                    self._demo_tables
                    if ctx.shared.agent_type == "scoped-demo"
                    else self._all_tables
                )
                | DYNAMIC_SQL_TABLES
            )
        )
        referenced = {m.group(1).lower() for m in _SQL_TABLE_RE.finditer(query)}
        cte_names = {m.group(1).lower() for m in _SQL_CTE_RE.finditer(query)}
        allowed_lower = {t.lower() for t in allowed}
        disallowed = sorted((referenced - cte_names) - allowed_lower)
        if disallowed:
            event = {
                "middleware": self.name,
                "blocked_tables": disallowed,
                "allowed_tables": sorted(allowed_lower),
            }
            ctx.state.middleware_trace.append(event)
            ctx.writer.emit("middleware_event", **event)
            return (
                f"[SQL_ALLOWLIST_ERROR] Access denied for table(s): {', '.join(disallowed)}. "
                f"Permitted tables: {', '.join(sorted(allowed_lower))}."
            )

        return call_next(ctx)
