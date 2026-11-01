from __future__ import annotations

import dataclasses
from typing import Any, Callable

_TYPE_ALIASES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
    "tuple": "array",
    "any": "string",
}


def sanitize_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize JSON Schema nodes so grammar enforcers and strict validators never fail on untyped fields."""
    if not isinstance(schema, dict):
        return {"type": "string"}

    out: dict[str, Any] = dict(schema)
    raw_type = out.get("type")
    if isinstance(raw_type, str) and raw_type.lower() in _TYPE_ALIASES:
        out["type"] = _TYPE_ALIASES[raw_type.lower()]
    elif raw_type is None and not any(
        k in out for k in ("anyOf", "oneOf", "allOf", "enum", "const")
    ):
        if "properties" in out:
            out["type"] = "object"
        elif "items" in out:
            out["type"] = "array"
        else:
            out["type"] = "string"

    if "properties" in out and isinstance(out["properties"], dict):
        out["properties"] = {
            k: sanitize_json_schema(v) if isinstance(v, dict) else {"type": "string"}
            for k, v in out["properties"].items()
        }
        if "required" in out and isinstance(out["required"], list):
            req_set = set(out["required"])
            out["required"] = [k for k in out["properties"] if k in req_set]

    if "items" in out and isinstance(out["items"], dict):
        out["items"] = sanitize_json_schema(out["items"])

    return out


@dataclasses.dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    domain: str = "general"
    mutating: bool = False
    project_scoped: bool = False
    source: str = "builtin"

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": sanitize_json_schema(self.parameters),
            },
        }


class ToolRegistry:
    """Central registry for built-in, plugin, and MCP-bridged tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        domain: str = "general",
        mutating: bool = False,
        project_scoped: bool = False,
        source: str = "builtin",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            spec = ToolSpec(
                name=name,
                description=description,
                parameters=sanitize_json_schema(parameters),
                handler=fn,
                domain=domain,
                mutating=mutating,
                project_scoped=project_scoped,
                source=source,
            )
            self._tools[name] = spec
            return fn

        return decorator

    def register_spec(self, spec: ToolSpec) -> None:
        spec.parameters = sanitize_json_schema(spec.parameters)
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self, domain: str | None = None) -> list[ToolSpec]:
        specs = list(self._tools.values())
        if domain is not None:
            specs = [s for s in specs if s.domain == domain]
        return sorted(specs, key=lambda s: s.name)

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"Unknown tool {name!r}")
        return spec.handler(**arguments)


default_tool_registry = ToolRegistry()
