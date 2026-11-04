from __future__ import annotations

import dataclasses
import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Callable

from pocket_genie.core.agent_configs import SPECIALIST_ROSTER, SpecialistConfig
from pocket_genie.registry.tools import ToolRegistry, ToolSpec, sanitize_json_schema
from pocket_genie.shared import PrecisionTier


@dataclasses.dataclass(slots=True)
class MCPServerConfig:
    name: str
    transport: str = "stdio"
    description: str = ""
    command: str | None = None
    args: list[str] = dataclasses.field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    default_tier: PrecisionTier = "t1"
    mutating_tools: set[str] = dataclasses.field(default_factory=set)


class MCPBridge:
    """Bridges stdio, HTTP JSON-RPC, or in-process MCP servers into isolated SpecialistConfig domains."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry
        self._servers: dict[str, MCPServerConfig] = {}
        self._inproc_handlers: dict[str, dict[str, Callable[..., Any]]] = {}

    @property
    def servers(self) -> dict[str, MCPServerConfig]:
        return dict(self._servers)

    def load_config_file(self, path: Path) -> list[MCPServerConfig]:
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        servers_dict = raw.get("mcpServers", raw.get("servers", {}))
        loaded: list[MCPServerConfig] = []
        for name, cfg in servers_dict.items():
            srv = MCPServerConfig(
                name=name,
                transport=cfg.get("transport", "http" if "url" in cfg else "stdio"),
                description=cfg.get("description", f"MCP Server {name}"),
                command=cfg.get("command"),
                args=list(cfg.get("args", [])),
                url=cfg.get("url"),
                env=dict(cfg.get("env", {})),
                default_tier=cfg.get("default_tier", "t1"),
                mutating_tools=set(cfg.get("mutating_tools", [])),
            )
            self._servers[name] = srv
            loaded.append(srv)
        return loaded

    def register_server_tools(
        self,
        server: MCPServerConfig,
        tool_definitions: list[dict[str, Any]],
        inproc_handlers: dict[str, Callable[..., Any]] | None = None,
    ) -> SpecialistConfig:
        """Register an MCP server's tools under domain `mcp:<server>` and create an isolated SpecialistConfig."""
        self._servers[server.name] = server
        if inproc_handlers:
            self._inproc_handlers[server.name] = dict(inproc_handlers)

        domain = f"mcp:{server.name}"
        for tdef in tool_definitions:
            raw_name = tdef["name"]
            qualified_name = f"mcp.{server.name}.{raw_name}"
            schema = sanitize_json_schema(
                tdef.get("inputSchema") or tdef.get("parameters") or {"type": "object", "properties": {}}
            )
            is_mutating = raw_name in server.mutating_tools or qualified_name in server.mutating_tools

            def _make_caller(srv: MCPServerConfig, r_name: str) -> Callable[..., Any]:
                def _call(**kwargs: Any) -> Any:
                    return self.invoke_mcp_tool(srv.name, r_name, kwargs)

                return _call

            spec = ToolSpec(
                name=qualified_name,
                description=tdef.get("description", f"MCP tool {qualified_name}"),
                parameters=schema,
                handler=_make_caller(server, raw_name),
                domain=domain,
                mutating=is_mutating,
                source=f"mcp:{server.name}",
            )
            self._tool_registry.register_spec(spec)

        specialist_id = f"mcp_{server.name}_agent"
        specialist = SpecialistConfig(
            agent_id=specialist_id,
            display_name=f"MCP Specialist ({server.name})",
            description=server.description or f"Executes tools provided by MCP server `{server.name}`.",
            tool_domains=(domain,),
            default_tier=server.default_tier,
            system_prompt=(
                f"You are the specialist for MCP server `{server.name}`. "
                f"Invoke only the `mcp.{server.name}.*` tools needed to complete the task."
            ),
        )
        SPECIALIST_ROSTER[specialist_id] = specialist
        return specialist

    def invoke_mcp_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        if server_name not in self._servers:
            raise KeyError(f"Unknown MCP server {server_name!r}")
        srv = self._servers[server_name]

        if server_name in self._inproc_handlers and tool_name in self._inproc_handlers[server_name]:
            return self._inproc_handlers[server_name][tool_name](**arguments)

        rpc_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        if srv.transport == "http" and srv.url:
            req = urllib.request.Request(
                srv.url,
                data=json.dumps(rpc_req).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if "error" in payload:
                raise RuntimeError(f"MCP error: {payload['error']}")
            return payload.get("result")

        if srv.transport == "stdio" and srv.command:
            proc = subprocess.run(
                [srv.command, *srv.args],
                input=json.dumps(rpc_req) + "\n",
                text=True,
                capture_output=True,
                timeout=20.0,
                check=False,
            )
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                payload = json.loads(line)
                if "error" in payload:
                    raise RuntimeError(f"MCP stdio error: {payload['error']}")
                if "result" in payload:
                    return payload["result"]
            raise RuntimeError(f"No valid JSON-RPC response from MCP server {server_name}: {proc.stderr}")

        raise RuntimeError(f"Unsupported or unconfigured MCP transport for {server_name}")
