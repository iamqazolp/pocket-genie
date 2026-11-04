from __future__ import annotations

from pathlib import Path

from pocket_genie.core import GenieExecutionGraph, GenieState, TieredAgentFactory
from pocket_genie.mcp import MCPBridge, MCPServerConfig
from pocket_genie.registry import ToolRegistry
from pocket_genie.shared import ChatMessage, GenieSharedContext


def test_mcp_bridge_registers_scoped_specialist_and_sanitizes_schemas(tmp_path: Path):
    reg = ToolRegistry()
    bridge = MCPBridge(reg)

    cfg_file = tmp_path / "mcp_servers.json"
    cfg_file.write_text(
        '{"mcpServers": {"github": {"command": "npx", "args": ["@modelcontextprotocol/server-github"], "mutating_tools": ["create_issue"]}}}',
        encoding="utf-8",
    )
    loaded = bridge.load_config_file(cfg_file)
    assert len(loaded) == 1
    assert loaded[0].name == "github"
    assert "create_issue" in loaded[0].mutating_tools

    srv = MCPServerConfig(
        name="pagerduty",
        transport="inproc",
        description="Incident management MCP server",
        mutating_tools={"ack_incident"},
    )
    specialist = bridge.register_server_tools(
        server=srv,
        tool_definitions=[
            {
                "name": "list_incidents",
                "description": "List active incidents",
                "inputSchema": {"type": "dict", "properties": {"query": {"description": "Filter string"}}},
            },
            {
                "name": "ack_incident",
                "description": "Acknowledge incident",
                "inputSchema": {"type": "object", "properties": {"incident_id": {"type": "str"}}},
            },
        ],
        inproc_handlers={
            "list_incidents": lambda query="": f"INC-402 (open, filter={query})",
            "ack_incident": lambda incident_id="": f"Acked {incident_id}",
        },
    )

    assert specialist.agent_id == "mcp_pagerduty_agent"
    list_spec = reg.get("mcp.pagerduty.list_incidents")
    ack_spec = reg.get("mcp.pagerduty.ack_incident")
    assert list_spec is not None and list_spec.mutating is False
    assert list_spec.parameters["properties"]["query"]["type"] == "string"
    assert ack_spec is not None and ack_spec.mutating is True

    graph = GenieExecutionGraph(factory=TieredAgentFactory(), tool_registry=reg)
    state = GenieState(
        session_id="sess-mcp",
        session_created_at="2026-11-07T23:00:00Z",
        messages=[ChatMessage(role="user", content="list active incidents oncall")],
    )
    shared = GenieSharedContext(
        session_id="sess-mcp",
        session_created_at="2026-11-07T23:00:00Z",
        workspace_dir=tmp_path,
        agent_type="direct-specialist",
        target_specialist="mcp_pagerduty_agent",
    )
    reply, _ = graph.invoke(state, shared)
    assert "INC-402" in reply.content
