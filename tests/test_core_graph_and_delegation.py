from __future__ import annotations

from pathlib import Path

from pocket_genie.core import (
    GenieExecutionGraph,
    GenieState,
    SessionMemoryStore,
    TieredAgentFactory,
    ToolCallChunkAccumulator,
)
from pocket_genie.registry import ToolRegistry
from pocket_genie.shared import ChatMessage, GenieSharedContext, ToolCallChunk


def test_tool_call_chunk_accumulator_reassembles_fragments():
    acc = ToolCallChunkAccumulator()
    acc.ingest(ToolCallChunk(index=0, id="call_abc", name="devops.git_status", arguments_fragment='{"short": '))
    acc.ingest(ToolCallChunk(index=0, id=None, name=None, arguments_fragment="true}"))
    calls = acc.finalize()
    assert len(calls) == 1
    assert calls[0].id == "call_abc"
    assert calls[0].name == "devops.git_status"
    assert calls[0].arguments == {"short": True}


def test_execution_graph_parallel_preprocessing_and_delegation():
    reg = ToolRegistry()

    @reg.register(
        name="devops.git_status",
        description="Check git status",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        domain="devops",
    )
    def git_status(query: str = "") -> str:
        return f"On branch main (clean) [{query}]"

    mem = SessionMemoryStore()
    mem.add_memory("Always run pytest -q")
    mem.record_session_summary("old-sess", "Completed W4A8 calibration")

    graph = GenieExecutionGraph(
        factory=TieredAgentFactory(),
        tool_registry=reg,
        memory_store=mem,
    )
    state = GenieState(
        session_id="sess-graph",
        session_created_at="2026-11-07T15:00:00Z",
        messages=[ChatMessage(role="user", content="Check repository git status")],
    )
    shared = GenieSharedContext(
        session_id="sess-graph",
        session_created_at="2026-11-07T15:00:00Z",
        workspace_dir=Path("."),
        project_id="pocket-genie",
        agent_type="supervisor",
    )

    reply, stream = graph.invoke(state, shared)
    assert "On branch main (clean)" in reply.content
    assert state.pinned_memories == ["Always run pytest -q"]
    assert state.recent_session_summaries == ["Completed W4A8 calibration"]
    assert "Check repository git status" in state.user_messages_summary
    assert len(state.full_history) >= 2

    event_types = [e.event_type for e in stream.events]
    assert "subagent_reasoning" in event_types
    assert "tool_message_chunk" in event_types
    assert "quant_telemetry" in event_types
    assert "final" in event_types
