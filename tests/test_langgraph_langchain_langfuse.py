from __future__ import annotations

from pathlib import Path

import pytest

from pocket_genie.core import (
    LANGGRAPH_AVAILABLE,
    GenieExecutionGraph,
    GenieState,
    TieredAgentFactory,
)
from pocket_genie.integrations import (
    LANGCHAIN_AVAILABLE,
    PocketGenieChatModel,
    chat_messages_to_langchain,
    langchain_to_chat_messages,
)
from pocket_genie.observability import LANGFUSE_AVAILABLE, GenieLangfuseTracer
from pocket_genie.registry import DeterministicReplayBackend, ToolRegistry
from pocket_genie.shared import ChatMessage, GenieSharedContext, ToolInvocation


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="langgraph not installed")
def test_langgraph_stategraph_compiles_with_parallel_fanout_and_checkpointer():
    reg = ToolRegistry()

    @reg.register(
        name="devops.git_status",
        description="Inspect repository git status",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        domain="devops",
    )
    def git_status(query: str = "") -> str:
        return f"clean-main:{query}"

    graph = GenieExecutionGraph(factory=TieredAgentFactory(), tool_registry=reg)
    assert graph.compiled_graph is not None
    assert graph.checkpointer is not None

    mermaid = graph.get_mermaid_graph()
    assert "user_messages_summary" in mermaid
    assert "recent_conversations" in mermaid
    assert "load_user_memories" in mermaid
    assert "route_dispatch" in mermaid
    assert "supervisor_node" in mermaid
    assert "capture_to_full_history" in mermaid

    state = GenieState(
        session_id="lg-thread-01",
        session_created_at="2026-11-10T10:00:00Z",
        messages=[ChatMessage(role="user", content="Check git status")],
    )
    shared = GenieSharedContext(
        session_id="lg-thread-01",
        session_created_at="2026-11-10T10:00:00Z",
        workspace_dir=Path("."),
        project_id="pocket-genie",
    )
    reply, _ = graph.invoke(state, shared)
    assert "clean-main" in reply.content

    ckpt = graph.compiled_graph.get_state({"configurable": {"thread_id": "lg-thread-01"}})
    assert ckpt is not None
    assert ckpt.values.get("session_id") == "lg-thread-01"


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-core not installed")
def test_langchain_structured_tool_and_chat_model_roundtrip():
    reg = ToolRegistry()

    @reg.register(
        name="calc.multiply",
        description="Multiply two integers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First factor"},
                "b": {"type": "integer", "description": "Second factor"},
            },
            "required": ["a", "b"],
        },
        domain="math",
    )
    def multiply(a: int, b: int) -> int:
        return a * b

    spec = reg.get("calc.multiply")
    assert spec is not None
    lc_tool = spec.to_langchain_tool()
    assert lc_tool.name == "calc.multiply"
    assert lc_tool.invoke({"a": 6, "b": 7}) == 42

    orig_msgs = [
        ChatMessage(role="system", content="You are precise."),
        ChatMessage(role="user", content="Compute 6 * 7"),
        ChatMessage(
            role="assistant",
            content="",
            reasoning="Calling calc.multiply",
            tool_calls=[ToolInvocation(id="tc_1", name="calc.multiply", arguments={"a": 6, "b": 7})],
        ),
        ChatMessage(role="tool", content="42", name="calc.multiply", tool_call_id="tc_1"),
    ]
    lc_msgs = chat_messages_to_langchain(orig_msgs)
    restored = langchain_to_chat_messages(lc_msgs)
    assert len(restored) == 4
    assert restored[2].tool_calls[0].arguments == {"a": 6, "b": 7}
    assert restored[2].reasoning == "Calling calc.multiply"
    assert restored[3].tool_call_id == "tc_1"

    chat_model = PocketGenieChatModel(backend=DeterministicReplayBackend(tier="t2"), constrain_schema=True)
    bound = chat_model.bind_tools([lc_tool])
    ai_resp = bound.invoke("Multiply numbers")
    assert ai_resp.tool_calls
    assert ai_resp.response_metadata["tier"] == "t2"
    assert ai_resp.response_metadata["quant_telemetry"]["profile"] == "P_mixed"


@pytest.mark.skipif(not LANGFUSE_AVAILABLE, reason="langfuse not installed")
def test_langfuse_otel_tracer_records_generations_tools_guardrails_and_scores():
    reg = ToolRegistry()

    @reg.register(
        name="devops.git_status",
        description="Check git status",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        domain="devops",
    )
    def git_status(query: str = "") -> str:
        return "branch main clean"

    tracer = GenieLangfuseTracer(in_memory_otel=True)
    assert tracer.enabled is True
    assert tracer.get_langchain_callback_handler() is not None

    factory = TieredAgentFactory()
    b_t1 = DeterministicReplayBackend(tier="t1")
    b_t1.push_response(ChatMessage(role="assistant", content='<tool_call>{"name": "corrupt'))
    factory.backends.register("t1", b_t1)

    graph = GenieExecutionGraph(factory=factory, tool_registry=reg, tracer=tracer)
    state = GenieState(
        session_id="lf-sess-01",
        session_created_at="2026-11-10T10:05:00Z",
        messages=[ChatMessage(role="user", content="Check git status")],
    )
    shared = GenieSharedContext(
        session_id="lf-sess-01",
        session_created_at="2026-11-10T10:05:00Z",
        workspace_dir=Path("."),
        project_id="pocket-genie",
    )
    reply, _ = graph.invoke(state, shared)
    tracer.flush()

    assert "branch main clean" in reply.content
    traces = tracer.list_traces()
    assert len(traces) == 1
    tr = traces[0]
    assert tr["active_tier"] == "t2"
    assert tr["profile"] == "P_mixed"
    assert tr["scores"]["effective_act_bits"] >= 8.0

    obs_types = {o["as_type"] for o in tr["observations"]}
    assert "generation" in obs_types
    assert "tool" in obs_types
    assert "guardrail" in obs_types

    otel_spans = tracer.exported_otel_spans()
    assert len(otel_spans) >= 3
