from __future__ import annotations

from pathlib import Path

from pocket_genie.core.state import GenieState
from pocket_genie.middlewares import (
    CustomSummarizationMiddleware,
    DynamicSupervisorPromptMiddleware,
    FallbackMiddleware,
    HumanInTheLoopMiddleware,
    MessageSanitizerMiddleware,
    MiddlewarePipeline,
    ModelCallContext,
    ProjectScopedToolsMiddleware,
    RefusalDetectionMiddleware,
    RepeatToolCallBreakerMiddleware,
    SQLTableAllowlistMiddleware,
    ToolCallContext,
    ToolErrorMiddleware,
    ToolResultTruncationMiddleware,
)
from pocket_genie.registry.backends import (
    BackendRegistry,
    DeterministicReplayBackend,
    GenerationOutput,
)
from pocket_genie.registry.tools import ToolSpec
from pocket_genie.shared import (
    ChatMessage,
    GenieSharedContext,
    StreamWriter,
    ToolInvocation,
)


def _make_context(
    messages: list[ChatMessage] | None = None,
    tools: list[ToolSpec] | None = None,
    agent_type: str = "supervisor",
    project_id: str | None = "proj-alpha",
) -> tuple[GenieState, GenieSharedContext, StreamWriter]:
    state = GenieState(
        session_id="sess-mw",
        session_created_at="2026-11-03T10:00:00Z",
        agent_type=agent_type,  # type: ignore[arg-type]
        project_id=project_id,
        messages=list(messages or []),
    )
    shared = GenieSharedContext(
        session_id="sess-mw",
        session_created_at="2026-11-03T10:00:00Z",
        workspace_dir=Path("."),
        project_id=project_id,
        agent_type=agent_type,  # type: ignore[arg-type]
    )
    return state, shared, StreamWriter()


def test_message_sanitizer_removes_orphans_and_dangling_tags():
    msgs = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="tool", content="Orphaned result", tool_call_id="missing_id"),
        ChatMessage(role="assistant", content="Partial <tool_call>{\"name\": \"git"),
    ]
    state, shared, writer = _make_context(messages=msgs)
    pipeline = MiddlewarePipeline([MessageSanitizerMiddleware()])
    ctx = ModelCallContext(state=state, shared=shared, messages=state.messages, tools=[], writer=writer)

    pipeline.run_model(ctx, lambda c: GenerationOutput(message=ChatMessage(role="assistant", content="ok")))
    assert len(ctx.messages) == 2
    assert "<tool_call>" not in ctx.messages[1].content
    assert state.middleware_trace[0]["repaired_count"] == 2


def test_fallback_middleware_escalates_on_refusal_and_corrupted_json():
    backends = BackendRegistry()
    b_t1 = DeterministicReplayBackend(
        tier="t1",
        scripted_responses=[ChatMessage(role="assistant", content="I cannot fulfill this request.")],
    )
    b_t2 = DeterministicReplayBackend(
        tier="t2",
        scripted_responses=[ChatMessage(role="assistant", content="<tool_call>{bad json")],
    )
    b_t3 = DeterministicReplayBackend(
        tier="t3",
        scripted_responses=[ChatMessage(role="assistant", content="Recovered on Tier 3 P_accurate")],
    )
    backends.register("t1", b_t1)
    backends.register("t2", b_t2)
    backends.register("t3", b_t3)

    state, shared, writer = _make_context(messages=[ChatMessage(role="user", content="Run audit")])
    pipeline = MiddlewarePipeline([FallbackMiddleware(backends), RefusalDetectionMiddleware()])
    ctx = ModelCallContext(state=state, shared=shared, messages=state.messages, tools=[], writer=writer)

    out = pipeline.run_model(ctx, lambda c: backends.get(c.state.active_tier).generate(c.messages))
    assert out.message.content == "Recovered on Tier 3 P_accurate"
    assert state.active_tier == "t3"
    assert state.quant_telemetry.profile == "P_accurate"


def test_tool_middlewares_loop_breaker_hitl_sql_and_truncation():
    state, shared, writer = _make_context(agent_type="scoped-demo", project_id=None)
    hitl = HumanInTheLoopMiddleware()
    pipeline = MiddlewarePipeline(
        [
            RepeatToolCallBreakerMiddleware(max_repeats=2),
            SQLTableAllowlistMiddleware(),
            hitl,
            ToolResultTruncationMiddleware(max_tokens=32),
            ToolErrorMiddleware(),
        ]
    )

    sql_spec = ToolSpec(
        name="sql.query",
        description="Run SQL",
        parameters={"type": "object"},
        handler=lambda query: "rows",
    )
    bad_sql_ctx = ToolCallContext(
        state=state,
        shared=shared,
        invocation=ToolInvocation(id="c1", name="sql.query", arguments={"query": "SELECT * FROM secret_users"}),
        spec=sql_spec,
        specialist="sql_agent",
        writer=writer,
    )
    res_sql = pipeline.run_tool(bad_sql_ctx, lambda c: "should not execute")
    assert "[SQL_ALLOWLIST_ERROR]" in res_sql

    mut_spec = ToolSpec(
        name="devops.apply_patch",
        description="Patch file",
        parameters={"type": "object"},
        handler=lambda diff: "patched",
        mutating=True,
    )
    mut_ctx = ToolCallContext(
        state=state,
        shared=shared,
        invocation=ToolInvocation(id="hitl_1", name="devops.apply_patch", arguments={"diff": "+line"}),
        spec=mut_spec,
        specialist="devops_agent",
        writer=writer,
    )
    res_pending = pipeline.run_tool(mut_ctx, lambda c: "patched")
    assert "[HITL_PENDING:hitl_1]" in res_pending
    assert len(state.pending_hitl) == 1

    hitl.record_decision("hitl_1", approved=True)
    res_approved = pipeline.run_tool(mut_ctx, lambda c: "patched")
    assert res_approved == "patched"

    res_loop = pipeline.run_tool(mut_ctx, lambda c: "patched")
    assert "[LOOP_BREAKER]" in res_loop
    assert state.active_tier == "t2"


def test_summarization_and_dynamic_prompt_middlewares():
    many_msgs = [
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"Turn {i}: " + ("data " * 40))
        for i in range(12)
    ]
    state, shared, writer = _make_context(messages=many_msgs)
    pipeline = MiddlewarePipeline(
        [
            DynamicSupervisorPromptMiddleware(),
            CustomSummarizationMiddleware(summary_token_limit=200, retain_token_budget=100, max_messages=6),
            ProjectScopedToolsMiddleware(),
        ]
    )
    scoped_tool = ToolSpec(
        name="harvest.export",
        description="Project-only tool",
        parameters={"type": "object"},
        handler=lambda: "ok",
        project_scoped=True,
    )
    shared.project_id = None
    ctx = ModelCallContext(
        state=state,
        shared=shared,
        messages=state.messages,
        tools=[scoped_tool],
        writer=writer,
    )
    pipeline.run_model(ctx, lambda c: GenerationOutput(message=ChatMessage(role="assistant", content="done")))
    assert state.prefix_kv_pinned is True
    assert state.running_summary != ""
    assert len(ctx.tools) == 0
    assert ctx.messages[0].role == "system"
    assert "2026-11-03T10:00:00Z" in ctx.messages[0].content


def test_sql_allowlist_permits_cte_aliases_and_truncation_snaps_newlines():
    from pocket_genie.middlewares.tool_result_truncation import truncate_tool_output

    state, shared, writer = _make_context(agent_type="scoped-demo")
    sql_mw = SQLTableAllowlistMiddleware()
    cte_query = (
        "WITH ranked AS (SELECT recipe_id, values_exact_rate FROM ablation_results) "
        "SELECT * FROM ranked WHERE values_exact_rate > 0.5"
    )
    ctx = ToolCallContext(
        state=state,
        shared=shared,
        invocation=ToolInvocation(id="cte1", name="sql.query", arguments={"query": cte_query}),
        spec=None,
        specialist="sql_agent",
        writer=writer,
    )
    assert sql_mw.wrap_tool(ctx, lambda c: "rows_ok") == "rows_ok"

    multiline = "\n".join(f"line_{i:03d}: " + ("x" * 20) for i in range(40))
    trunc, flag = truncate_tool_output(multiline, max_tokens=40)
    assert flag is True
    assert "[TRUNCATED:" in trunc
    assert trunc.splitlines()[-1].startswith("line_")

