from __future__ import annotations

import dataclasses
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

from pocket_genie.core.agent_configs import get_specialist_config
from pocket_genie.core.delegation import execute_specialist_subgraph
from pocket_genie.core.memory import (
    SessionMemoryStore,
    estimate_message_tokens,
    summarize_user_queries,
)
from pocket_genie.core.state import GenieState
from pocket_genie.core.supervisor_agent import SupervisorAgent
from pocket_genie.core.tiered_agent_factory import TieredAgentFactory
from pocket_genie.integrations.decision_router import GeminiDecisionClient, TriageDecision
from pocket_genie.middlewares import (
    CustomSummarizationMiddleware,
    DynamicSupervisorPromptMiddleware,
    FallbackMiddleware,
    HumanInTheLoopMiddleware,
    LangfuseTracingMiddleware,
    MessageSanitizerMiddleware,
    MiddlewarePipeline,
    ProjectScopedToolsMiddleware,
    RefusalDetectionMiddleware,
    RepeatToolCallBreakerMiddleware,
    SQLTableAllowlistMiddleware,
    ToolErrorMiddleware,
    ToolResultTruncationMiddleware,
)
from pocket_genie.observability.langfuse_tracer import GenieLangfuseTracer
from pocket_genie.registry.skills import SkillRegistry
from pocket_genie.registry.tools import ToolRegistry, default_tool_registry
from pocket_genie.shared import ChatMessage, GenieSharedContext, StreamWriter

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    InMemorySaver = None  # type: ignore[assignment,misc]
    END = "__end__"
    START = "__start__"
    StateGraph = None  # type: ignore[assignment,misc]


class GenieGraphState(TypedDict, total=False):
    run_id: str
    session_id: str
    agent_type: str
    start_idx: int
    user_messages_summary: str
    recent_session_summaries: list[str]
    pinned_memories: list[str]
    decision_triage: dict[str, Any]
    route_key: str
    final_content: str


@dataclasses.dataclass(slots=True)
class _ActiveGraphRun:
    state: GenieState
    shared: GenieSharedContext
    writer: StreamWriter
    final_msg: ChatMessage | None = None


def user_messages_summary_node(state: GenieState) -> str:
    return summarize_user_queries(state.messages, previous_summary=state.user_messages_summary)


def recent_conversations_node(state: GenieState, memory_store: SessionMemoryStore) -> list[str]:
    return memory_store.recent_summaries(exclude_session_id=state.session_id, limit=3)


def load_user_memories_node(memory_store: SessionMemoryStore) -> list[str]:
    return memory_store.list_memories()


class GenieExecutionGraph:
    """LangGraph-compiled StateGraph with parallel 4-node fan-out (including Gemini Decision System-One triage), conditional routing, checkpointer, and Langfuse tracing."""

    def __init__(
        self,
        factory: TieredAgentFactory | None = None,
        tool_registry: ToolRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        memory_store: SessionMemoryStore | None = None,
        hitl_middleware: HumanInTheLoopMiddleware | None = None,
        tracer: GenieLangfuseTracer | None = None,
        decision_client: GeminiDecisionClient | None = None,
        enable_decision_routing: bool = False,
    ) -> None:
        self.factory = factory or TieredAgentFactory()
        self.tool_registry = tool_registry or default_tool_registry
        self.skill_registry = skill_registry or SkillRegistry()
        self.memory_store = memory_store or SessionMemoryStore()
        self.hitl = hitl_middleware or HumanInTheLoopMiddleware()
        self.tracer = tracer or GenieLangfuseTracer()
        self.decision_client = decision_client or GeminiDecisionClient()
        self.enable_decision_routing = enable_decision_routing
        self.loop_breaker = RepeatToolCallBreakerMiddleware(max_repeats=2)
        self._active_runs: dict[str, _ActiveGraphRun] = {}

        self.pipeline = MiddlewarePipeline(
            [
                MessageSanitizerMiddleware(),
                DynamicSupervisorPromptMiddleware(self.skill_registry),
                CustomSummarizationMiddleware(),
                ProjectScopedToolsMiddleware(),
                LangfuseTracingMiddleware(self.tracer),
                FallbackMiddleware(self.factory.backends),
                RefusalDetectionMiddleware(),
                self.loop_breaker,
                SQLTableAllowlistMiddleware(),
                self.hitl,
                ToolResultTruncationMiddleware(max_tokens=512),
                ToolErrorMiddleware(),
            ]
        )

        self.checkpointer: Any | None = InMemorySaver() if LANGGRAPH_AVAILABLE else None
        self.compiled_graph: Any | None = self._build_langgraph() if LANGGRAPH_AVAILABLE else None

    def _build_langgraph(self) -> Any:
        builder = StateGraph(GenieGraphState)

        def _lg_summary(gs: GenieGraphState) -> dict[str, Any]:
            st = self._active_runs[gs["run_id"]].state
            val = user_messages_summary_node(st)
            st.user_messages_summary = val
            return {"user_messages_summary": val}

        def _lg_recent(gs: GenieGraphState) -> dict[str, Any]:
            st = self._active_runs[gs["run_id"]].state
            val = recent_conversations_node(st, self.memory_store)
            st.recent_session_summaries = val
            return {"recent_session_summaries": val}

        def _lg_memories(gs: GenieGraphState) -> dict[str, Any]:
            st = self._active_runs[gs["run_id"]].state
            val = load_user_memories_node(self.memory_store)
            st.pinned_memories = val
            return {"pinned_memories": val}

        def _lg_decision_triage(gs: GenieGraphState) -> dict[str, Any]:
            run = self._active_runs[gs["run_id"]]
            decision = self._run_decision_triage(run.state)
            return {"decision_triage": decision.to_dict()}

        def _lg_route_dispatch(gs: GenieGraphState) -> dict[str, Any]:
            run = self._active_runs[gs["run_id"]]
            rk = self._apply_decision_and_determine_route(run.state, run.shared, run.writer)
            return {"route_key": rk, "agent_type": run.shared.agent_type}

        def _select_route(gs: GenieGraphState) -> str:
            return str(gs.get("route_key", "supervisor_node"))

        def _lg_decision_block(gs: GenieGraphState) -> dict[str, Any]:
            run = self._active_runs[gs["run_id"]]
            msg = self._execute_decision_guardrail_block(run.state, run.writer)
            run.final_msg = msg
            return {"final_content": msg.content}

        def _lg_direct_specialist(gs: GenieGraphState) -> dict[str, Any]:
            run = self._active_runs[gs["run_id"]]
            msg = self._execute_direct_specialist(run.state, run.shared, run.writer)
            run.final_msg = msg
            return {"final_content": msg.content}

        def _lg_red_agent(gs: GenieGraphState) -> dict[str, Any]:
            run = self._active_runs[gs["run_id"]]
            msg = self._execute_red_agent(run.state, run.shared, run.writer)
            run.final_msg = msg
            return {"final_content": msg.content}

        def _lg_supervisor(gs: GenieGraphState) -> dict[str, Any]:
            run = self._active_runs[gs["run_id"]]
            msg = self._execute_supervisor_route(run.state, run.shared, run.writer)
            run.final_msg = msg
            return {"final_content": msg.content}

        def _lg_capture(gs: GenieGraphState) -> dict[str, Any]:
            run = self._active_runs[gs["run_id"]]
            self._capture_to_full_history(run.state, start_idx=int(gs.get("start_idx", 0)))
            return {}

        builder.add_node("user_messages_summary", _lg_summary)
        builder.add_node("recent_conversations", _lg_recent)
        builder.add_node("load_user_memories", _lg_memories)
        builder.add_node("decision_system_one_triage", _lg_decision_triage)
        builder.add_node("route_dispatch", _lg_route_dispatch)
        builder.add_node("decision_guardrail_block_node", _lg_decision_block)
        builder.add_node("supervisor_node", _lg_supervisor)
        builder.add_node("direct_specialist_node", _lg_direct_specialist)
        builder.add_node("red_agent_node", _lg_red_agent)
        builder.add_node("capture_to_full_history", _lg_capture)

        builder.add_edge(START, "user_messages_summary")
        builder.add_edge(START, "recent_conversations")
        builder.add_edge(START, "load_user_memories")
        builder.add_edge(START, "decision_system_one_triage")

        builder.add_edge("user_messages_summary", "route_dispatch")
        builder.add_edge("recent_conversations", "route_dispatch")
        builder.add_edge("load_user_memories", "route_dispatch")
        builder.add_edge("decision_system_one_triage", "route_dispatch")

        builder.add_conditional_edges(
            "route_dispatch",
            _select_route,
            {
                "decision_guardrail_block_node": "decision_guardrail_block_node",
                "supervisor_node": "supervisor_node",
                "direct_specialist_node": "direct_specialist_node",
                "red_agent_node": "red_agent_node",
            },
        )

        builder.add_edge("decision_guardrail_block_node", "capture_to_full_history")
        builder.add_edge("supervisor_node", "capture_to_full_history")
        builder.add_edge("direct_specialist_node", "capture_to_full_history")
        builder.add_edge("red_agent_node", "capture_to_full_history")
        builder.add_edge("capture_to_full_history", END)

        return builder.compile(checkpointer=self.checkpointer)

    def get_mermaid_graph(self) -> str:
        if self.compiled_graph is not None:
            return str(self.compiled_graph.get_graph().draw_mermaid())
        return (
            "flowchart TD\n"
            "  START --> user_messages_summary & recent_conversations & load_user_memories & decision_system_one_triage\n"
            "  user_messages_summary & recent_conversations & load_user_memories & decision_system_one_triage --> route_dispatch\n"
            "  route_dispatch --> decision_guardrail_block_node & supervisor_node & direct_specialist_node & red_agent_node\n"
            "  decision_guardrail_block_node & supervisor_node & direct_specialist_node & red_agent_node --> capture_to_full_history --> END\n"
        )

    def _run_decision_triage(self, state: GenieState) -> TriageDecision:
        last_user = next((m.content for m in reversed(state.messages) if m.role == "user"), "")
        decision = self.decision_client.evaluate_turn(last_user)
        state.decision_triage = decision.to_dict()
        return decision

    def _apply_decision_and_determine_route(
        self,
        state: GenieState,
        shared: GenieSharedContext,
        writer: StreamWriter,
    ) -> str:
        triage = state.decision_triage
        self.tracer.record_guardrail_event(guardrail_name="decision_system_one", details=triage)
        writer.emit("middleware_event", middleware="decision_system_one", **triage)

        if triage.get("is_injection_blocked"):
            state.middleware_trace.append(
                {
                    "middleware": "decision_system_one",
                    "blocked": True,
                    "injection_prob": triage.get("injection_prob"),
                }
            )
            return "decision_guardrail_block_node"

        use_decision_routing = self.enable_decision_routing or triage.get("source") == "gemini"
        if use_decision_routing:
            tier_rank = {"t1": 0, "t2": 1, "t3": 2}
            rec_tier = str(triage.get("recommended_tier", "t1"))
            if rec_tier in tier_rank and tier_rank[rec_tier] > tier_rank.get(state.active_tier, 0):
                prev_t = state.active_tier
                state.active_tier = rec_tier  # type: ignore[assignment]
                ev = {
                    "middleware": "decision_tier_preescalation",
                    "from_tier": prev_t,
                    "to_tier": rec_tier,
                    "complexity_score": triage.get("complexity_score"),
                }
                state.middleware_trace.append(ev)
                writer.emit("middleware_event", **ev)

            chosen_spec = str(triage.get("specialist", "supervisor"))
            conf = float(triage.get("route_confidence", 0.0))
            if (
                shared.agent_type == "supervisor"
                and chosen_spec != "supervisor"
                and conf >= 0.75
            ):
                shared.target_specialist = chosen_spec
                return "direct_specialist_node"

        if shared.agent_type == "direct-specialist" and shared.target_specialist:
            return "direct_specialist_node"
        if shared.agent_type == "red-agent":
            return "red_agent_node"
        return "supervisor_node"

    def _execute_decision_guardrail_block(self, state: GenieState, writer: StreamWriter) -> ChatMessage:
        prob = float(state.decision_triage.get("injection_prob", 0.9))
        reply = ChatMessage(
            role="assistant",
            content=(
                f"[DECISION_SYSTEM_ONE_BLOCK] Request halted by Gemini Decision System-One guardrail "
                f"(prompt_injection={prob:.2f})."
            ),
            reasoning="System-One Gemini Decision noul predicate flagged prompt injection prior to LLM invocation.",
        )
        state.messages.append(reply)
        writer.emit("ai_message_chunk", content=reply.content)
        return reply

    def _run_parallel_preprocessing(self, state: GenieState) -> None:
        with ThreadPoolExecutor(max_workers=4) as pool:
            f_sum = pool.submit(user_messages_summary_node, state)
            f_rec = pool.submit(recent_conversations_node, state, self.memory_store)
            f_mem = pool.submit(load_user_memories_node, self.memory_store)
            f_decision = pool.submit(self._run_decision_triage, state)
            state.user_messages_summary = f_sum.result()
            state.recent_session_summaries = f_rec.result()
            state.pinned_memories = f_mem.result()
            f_decision.result()

    def _execute_direct_specialist(
        self,
        state: GenieState,
        shared: GenieSharedContext,
        writer: StreamWriter,
    ) -> ChatMessage:
        cfg = get_specialist_config(shared.target_specialist or "devops_agent")
        last_user = next((m.content for m in reversed(state.messages) if m.role == "user"), "")
        obs = execute_specialist_subgraph(
            config=cfg,
            task=last_user,
            context="",
            tier=state.active_tier,
            state=state,
            shared=shared,
            factory=self.factory,
            tool_registry=self.tool_registry,
            pipeline=self.pipeline,
            writer=writer,
        )
        reply = ChatMessage(role="assistant", content=obs)
        state.messages.append(reply)
        writer.emit("ai_message_chunk", content=reply.content)
        return reply

    def _execute_red_agent(
        self,
        state: GenieState,
        shared: GenieSharedContext,
        writer: StreamWriter,
    ) -> ChatMessage:
        cfg = get_specialist_config("red_teaming_agent")
        last_user = next((m.content for m in reversed(state.messages) if m.role == "user"), "")
        obs = execute_specialist_subgraph(
            config=cfg,
            task=last_user,
            context="red-team-evaluation",
            tier="t2",
            state=state,
            shared=shared,
            factory=self.factory,
            tool_registry=self.tool_registry,
            pipeline=self.pipeline,
            writer=writer,
        )
        reply = ChatMessage(role="assistant", content=obs)
        state.messages.append(reply)
        writer.emit("ai_message_chunk", content=reply.content)
        return reply

    def _execute_supervisor_route(
        self,
        state: GenieState,
        shared: GenieSharedContext,
        writer: StreamWriter,
    ) -> ChatMessage:
        agent_type = shared.agent_type
        if agent_type == "scoped-demo":
            sup = SupervisorAgent(
                factory=self.factory,
                tool_registry=self.tool_registry,
                pipeline=self.pipeline,
                specialist_ids=("knowledge_base_agent", "sql_agent", "device_agent"),
                direct_domains=("visualization",),
            )
            return sup.run_turn(state=state, shared=shared, writer=writer)

        if agent_type == "run-assistant":
            sup = SupervisorAgent(
                factory=self.factory,
                tool_registry=self.tool_registry,
                pipeline=self.pipeline,
                specialist_ids=("run_assistant_agent", "devops_agent", "sql_agent"),
            )
            return sup.run_turn(state=state, shared=shared, writer=writer)

        sup = SupervisorAgent(
            factory=self.factory,
            tool_registry=self.tool_registry,
            pipeline=self.pipeline,
            specialist_ids=("devops_agent", "knowledge_base_agent", "sql_agent", "device_agent"),
        )
        return sup.run_turn(state=state, shared=shared, writer=writer)

    def _route_and_stream(
        self,
        state: GenieState,
        shared: GenieSharedContext,
        writer: StreamWriter,
    ) -> ChatMessage:
        rk = self._apply_decision_and_determine_route(state, shared, writer)
        if rk == "decision_guardrail_block_node":
            return self._execute_decision_guardrail_block(state, writer)
        if rk == "direct_specialist_node":
            return self._execute_direct_specialist(state, shared, writer)
        if rk == "red_agent_node":
            return self._execute_red_agent(state, shared, writer)
        return self._execute_supervisor_route(state, shared, writer)

    def _capture_to_full_history(self, state: GenieState, start_idx: int) -> None:
        new_msgs = state.messages[start_idx:]
        state.full_history.extend(new_msgs)
        total_toks = max(128, sum(estimate_message_tokens(m) for m in state.full_history))
        fp16_mb, comp_mb, _, _, _ = self.factory.kv_cache.estimate_memory_mb(total_toks)
        state.quant_telemetry.kv_cache_fp16_mb = fp16_mb
        state.quant_telemetry.kv_cache_quant_mb = comp_mb
        if state.user_messages_summary:
            self.memory_store.record_session_summary(state.session_id, state.user_messages_summary)

    def invoke(
        self,
        state: GenieState,
        shared: GenieSharedContext,
        writer: StreamWriter | None = None,
    ) -> tuple[ChatMessage, StreamWriter]:
        stream = writer or StreamWriter()
        start_len = len(state.messages)
        last_user = next((m.content for m in reversed(state.messages) if m.role == "user"), "")

        with self.tracer.trace_turn(
            session_id=state.session_id,
            name=f"genie:{shared.agent_type}",
            input_prompt=last_user,
            agent_type=shared.agent_type,
            tier=state.active_tier,
        ) as trace_rec:
            if self.compiled_graph is not None:
                run_id = uuid.uuid4().hex
                run_ctx = _ActiveGraphRun(state=state, shared=shared, writer=stream)
                self._active_runs[run_id] = run_ctx
                try:
                    callbacks = []
                    lf_cb = self.tracer.get_langchain_callback_handler(trace_id=trace_rec.trace_id)
                    if lf_cb is not None:
                        callbacks.append(lf_cb)

                    config: dict[str, Any] = {"configurable": {"thread_id": state.session_id}}
                    if callbacks:
                        config["callbacks"] = callbacks

                    self.compiled_graph.invoke(
                        {
                            "run_id": run_id,
                            "session_id": state.session_id,
                            "agent_type": shared.agent_type,
                            "start_idx": start_len,
                        },
                        config=config,
                    )
                    assert run_ctx.final_msg is not None
                    final_msg = run_ctx.final_msg
                finally:
                    self._active_runs.pop(run_id, None)
            else:
                self._run_parallel_preprocessing(state)
                final_msg = self._route_and_stream(state=state, shared=shared, writer=stream)
                self._capture_to_full_history(state=state, start_idx=start_len)

            trace_rec.output_reply = final_msg.content
            trace_rec.active_tier = state.active_tier
            trace_rec.profile = state.quant_telemetry.profile

        stream.emit(
            "final",
            content=final_msg.content,
            tier=state.active_tier,
            telemetry=state.quant_telemetry.to_dict(),
        )
        return final_msg, stream
