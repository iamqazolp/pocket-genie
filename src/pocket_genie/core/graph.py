from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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
from pocket_genie.middlewares import (
    CustomSummarizationMiddleware,
    DynamicSupervisorPromptMiddleware,
    FallbackMiddleware,
    HumanInTheLoopMiddleware,
    MessageSanitizerMiddleware,
    MiddlewarePipeline,
    ProjectScopedToolsMiddleware,
    RefusalDetectionMiddleware,
    RepeatToolCallBreakerMiddleware,
    SQLTableAllowlistMiddleware,
    ToolErrorMiddleware,
    ToolResultTruncationMiddleware,
)
from pocket_genie.registry.skills import SkillRegistry
from pocket_genie.registry.tools import ToolRegistry, default_tool_registry
from pocket_genie.shared import ChatMessage, GenieSharedContext, StreamWriter


def user_messages_summary_node(state: GenieState) -> str:
    return summarize_user_queries(state.messages, previous_summary=state.user_messages_summary)


def recent_conversations_node(state: GenieState, memory_store: SessionMemoryStore) -> list[str]:
    return memory_store.recent_summaries(exclude_session_id=state.session_id, limit=3)


def load_user_memories_node(memory_store: SessionMemoryStore) -> list[str]:
    return memory_store.list_memories()


class GenieExecutionGraph:
    """Main execution StateGraph with parallel 3-node pre-processing, multi-route dispatch, and history capture."""

    def __init__(
        self,
        factory: TieredAgentFactory | None = None,
        tool_registry: ToolRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        memory_store: SessionMemoryStore | None = None,
        hitl_middleware: HumanInTheLoopMiddleware | None = None,
    ) -> None:
        self.factory = factory or TieredAgentFactory()
        self.tool_registry = tool_registry or default_tool_registry
        self.skill_registry = skill_registry or SkillRegistry()
        self.memory_store = memory_store or SessionMemoryStore()
        self.hitl = hitl_middleware or HumanInTheLoopMiddleware()
        self.loop_breaker = RepeatToolCallBreakerMiddleware(max_repeats=2)

        self.pipeline = MiddlewarePipeline(
            [
                MessageSanitizerMiddleware(),
                DynamicSupervisorPromptMiddleware(self.skill_registry),
                CustomSummarizationMiddleware(),
                ProjectScopedToolsMiddleware(),
                FallbackMiddleware(self.factory.backends),
                RefusalDetectionMiddleware(),
                self.loop_breaker,
                SQLTableAllowlistMiddleware(),
                self.hitl,
                ToolResultTruncationMiddleware(max_tokens=512),
                ToolErrorMiddleware(),
            ]
        )

    def _run_parallel_preprocessing(self, state: GenieState) -> None:
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_sum = pool.submit(user_messages_summary_node, state)
            f_rec = pool.submit(recent_conversations_node, state, self.memory_store)
            f_mem = pool.submit(load_user_memories_node, self.memory_store)
            state.user_messages_summary = f_sum.result()
            state.recent_session_summaries = f_rec.result()
            state.pinned_memories = f_mem.result()

    def _route_and_stream(
        self,
        state: GenieState,
        shared: GenieSharedContext,
        writer: StreamWriter,
    ) -> ChatMessage:
        agent_type = shared.agent_type
        if agent_type == "direct-specialist" and shared.target_specialist:
            cfg = get_specialist_config(shared.target_specialist)
            last_user = next(
                (m.content for m in reversed(state.messages) if m.role == "user"),
                "",
            )
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

        if agent_type == "scoped-demo":
            sup = SupervisorAgent(
                factory=self.factory,
                tool_registry=self.tool_registry,
                pipeline=self.pipeline,
                specialist_ids=("knowledge_base_agent", "sql_agent", "device_agent"),
                direct_domains=("visualization",),
            )
            return sup.run_turn(state=state, shared=shared, writer=writer)

        if agent_type == "red-agent":
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
            return reply

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

    def _capture_to_full_history(self, state: GenieState, start_idx: int) -> None:
        new_msgs = state.messages[start_idx:]
        state.full_history.extend(new_msgs)
        total_toks = sum(estimate_message_tokens(m) for m in state.full_history)
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
        self._run_parallel_preprocessing(state)
        final_msg = self._route_and_stream(state=state, shared=shared, writer=stream)
        self._capture_to_full_history(state=state, start_idx=start_len)
        stream.emit(
            "final",
            content=final_msg.content,
            tier=state.active_tier,
            telemetry=state.quant_telemetry.to_dict(),
        )
        return final_msg, stream
