from __future__ import annotations

import json
from typing import Sequence

from pocket_genie.core.agent_configs import list_specialist_configs
from pocket_genie.core.delegation import (
    ToolCallChunkAccumulator,
    create_agent_as_tool,
    extract_reasoning_from_message,
)
from pocket_genie.core.state import GenieState
from pocket_genie.core.tiered_agent_factory import TieredAgentFactory
from pocket_genie.middlewares.base import (
    MiddlewarePipeline,
    ModelCallContext,
    ToolCallContext,
)
from pocket_genie.registry.tools import ToolRegistry, ToolSpec
from pocket_genie.shared import ChatMessage, GenieSharedContext, StreamWriter


class SupervisorAgent:
    """Tiered orchestrator combining delegated specialist tools with direct supervisor utilities."""

    def __init__(
        self,
        factory: TieredAgentFactory,
        tool_registry: ToolRegistry,
        pipeline: MiddlewarePipeline,
        specialist_ids: Sequence[str] | None = None,
        direct_domains: tuple[str, ...] = ("memory", "visualization", "sql"),
    ) -> None:
        self.factory = factory
        self.tool_registry = tool_registry
        self.pipeline = pipeline
        self.specialist_configs = list_specialist_configs(specialist_ids)
        self.direct_domains = direct_domains

    def build_supervisor_tools(
        self,
        state: GenieState,
        shared: GenieSharedContext,
        writer: StreamWriter,
    ) -> list[ToolSpec]:
        tools: list[ToolSpec] = []
        tier = state.active_tier
        for cfg in self.specialist_configs:
            tools.append(
                create_agent_as_tool(
                    config=cfg,
                    tier=tier,
                    state=state,
                    shared=shared,
                    factory=self.factory,
                    tool_registry=self.tool_registry,
                    pipeline=self.pipeline,
                    writer=writer,
                )
            )
        for domain in self.direct_domains:
            tools.extend(self.tool_registry.list_tools(domain=domain))
        return tools

    def run_turn(
        self,
        state: GenieState,
        shared: GenieSharedContext,
        writer: StreamWriter,
    ) -> ChatMessage:
        sup_tools = self.build_supervisor_tools(state=state, shared=shared, writer=writer)
        tool_map = {t.name: t for t in sup_tools}

        model_ctx = ModelCallContext(
            state=state,
            shared=shared,
            messages=state.messages,
            tools=sup_tools,
            writer=writer,
            metadata={"agent": "supervisor"},
        )

        if state.messages and state.messages[0].role == "system":
            schemas_str = json.dumps([t.to_openai_tool() for t in sup_tools], sort_keys=True)
            hit = self.factory.kv_cache.pin_prefix(state.messages[0].content, schemas_str)
            state.quant_telemetry.prefix_kv_hit = hit

        output = self.pipeline.run_model(
            model_ctx,
            lambda c: self.factory.get_backend(c.state.active_tier).generate(
                c.messages,
                tools=[t.to_openai_tool() for t in c.tools] if c.tools else None,
                constrain_schema=(c.state.active_tier != "t1"),
            ),
        )

        reasoning = extract_reasoning_from_message(output.message)
        if reasoning:
            writer.emit("subagent_reasoning", specialist="supervisor", tier=state.active_tier, reasoning=reasoning)

        acc = ToolCallChunkAccumulator()
        for ch in output.streamed_chunks:
            cid = acc.ingest(ch)
            writer.emit("tool_message_chunk", specialist="supervisor", call_id=cid, fragment=ch.arguments_fragment)

        invocations = output.message.tool_calls or (acc.finalize() if output.streamed_chunks else [])
        state.messages.append(output.message)

        if not invocations:
            writer.emit("ai_message_chunk", content=output.message.content)
            return output.message

        observations: list[str] = []
        for inv in invocations:
            spec = tool_map.get(inv.name) or self.tool_registry.get(inv.name)
            if spec is not None and spec.domain == "delegation":
                obs = spec.handler(**inv.arguments)
            else:
                t_ctx = ToolCallContext(
                    state=state,
                    shared=shared,
                    invocation=inv,
                    spec=spec,
                    specialist="supervisor",
                    writer=writer,
                )

                def _exec_direct(tc: ToolCallContext) -> str:
                    if tc.spec is None:
                        raise KeyError(f"Unknown tool `{tc.invocation.name}`")
                    res = tc.spec.handler(**tc.invocation.arguments)
                    return res if isinstance(res, str) else json.dumps(res, indent=2, default=str)

                obs = self.pipeline.run_tool(t_ctx, _exec_direct)

            if obs.startswith("[DELEGATION_ERROR]") and state.active_tier == "t1":
                state.active_tier = "t2"
                writer.emit(
                    "middleware_event",
                    middleware="supervisor_recovery",
                    escalated_tier="t2",
                    reason=obs,
                )

            tool_msg = ChatMessage(
                role="tool",
                content=obs,
                name=inv.name,
                tool_call_id=inv.id,
            )
            state.messages.append(tool_msg)
            observations.append(obs)

        synthesis = ChatMessage(
            role="assistant",
            content="\n\n".join(observations),
            reasoning="Synthesized specialist and direct tool observations.",
        )
        state.messages.append(synthesis)
        writer.emit("ai_message_chunk", content=synthesis.content)
        return synthesis
