from __future__ import annotations

import json

from pocket_genie.core.agent_configs import SpecialistConfig
from pocket_genie.core.state import GenieState
from pocket_genie.core.tiered_agent_factory import TieredAgentFactory
from pocket_genie.middlewares.base import (
    MiddlewarePipeline,
    ModelCallContext,
    ToolCallContext,
)
from pocket_genie.registry.tools import ToolRegistry, ToolSpec
from pocket_genie.shared import (
    ChatMessage,
    GenieSharedContext,
    PrecisionTier,
    StreamWriter,
    ToolCallChunk,
    ToolInvocation,
)


class ToolCallChunkAccumulator:
    """Progressively reassembles fragmented streaming tool_call_chunks by index and call ID."""

    def __init__(self) -> None:
        self.chunk_index_to_id: dict[int, str] = {}
        self.names_by_id: dict[str, str] = {}
        self.args_buffer_by_id: dict[str, list[str]] = {}

    def ingest(self, chunk: ToolCallChunk) -> str:
        if chunk.id:
            self.chunk_index_to_id[chunk.index] = chunk.id
        call_id = self.chunk_index_to_id.get(chunk.index) or f"call_{chunk.index}"
        self.chunk_index_to_id[chunk.index] = call_id
        if chunk.name:
            self.names_by_id[call_id] = chunk.name
        if chunk.arguments_fragment:
            self.args_buffer_by_id.setdefault(call_id, []).append(chunk.arguments_fragment)
        return call_id

    def finalize(self) -> list[ToolInvocation]:
        invocations: list[ToolInvocation] = []
        for idx in sorted(self.chunk_index_to_id.keys()):
            call_id = self.chunk_index_to_id[idx]
            name = self.names_by_id.get(call_id, "unknown")
            raw_json = "".join(self.args_buffer_by_id.get(call_id, [])) or "{}"
            try:
                parsed = json.loads(raw_json)
                if not isinstance(parsed, dict):
                    parsed = {"value": parsed}
            except json.JSONDecodeError:
                parsed = {"_raw": raw_json}
            invocations.append(ToolInvocation(id=call_id, name=name, arguments=parsed))
        return invocations


def extract_reasoning_from_message(msg: ChatMessage) -> str | None:
    if msg.reasoning and msg.reasoning.strip():
        return msg.reasoning.strip()
    return None


def execute_specialist_subgraph(
    config: SpecialistConfig,
    task: str,
    context: str,
    tier: PrecisionTier,
    state: GenieState,
    shared: GenieSharedContext,
    factory: TieredAgentFactory,
    tool_registry: ToolRegistry,
    pipeline: MiddlewarePipeline,
    writer: StreamWriter,
    max_steps: int = 3,
) -> str:
    """Run an isolated specialist ReAct sub-graph with domain-scoped tools, streaming chunks, and error recovery."""
    scoped_tools: list[ToolSpec] = []
    for domain in config.tool_domains:
        scoped_tools.extend(tool_registry.list_tools(domain=domain))

    sub_messages = [
        ChatMessage(role="system", content=config.system_prompt),
        ChatMessage(
            role="user",
            content=f"Task: {task}\nContext: {context}" if context else task,
        ),
    ]

    prev_tier = state.active_tier
    state.active_tier = tier
    all_observations: list[str] = []
    try:
        for _step_idx in range(max(1, max_steps)):
            model_ctx = ModelCallContext(
                state=state,
                shared=shared,
                messages=sub_messages,
                tools=scoped_tools,
                writer=writer,
                metadata={"specialist": config.agent_id},
            )
            output = pipeline.run_model(
                model_ctx,
                lambda c: factory.get_backend(c.state.active_tier).generate(
                    c.messages,
                    tools=[t.to_openai_tool() for t in c.tools] if c.tools else None,
                    constrain_schema=(c.state.active_tier != "t1"),
                ),
            )

            reasoning = extract_reasoning_from_message(output.message)
            if reasoning:
                writer.emit(
                    "subagent_reasoning",
                    specialist=config.agent_id,
                    tier=state.active_tier,
                    reasoning=reasoning,
                )

            accumulator = ToolCallChunkAccumulator()
            for ch in output.streamed_chunks:
                cid = accumulator.ingest(ch)
                writer.emit(
                    "tool_message_chunk",
                    specialist=config.agent_id,
                    call_id=cid,
                    fragment=ch.arguments_fragment,
                )

            invocations = output.message.tool_calls or (
                accumulator.finalize() if output.streamed_chunks else []
            )
            if not invocations:
                if output.message.content:
                    return output.message.content
                if all_observations:
                    return "\n".join(all_observations)
                return f"[{config.agent_id}] Completed with no tool calls."

            sub_messages.append(output.message)
            step_observations: list[str] = []
            halted = False
            for inv in invocations:
                spec = tool_registry.get(inv.name)
                tool_ctx = ToolCallContext(
                    state=state,
                    shared=shared,
                    invocation=inv,
                    spec=spec,
                    specialist=config.agent_id,
                    writer=writer,
                )

                def _invoke_tool(tc: ToolCallContext) -> str:
                    if tc.spec is None:
                        raise KeyError(
                            f"Specialist `{config.agent_id}` called unregistered tool `{tc.invocation.name}`"
                        )
                    res = tc.spec.handler(**tc.invocation.arguments)
                    return res if isinstance(res, str) else json.dumps(res, indent=2, default=str)

                obs = pipeline.run_tool(tool_ctx, _invoke_tool)
                writer.emit(
                    "tool_message",
                    specialist=config.agent_id,
                    tool=inv.name,
                    observation=obs,
                )
                formatted_obs = f"[{inv.name}] {obs}"
                step_observations.append(formatted_obs)
                all_observations.append(formatted_obs)
                sub_messages.append(
                    ChatMessage(
                        role="tool",
                        content=obs,
                        name=inv.name,
                        tool_call_id=inv.id,
                    )
                )
                if obs.startswith(("[HITL_PENDING:", "[HITL_REJECTED]", "[LOOP_BREAKER]")):
                    halted = True

            if halted:
                return "\n".join(step_observations)

        return "\n".join(all_observations)
    except Exception as exc:
        err = f"[DELEGATION_ERROR] Specialist `{config.agent_id}` failed on tier {tier}: {exc}"
        writer.emit(
            "middleware_event",
            middleware="delegation",
            specialist=config.agent_id,
            error=str(exc),
        )
        return err
    finally:
        state.active_tier = prev_tier


def create_agent_as_tool(
    config: SpecialistConfig,
    tier: PrecisionTier,
    state: GenieState,
    shared: GenieSharedContext,
    factory: TieredAgentFactory,
    tool_registry: ToolRegistry,
    pipeline: MiddlewarePipeline,
    writer: StreamWriter,
) -> ToolSpec:
    """Wrap a SpecialistConfig into a typed delegation tool `create_task_<agent_id>_t<tier>`."""
    tool_name = f"create_task_{config.agent_id}_{tier}"

    def _delegate_handler(task: str, context: str = "") -> str:
        return execute_specialist_subgraph(
            config=config,
            task=task,
            context=context,
            tier=tier,
            state=state,
            shared=shared,
            factory=factory,
            tool_registry=tool_registry,
            pipeline=pipeline,
            writer=writer,
        )

    return ToolSpec(
        name=tool_name,
        description=f"[{config.display_name} ({tier})] {config.description}",
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": f"Actionable instruction for {config.display_name}.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional scoped context or file/table hints.",
                },
            },
            "required": ["task"],
        },
        handler=_delegate_handler,
        domain="delegation",
        mutating=False,
    )
