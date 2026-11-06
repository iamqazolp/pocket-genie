from __future__ import annotations

import time
from typing import Callable

from pocket_genie.middlewares.base import BaseMiddleware, ModelCallContext, ToolCallContext
from pocket_genie.observability.langfuse_tracer import GenieLangfuseTracer
from pocket_genie.registry.backends import GenerationOutput


class LangfuseTracingMiddleware(BaseMiddleware):
    """Middleware stage recording model generations, quantization telemetry, and tool spans into Langfuse."""

    name = "langfuse_tracing"

    def __init__(self, tracer: GenieLangfuseTracer) -> None:
        self.tracer = tracer

    def wrap_model(
        self,
        ctx: ModelCallContext,
        call_next: Callable[[ModelCallContext], GenerationOutput],
    ) -> GenerationOutput:
        t0 = time.perf_counter()
        prev_mw_len = len(ctx.state.middleware_trace)
        output = call_next(ctx)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        agent_label = str(
            ctx.metadata.get("specialist")
            or ctx.metadata.get("agent")
            or ctx.shared.agent_type
        )
        self.tracer.record_generation(
            name=f"llm:{agent_label}:{output.telemetry.tier}",
            model=f"{output.telemetry.tier}/{output.telemetry.profile}",
            input_messages=[m.to_dict() for m in ctx.messages],
            output_message=output.message.to_dict(),
            telemetry=output.telemetry.to_dict(),
            latency_ms=elapsed_ms,
        )

        for mw_ev in ctx.state.middleware_trace[prev_mw_len:]:
            self.tracer.record_guardrail_event(
                guardrail_name=str(mw_ev.get("middleware", "guardrail")),
                details=mw_ev,
            )
        return output

    def wrap_tool(
        self,
        ctx: ToolCallContext,
        call_next: Callable[[ToolCallContext], str],
    ) -> str:
        t0 = time.perf_counter()
        prev_mw_len = len(ctx.state.middleware_trace)
        obs = call_next(ctx)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.tracer.record_tool_span(
            tool_name=ctx.invocation.name,
            arguments=ctx.invocation.arguments,
            observation=obs,
            specialist=ctx.specialist,
            tier=ctx.state.active_tier,
            latency_ms=elapsed_ms,
        )

        for mw_ev in ctx.state.middleware_trace[prev_mw_len:]:
            self.tracer.record_guardrail_event(
                guardrail_name=str(mw_ev.get("middleware", "guardrail")),
                details=mw_ev,
            )
        return obs
