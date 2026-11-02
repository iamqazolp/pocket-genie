from __future__ import annotations

from typing import Callable

from pocket_genie.middlewares.base import BaseMiddleware, ModelCallContext
from pocket_genie.middlewares.refusal_detection import is_refusal_or_corrupted
from pocket_genie.registry.backends import BackendRegistry, GenerationOutput
from pocket_genie.shared import PrecisionTier

_TIER_ORDER: tuple[PrecisionTier, ...] = ("t1", "t2", "t3")


class FallbackMiddleware(BaseMiddleware):
    """Seamless tier escalation (t1 -> t2 -> t3) upon model refusal, schema corruption, or backend error."""

    name = "fallback"

    def __init__(self, backends: BackendRegistry) -> None:
        self._backends = backends

    def wrap_model(
        self,
        ctx: ModelCallContext,
        call_next: Callable[[ModelCallContext], GenerationOutput],
    ) -> GenerationOutput:
        start_idx = _TIER_ORDER.index(ctx.state.active_tier)
        last_exc: Exception | None = None

        for idx in range(start_idx, len(_TIER_ORDER)):
            tier = _TIER_ORDER[idx]
            ctx.state.active_tier = tier
            try:
                output = (
                    call_next(ctx)
                    if idx == start_idx
                    else self._backends.get(tier).generate(
                        ctx.messages,
                        tools=[t.to_openai_tool() for t in ctx.tools] if ctx.tools else None,
                        constrain_schema=(tier != "t1"),
                    )
                )
            except Exception as exc:
                last_exc = exc
                self._emit_escalation(ctx, from_tier=tier, reason=f"exception:{type(exc).__name__}")
                continue

            flagged, reason = is_refusal_or_corrupted(output)
            if not flagged or idx == len(_TIER_ORDER) - 1:
                ctx.state.quant_telemetry = output.telemetry
                ctx.writer.emit("quant_telemetry", **output.telemetry.to_dict())
                return output

            self._emit_escalation(ctx, from_tier=tier, reason=reason)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("All model fallback tiers exhausted")

    def _emit_escalation(self, ctx: ModelCallContext, from_tier: PrecisionTier, reason: str) -> None:
        next_idx = min(_TIER_ORDER.index(from_tier) + 1, len(_TIER_ORDER) - 1)
        to_tier = _TIER_ORDER[next_idx]
        event = {
            "middleware": self.name,
            "from_tier": from_tier,
            "to_tier": to_tier,
            "reason": reason,
        }
        ctx.state.middleware_trace.append(event)
        ctx.writer.emit("middleware_event", **event)
