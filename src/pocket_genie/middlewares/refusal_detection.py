from __future__ import annotations

from pocket_genie.middlewares.base import BaseMiddleware, ModelCallContext
from pocket_genie.registry.backends import GenerationOutput

_REFUSAL_MARKERS = (
    "i cannot fulfill this request",
    "i am unable to assist with",
    "as an ai, i cannot",
    "[refusal]",
)


def is_refusal_or_corrupted(output: GenerationOutput) -> tuple[bool, str]:
    msg = output.message
    lower = msg.content.strip().lower()
    for marker in _REFUSAL_MARKERS:
        if marker in lower:
            return True, "model_refusal"
    if not msg.content.strip() and not msg.tool_calls:
        return True, "empty_generation"
    if "<tool_call>" in msg.content and not msg.tool_calls:
        return True, "corrupted_tool_json"
    return False, "ok"


class RefusalDetectionMiddleware(BaseMiddleware):
    name = "refusal_detection"

    def after_model(self, ctx: ModelCallContext, output: GenerationOutput) -> GenerationOutput:
        flagged, reason = is_refusal_or_corrupted(output)
        if flagged:
            ctx.metadata["fault_reason"] = reason
            event = {
                "middleware": self.name,
                "reason": reason,
                "tier": ctx.state.active_tier,
            }
            ctx.state.middleware_trace.append(event)
            ctx.writer.emit("middleware_event", **event)
        return output
