from __future__ import annotations

import hashlib
import json
from typing import Callable

from pocket_genie.middlewares.base import BaseMiddleware, ToolCallContext


def fingerprint_invocation(tool_name: str, arguments: dict) -> str:
    canonical = json.dumps({"name": tool_name, "args": arguments}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class RepeatToolCallBreakerMiddleware(BaseMiddleware):
    """Prevents cyclical identical tool invocations and requests precision tier escalation."""

    name = "repeat_tool_call_breaker"

    def __init__(self, max_repeats: int = 2) -> None:
        self.max_repeats = max_repeats
        self._counts: dict[str, int] = {}

    def reset(self) -> None:
        self._counts.clear()

    def wrap_tool(
        self,
        ctx: ToolCallContext,
        call_next: Callable[[ToolCallContext], str],
    ) -> str:
        fp = f"{ctx.state.session_id}:{fingerprint_invocation(ctx.invocation.name, ctx.invocation.arguments)}"
        count = self._counts.get(fp, 0) + 1
        self._counts[fp] = count

        if count > self.max_repeats:
            if ctx.state.active_tier == "t1":
                ctx.state.active_tier = "t2"
            elif ctx.state.active_tier == "t2":
                ctx.state.active_tier = "t3"
            event = {
                "middleware": self.name,
                "tool": ctx.invocation.name,
                "repeat_count": count,
                "escalated_tier": ctx.state.active_tier,
            }
            ctx.state.middleware_trace.append(event)
            ctx.writer.emit("middleware_event", **event)
            return (
                f"[LOOP_BREAKER] Halted repeated call #{count} to `{ctx.invocation.name}` "
                f"with identical arguments. Escalated precision tier to {ctx.state.active_tier}."
            )

        return call_next(ctx)
