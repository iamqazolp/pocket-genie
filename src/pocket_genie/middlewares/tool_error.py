from __future__ import annotations

from typing import Callable

from pocket_genie.middlewares.base import BaseMiddleware, ToolCallContext


class ToolErrorMiddleware(BaseMiddleware):
    """Catches runtime exceptions from tools and formats them into safe [TOOL_ERROR] strings."""

    name = "tool_error"

    def wrap_tool(
        self,
        ctx: ToolCallContext,
        call_next: Callable[[ToolCallContext], str],
    ) -> str:
        try:
            return call_next(ctx)
        except Exception as exc:
            err_msg = f"[TOOL_ERROR] Tool `{ctx.invocation.name}` failed ({type(exc).__name__}): {exc}"
            event = {
                "middleware": self.name,
                "tool": ctx.invocation.name,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            ctx.state.middleware_trace.append(event)
            ctx.writer.emit("middleware_event", **event)
            return err_msg
