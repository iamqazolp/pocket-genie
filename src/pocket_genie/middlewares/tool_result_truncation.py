from __future__ import annotations

from typing import Callable

from pocket_genie.core.memory import estimate_tokens
from pocket_genie.middlewares.base import BaseMiddleware, ToolCallContext


def truncate_tool_output(text: str, max_tokens: int = 512) -> tuple[str, bool]:
    """Cap oversized tool outputs to protect edge context windows."""
    est = estimate_tokens(text)
    if est <= max_tokens:
        return text, False

    max_chars = max(64, max_tokens * 4)
    head_chars = int(max_chars * 0.7)
    tail_chars = max(16, max_chars - head_chars)
    head_part = text[:head_chars]
    tail_part = text[-tail_chars:]
    if "\n" in head_part:
        last_nl = head_part.rfind("\n")
        if last_nl >= head_chars // 2:
            head_part = head_part[:last_nl]
    if "\n" in tail_part:
        first_nl = tail_part.find("\n")
        if 0 <= first_nl <= len(tail_part) // 2:
            tail_part = tail_part[first_nl + 1 :]
    omitted_chars = max(0, len(text) - (len(head_part) + len(tail_part)))
    truncated = (
        f"{head_part}\n"
        f"... [TRUNCATED: {omitted_chars} chars omitted to fit {max_tokens}-token budget] ...\n"
        f"{tail_part}"
    )
    return truncated, True


class ToolResultTruncationMiddleware(BaseMiddleware):
    name = "tool_result_truncation"

    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens

    def wrap_tool(
        self,
        ctx: ToolCallContext,
        call_next: Callable[[ToolCallContext], str],
    ) -> str:
        raw = str(call_next(ctx))
        out, was_truncated = truncate_tool_output(raw, max_tokens=self.max_tokens)
        if was_truncated:
            event = {
                "middleware": self.name,
                "tool": ctx.invocation.name,
                "original_chars": len(raw),
                "truncated_chars": len(out),
            }
            ctx.state.middleware_trace.append(event)
            ctx.writer.emit("middleware_event", **event)
        return out
