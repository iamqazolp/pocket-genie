from __future__ import annotations

from pocket_genie.core.memory import RunningSummary, estimate_message_tokens
from pocket_genie.middlewares.base import BaseMiddleware, ModelCallContext
from pocket_genie.shared import ChatMessage


class CustomSummarizationMiddleware(BaseMiddleware):
    """Sliding-window conversation chunker that compacts older turns into a RunningSummary."""

    name = "summarization"

    def __init__(
        self,
        summary_token_limit: int = 4000,
        retain_token_budget: int = 2000,
        max_messages: int = 50,
    ) -> None:
        self.summary_token_limit = summary_token_limit
        self.retain_token_budget = retain_token_budget
        self.max_messages = max_messages
        self._running = RunningSummary()

    def before_model(self, ctx: ModelCallContext) -> None:
        total_tokens = sum(estimate_message_tokens(m) for m in ctx.messages)
        if total_tokens <= self.summary_token_limit and len(ctx.messages) <= self.max_messages:
            return

        system_msgs = [m for m in ctx.messages if m.role == "system"]
        conv_msgs = [m for m in ctx.messages if m.role != "system"]
        if len(conv_msgs) <= 2:
            return

        retained: list[ChatMessage] = []
        retained_tokens = 0
        for msg in reversed(conv_msgs):
            msg_tokens = estimate_message_tokens(msg)
            if retained and (retained_tokens + msg_tokens > self.retain_token_budget):
                break
            retained.append(msg)
            retained_tokens += msg_tokens
        retained.reverse()

        evicted_count = len(conv_msgs) - len(retained)
        if evicted_count <= 0:
            evicted = conv_msgs[:-2]
            retained = conv_msgs[-2:]
        else:
            evicted = conv_msgs[:evicted_count]

        summary_str = self._running.update(evicted)
        ctx.state.running_summary = summary_str

        summary_msg = ChatMessage(
            role="system",
            content=f"[RunningSummary ({self._running.compacted_message_count} turns compacted)]: {summary_str}",
            metadata={"compacted": True},
        )
        ctx.messages[:] = [*system_msgs, summary_msg, *retained]
        event = {
            "middleware": self.name,
            "evicted_messages": len(evicted),
            "retained_messages": len(retained),
            "total_tokens_before": total_tokens,
        }
        ctx.state.middleware_trace.append(event)
        ctx.writer.emit("middleware_event", **event)
