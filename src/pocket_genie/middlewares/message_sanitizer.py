from __future__ import annotations

import re

from pocket_genie.middlewares.base import BaseMiddleware, ModelCallContext
from pocket_genie.shared import ChatMessage

_DANGLING_TOOL_TAG_RE = re.compile(r"<tool_call>(?:(?!</tool_call>).)*\Z", re.DOTALL)


def sanitize_messages(messages: list[ChatMessage]) -> tuple[list[ChatMessage], int]:
    """Remove orphaned tool messages and strip unclosed <tool_call> tags from interrupted turns."""
    valid_call_ids: set[str] = set()
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.id:
                    valid_call_ids.add(tc.id)

    resolved_tool_ids: set[str] = {
        msg.tool_call_id
        for msg in messages
        if msg.role == "tool" and msg.tool_call_id in valid_call_ids
    }

    cleaned: list[ChatMessage] = []
    removed_or_repaired = 0

    for msg in messages:
        if msg.role == "tool":
            if not msg.tool_call_id or msg.tool_call_id not in valid_call_ids:
                removed_or_repaired += 1
                continue
            cleaned.append(msg)
            continue

        content = msg.content
        if "<tool_call>" in content and "</tool_call>" not in content:
            content = _DANGLING_TOOL_TAG_RE.sub("", content).strip()
            removed_or_repaired += 1

        if msg.role == "assistant" and msg.tool_calls:
            kept_calls = [tc for tc in msg.tool_calls if tc.id in resolved_tool_ids]
            if len(kept_calls) != len(msg.tool_calls):
                removed_or_repaired += len(msg.tool_calls) - len(kept_calls)
            if not kept_calls and not content:
                continue
            cleaned.append(
                ChatMessage(
                    role=msg.role,
                    content=content,
                    name=msg.name,
                    tool_call_id=msg.tool_call_id,
                    tool_calls=kept_calls,
                    reasoning=msg.reasoning,
                    metadata=dict(msg.metadata),
                )
            )
            continue

        cleaned.append(
            ChatMessage(
                role=msg.role,
                content=content,
                name=msg.name,
                tool_call_id=msg.tool_call_id,
                tool_calls=list(msg.tool_calls),
                reasoning=msg.reasoning,
                metadata=dict(msg.metadata),
            )
        )

    return cleaned, removed_or_repaired


class MessageSanitizerMiddleware(BaseMiddleware):
    name = "message_sanitizer"

    def before_model(self, ctx: ModelCallContext) -> None:
        cleaned, repaired = sanitize_messages(ctx.messages)
        ctx.messages[:] = cleaned
        if repaired > 0:
            event = {"middleware": self.name, "repaired_count": repaired}
            ctx.state.middleware_trace.append(event)
            ctx.writer.emit("middleware_event", **event)
