from __future__ import annotations

import dataclasses
from typing import Iterable

from pocket_genie.shared import ChatMessage


@dataclasses.dataclass(slots=True)
class RunningSummary:
    summary_text: str = ""
    compacted_message_count: int = 0
    compacted_token_estimate: int = 0

    def update(self, evicted_messages: Iterable[ChatMessage]) -> str:
        snippets: list[str] = []
        added_tokens = 0
        count = 0
        for msg in evicted_messages:
            count += 1
            text = " ".join(msg.content.strip().split())
            if not text and msg.tool_calls:
                calls = ", ".join(tc.name for tc in msg.tool_calls)
                text = f"invoked tools [{calls}]"
            if len(text) > 140:
                text = text[:137] + "..."
            snippets.append(f"[{msg.role}] {text}")
            added_tokens += estimate_message_tokens(msg)

        if snippets:
            delta = " | ".join(snippets)
            if self.summary_text:
                self.summary_text = f"{self.summary_text} || {delta}"
            else:
                self.summary_text = delta
            if len(self.summary_text) > 1200:
                self.summary_text = "..." + self.summary_text[-1197:]
        self.compacted_message_count += count
        self.compacted_token_estimate += added_tokens
        return self.summary_text


class SessionMemoryStore:
    """In-memory and optional file-backed store for user memories and session summaries."""

    def __init__(self) -> None:
        self._user_memories: list[str] = []
        self._session_summaries: list[tuple[str, str]] = []
        self._session_cache: dict[str, list[str]] = {}

    def add_memory(self, fact: str) -> str:
        cleaned = fact.strip()
        if cleaned and cleaned not in self._user_memories:
            self._user_memories.append(cleaned)
        return cleaned

    def list_memories(self) -> list[str]:
        return list(self._user_memories)

    def record_session_summary(self, session_id: str, summary: str) -> None:
        cleaned = summary.strip()
        if not cleaned:
            return
        self._session_summaries = [
            (sid, s) for sid, s in self._session_summaries if sid != session_id
        ]
        self._session_summaries.append((session_id, cleaned))
        self._session_cache.clear()

    def recent_summaries(self, exclude_session_id: str | None = None, limit: int = 3) -> list[str]:
        cache_key = f"{exclude_session_id}:{limit}"
        if cache_key in self._session_cache:
            return list(self._session_cache[cache_key])
        filtered = [
            s for sid, s in reversed(self._session_summaries) if sid != exclude_session_id
        ][:limit]
        self._session_cache[cache_key] = filtered
        return list(filtered)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_message_tokens(msg: ChatMessage) -> int:
    base = estimate_tokens(msg.content) + 4
    if msg.reasoning:
        base += estimate_tokens(msg.reasoning)
    for tc in msg.tool_calls:
        base += estimate_tokens(tc.name) + estimate_tokens(str(tc.arguments))
    return base


def summarize_user_queries(messages: Iterable[ChatMessage], previous_summary: str = "") -> str:
    queries = [
        " ".join(m.content.strip().split())
        for m in messages
        if m.role == "user" and m.content.strip()
    ]
    if not queries:
        return previous_summary
    latest = "; ".join(q[:100] for q in queries[-4:])
    return f"User objectives: {latest}"
