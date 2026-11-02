"""Initial single-file prototype loop before splitting into registries and specialist subgraphs."""
from __future__ import annotations

import json
from typing import Any, Callable

from pocket_genie.shared import ChatMessage, ToolInvocation


class PrototypeAgentRunner:
    def __init__(self) -> None:
        self.tools: dict[str, Callable[..., Any]] = {}
        self.schemas: dict[str, dict[str, Any]] = {}
        self.history: list[ChatMessage] = []

    def register_tool(self, name: str, description: str, fn: Callable[..., Any]) -> None:
        self.tools[name] = fn
        self.schemas[name] = {"name": name, "description": description}

    def sanitize_history(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        cleaned: list[ChatMessage] = []
        known_ids: set[str] = set()
        for msg in messages:
            for tc in msg.tool_calls:
                known_ids.add(tc.id)
            if msg.role == "tool" and msg.tool_call_id not in known_ids:
                continue
            cleaned.append(msg)
        return cleaned

    def execute_turn(self, prompt: str, tool_call: ToolInvocation | None = None) -> ChatMessage:
        self.history.append(ChatMessage(role="user", content=prompt))
        self.history = self.sanitize_history(self.history)
        if tool_call is None:
            reply = ChatMessage(role="assistant", content=f"Echo: {prompt}")
            self.history.append(reply)
            return reply
        fn = self.tools.get(tool_call.name)
        if fn is None:
            obs = f"[ERROR] Unknown tool: {tool_call.name}"
        else:
            try:
                raw = fn(**tool_call.arguments)
                obs = raw if isinstance(raw, str) else json.dumps(raw)
            except Exception as exc:
                obs = f"[ERROR] {exc}"
        if len(obs) > 1200:
            obs = obs[:600] + "\n...[truncated]...\n" + obs[-400:]
        self.history.append(ChatMessage(role="tool", content=obs, name=tool_call.name, tool_call_id=tool_call.id))
        reply = ChatMessage(role="assistant", content=obs)
        self.history.append(reply)
        return reply
