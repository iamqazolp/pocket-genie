"""Prototype agent loop backed by ToolRegistry."""
from __future__ import annotations

import json

from pocket_genie.registry.tools import ToolRegistry
from pocket_genie.shared import ChatMessage, ToolInvocation


class PrototypeAgentRunner:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry()
        self.history: list[ChatMessage] = []

    def execute_turn(self, prompt: str, tool_call: ToolInvocation | None = None) -> ChatMessage:
        self.history.append(ChatMessage(role="user", content=prompt))
        if tool_call is None:
            reply = ChatMessage(role="assistant", content=f"Ack: {prompt}")
            self.history.append(reply)
            return reply
        res = self.registry.execute(tool_call.name, tool_call.arguments)
        obs = res if isinstance(res, str) else json.dumps(res)
        reply = ChatMessage(role="assistant", content=obs)
        self.history.append(reply)
        return reply
