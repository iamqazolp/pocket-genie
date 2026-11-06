from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable, Literal

AgentType = Literal[
    "supervisor",
    "scoped-demo",
    "red-agent",
    "run-assistant",
    "direct-specialist",
]

PrecisionTier = Literal["t1", "t2", "t3"]
PrecisionProfileName = Literal["P_fast", "P_mixed", "P_accurate"]


@dataclasses.dataclass(slots=True)
class ToolCallChunk:
    index: int
    id: str | None = None
    name: str | None = None
    arguments_fragment: str = ""


@dataclasses.dataclass(slots=True)
class ToolInvocation:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclasses.dataclass(slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolInvocation] = dataclasses.field(default_factory=list)
    reasoning: str | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ]
        if self.reasoning:
            payload["reasoning"] = self.reasoning
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


StreamEventType = Literal[
    "ai_message_chunk",
    "tool_message_chunk",
    "tool_message",
    "subagent_reasoning",
    "middleware_event",
    "quant_telemetry",
    "hitl_request",
    "final",
]


@dataclasses.dataclass(slots=True)
class StreamEvent:
    event_type: StreamEventType
    data: dict[str, Any]


class StreamWriter:
    """Collects and dispatches incremental execution events to SSE or CLI listeners."""

    def __init__(self, callback: Callable[[StreamEvent], None] | None = None) -> None:
        self._events: list[StreamEvent] = []
        self._callback = callback

    @property
    def events(self) -> list[StreamEvent]:
        return list(self._events)

    def emit(self, event_type: StreamEventType, **data: Any) -> StreamEvent:
        ev = StreamEvent(event_type=event_type, data=data)
        self._events.append(ev)
        if self._callback is not None:
            self._callback(ev)
        return ev


@dataclasses.dataclass(slots=True)
class QuantTelemetrySnapshot:
    tier: PrecisionTier = "t1"
    profile: PrecisionProfileName = "P_fast"
    weight_bits: float = 4.0
    effective_act_bits: float = 8.0
    promoted_tokens: int = 0
    total_tokens: int = 0
    min_logit_margin: float = 5.0
    mean_entropy: float = 0.15
    prefix_kv_hit: bool = True
    kv_cache_fp16_mb: float = 0.0
    kv_cache_quant_mb: float = 0.0
    weight_footprint_mb: float = 315.1
    fallback_overhead_mb: float = 0.0

    @property
    def promoted_token_ratio(self) -> float:
        if self.total_tokens <= 0:
            return 0.0
        return self.promoted_tokens / float(self.total_tokens)

    @property
    def kv_memory_saved_mb(self) -> float:
        return max(0.0, self.kv_cache_fp16_mb - self.kv_cache_quant_mb)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["promoted_token_ratio"] = round(self.promoted_token_ratio, 4)
        d["kv_memory_saved_mb"] = round(self.kv_memory_saved_mb, 3)
        return d


@dataclasses.dataclass(slots=True)
class GenieSharedContext:
    session_id: str
    session_created_at: str
    workspace_dir: Path
    project_id: str | None = None
    agent_type: AgentType = "supervisor"
    target_specialist: str | None = None
    active_tier: PrecisionTier = "t1"
    auto_approve_hitl: bool = False
