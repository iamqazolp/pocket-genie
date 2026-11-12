from __future__ import annotations

import dataclasses
from typing import Any

from pocket_genie.shared import (
    AgentType,
    ChatMessage,
    PrecisionTier,
    QuantTelemetrySnapshot,
)


@dataclasses.dataclass(slots=True)
class HITLApprovalRequest:
    request_id: str
    tool_name: str
    arguments: dict[str, Any]
    specialist: str
    tier: PrecisionTier
    reason: str
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(slots=True)
class GenieState:
    session_id: str
    session_created_at: str
    agent_type: AgentType = "supervisor"
    project_id: str | None = None
    target_specialist: str | None = None
    messages: list[ChatMessage] = dataclasses.field(default_factory=list)
    full_history: list[ChatMessage] = dataclasses.field(default_factory=list)
    user_messages_summary: str = ""
    recent_session_summaries: list[str] = dataclasses.field(default_factory=list)
    pinned_memories: list[str] = dataclasses.field(default_factory=list)
    running_summary: str = ""
    prefix_kv_pinned: bool = False
    active_tier: PrecisionTier = "t1"
    quant_telemetry: QuantTelemetrySnapshot = dataclasses.field(
        default_factory=QuantTelemetrySnapshot
    )
    pending_hitl: list[HITLApprovalRequest] = dataclasses.field(default_factory=list)
    middleware_trace: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    decision_triage: dict[str, Any] = dataclasses.field(default_factory=dict)
