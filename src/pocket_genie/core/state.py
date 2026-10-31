from __future__ import annotations

import dataclasses
from typing import Any

from pocket_genie.shared import (
    AgentType,
    ChatMessage,
    PrecisionProfileName,
    PrecisionTier,
)


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
