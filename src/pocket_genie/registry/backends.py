from __future__ import annotations

import dataclasses
import json
import urllib.request
from typing import Any, Protocol

from pocket_genie.core.state import QuantTelemetrySnapshot
from pocket_genie.shared import (
    ChatMessage,
    PrecisionTier,
    ToolCallChunk,
    ToolInvocation,
)


@dataclasses.dataclass(slots=True)
class GenerationOutput:
    message: ChatMessage
    streamed_chunks: list[ToolCallChunk] = dataclasses.field(default_factory=list)
    telemetry: QuantTelemetrySnapshot = dataclasses.field(
        default_factory=QuantTelemetrySnapshot
    )


class ModelBackend(Protocol):
    name: str
    tier: PrecisionTier

    def generate(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        constrain_schema: bool = False,
    ) -> GenerationOutput: ...


class DeterministicReplayBackend:
    """Deterministic backend for fast CI verification, Termux offline execution, and fault testing."""

    def __init__(
        self,
        tier: PrecisionTier = "t1",
        scripted_responses: list[CallableResponse | GenerationOutput] | None = None,
    ) -> None:
        self.name = f"deterministic-{tier}"
        self.tier = tier
        self._queue: list[ Any ] = list(scripted_responses or [])

    def push_response(self, response: Any) -> None:
        self._queue.append(response)

    def generate(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        constrain_schema: bool = False,
    ) -> GenerationOutput:
        profile_map = {"t1": "P_fast", "t2": "P_mixed", "t3": "P_accurate"}
        act_bits_map = {"t1": 8.0, "t2": 8.38, "t3": 16.0}
        telemetry = QuantTelemetrySnapshot(
            tier=self.tier,
            profile=profile_map[self.tier],  # type: ignore[arg-type]
            weight_bits=4.0,
            effective_act_bits=act_bits_map[self.tier],
            promoted_tokens=0 if self.tier == "t1" else (4 if self.tier == "t2" else 32),
            total_tokens=32,
            min_logit_margin=4.2 if self.tier == "t1" else 6.1,
            mean_entropy=0.12 if self.tier != "t1" else 0.21,
            prefix_kv_hit=True,
            kv_cache_fp16_mb=12.4,
            kv_cache_quant_mb=3.6,
        )

        if self._queue:
            item = self._queue.pop(0)
            if callable(item):
                return item(messages, tools, constrain_schema, telemetry)
            if isinstance(item, GenerationOutput):
                return item
            if isinstance(item, ChatMessage):
                return GenerationOutput(message=item, telemetry=telemetry)

        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        if tools:
            q_lower = last_user.lower()
            chosen = tools[0]["function"]
            best_score = -1
            for t in tools:
                fn = t["function"]
                t_name = fn["name"].lower()
                t_desc = fn.get("description", "").lower()
                score = sum(
                    2 for tok in q_lower.replace(".", " ").replace("_", " ").split()
                    if len(tok) >= 3 and (tok in t_name or tok in t_desc)
                )
                if score > best_score:
                    best_score = score
                    chosen = fn

            first_tool = chosen["name"]
            args: dict[str, Any] = {}
            props = chosen.get("parameters", {}).get("properties", {})
            if "query" in props:
                args["query"] = last_user or "status"
            elif "task" in props:
                args["task"] = last_user or "inspect workspace"
            raw_json = json.dumps(args)
            mid = max(1, len(raw_json) // 2)
            chunks = [
                ToolCallChunk(index=0, id="call_01", name=first_tool, arguments_fragment=raw_json[:mid]),
                ToolCallChunk(index=0, id=None, name=None, arguments_fragment=raw_json[mid:]),
            ]
            msg = ChatMessage(
                role="assistant",
                content="",
                reasoning=f"Selected `{first_tool}` on tier {self.tier} ({telemetry.profile}).",
                tool_calls=[ToolInvocation(id="call_01", name=first_tool, arguments=args)],
            )
            return GenerationOutput(message=msg, streamed_chunks=chunks, telemetry=telemetry)

        return GenerationOutput(
            message=ChatMessage(
                role="assistant",
                content=f"[{self.tier}/{telemetry.profile}] Completed analysis for: {last_user}",
                reasoning="Synthesizing final response from specialist observations.",
            ),
            telemetry=telemetry,
        )


CallableResponse = Any


class OpenAICompatibleBackend:
    """HTTP client backend for local llama.cpp, Ollama, vLLM, or OpenAI-compatible endpoints."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        model: str = "qwen2.5:0.5b",
        api_key: str = "local",
        tier: PrecisionTier = "t1",
        timeout: float = 30.0,
    ) -> None:
        self.name = f"openai-compat:{model}"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.tier = tier
        self.timeout = timeout

    def generate(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        constrain_schema: bool = False,
    ) -> GenerationOutput:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": 0.0,
        }
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        choice = body["choices"][0]["message"]
        invocations: list[ToolInvocation] = []
        for idx, tc in enumerate(choice.get("tool_calls") or []):
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            invocations.append(
                ToolInvocation(
                    id=tc.get("id") or f"call_{idx}",
                    name=fn.get("name", ""),
                    arguments=parsed_args,
                )
            )
        msg = ChatMessage(
            role="assistant",
            content=choice.get("content") or "",
            tool_calls=invocations,
        )
        return GenerationOutput(
            message=msg,
            telemetry=QuantTelemetrySnapshot(tier=self.tier),
        )


class BackendRegistry:
    """Manages model backends mapped to precision tiers t1, t2, and t3."""

    def __init__(self) -> None:
        self._backends: dict[PrecisionTier, ModelBackend] = {
            "t1": DeterministicReplayBackend(tier="t1"),
            "t2": DeterministicReplayBackend(tier="t2"),
            "t3": DeterministicReplayBackend(tier="t3"),
        }

    def register(self, tier: PrecisionTier, backend: ModelBackend) -> None:
        self._backends[tier] = backend

    def get(self, tier: PrecisionTier) -> ModelBackend:
        return self._backends[tier]
