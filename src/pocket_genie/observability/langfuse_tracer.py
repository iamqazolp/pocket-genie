from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import os
import time
import uuid
from collections import deque
from typing import Any, Iterator

try:
    from langfuse import Langfuse, propagate_attributes
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None  # type: ignore[assignment,misc]
    propagate_attributes = None  # type: ignore[assignment]
    LangfuseCallbackHandler = None  # type: ignore[assignment,misc]
    TracerProvider = None  # type: ignore[assignment,misc]
    SimpleSpanProcessor = None  # type: ignore[assignment,misc]
    InMemorySpanExporter = None  # type: ignore[assignment,misc]


@dataclasses.dataclass(slots=True)
class ObservationRecord:
    observation_id: str
    as_type: str
    name: str
    started_at: str
    latency_ms: float
    model: str | None = None
    input_data: Any = None
    output_data: Any = None
    usage: dict[str, int] = dataclasses.field(default_factory=dict)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(slots=True)
class TraceRecord:
    trace_id: str
    session_id: str
    name: str
    started_at: str
    latency_ms: float = 0.0
    input_prompt: str = ""
    output_reply: str = ""
    active_tier: str = "t1"
    profile: str = "P_fast"
    observations: list[ObservationRecord] = dataclasses.field(default_factory=list)
    scores: dict[str, float] = dataclasses.field(default_factory=dict)
    tags: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "name": self.name,
            "started_at": self.started_at,
            "latency_ms": round(self.latency_ms, 2),
            "input_prompt": self.input_prompt,
            "output_reply": self.output_reply,
            "active_tier": self.active_tier,
            "profile": self.profile,
            "observations": [o.to_dict() for o in self.observations],
            "scores": dict(self.scores),
            "tags": list(self.tags),
        }


class GenieLangfuseTracer:
    """Dual-mode Langfuse observability tracer recording agent spans, quant telemetry, and LangChain callbacks."""

    def __init__(
        self,
        client: Any | None = None,
        in_memory_otel: bool = False,
        max_local_traces: int = 50,
    ) -> None:
        self._local_traces: deque[TraceRecord] = deque(maxlen=max_local_traces)
        self._active_trace: TraceRecord | None = None
        self._active_otel_span: Any | None = None
        self._memory_exporter: Any | None = None
        self._public_key: str | None = None
        self.client: Any | None = client

        if self.client is None and LANGFUSE_AVAILABLE:
            pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
            sk = os.environ.get("LANGFUSE_SECRET_KEY")
            host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
            if in_memory_otel:
                self._memory_exporter = InMemorySpanExporter()
                provider = TracerProvider()
                provider.add_span_processor(SimpleSpanProcessor(self._memory_exporter))
                self._public_key = pk or "pk-lf-pocket-genie-local"
                self.client = Langfuse(
                    public_key=self._public_key,
                    secret_key=sk or "sk-lf-pocket-genie-local",
                    host=host or "http://127.0.0.1:3000",
                    tracer_provider=provider,
                )
            elif pk and sk:
                self._public_key = pk
                self.client = Langfuse(
                    public_key=pk,
                    secret_key=sk,
                    host=host,
                )

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @property
    def active_trace(self) -> TraceRecord | None:
        return self._active_trace

    def list_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        items = list(self._local_traces)[-limit:]
        return [t.to_dict() for t in reversed(items)]

    def exported_otel_spans(self) -> list[Any]:
        if self._memory_exporter is None:
            return []
        return list(self._memory_exporter.get_finished_spans())

    def get_langchain_callback_handler(self, trace_id: str | None = None) -> Any | None:
        """Return a real langfuse.langchain.CallbackHandler bound to the active trace context when enabled."""
        if not LANGFUSE_AVAILABLE or not self.enabled or LangfuseCallbackHandler is None:
            return None
        trace_ctx = {"trace_id": trace_id} if trace_id and len(trace_id) == 32 else None
        return LangfuseCallbackHandler(
            public_key=self._public_key,
            trace_context=trace_ctx,  # type: ignore[arg-type]
        )

    @contextlib.contextmanager
    def trace_turn(
        self,
        *,
        session_id: str,
        name: str = "pocket-genie-turn",
        input_prompt: str = "",
        agent_type: str = "supervisor",
        tier: str = "t1",
    ) -> Iterator[TraceRecord]:
        trace_id = uuid.uuid4().hex
        now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        record = TraceRecord(
            trace_id=trace_id,
            session_id=session_id,
            name=name,
            started_at=now_iso,
            input_prompt=input_prompt,
            active_tier=tier,
            tags=[f"agent:{agent_type}", f"start_tier:{tier}"],
        )
        prev_trace = self._active_trace
        self._active_trace = record
        t0 = time.perf_counter()

        if self.client is not None and propagate_attributes is not None:
            with (
                self.client.start_as_current_observation(
                    name=name,
                    as_type="agent",
                    input={"prompt": input_prompt, "agent_type": agent_type, "tier": tier},
                    metadata={"session_id": session_id, "agent_type": agent_type},
                ) as root_obs,
                propagate_attributes(
                    session_id=session_id,
                    tags=record.tags,
                ),
            ):
                prev_span = self._active_otel_span
                self._active_otel_span = root_obs
                try:
                    yield record
                finally:
                    record.latency_ms = (time.perf_counter() - t0) * 1000.0
                    root_obs.update(
                        output={
                            "reply": record.output_reply,
                            "active_tier": record.active_tier,
                            "profile": record.profile,
                        },
                        metadata={
                            "scores": record.scores,
                            "latency_ms": round(record.latency_ms, 2),
                        },
                    )
                    self._active_otel_span = prev_span
                    self._active_trace = prev_trace
                    self._local_traces.append(record)
        else:
            try:
                yield record
            finally:
                record.latency_ms = (time.perf_counter() - t0) * 1000.0
                self._active_trace = prev_trace
                self._local_traces.append(record)

    def record_generation(
        self,
        *,
        name: str,
        model: str,
        input_messages: list[dict[str, Any]],
        output_message: dict[str, Any],
        telemetry: dict[str, Any],
        latency_ms: float,
    ) -> ObservationRecord:
        prompt_chars = sum(len(str(m.get("content", ""))) for m in input_messages)
        prompt_toks = max(1, prompt_chars // 4)
        comp_toks = max(1, int(telemetry.get("total_tokens") or 1))
        usage = {
            "input": prompt_toks,
            "output": comp_toks,
            "total": prompt_toks + comp_toks,
        }
        obs = ObservationRecord(
            observation_id=uuid.uuid4().hex[:16],
            as_type="generation",
            name=name,
            started_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            latency_ms=round(latency_ms, 2),
            model=model,
            input_data=input_messages_summary_or_raw(input_messages),
            output_data=output_message,
            usage=usage,
            metadata=dict(telemetry),
        )
        if self._active_trace is not None:
            self._active_trace.observations.append(obs)
            self._active_trace.active_tier = str(telemetry.get("tier", self._active_trace.active_tier))
            self._active_trace.profile = str(telemetry.get("profile", self._active_trace.profile))
            self._active_trace.scores["effective_act_bits"] = float(telemetry.get("effective_act_bits", 8.0))
            self._active_trace.scores["promoted_token_ratio"] = float(telemetry.get("promoted_token_ratio", 0.0))
            self._active_trace.scores["kv_memory_saved_mb"] = float(telemetry.get("kv_memory_saved_mb", 0.0))

        if self.client is not None:
            gen = self.client.start_observation(
                name=name,
                as_type="generation",
                model=model,
                input=obs.input_data,
                output=output_message,
                usage_details=usage,
                metadata=dict(telemetry),
            )
            gen.end()
        return obs

    def record_tool_span(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        observation: str,
        specialist: str,
        tier: str,
        latency_ms: float,
    ) -> ObservationRecord:
        obs = ObservationRecord(
            observation_id=uuid.uuid4().hex[:16],
            as_type="tool",
            name=f"tool:{tool_name}",
            started_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            latency_ms=round(latency_ms, 2),
            input_data=dict(arguments),
            output_data=observation[:600],
            metadata={"specialist": specialist, "tier": tier},
        )
        if self._active_trace is not None:
            self._active_trace.observations.append(obs)
        if self.client is not None:
            sp = self.client.start_observation(
                name=f"tool:{tool_name}",
                as_type="tool",
                input=arguments,
                output=observation[:600],
                metadata={"specialist": specialist, "tier": tier, "latency_ms": round(latency_ms, 2)},
            )
            sp.end()
        return obs

    def record_guardrail_event(
        self,
        *,
        guardrail_name: str,
        details: dict[str, Any],
    ) -> ObservationRecord:
        obs = ObservationRecord(
            observation_id=uuid.uuid4().hex[:16],
            as_type="guardrail",
            name=f"guardrail:{guardrail_name}",
            started_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            latency_ms=0.0,
            metadata=dict(details),
        )
        if self._active_trace is not None:
            self._active_trace.observations.append(obs)
        if self.client is not None:
            gr = self.client.start_observation(
                name=f"guardrail:{guardrail_name}",
                as_type="guardrail",
                metadata=dict(details),
            )
            gr.end()
        return obs

    def flush(self) -> None:
        if self.client is not None and hasattr(self.client, "flush"):
            self.client.flush()


def input_messages_summary_or_raw(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for m in messages[-6:]:
        content = str(m.get("content", ""))
        compact.append(
            {
                "role": m.get("role", "user"),
                "content": content if len(content) <= 400 else content[:397] + "...",
            }
        )
    return compact
