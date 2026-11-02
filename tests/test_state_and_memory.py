from __future__ import annotations

from pocket_genie.core.memory import (
    RunningSummary,
    SessionMemoryStore,
    estimate_message_tokens,
    summarize_user_queries,
)
from pocket_genie.core.state import GenieState, QuantTelemetrySnapshot
from pocket_genie.shared import ChatMessage, StreamWriter


def test_running_summary_and_memory_store():
    summary = RunningSummary()
    msgs = [
        ChatMessage(role="user", content="Inspect git status on the repository"),
        ChatMessage(role="assistant", content="Repo has 2 modified files in src/"),
    ]
    text = summary.update(msgs)
    assert "Inspect git status" in text
    assert summary.compacted_message_count == 2
    assert summary.compacted_token_estimate > 0

    store = SessionMemoryStore()
    store.add_memory("Prefer concise diffs and pytest -q")
    store.record_session_summary("s1", "Audited W4A8 outlier layers")
    store.record_session_summary("s2", "Added SQL allowlist validation")
    store.record_session_summary("s3", "Configured Termux battery alert")
    store.record_session_summary("s4", "Current active session")

    recents = store.recent_summaries(exclude_session_id="s4", limit=3)
    assert len(recents) == 3
    assert recents[0] == "Configured Termux battery alert"
    assert store.list_memories() == ["Prefer concise diffs and pytest -q"]


def test_quant_telemetry_and_stream_writer():
    snap = QuantTelemetrySnapshot(
        tier="t2",
        profile="P_mixed",
        promoted_tokens=6,
        total_tokens=48,
        kv_cache_fp16_mb=24.0,
        kv_cache_quant_mb=6.5,
    )
    d = snap.to_dict()
    assert d["promoted_token_ratio"] == 0.125
    assert d["kv_memory_saved_mb"] == 17.5

    writer = StreamWriter()
    writer.emit("quant_telemetry", **d)
    assert len(writer.events) == 1
    assert writer.events[0].event_type == "quant_telemetry"

    state = GenieState(
        session_id="sess-01",
        session_created_at="2026-10-31T08:00:00Z",
        messages=[ChatMessage(role="user", content="Run unit tests")],
    )
    assert estimate_message_tokens(state.messages[0]) > 0
    assert "Run unit tests" in summarize_user_queries(state.messages)
