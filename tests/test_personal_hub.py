"""Tests for Personal Hub tools."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock

from pocket_genie.registry.tools import ToolRegistry
from pocket_genie.tools.personal_hub import (
    PersonalHubStore,
    _parse_macos_reminders_output,
    _probe_macos_calendar_and_reminders,
    compute_countdown,
    register_personal_tools,
)


def test_compute_countdown_future_and_overdue() -> None:
    now = dt.datetime(2026, 11, 11, 12, 0, 0)
    future = "2026-11-13T18:00:00"
    hours, label = compute_countdown(future, now=now)
    assert hours > 0
    assert "2d 6h left" in label

    past = "2026-11-11T10:00:00"
    past_hours, past_label = compute_countdown(past, now=now)
    assert past_hours < 0
    assert "OVERDUE by 2.0h" in past_label


def test_parse_macos_reminders_and_probe_disabled() -> None:
    parsed = _parse_macos_reminders_output("Buy milk\n\nCall client\n")
    assert parsed == ["Buy milk", "Call client"]

    probe = _probe_macos_calendar_and_reminders()
    assert probe == {"calendar": [], "reminders": []}


def test_personal_hub_store_crud(tmp_path: Path) -> None:
    db_file = tmp_path / "test_hub.db"
    store = PersonalHubStore(db_path=db_file)

    d1 = store.add_deadline("Ship feature", "2026-12-01T00:00:00", priority="high")
    assert d1["id"] is not None
    deadlines = store.list_deadlines(include_completed=False)
    assert len(deadlines) == 1
    assert "left" in deadlines[0]["countdown"]

    completed = store.complete_deadline(d1["id"])
    assert completed is True
    assert len(store.list_deadlines(include_completed=False)) == 0
    assert len(store.list_deadlines(include_completed=True)) == 1

    ev = store.add_event("Team Standup", "2026-11-12T09:00:00", location="Room A")
    assert ev["title"] == "Team Standup"
    events = store.list_events()
    assert len(events) == 1

    msg = store.enqueue_message(
        sender="Alice",
        body="Can we review?",
        urgency=0.9,
        tone="warm_collaborative",
        priority_score=8.5,
        draft_reply="Sure!",
    )
    assert msg["sender"] == "Alice"
    messages = store.list_messages()
    assert len(messages) == 1


def test_register_personal_tools(tmp_path: Path) -> None:
    registry = ToolRegistry()
    mock_jev = MagicMock()
    mock_jev.decide.return_value = {
        "answers": {
            "urgency": 0.8,
            "tone": "warm_collaborative",
            "priority": 9.0,
        }
    }

    store = register_personal_tools(
        registry=registry,
        workspace_dir=tmp_path,
        jev_client=mock_jev,
    )
    assert store is not None

    expected_tools = [
        "personal.add_deadline",
        "personal.list_deadlines",
        "personal.complete_deadline",
        "personal.add_event",
        "personal.agenda",
        "personal.triage_message",
        "personal.list_messages",
    ]
    for tool_name in expected_tools:
        assert registry.get(tool_name) is not None

    res_deadline = registry.execute(
        "personal.add_deadline",
        {"title": "File taxes", "due_at": "2026-12-31T23:59:59"},
    )
    assert res_deadline["title"] == "File taxes"
    d_id = res_deadline["id"]

    res_agenda = registry.execute("personal.agenda", {})
    assert len(res_agenda["deadlines"]) == 1

    res_triage = registry.execute(
        "personal.triage_message",
        {"sender": "Bob", "body": "Need urgent feedback on PR"},
    )
    assert res_triage["urgency"] == 0.8
    assert "Hi Bob" in res_triage["draft_reply"]

    res_comp = registry.execute("personal.complete_deadline", {"deadline_id": d_id})
    assert res_comp["success"] is True
