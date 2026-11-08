"""Personal hub tools: deadlines, agenda, calendar/reminder probes, message triage."""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from pocket_genie.integrations.jev_router import JevDecisionClient
from pocket_genie.registry.tools import ToolRegistry


def compute_countdown(due_at_iso: str, now: dt.datetime | None = None) -> tuple[float, str]:
    target = dt.datetime.fromisoformat(due_at_iso)
    if target.tzinfo is not None:
        target = target.astimezone(dt.timezone.utc).replace(tzinfo=None)
    current = now or dt.datetime.utcnow()
    if current.tzinfo is not None:
        current = current.astimezone(dt.timezone.utc).replace(tzinfo=None)
    diff = (target - current).total_seconds()
    hours_left = diff / 3600.0
    if diff < 0:
        return hours_left, f"OVERDUE by {abs(diff) / 3600.0:.1f}h"
    return hours_left, f"{int(diff // 86400)}d {int((diff % 86400) // 3600)}h left"


def _parse_macos_reminders_output(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _probe_macos_calendar_and_reminders() -> dict[str, list[str]]:
    empty: dict[str, list[str]] = {"calendar": [], "reminders": []}
    if (
        "PYTEST_CURRENT_TEST" in os.environ
        or os.environ.get("GENIE_DISABLE_OS_EFFECTS") == "1"
        or sys.platform != "darwin"
    ):
        return empty
    try:
        cmd = ["osascript", "-e", 'tell app "Reminders" to get name of (reminders whose completed is false)']
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0, check=False)
        items = [x.strip() for x in proc.stdout.split(",") if x.strip()] if proc.returncode == 0 else []
        return {"calendar": [], "reminders": items}
    except Exception:
        return empty


class PersonalHubStore:
    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS deadlines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, due_at TEXT,
                    priority TEXT DEFAULT 'medium', completed INTEGER DEFAULT 0, created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS schedule_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, start_at TEXT,
                    end_at TEXT, location TEXT, created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS message_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, sender TEXT, body TEXT,
                    urgency REAL, tone TEXT, priority_score REAL, draft_reply TEXT,
                    status TEXT DEFAULT 'pending', created_at TEXT
                );
                """
            )

    def add_deadline(self, title: str, due_at: str, priority: str = "medium") -> dict[str, Any]:
        now = dt.datetime.utcnow().isoformat()
        with self.conn:
            c = self.conn.execute(
                "INSERT INTO deadlines (title, due_at, priority, completed, created_at) VALUES (?, ?, ?, 0, ?)",
                (title, due_at, priority, now),
            )
            return {"id": c.lastrowid, "title": title, "due_at": due_at, "priority": priority, "completed": 0}

    def list_deadlines(self, include_completed: bool = False) -> list[dict[str, Any]]:
        q = "SELECT * FROM deadlines" + ("" if include_completed else " WHERE completed = 0") + " ORDER BY due_at ASC"
        rows = [dict(r) for r in self.conn.execute(q).fetchall()]
        for r in rows:
            _, r["countdown"] = compute_countdown(r["due_at"])
        return rows

    def complete_deadline(self, deadline_id: int) -> bool:
        with self.conn:
            return self.conn.execute("UPDATE deadlines SET completed = 1 WHERE id = ?", (deadline_id,)).rowcount > 0

    def add_event(self, title: str, start_at: str, end_at: str | None = None, location: str | None = None) -> dict[str, Any]:
        now = dt.datetime.utcnow().isoformat()
        with self.conn:
            c = self.conn.execute(
                "INSERT INTO schedule_events (title, start_at, end_at, location, created_at) VALUES (?, ?, ?, ?, ?)",
                (title, start_at, end_at, location, now),
            )
            return {"id": c.lastrowid, "title": title, "start_at": start_at, "end_at": end_at, "location": location}

    def list_events(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM schedule_events ORDER BY start_at ASC").fetchall()]

    def enqueue_message(
        self,
        sender: str,
        body: str,
        urgency: float,
        tone: str,
        priority_score: float,
        draft_reply: str,
        status: str = "pending",
    ) -> dict[str, Any]:
        now = dt.datetime.utcnow().isoformat()
        with self.conn:
            c = self.conn.execute(
                "INSERT INTO message_queue (sender, body, urgency, tone, priority_score, draft_reply, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sender, body, urgency, tone, priority_score, draft_reply, status, now),
            )
            return {"id": c.lastrowid, "sender": sender, "body": body, "urgency": urgency, "tone": tone, "priority_score": priority_score, "draft_reply": draft_reply, "status": status}

    def list_messages(self, status: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM message_queue WHERE status = ? ORDER BY id DESC" if status else "SELECT * FROM message_queue ORDER BY id DESC"
        return [dict(r) for r in self.conn.execute(q, (status,) if status else ()).fetchall()]


def register_personal_tools(
    registry: ToolRegistry,
    workspace_dir: Path,
    store: PersonalHubStore | None = None,
    jev_client: JevDecisionClient | None = None,
) -> PersonalHubStore:
    if store is None:
        db = ":memory:" if "PYTEST_CURRENT_TEST" in os.environ else str(workspace_dir / ".genie_personal.db")
        store = PersonalHubStore(db_path=db)
    jev = jev_client or JevDecisionClient()

    @registry.register(
        name="personal.add_deadline", description="Add a task or deadline with ISO timestamp.",
        parameters={"type": "object", "properties": {"title": {"type": "string"}, "due_at": {"type": "string"}, "priority": {"type": "string"}}, "required": ["title", "due_at"]},
        domain="personal", mutating=False,
    )
    def add_deadline(title: str, due_at: str, priority: str = "medium") -> dict[str, Any]:
        return store.add_deadline(title=title, due_at=due_at, priority=priority)

    @registry.register(
        name="personal.list_deadlines", description="List active or completed deadlines with countdown.",
        parameters={"type": "object", "properties": {"include_completed": {"type": "boolean"}}, "required": []},
        domain="personal", mutating=False,
    )
    def list_deadlines(include_completed: bool = False) -> list[dict[str, Any]]:
        return store.list_deadlines(include_completed=include_completed)

    @registry.register(
        name="personal.complete_deadline", description="Mark a deadline as completed by ID.",
        parameters={"type": "object", "properties": {"deadline_id": {"type": "integer"}}, "required": ["deadline_id"]},
        domain="personal", mutating=False,
    )
    def complete_deadline(deadline_id: int) -> dict[str, Any]:
        return {"success": store.complete_deadline(deadline_id=deadline_id), "deadline_id": deadline_id}

    @registry.register(
        name="personal.add_event", description="Add a scheduled event/meeting.",
        parameters={"type": "object", "properties": {"title": {"type": "string"}, "start_at": {"type": "string"}, "end_at": {"type": "string"}, "location": {"type": "string"}}, "required": ["title", "start_at"]},
        domain="personal", mutating=False,
    )
    def add_event(title: str, start_at: str, end_at: str | None = None, location: str | None = None) -> dict[str, Any]:
        return store.add_event(title=title, start_at=start_at, end_at=end_at, location=location)

    @registry.register(
        name="personal.agenda", description="View combined agenda of deadlines, events, and native reminders.",
        parameters={"type": "object", "properties": {"include_os": {"type": "boolean"}}, "required": []},
        domain="personal", mutating=False,
    )
    def agenda(include_os: bool = True) -> dict[str, Any]:
        os_data = _probe_macos_calendar_and_reminders() if include_os else {"calendar": [], "reminders": []}
        return {"deadlines": store.list_deadlines(include_completed=False), "events": store.list_events(), "os_sync": os_data}

    @registry.register(
        name="personal.triage_message", description="Triage incoming message for urgency, tone, priority, and draft reply.",
        parameters={"type": "object", "properties": {"sender": {"type": "string"}, "body": {"type": "string"}, "draft_reply": {"type": "string"}}, "required": ["sender", "body"]},
        domain="personal", mutating=False,
    )
    def triage_message(sender: str, body: str, draft_reply: str = "") -> dict[str, Any]:
        qs: dict[str, dict[str, Any]] = {
            "urgency": {
                "type": "noul",
                "instructions": (
                    "The message contains an urgent deadline, blocker, interview schedule, "
                    "or time-sensitive request requiring a fast reply."
                ),
            },
            "tone": {
                "type": "choice",
                "criteria": {
                    "concise_direct": "Quick technical coordination, PR review, or status check.",
                    "warm_collaborative": "Teammate, friend, mentoring, or collaborative follow-up.",
                    "formal_professional": "Recruiter, interview panel, executive, or external partner.",
                },
            },
            "priority": {
                "type": "score",
                "criteria": [
                    "Low-priority FYI or newsletter (0.0).",
                    "Standard async personal or engineering coordination (0.5).",
                    "High-priority action item, deadline, or interview message (1.0).",
                ],
            },
        }
        try:
            ans = jev.decide(state=f"Sender: {sender}\nBody: {body}", questions=qs).get("answers", {})
            raw_urg = ans.get("urgency", 0.5)
            urgency = float(raw_urg.get("noul", 0.5) if isinstance(raw_urg, dict) else raw_urg)
            raw_tone = ans.get("tone", "concise_direct")
            tone = str(
                raw_tone.get("choice", "concise_direct") if isinstance(raw_tone, dict) else raw_tone
            )
            raw_prio = ans.get("priority", 0.5)
            priority = float(
                raw_prio.get("score", 0.5) * 10.0 if isinstance(raw_prio, dict) else raw_prio
            )
        except Exception:
            urgency, tone, priority = 0.5, "concise_direct", 5.0

        if not draft_reply:
            if tone == "warm_collaborative":
                draft_reply = f"Hi {sender}, thanks for reaching out! Let's follow up on this shortly."
            elif tone == "formal_professional":
                draft_reply = f"Dear {sender}, thank you for your note. I will review and reply soon."
            else:
                draft_reply = f"Acknowledged. Will get back to you regarding: {body[:60]}."

        return store.enqueue_message(sender, body, urgency, tone, priority, draft_reply)

    @registry.register(
        name="personal.list_messages", description="List triaged incoming messages in personal queue.",
        parameters={"type": "object", "properties": {"status": {"type": "string"}}, "required": []},
        domain="personal", mutating=False,
    )
    def list_messages(status: str | None = None) -> list[dict[str, Any]]:
        return store.list_messages(status=status)

    return store
