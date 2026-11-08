from __future__ import annotations

import json

from pocket_genie.registry.tools import ToolRegistry
from pocket_genie.tools.device_termux import (
    _parse_macos_pmset,
    register_device_tools,
)


def test_parse_macos_pmset_battery_discharging() -> None:
    sample = (
        "Now drawing from 'Battery Power'\n"
        " -InternalBattery-0 (id=1234567)\t84%; discharging; 3:45 remaining present: true\n"
    )
    parsed = _parse_macos_pmset(sample)
    assert parsed == {
        "source": "macos_pmset",
        "status": "DISCHARGING",
        "percentage": 84,
        "remaining": "3:45",
    }


def test_parse_macos_pmset_ac_charging_and_charged() -> None:
    charging_sample = (
        "Now drawing from 'AC Power'\n"
        " -InternalBattery-0 (id=1234567)\t62%; charging; 1:12 remaining present: true\n"
    )
    assert _parse_macos_pmset(charging_sample) == {
        "source": "macos_pmset",
        "status": "CHARGING",
        "percentage": 62,
        "remaining": "1:12",
    }

    charged_sample = (
        "Now drawing from 'AC Power'\n"
        " -InternalBattery-0 (id=1234567)\t100%; charged; 0:00 remaining present: true\n"
    )
    assert _parse_macos_pmset(charged_sample) == {
        "source": "macos_pmset",
        "status": "AC_POWER",
        "percentage": 100,
        "remaining": "0:00",
    }

    assert _parse_macos_pmset("") is None
    assert _parse_macos_pmset("unrecognized output") is None


def test_system_telemetry_json_structure() -> None:
    registry = ToolRegistry()
    register_device_tools(registry)

    raw = registry.execute("device.system_telemetry", {"query": ""})
    payload = json.loads(raw)
    assert "total_ram_gb" in payload
    assert isinstance(payload["total_ram_gb"], float)
    assert payload["total_ram_gb"] >= 0.0
    assert "battery" in payload
    assert isinstance(payload["battery"], dict)
    assert "percentage" in payload["battery"]
    assert "status" in payload["battery"]
    assert payload["cpu_count"] >= 1
    assert payload["disk_free_gb"] > 0


def test_clipboard_set_and_get_round_trip() -> None:
    registry = ToolRegistry()
    state = register_device_tools(registry)

    assert registry.execute("device.clipboard_get", {}) == ""
    msg = registry.execute("device.clipboard_set", {"text": "edge-genie-payload"})
    assert "Copied 18 chars" in msg
    assert state["clipboard"] == "edge-genie-payload"
    assert registry.execute("device.clipboard_get", {}) == "edge-genie-payload"


def test_send_notification_records_state() -> None:
    registry = ToolRegistry()
    state = register_device_tools(registry)

    out = registry.execute(
        "device.send_notification",
        {"title": "Thermal Alert", "message": "NPU load nominal"},
    )
    assert "Notification dispatched: Thermal Alert — NPU load nominal" in out
    assert state["notifications"] == [
        {"title": "Thermal Alert", "message": "NPU load nominal"}
    ]


def test_top_processes_returns_valid_json_list() -> None:
    registry = ToolRegistry()
    register_device_tools(registry)

    raw = registry.execute("device.top_processes", {"limit": 5})
    processes = json.loads(raw)
    assert isinstance(processes, list)
    assert 1 <= len(processes) <= 5
    first = processes[0]
    assert isinstance(first["pid"], int)
    assert isinstance(first["cpu_pct"], float)
    assert isinstance(first["mem_pct"], float)
    assert isinstance(first["command"], str)
    assert first["command"] != ""
