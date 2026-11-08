from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from typing import Any

from pocket_genie.registry.tools import ToolRegistry


def _probe_termux_battery() -> dict[str, Any] | None:
    if shutil.which("termux-battery-status") is None:
        return None
    try:
        proc = subprocess.run(
            ["termux-battery-status"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout)
    except Exception:
        return None
    return None


def register_device_tools(registry: ToolRegistry) -> dict[str, Any]:
    device_state: dict[str, Any] = {
        "notifications": [],
        "clipboard": "",
    }

    @registry.register(
        name="device.system_telemetry",
        description="Retrieve host/mobile CPU, load average, disk headroom, and Android Termux battery status.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional metric filter."},
            },
        },
        domain="device",
        mutating=False,
    )
    def system_telemetry(query: str = "") -> str:
        usage = shutil.disk_usage(".")
        load_avg = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
        battery = _probe_termux_battery() or {
            "status": "HOST_AC",
            "percentage": 100,
            "temperature": 31.5,
        }
        payload = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count() or 1,
            "load_avg_1m": round(load_avg[0], 2),
            "disk_free_gb": round(usage.free / (1024**3), 2),
            "battery": battery,
        }
        return json.dumps(payload, indent=2)

    @registry.register(
        name="device.send_notification",
        description="Dispatch a notification to Android Termux or mobile console (requires HITL approval).",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Notification title."},
                "message": {"type": "string", "description": "Notification body text."},
            },
            "required": ["title", "message"],
        },
        domain="device",
        mutating=True,
    )
    def send_notification(title: str, message: str) -> str:
        entry = {"title": title, "message": message}
        device_state["notifications"].append(entry)
        if shutil.which("termux-notification") is not None:
            subprocess.run(
                ["termux-notification", "--title", title, "--content", message],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
        return f"Notification dispatched: {title} — {message}"

    @registry.register(
        name="device.clipboard_set",
        description="Copy text to the device clipboard (requires HITL approval).",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Clipboard text payload."},
            },
            "required": ["text"],
        },
        domain="device",
        mutating=True,
    )
    def clipboard_set(text: str) -> str:
        device_state["clipboard"] = text
        return f"Copied {len(text)} chars to device clipboard."

    return device_state
