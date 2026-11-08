from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pocket_genie.registry.tools import ToolRegistry


def _os_effects_enabled() -> bool:
    if os.environ.get("GENIE_DISABLE_OS_EFFECTS") == "1":
        return False
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    return True


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
            data = json.loads(proc.stdout)
            if isinstance(data, dict):
                data.setdefault("source", "termux")
                return data
    except Exception:
        return None
    return None


def _parse_macos_pmset(output: str) -> dict[str, Any] | None:
    if not output or not output.strip():
        return None
    lower = output.lower()
    pct_match = re.search(r"(\d{1,3})%", output)
    if pct_match is None and "ac power" not in lower and "battery power" not in lower:
        return None

    percentage = int(pct_match.group(1)) if pct_match else 100
    rem_match = re.search(r"(\d+:\d{2})\s+remaining", lower)
    remaining = rem_match.group(1) if rem_match else "unknown"

    if "discharging" in lower:
        status = "DISCHARGING"
    elif "not charging" in lower or "charged" in lower:
        status = "AC_POWER"
    elif "charging" in lower or "finishing charge" in lower:
        status = "CHARGING"
    elif "ac power" in lower:
        status = "AC_POWER"
    else:
        status = "DISCHARGING"

    return {
        "source": "macos_pmset",
        "status": status,
        "percentage": percentage,
        "remaining": remaining,
    }


def _probe_battery() -> dict[str, Any]:
    termux_batt = _probe_termux_battery()
    if termux_batt is not None:
        return termux_batt

    if shutil.which("pmset") is not None:
        try:
            proc = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout:
                parsed = _parse_macos_pmset(proc.stdout)
                if parsed is not None:
                    return parsed
        except Exception:
            pass

    linux_cap = Path("/sys/class/power_supply/BAT0/capacity")
    if linux_cap.exists():
        try:
            pct = int(linux_cap.read_text(encoding="utf-8").strip())
            linux_status_path = Path("/sys/class/power_supply/BAT0/status")
            status = (
                linux_status_path.read_text(encoding="utf-8").strip().upper()
                if linux_status_path.exists()
                else "UNKNOWN"
            )
            return {
                "source": "linux_sysfs",
                "status": status,
                "percentage": pct,
            }
        except Exception:
            pass

    return {
        "source": "host_fallback",
        "status": "HOST_AC",
        "percentage": 100,
        "temperature": 31.5,
    }


def _probe_memory_gb() -> dict[str, float]:
    try:
        if hasattr(os, "sysconf"):
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            if (
                isinstance(page_size, int)
                and isinstance(phys_pages, int)
                and page_size > 0
                and phys_pages > 0
            ):
                return {"total_ram_gb": round((page_size * phys_pages) / (1024**3), 2)}
    except (ValueError, OSError):
        pass
    return {"total_ram_gb": 0.0}


def _collect_top_processes(limit: int = 8) -> list[dict[str, Any]]:
    capped = max(1, min(25, int(limit)))
    if shutil.which("ps") is None:
        return []
    cmd = (
        ["ps", "-Ao", "pid,pcpu,pmem,comm", "-r"]
        if sys.platform == "darwin"
        else ["ps", "-eo", "pid,pcpu,pmem,comm", "--sort=-pcpu"]
    )
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            return []
        rows: list[dict[str, Any]] = []
        for raw_line in proc.stdout.splitlines()[1:]:
            parts = raw_line.strip().split(None, 3)
            if len(parts) < 4:
                continue
            try:
                pid = int(parts[0])
                cpu_pct = float(parts[1].replace(",", "."))
                mem_pct = float(parts[2].replace(",", "."))
            except ValueError:
                continue
            rows.append(
                {
                    "pid": pid,
                    "cpu_pct": cpu_pct,
                    "mem_pct": mem_pct,
                    "command": parts[3].strip(),
                }
            )
            if len(rows) >= capped:
                break
        return rows
    except Exception:
        return []


def register_device_tools(registry: ToolRegistry) -> dict[str, Any]:
    device_state: dict[str, Any] = {
        "notifications": [],
        "clipboard": "",
    }

    @registry.register(
        name="device.system_telemetry",
        description="Retrieve host/mobile CPU, load average, RAM, disk headroom, and battery status.",
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
        mem = _probe_memory_gb()
        battery = _probe_battery()
        payload = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count() or 1,
            "load_avg_1m": round(load_avg[0], 2),
            "disk_free_gb": round(usage.free / (1024**3), 2),
            "total_ram_gb": mem["total_ram_gb"],
            "battery": battery,
        }
        return json.dumps(payload, indent=2)

    @registry.register(
        name="device.send_notification",
        description="Dispatch a notification to Android Termux, macOS, or Linux desktop (requires HITL approval).",
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
        if _os_effects_enabled():
            try:
                if shutil.which("termux-notification") is not None:
                    subprocess.run(
                        ["termux-notification", "--title", title, "--content", message],
                        capture_output=True,
                        text=True,
                        timeout=3.0,
                        check=False,
                    )
                elif sys.platform == "darwin" and shutil.which("osascript") is not None:
                    subprocess.run(
                        [
                            "osascript",
                            "-e",
                            "on run argv",
                            "-e",
                            "display notification (item 2 of argv) with title (item 1 of argv)",
                            "-e",
                            "end run",
                            "--",
                            title,
                            message,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=3.0,
                        check=False,
                    )
                elif shutil.which("notify-send") is not None:
                    subprocess.run(
                        ["notify-send", title, message],
                        capture_output=True,
                        text=True,
                        timeout=3.0,
                        check=False,
                    )
            except Exception:
                pass
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
        if _os_effects_enabled():
            try:
                if shutil.which("termux-clipboard-set") is not None:
                    subprocess.run(
                        ["termux-clipboard-set"],
                        input=text,
                        capture_output=True,
                        text=True,
                        timeout=2.0,
                        check=False,
                    )
                elif sys.platform == "darwin" and shutil.which("pbcopy") is not None:
                    subprocess.run(
                        ["pbcopy"],
                        input=text,
                        capture_output=True,
                        text=True,
                        timeout=2.0,
                        check=False,
                    )
                elif shutil.which("wl-copy") is not None:
                    subprocess.run(
                        ["wl-copy"],
                        input=text,
                        capture_output=True,
                        text=True,
                        timeout=2.0,
                        check=False,
                    )
                elif shutil.which("xclip") is not None:
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=text,
                        capture_output=True,
                        text=True,
                        timeout=2.0,
                        check=False,
                    )
            except Exception:
                pass
        return f"Copied {len(text)} chars to device clipboard."

    @registry.register(
        name="device.clipboard_get",
        description="Read text from the device or host OS clipboard.",
        parameters={
            "type": "object",
            "properties": {},
        },
        domain="device",
        mutating=False,
    )
    def clipboard_get() -> str:
        if device_state["clipboard"]:
            return str(device_state["clipboard"])
        if _os_effects_enabled():
            try:
                cmd: list[str] | None = None
                if shutil.which("termux-clipboard-get") is not None:
                    cmd = ["termux-clipboard-get"]
                elif sys.platform == "darwin" and shutil.which("pbpaste") is not None:
                    cmd = ["pbpaste"]
                if cmd is not None:
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=2.0,
                        check=False,
                    )
                    if proc.returncode == 0:
                        device_state["clipboard"] = proc.stdout
            except Exception:
                pass
        return str(device_state["clipboard"])

    @registry.register(
        name="device.top_processes",
        description="List top CPU/memory processes on the host or edge device.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of top processes to return (1-25).",
                },
            },
        },
        domain="device",
        mutating=False,
    )
    def top_processes(limit: int = 8) -> str:
        rows = _collect_top_processes(limit=limit)
        return json.dumps(rows, indent=2)

    return device_state
