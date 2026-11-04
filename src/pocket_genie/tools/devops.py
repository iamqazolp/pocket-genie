from __future__ import annotations

import subprocess
from pathlib import Path

from pocket_genie.registry.tools import ToolRegistry


def _resolve_safe_path(workspace_dir: Path, rel_path: str) -> Path:
    root = workspace_dir.resolve()
    target = (root / rel_path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Path {rel_path!r} escapes workspace boundary {root}")
    return target


def register_devops_tools(registry: ToolRegistry, workspace_dir: Path) -> None:
    @registry.register(
        name="devops.git_status",
        description="Inspect git branch and working tree status in the active repository.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional sub-path filter."},
            },
        },
        domain="devops",
        mutating=False,
    )
    def git_status(query: str = "") -> str:
        cmd = ["git", "-C", str(workspace_dir), "status", "--short", "--branch"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0, check=False)
        if proc.returncode != 0:
            return f"git status exited with {proc.returncode}: {proc.stderr.strip()}"
        return proc.stdout.strip() or "Clean working tree."

    @registry.register(
        name="devops.git_diff",
        description="Show unstaged and staged git diff statistics or patch hunks.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Optional relative file path."},
            },
        },
        domain="devops",
        mutating=False,
    )
    def git_diff(path: str = "") -> str:
        cmd = ["git", "-C", str(workspace_dir), "diff"]
        if path:
            _resolve_safe_path(workspace_dir, path)
            cmd.extend(["--", path])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0, check=False)
        return proc.stdout.strip() or "No diff hunks."

    @registry.register(
        name="devops.read_file_chunk",
        description="Read a bounded line range from a file inside the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path inside workspace."},
                "start_line": {"type": "integer", "description": "1-based start line."},
                "end_line": {"type": "integer", "description": "1-based inclusive end line."},
            },
            "required": ["path"],
        },
        domain="devops",
        mutating=False,
    )
    def read_file_chunk(path: str, start_line: int = 1, end_line: int = 80) -> str:
        target = _resolve_safe_path(workspace_dir, path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        lines = target.read_text(encoding="utf-8").splitlines()
        s_idx = max(0, int(start_line) - 1)
        e_idx = min(len(lines), max(s_idx + 1, int(end_line)))
        numbered = [f"{i + 1:4d}: {lines[i]}" for i in range(s_idx, e_idx)]
        return "\n".join(numbered)

    @registry.register(
        name="devops.run_pytest",
        description="Run pytest on a scoped test file or expression inside the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Test file or directory (default: tests)."},
            },
        },
        domain="devops",
        mutating=False,
    )
    def run_pytest(target: str = "tests") -> str:
        _resolve_safe_path(workspace_dir, target)
        proc = subprocess.run(
            ["python3", "-m", "pytest", "-q", target],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        return (proc.stdout + "\n" + proc.stderr).strip()

    @registry.register(
        name="devops.apply_patch",
        description="Write or update a file inside the workspace (requires HITL approval).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path to write."},
                "content": {"type": "string", "description": "Updated file content."},
            },
            "required": ["path", "content"],
        },
        domain="devops",
        mutating=True,
        project_scoped=True,
    )
    def apply_patch(path: str, content: str) -> str:
        target = _resolve_safe_path(workspace_dir, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}."
