from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path

from pocket_genie.registry.tools import ToolRegistry

_IGNORED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".local-notes",
}


def _resolve_safe_path(workspace_dir: Path, rel_path: str) -> Path:
    root = workspace_dir.resolve()
    target = (root / rel_path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Path {rel_path!r} escapes workspace boundary {root}")
    return target


def _iter_workspace_files(root: Path, target: Path) -> list[tuple[Path, str]]:
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target.relative_to(root)}")
    if target.is_file():
        return [(target, target.relative_to(root).as_posix())]

    collected: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(target):
        dirnames[:] = sorted(
            d for d in dirnames if d not in _IGNORED_DIRS and not d.startswith(".")
        )
        base = Path(dirpath)
        for fname in sorted(filenames):
            fpath = base / fname
            if fpath.is_file():
                rel_str = fpath.resolve().relative_to(root).as_posix()
                collected.append((fpath, rel_str))
    return collected


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
        name="devops.git_log",
        description="Show recent git commit history in oneline decorated format.",
        parameters={
            "type": "object",
            "properties": {
                "max_count": {
                    "type": "integer",
                    "description": "Maximum number of commits to return (default: 10).",
                },
                "path": {
                    "type": "string",
                    "description": "Optional relative file or directory path filter.",
                },
            },
        },
        domain="devops",
        mutating=False,
    )
    def git_log(max_count: int = 10, path: str = "") -> str:
        bounded_count = max(1, min(100, int(max_count)))
        cmd = [
            "git",
            "-C",
            str(workspace_dir),
            "log",
            "-n",
            str(bounded_count),
            "--oneline",
            "--decorate",
        ]
        if path:
            _resolve_safe_path(workspace_dir, path)
            cmd.extend(["--", path])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0, check=False)
        if proc.returncode != 0:
            return "No commit history."
        return proc.stdout.strip() or "No commit history."

    @registry.register(
        name="devops.list_files",
        description="List files in the workspace with byte sizes, filtered by glob pattern.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative directory or file path inside workspace (default: '.').",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match filenames or relative paths (default: '*').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of files to return (default: 80).",
                },
            },
        },
        domain="devops",
        mutating=False,
    )
    def list_files(path: str = ".", pattern: str = "*", max_results: int = 80) -> str:
        root = workspace_dir.resolve()
        target = _resolve_safe_path(workspace_dir, path)
        cap = max(1, int(max_results))
        entries: list[str] = []
        for fpath, rel_str in _iter_workspace_files(root, target):
            if fnmatch.fnmatch(fpath.name, pattern) or fnmatch.fnmatch(rel_str, pattern):
                size = fpath.stat().st_size
                entries.append(f"{rel_str} ({size}B)")
                if len(entries) >= cap:
                    break
        return "\n".join(entries) if entries else "No matching files."

    @registry.register(
        name="devops.search_code",
        description="Search UTF-8 text files in the workspace by regex or substring pattern.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex or substring pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Relative directory or file path to search (default: '.').",
                },
                "file_glob": {
                    "type": "string",
                    "description": "Glob filter for filenames (default: '*').",
                },
                "max_matches": {
                    "type": "integer",
                    "description": "Maximum number of matching lines to return (default: 50).",
                },
            },
            "required": ["pattern"],
        },
        domain="devops",
        mutating=False,
    )
    def search_code(
        pattern: str,
        path: str = ".",
        file_glob: str = "*",
        max_matches: int = 50,
    ) -> str:
        root = workspace_dir.resolve()
        target = _resolve_safe_path(workspace_dir, path)
        cap = max(1, int(max_matches))
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        matches: list[str] = []
        for fpath, rel_str in _iter_workspace_files(root, target):
            if not (fnmatch.fnmatch(fpath.name, file_glob) or fnmatch.fnmatch(rel_str, file_glob)):
                continue
            try:
                raw = fpath.read_bytes()
                if b"\x00" in raw:
                    continue
                text = raw.decode("utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            for idx, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{rel_str}:{idx}: {line}")
                    if len(matches) >= cap:
                        return "\n".join(matches)

        return "\n".join(matches) if matches else "No matches found."

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
                "target": {
                    "type": "string",
                    "description": "Test file or directory (default: tests).",
                },
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
        description="Write or surgically update a file inside the workspace (requires HITL approval).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path to write or patch."},
                "content": {
                    "type": "string",
                    "description": "Full file content when writing an entire file.",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact existing snippet to replace for surgical edits.",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement snippet when old_text is provided.",
                },
            },
            "required": ["path"],
        },
        domain="devops",
        mutating=True,
        project_scoped=True,
    )
    def apply_patch(
        path: str,
        content: str = "",
        old_text: str = "",
        new_text: str = "",
    ) -> str:
        target = _resolve_safe_path(workspace_dir, path)
        if old_text:
            if not target.exists() or not target.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            existing_content = target.read_text(encoding="utf-8")
            if old_text not in existing_content:
                raise ValueError(f"old_text not found in {path}")
            updated_content = existing_content.replace(old_text, new_text, 1)
            target.write_text(updated_content, encoding="utf-8")
            return f"Replaced snippet ({len(old_text)} -> {len(new_text)} chars) in {path}."

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}."
