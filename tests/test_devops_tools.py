from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pocket_genie.registry.tools import ToolRegistry
from pocket_genie.tools.devops import register_devops_tools


def _make_registry(workspace: Path) -> ToolRegistry:
    registry = ToolRegistry()
    register_devops_tools(registry, workspace)
    return registry


def test_list_files_filters_ignored_dirs_and_glob(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "src" / "notes.md").write_text("# Notes\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("ignored = 1\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.py").write_text("cached = 1\n", encoding="utf-8")

    registry = _make_registry(tmp_path)

    all_out = registry.execute("devops.list_files", {"path": ".", "pattern": "*"})
    assert "src/app.py (12B)" in all_out
    assert "src/notes.md (8B)" in all_out
    assert ".venv" not in all_out
    assert "__pycache__" not in all_out

    py_out = registry.execute("devops.list_files", {"path": ".", "pattern": "*.py"})
    assert "src/app.py (12B)" in py_out
    assert "notes.md" not in py_out

    capped_out = registry.execute(
        "devops.list_files", {"path": ".", "pattern": "*", "max_results": 1}
    )
    assert len(capped_out.splitlines()) == 1


def test_search_code_regex_and_literal_fallback(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "service.py").write_text(
        "def compute_total(items):\n    return sum(items)  # [TODO-1]\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "binary.bin").write_bytes(b"\x00\x01\x02compute_total\x00")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("compute_total\n", encoding="utf-8")

    registry = _make_registry(tmp_path)

    match_out = registry.execute(
        "devops.search_code",
        {"pattern": r"def\s+compute_total", "path": ".", "file_glob": "*.py"},
    )
    assert "pkg/service.py:1: def compute_total(items):" in match_out
    assert "binary.bin" not in match_out
    assert ".git" not in match_out

    # Invalid regex with unclosed bracket should safely fall back to literal search
    literal_out = registry.execute(
        "devops.search_code",
        {"pattern": "[TODO-1", "path": "."},
    )
    assert "pkg/service.py:2:" in literal_out

    empty_out = registry.execute(
        "devops.search_code",
        {"pattern": "non_existent_symbol_xyz", "path": "."},
    )
    assert empty_out == "No matches found."


def test_git_log_on_temp_repo(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)
    # Before git init, should gracefully return "No commit history."
    assert registry.execute("devops.git_log", {"max_count": 5}) == "No commit history."

    subprocess.run(["git", "-C", str(tmp_path), "init"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test User"],
        check=True,
        capture_output=True,
    )
    sample = tmp_path / "hello.py"
    sample.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "hello.py"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "Initial commit"],
        check=True,
        capture_output=True,
    )

    log_out = registry.execute("devops.git_log", {"max_count": 5, "path": "hello.py"})
    assert "Initial commit" in log_out


def test_apply_patch_full_and_surgical_modes(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)

    write_msg = registry.execute(
        "devops.apply_patch",
        {"path": "sub/module.py", "content": "def greet():\n    return 'hello'\n"},
    )
    assert "Wrote 32 bytes to sub/module.py." == write_msg
    target = tmp_path / "sub" / "module.py"
    assert target.read_text(encoding="utf-8") == "def greet():\n    return 'hello'\n"

    replace_msg = registry.execute(
        "devops.apply_patch",
        {
            "path": "sub/module.py",
            "old_text": "return 'hello'",
            "new_text": "return 'world'",
        },
    )
    assert "sub/module.py" in replace_msg
    assert "Replaced" in replace_msg
    assert target.read_text(encoding="utf-8") == "def greet():\n    return 'world'\n"

    with pytest.raises(ValueError, match="old_text not found in sub/module.py"):
        registry.execute(
            "devops.apply_patch",
            {
                "path": "sub/module.py",
                "old_text": "return 'missing'",
                "new_text": "return 'never'",
            },
        )


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("devops.list_files", {"path": "../outside.txt"}),
        ("devops.search_code", {"pattern": "secret", "path": "../outside.txt"}),
        ("devops.git_log", {"path": "../outside.txt"}),
        ("devops.apply_patch", {"path": "../outside.txt", "content": "bad"}),
        ("devops.read_file_chunk", {"path": "../outside.txt"}),
        ("devops.git_diff", {"path": "../outside.txt"}),
    ],
)
def test_path_traversal_rejected(tmp_path: Path, tool_name: str, args: dict[str, str]) -> None:
    registry = _make_registry(tmp_path)
    with pytest.raises(ValueError, match="escapes workspace boundary"):
        registry.execute(tool_name, args)
