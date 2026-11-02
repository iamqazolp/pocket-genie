from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from pocket_genie.shared import PrecisionTier


@dataclasses.dataclass(slots=True)
class SkillSpec:
    name: str
    description: str
    body: str
    path: Path
    recommended_tier: PrecisionTier = "t1"
    tags: list[str] = dataclasses.field(default_factory=list)


def _parse_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw_text.strip()

    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}, raw_text.strip()

    fm_block = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1 :]).strip()
    if yaml is not None:
        parsed = yaml.safe_load(fm_block) or {}
        if isinstance(parsed, dict):
            return parsed, body

    meta: dict[str, Any] = {}
    for line in lines[1:end_idx]:
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip("\"'")
    return meta, body


class SkillRegistry:
    """Discovers markdown skill playbooks with YAML frontmatter and loads bodies on demand."""

    def __init__(self, search_dirs: list[Path] | None = None) -> None:
        builtin_dir = Path(__file__).resolve().parent.parent / "skills"
        user_dir = Path.home() / ".pocket-genie" / "skills"
        self._search_dirs = search_dirs if search_dirs is not None else [builtin_dir, user_dir]
        self._skills: dict[str, SkillSpec] = {}
        self.refresh()

    def refresh(self) -> None:
        self._skills.clear()
        for directory in self._search_dirs:
            if not directory.exists() or not directory.is_dir():
                continue
            for md_path in sorted(directory.glob("*.md")):
                raw = md_path.read_text(encoding="utf-8")
                meta, body = _parse_frontmatter(raw)
                name = str(meta.get("name") or md_path.stem)
                desc = str(meta.get("description") or f"Playbook {name}")
                tier_raw = str(meta.get("recommended_tier", "t1"))
                tier: PrecisionTier = tier_raw if tier_raw in ("t1", "t2", "t3") else "t1"  # type: ignore[assignment]
                tags_raw = meta.get("tags", [])
                tags = [str(t) for t in tags_raw] if isinstance(tags_raw, list) else []
                self._skills[name] = SkillSpec(
                    name=name,
                    description=desc,
                    body=body,
                    path=md_path,
                    recommended_tier=tier,
                    tags=tags,
                )

    def register_skill(self, spec: SkillSpec) -> None:
        self._skills[spec.name] = spec

    def get(self, name: str) -> SkillSpec | None:
        return self._skills.get(name)

    def list_skills(self) -> list[SkillSpec]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    def render_catalog(self) -> str:
        """Render a static, deterministic catalog suitable for prefix-cached system prompts."""
        skills = self.list_skills()
        if not skills:
            return "No skill playbooks registered."
        lines = ["Available Skill Playbooks (invoke `load_skill` to read full instructions):"]
        for s in skills:
            tag_suffix = f" [tags: {', '.join(s.tags)}]" if s.tags else ""
            lines.append(f"- {s.name} (tier={s.recommended_tier}): {s.description}{tag_suffix}")
        return "\n".join(lines)
