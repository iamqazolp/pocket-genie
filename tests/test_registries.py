from __future__ import annotations

from pocket_genie.registry import (
    BackendRegistry,
    SkillRegistry,
    ToolRegistry,
    sanitize_json_schema,
)
from pocket_genie.shared import ChatMessage


def test_tool_registry_and_schema_sanitizer():
    reg = ToolRegistry()

    @reg.register(
        name="demo.echo",
        description="Echo payload",
        parameters={
            "type": "dict",
            "properties": {
                "text": {"description": "Untyped text field"},
                "count": {"type": "int"},
            },
            "required": ["text", "missing_field"],
        },
        domain="demo",
        mutating=True,
    )
    def echo_tool(text: str, count: int = 1) -> str:
        return f"{text}:{count}"

    spec = reg.get("demo.echo")
    assert spec is not None
    assert spec.mutating is True
    assert spec.parameters["type"] == "object"
    assert spec.parameters["properties"]["text"]["type"] == "string"
    assert spec.parameters["properties"]["count"]["type"] == "integer"
    assert spec.parameters["required"] == ["text"]
    assert reg.execute("demo.echo", {"text": "ok", "count": 3}) == "ok:3"


def test_skill_registry_loads_builtin_playbooks():
    skills = SkillRegistry()
    names = [s.name for s in skills.list_skills()]
    assert "repo-ci-triage" in names
    assert "quant-diagnosis" in names
    assert "deep-investigation" in names

    triage = skills.get("repo-ci-triage")
    assert triage is not None
    assert triage.recommended_tier == "t2"
    assert "devops.run_pytest" in triage.body

    catalog = skills.render_catalog()
    assert "repo-ci-triage (tier=t2)" in catalog


def test_backend_registry_default_tiers():
    backends = BackendRegistry()
    out_t1 = backends.get("t1").generate([ChatMessage(role="user", content="Check status")])
    out_t3 = backends.get("t3").generate([ChatMessage(role="user", content="Check status")])
    assert out_t1.telemetry.profile == "P_fast"
    assert out_t3.telemetry.profile == "P_accurate"
    assert out_t3.telemetry.effective_act_bits == 16.0
