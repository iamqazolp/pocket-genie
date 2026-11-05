from __future__ import annotations

from pocket_genie.registry import (
    BackendRegistry,
    OpenAICompatibleBackend,
    SkillRegistry,
    ToolRegistry,
    parse_tool_calls_from_text,
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


def test_local_llm_configuration_and_tool_call_parser():
    raw_qwen_output = (
        'Checking status first.\n'
        '<tool_call>\n'
        '{"name": "devops.git_status", "arguments": {"repo_path": "."}}\n'
        '</tool_call>'
    )
    calls = parse_tool_calls_from_text(raw_qwen_output)
    assert len(calls) == 1
    assert calls[0].name == "devops.git_status"
    assert calls[0].arguments == {"repo_path": "."}

    backends = BackendRegistry()
    backends.configure_local_llm(
        base_url="http://127.0.0.1:8080/v1",
        primary_model="qwen2.5-coder:3b",
        fallback_model="qwen2.5-coder:7b",
    )
    t2 = backends.get("t2")
    t3 = backends.get("t3")
    assert isinstance(t2, OpenAICompatibleBackend)
    assert t2.model == "qwen2.5-coder:3b"
    assert t2.tier == "t2"
    assert isinstance(t3, OpenAICompatibleBackend)
    assert t3.model == "qwen2.5-coder:7b"
    assert t3.tier == "t3"

