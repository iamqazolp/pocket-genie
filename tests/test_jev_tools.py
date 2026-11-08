from __future__ import annotations

import json
from typing import Any

import pytest

from pocket_genie.integrations.jev_router import OPENROUTER_DECISIONS_URL, JevDecisionClient
from pocket_genie.registry.tools import ToolRegistry
from pocket_genie.tools.jev_tools import register_jev_tools


def test_jev_decide_with_mocked_openrouter_transport() -> None:
    captured: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def fake_transport(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        _timeout: float,
    ) -> dict[str, Any]:
        captured.append((url, payload, headers))
        return {
            "model": "typesafe/jev-latest",
            "answers": {
                "contains_sql": {"type": "noul", "noul": 0.93},
                "domain_class": {
                    "type": "choice",
                    "choice": "analytics",
                    "confidence": 0.89,
                    "probabilities": {"analytics": 0.89, "devops": 0.07, "device": 0.04},
                },
                "query_complexity": {"type": "score", "score": 0.65},
            },
            "usage": {"input_tokens": 180, "output_tokens": 3, "cost": 0.000008},
        }

    client = JevDecisionClient(api_key="sk-or-test-key", transport=fake_transport)
    questions = {
        "contains_sql": {
            "type": "noul",
            "instructions": "The input contains a SQL SELECT query.",
        },
        "domain_class": {
            "type": "choice",
            "criteria": {
                "analytics": "SQL tables, metrics, and quantization ablations.",
                "devops": "Git diffs, pytest suites, and workspace patches.",
                "device": "Battery, CPU, and Android Termux hardware.",
            },
        },
        "query_complexity": {
            "type": "score",
            "criteria": [
                "Trivial single-column lookup.",
                "Filtered aggregation or join across tables.",
                "Multi-stage window function and kurtosis analysis.",
            ],
        },
    }

    res = client.decide(
        state="SELECT recipe_id, AVG(json_valid_rate) FROM ablation_results GROUP BY recipe_id",
        questions=questions,
    )

    assert len(captured) == 1
    url, payload, headers = captured[0]
    assert url == OPENROUTER_DECISIONS_URL
    assert headers["Authorization"] == "Bearer sk-or-test-key"
    assert payload["questions"] == questions

    assert res["source"] == "openrouter"
    assert res["model"] == "typesafe/jev-latest"
    assert res["answers"]["contains_sql"]["noul"] == 0.93
    assert res["answers"]["contains_sql"]["probabilities"] == {"true": 0.93, "false": 0.07}
    assert res["answers"]["domain_class"]["choice"] == "analytics"
    assert res["answers"]["domain_class"]["confidence"] == 0.89
    assert res["answers"]["query_complexity"]["score"] == 0.65
    assert res["usage"]["input_tokens"] == 180


def test_jev_decide_offline_calibrated_fallback_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = JevDecisionClient(api_key=None, transport=None)

    questions = {
        "is_sql": {
            "type": "noul",
            "instructions": "The query uses SQL SELECT syntax on ablation_results.",
        },
        "target_module": {
            "type": "choice",
            "criteria": {
                "sql_module": "SQL SELECT query on ablation_results table.",
                "git_module": "Git branch diff and commit status.",
            },
        },
        "depth": {
            "type": "score",
            "criteria": [
                "Simple lookup.",
                "SQL SELECT query on ablation_results with kurtosis filter.",
            ],
        },
    }

    out = client.decide(
        state="SELECT * FROM ablation_results WHERE kurtosis > 3.0",
        questions=questions,
    )
    assert out["source"] == "local_calibrated"
    assert out["answers"]["is_sql"]["noul"] >= 0.60
    assert out["answers"]["target_module"]["choice"] == "sql_module"
    assert out["answers"]["depth"]["score"] >= 0.50
    assert out["answers"]["depth"]["rubric_index"] == 1

    with pytest.raises(ValueError, match="invalid type"):
        client.decide(state="test", questions={"bad": {"type": "unknown"}})


def test_jev_decide_tool_invocation_via_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reg = ToolRegistry()
    register_jev_tools(reg)

    spec = reg.get("jev.decide")
    assert spec is not None
    assert spec.domain == "jev"
    assert spec.mutating is False

    raw_json = reg.execute(
        "jev.decide",
        {
            "state": "pytest -q tests/test_quant_runtime.py",
            "questions_json": json.dumps(
                {
                    "is_test_run": {
                        "type": "noul",
                        "instructions": "The command executes pytest unit tests.",
                    },
                    "tool_category": {
                        "type": "choice",
                        "criteria": {
                            "testing": "Running pytest unit test suites.",
                            "database": "Executing SQL SELECT queries.",
                        },
                    },
                }
            ),
        },
    )
    parsed = json.loads(raw_json)
    assert parsed["answers"]["is_test_run"]["noul"] >= 0.60
    assert parsed["answers"]["tool_category"]["choice"] == "testing"

    with pytest.raises(ValueError, match="Invalid questions_json"):
        reg.execute("jev.decide", {"state": "hello", "questions_json": "{not-valid-json"})


def test_jev_audit_patch_distinguishes_clean_vs_dangerous_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reg = ToolRegistry()
    register_jev_tools(reg)

    clean_diff = """
    @@ -10,4 +10,7 @@
    +def clamp_probability(val: float) -> float:
    +    return max(0.0, min(1.0, val))
    """
    clean_report = json.loads(
        reg.execute(
            "jev.audit_patch",
            {"diff_or_code": clean_diff, "context": "Add typed probability clamp helper"},
        )
    )
    assert clean_report["recommendation"] == "APPROVE"
    assert clean_report["summary"]["security_hazard"] < 0.20
    assert clean_report["summary"]["breaking_risk"] < 0.20
    assert clean_report["summary"]["patch_quality"] >= 0.75

    dangerous_diff = """
    @@ -22,3 +22,5 @@
    +    os.system("rm -rf /")
    +    result = eval(user_input)
    """
    danger_report = json.loads(
        reg.execute(
            "jev.audit_patch",
            {"diff_or_code": dangerous_diff, "context": "Execute user command"},
        )
    )
    assert danger_report["recommendation"] == "BLOCK"
    assert danger_report["summary"]["security_hazard"] >= 0.80
    assert danger_report["summary"]["breaking_risk"] >= 0.75
    assert danger_report["summary"]["patch_quality"] <= 0.20
