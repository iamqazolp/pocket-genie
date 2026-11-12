from __future__ import annotations

import json
from typing import Any

from pocket_genie.integrations.decision_router import GeminiDecisionClient
from pocket_genie.registry.tools import ToolRegistry


def build_patch_audit_questions() -> dict[str, dict[str, Any]]:
    """Construct the 3-question Gemini Decision Decisions rubric for automated code/diff review."""
    return {
        "breaking_risk": {
            "type": "noul",
            "instructions": (
                "The diff or code snippet introduces breaking API changes, syntax errors, "
                "or unhandled runtime exceptions."
            ),
        },
        "security_hazard": {
            "type": "noul",
            "instructions": (
                "The diff or code snippet introduces command injection, arbitrary code execution, "
                "path traversal, hardcoded secrets, or guardrail bypasses."
            ),
        },
        "patch_quality": {
            "type": "score",
            "criteria": [
                "Risky, destructive, or untested patch with unsafe side effects (0.0).",
                "Moderately complex change that lacks explicit type annotations or boundary checks (0.5).",
                "Clean, well-typed, minimal, and safe code change following repository conventions (1.0).",
            ],
        },
    }


def register_decision_tools(
    registry: ToolRegistry,
    client: GeminiDecisionClient | None = None,
) -> GeminiDecisionClient:
    """Register Gemini Decision System-One decision and patch-auditing tools."""
    decision_client = client or GeminiDecisionClient()

    @registry.register(
        name="decision.decide",
        description=(
            "Evaluate arbitrary state text or JSON against structured Gemini Decision questions "
            "(noul binary probabilities, choice classification, and score rubrics) in a single pass."
        ),
        parameters={
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "Text, JSON payload, log snippet, or code diff to evaluate.",
                },
                "questions_json": {
                    "type": "string",
                    "description": (
                        "JSON object mapping question IDs to question specs with "
                        "type in {'noul', 'choice', 'score'}."
                    ),
                },
            },
            "required": ["state", "questions_json"],
        },
        domain="decision",
        mutating=False,
    )
    def decision_decide(state: str, questions_json: str) -> str:
        try:
            parsed = json.loads(questions_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid questions_json payload: {exc}") from exc

        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("questions_json must decode to a non-empty JSON object.")

        result = decision_client.decide(state=state, questions=parsed)
        return json.dumps(result, indent=2)

    @registry.register(
        name="decision.audit_patch",
        description=(
            "Audit a git diff or code snippet with Gemini Decision across breaking_risk (noul), "
            "security_hazard (noul), and patch_quality (score) before applying a patch."
        ),
        parameters={
            "type": "object",
            "properties": {
                "diff_or_code": {
                    "type": "string",
                    "description": "Git diff hunk or source code snippet to audit.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context describing the target file or patch intent.",
                },
            },
            "required": ["diff_or_code"],
        },
        domain="decision",
        mutating=False,
    )
    def decision_audit_patch(diff_or_code: str, context: str = "") -> str:
        eval_state: dict[str, str] | str = (
            {"context": context, "diff_or_code": diff_or_code} if context else diff_or_code
        )
        decision = decision_client.decide(
            state=eval_state,
            questions=build_patch_audit_questions(),
        )
        answers = decision.get("answers") or {}
        breaking_prob = float((answers.get("breaking_risk") or {}).get("noul") or 0.0)
        security_prob = float((answers.get("security_hazard") or {}).get("noul") or 0.0)
        quality_score = float((answers.get("patch_quality") or {}).get("score") or 0.0)

        if security_prob >= 0.70 or breaking_prob >= 0.75 or quality_score < 0.25:
            recommendation = "BLOCK"
        elif security_prob >= 0.35 or breaking_prob >= 0.40 or quality_score < 0.60:
            recommendation = "REVIEW_CAREFULLY"
        else:
            recommendation = "APPROVE"

        report = {
            "model": decision.get("model"),
            "source": decision.get("source"),
            "recommendation": recommendation,
            "summary": {
                "breaking_risk": round(breaking_prob, 4),
                "security_hazard": round(security_prob, 4),
                "patch_quality": round(quality_score, 4),
            },
            "answers": answers,
            "latency_ms": decision.get("latency_ms", 0.0),
        }
        return json.dumps(report, indent=2)

    return decision_client
