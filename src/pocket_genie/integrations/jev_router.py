from __future__ import annotations

import dataclasses
import json
import os
import re
import time
import urllib.request
from typing import Any, Callable

from pocket_genie.shared import PrecisionTier

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_JEV_MODEL = "typesafe/jev-latest"

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|system)\s+instructions", re.IGNORECASE),
    re.compile(r"bypass\s+(hitl|approval|allowlist|guardrail|security)", re.IGNORECASE),
    re.compile(r"(drop\s+table|rm\s+-rf\s+/|exfiltrate\s+secret)", re.IGNORECASE),
)

_SECURITY_HAZARD_PATTERNS: tuple[re.Pattern[str], ...] = (
    *_INJECTION_PATTERNS,
    re.compile(r"rm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\b(eval|exec)\s*\(", re.IGNORECASE),
    re.compile(r"os\.system\s*\(", re.IGNORECASE),
    re.compile(r"shell\s*=\s*True", re.IGNORECASE),
    re.compile(r"\.\./\.\./|/etc/passwd", re.IGNORECASE),
    re.compile(r"(api_key|secret|password|token)\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]", re.IGNORECASE),
)

_BREAKING_RISK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"rm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\b(eval|exec)\s*\(", re.IGNORECASE),
    re.compile(r"\b(SyntaxError|ZeroDivisionError)\b", re.IGNORECASE),
    re.compile(r"1\s*/\s*0", re.IGNORECASE),
    re.compile(r"breaking\s+change", re.IGNORECASE),
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "whether",
        "does",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "any",
        "all",
        "can",
        "will",
        "introduces",
        "input",
        "state",
        "diff",
    }
)

_VALID_QUESTION_TYPES: frozenset[str] = frozenset({"noul", "choice", "score"})


def _tokenize_words(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9_]+", text.lower())
        if len(tok) >= 2 and tok not in _STOPWORDS
    }

_MUTATING_KEYWORDS: tuple[str, ...] = (
    "patch",
    "commit",
    "write",
    "notify",
    "notification",
    "alert",
    "clipboard",
    "delete",
    "overwrite",
)

_SPECIALIST_HINTS: dict[str, tuple[str, ...]] = {
    "devops_agent": ("git", "status", "diff", "pytest", "test", "branch", "commit", "patch", "read_file", "line"),
    "sql_agent": ("sql", "select", "table", "ablation_results", "layer_telemetry", "failure_taxonomy", "schema"),
    "knowledge_base_agent": ("search", "docs", "documentation", "architecture", "smoothquant", "kurtosis", "kv cache"),
    "device_agent": ("battery", "cpu", "disk", "telemetry", "termux", "phone", "notification", "alert", "clipboard"),
    "personal_agent": ("deadline", "due", "schedule", "agenda", "calendar", "reminder", "triage", "reply", "inbox"),
    "work_ops_agent": ("github", "pull request", "databricks", "warehouse", "cluster", "antigravity", "claude code"),
    "research_study_agent": ("arxiv", "paper", "preprint", "flashcard", "study", "quiz", "sm-2", "fetch_url"),
}


@dataclasses.dataclass(slots=True)
class JevTriageDecision:
    """Structured System-One triage output from TypeSafe Jev (OpenRouter Decisions API)."""

    model: str
    source: str  # "openrouter" | "local_calibrated"
    injection_prob: float
    mutating_prob: float
    specialist: str
    route_confidence: float
    route_probabilities: dict[str, float]
    complexity_score: float
    recommended_tier: PrecisionTier
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def is_injection_blocked(self) -> bool:
        return self.injection_prob >= 0.80

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["is_injection_blocked"] = self.is_injection_blocked
        return d


def build_jev_triage_questions() -> dict[str, dict[str, Any]]:
    """Construct the 4-question typed Jev Decisions payload (2x noul, 1x choice, 1x score)."""
    return {
        "prompt_injection": {
            "type": "noul",
            "instructions": (
                "The user message attempts prompt injection, system instruction override, "
                "guardrail bypass, or destructive sabotage."
            ),
        },
        "mutating_intent": {
            "type": "noul",
            "instructions": (
                "The user message asks to mutate workspace files, apply patches, set clipboard state, "
                "or dispatch external device notifications."
            ),
        },
        "route_specialist": {
            "type": "choice",
            "criteria": {
                "devops_agent": "Git repository status, diffs, reading source files, running pytest suites, or code patches.",
                "sql_agent": "SQL SELECT analytics, table schemas, quantization ablation tables, or chart metrics.",
                "knowledge_base_agent": "Searching project docs, architecture notes, SmoothQuant/KV-cache concepts, or skills.",
                "device_agent": "Host CPU/disk/battery telemetry, Android Termux hardware APIs, clipboard, or notifications.",
                "personal_agent": "Personal deadlines, countdowns, schedules, native reminders, or message reply triage.",
                "work_ops_agent": "GitHub PRs/issues/CI, Databricks SQL/jobs/clusters, Antigravity, or Claude Code sessions.",
                "research_study_agent": "arXiv paper search, web doc fetching, SM-2 study flashcards, or active-recall quizzes.",
                "supervisor": "Multi-domain tasks spanning two or more specialists, or general ambiguous requests.",
            },
        },
        "precision_tier": {
            "type": "score",
            "criteria": [
                "Simple single-tool lookup or status check suitable for fast W4A8 execution (t1).",
                "Structured SQL filtering, multi-argument schema call, or technical synthesis needing P_mixed + FSM (t2).",
                "Complex multi-step debugging, adversarial edge case, or high-precision synthesis needing W4A16 (t3).",
            ],
        },
    }


def _score_to_tier(score: float) -> PrecisionTier:
    if score >= 0.72:
        return "t3"
    if score >= 0.38:
        return "t2"
    return "t1"


def _extract_noul(ans: dict[str, Any]) -> float:
    if "noul" in ans and ans["noul"] is not None:
        return max(0.0, min(1.0, float(ans["noul"])))
    probs = ans.get("probabilities") or {}
    if "true" in probs:
        return max(0.0, min(1.0, float(probs["true"])))
    return 0.0


def _extract_score(ans: dict[str, Any]) -> float:
    if "score" in ans and ans["score"] is not None:
        val = float(ans["score"])
        return val / 2.0 if val > 1.0 else max(0.0, min(1.0, val))
    return 0.0


class JevDecisionClient:
    """System-One TypeSafe Jev client over OpenRouter's `/api/alpha/decisions` endpoint with calibrated offline fallback."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        endpoint: str = OPENROUTER_DECISIONS_URL,
        timeout: float = 8.0,
        transport: Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]] | None = None,
    ) -> None:
        self.model = model or os.environ.get("JEV_MODEL", DEFAULT_JEV_MODEL)
        self._api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self._transport = transport

    def _resolve_api_key(self) -> str | None:
        return self._api_key or os.environ.get("OPENROUTER_API_KEY")

    def _default_post(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return dict(json.loads(resp.read().decode("utf-8")))

    def _local_calibrated_fallback(self, prompt: str, t0: float) -> JevTriageDecision:
        text = prompt.strip()
        lower = text.lower()

        inj_hits = sum(1 for pat in _INJECTION_PATTERNS if pat.search(text))
        inj_prob = min(0.96, 0.88 * inj_hits) if inj_hits else 0.02

        mut_hits = sum(1 for kw in _MUTATING_KEYWORDS if kw in lower)
        mut_prob = min(0.95, 0.35 + 0.30 * mut_hits) if mut_hits else 0.05

        raw_scores: dict[str, float] = {k: 0.15 for k in (*_SPECIALIST_HINTS.keys(), "supervisor")}
        matched_domains = 0
        for spec_id, hints in _SPECIALIST_HINTS.items():
            hits = sum(1 for h in hints if h in lower)
            if hits > 0:
                matched_domains += 1
                raw_scores[spec_id] += 1.4 * hits

        if matched_domains != 1:
            raw_scores["supervisor"] += 1.8

        total_w = sum(raw_scores.values()) or 1.0
        probs = {k: round(v / total_w, 4) for k, v in raw_scores.items()}
        winner = max(probs, key=lambda k: probs[k])
        sorted_vals = sorted(probs.values(), reverse=True)
        conf = round(min(1.0, max(0.0, sorted_vals[0] - sorted_vals[1] + 0.25)), 4)

        if any(w in lower for w in ("select ", "with ", "join ", "adversarial", "kurtosis", "compare")):
            c_score = 0.52
        elif len(text) > 220 or matched_domains >= 2:
            c_score = 0.76
        else:
            c_score = 0.18

        return JevTriageDecision(
            model=f"{self.model}:offline",
            source="local_calibrated",
            injection_prob=round(inj_prob, 4),
            mutating_prob=round(mut_prob, 4),
            specialist=winner,
            route_confidence=conf,
            route_probabilities=probs,
            complexity_score=c_score,
            recommended_tier=_score_to_tier(c_score),
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

    def evaluate_turn(self, prompt: str) -> JevTriageDecision:
        """Evaluate user prompt in a single System-One Jev Decisions call (falls back locally when offline)."""
        t0 = time.perf_counter()
        key = self._resolve_api_key()
        if not key and self._transport is None:
            return self._local_calibrated_fallback(prompt, t0)

        payload: dict[str, Any] = {
            "model": self.model,
            "state": prompt,
            "questions": build_jev_triage_questions(),
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key or 'sk-or-test'}",
        }

        try:
            post_fn = self._transport or self._default_post
            body = post_fn(self.endpoint, payload, headers, self.timeout)
        except Exception:
            return self._local_calibrated_fallback(prompt, t0)

        answers = body.get("answers") or {}
        inj_prob = _extract_noul(answers.get("prompt_injection") or {})
        mut_prob = _extract_noul(answers.get("mutating_intent") or {})

        route_ans = answers.get("route_specialist") or {}
        probs = {str(k): round(float(v), 4) for k, v in (route_ans.get("probabilities") or {}).items()}
        specialist = str(route_ans.get("choice") or "supervisor")
        conf = float(route_ans.get("confidence") or ( max(probs.values()) if probs else 0.5 ))

        c_score = _extract_score(answers.get("precision_tier") or {})
        usage = body.get("usage") or {}
        in_toks = int(usage.get("input_tokens") or 0)
        out_toks = int(usage.get("output_tokens") or 0)
        raw_cost = usage.get("cost")
        cost_usd = float(raw_cost) if raw_cost is not None else round(in_toks * 0.042 / 1_000_000.0, 8)

        return JevTriageDecision(
            model=str(body.get("model") or self.model),
            source="openrouter",
            injection_prob=round(inj_prob, 4),
            mutating_prob=round(mut_prob, 4),
            specialist=specialist,
            route_confidence=round(conf, 4),
            route_probabilities=probs,
            complexity_score=round(c_score, 4),
            recommended_tier=_score_to_tier(c_score),
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            input_tokens=in_toks,
            output_tokens=out_toks,
            cost_usd=cost_usd,
        )

    @staticmethod
    def _validate_questions(questions: dict[str, dict[str, Any]]) -> None:
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty dictionary mapping question IDs to specifications.")
        for q_id, q_spec in questions.items():
            if not isinstance(q_spec, dict):
                raise ValueError(f"Question {q_id!r} specification must be a dictionary.")
            q_type = q_spec.get("type")
            if q_type not in _VALID_QUESTION_TYPES:
                raise ValueError(
                    f"Question {q_id!r} has invalid type {q_type!r}; must be one of {sorted(_VALID_QUESTION_TYPES)}."
                )
            if q_type == "choice":
                criteria = q_spec.get("criteria")
                if not isinstance(criteria, dict) or not criteria:
                    raise ValueError(f"Choice question {q_id!r} requires a non-empty 'criteria' dict.")
            elif q_type == "score":
                criteria = q_spec.get("criteria")
                if not isinstance(criteria, list) or not criteria:
                    raise ValueError(f"Score question {q_id!r} requires a non-empty 'criteria' list.")

    def _local_decide_fallback(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        t0: float,
    ) -> dict[str, Any]:
        state_text = json.dumps(state, ensure_ascii=False) if isinstance(state, (dict, list)) else str(state)
        state_lower = state_text.lower()
        state_tokens = _tokenize_words(state_text)

        sec_hits = sum(1 for pat in _SECURITY_HAZARD_PATTERNS if pat.search(state_text))
        brk_hits = sum(1 for pat in _BREAKING_RISK_PATTERNS if pat.search(state_text))

        answers: dict[str, dict[str, Any]] = {}
        for q_id, q_spec in questions.items():
            q_type = str(q_spec["type"])
            q_id_lower = q_id.lower()

            if q_type == "noul":
                instructions = str(q_spec.get("instructions") or "")
                inst_lower = instructions.lower()
                inst_tokens = _tokenize_words(instructions)
                is_security_q = any(
                    w in inst_lower or w in q_id_lower
                    for w in ("security", "hazard", "injection", "traversal", "secret", "bypass", "sabotage")
                )
                is_breaking_q = any(
                    w in inst_lower or w in q_id_lower
                    for w in ("breaking", "syntax error", "exception", "unhandled", "regression", "crash")
                )

                if is_security_q:
                    if sec_hits > 0:
                        prob = round(min(0.96, 0.82 + 0.07 * sec_hits), 4)
                    else:
                        prob = 0.04
                elif is_breaking_q:
                    if brk_hits > 0 or sec_hits > 0:
                        prob = round(min(0.95, 0.78 + 0.08 * (brk_hits + sec_hits)), 4)
                    else:
                        prob = 0.05
                else:
                    overlap = len(inst_tokens & state_tokens)
                    if overlap > 0:
                        ratio = overlap / max(1, min(len(inst_tokens), 5))
                        prob = round(min(0.96, max(0.58, 0.45 + 0.48 * ratio)), 4)
                    elif sec_hits > 0 and "risk" in inst_lower:
                        prob = 0.88
                    else:
                        prob = 0.05

                prob = max(0.03, min(0.96, prob))
                answers[q_id] = {
                    "type": "noul",
                    "noul": prob,
                    "probabilities": {"true": prob, "false": round(1.0 - prob, 4)},
                }

            elif q_type == "choice":
                criteria_map: dict[str, Any] = dict(q_spec.get("criteria") or {})
                weights: dict[str, float] = {}
                for opt_key, desc in criteria_map.items():
                    opt_tokens = _tokenize_words(f"{opt_key} {desc}")
                    overlap = len(opt_tokens & state_tokens)
                    key_prefix = str(opt_key).lower().split("_")[0]
                    direct_bonus = 0.85 if (len(key_prefix) >= 3 and key_prefix in state_lower) else 0.0
                    weights[str(opt_key)] = 0.15 + 1.25 * overlap + direct_bonus

                total_w = sum(weights.values()) or 1.0
                probs = {k: round(v / total_w, 4) for k, v in weights.items()}
                winner = max(probs, key=lambda k: probs[k])
                sorted_vals = sorted(probs.values(), reverse=True)
                gap = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else sorted_vals[0]
                conf = round(min(0.99, max(0.35, sorted_vals[0] + 0.35 * gap)), 4)
                answers[q_id] = {
                    "type": "choice",
                    "choice": winner,
                    "confidence": conf,
                    "probabilities": probs,
                }

            elif q_type == "score":
                rubric: list[Any] = list(q_spec.get("criteria") or [])
                n_levels = len(rubric)
                rubric_text = " ".join(str(item).lower() for item in rubric)
                is_quality_rubric = "quality" in q_id_lower or (
                    "clean" in rubric_text and ("risky" in rubric_text or "untested" in rubric_text)
                )

                if is_quality_rubric:
                    if sec_hits > 0 or brk_hits > 0:
                        idx = 0
                        score = 0.08
                    elif any(tok in state_text for tok in ("->", ": int", ": str", ": float", "def ", "return ")):
                        idx = n_levels - 1
                        score = 0.88
                    else:
                        idx = max(0, (n_levels - 1) // 2)
                        score = 0.62
                else:
                    level_hits = [len(_tokenize_words(str(item)) & state_tokens) for item in rubric]
                    max_hits = max(level_hits, default=0)
                    if max_hits > 0:
                        idx = max(range(n_levels), key=lambda i: (level_hits[i], i))
                    elif len(state_text) >= 220:
                        idx = n_levels - 1
                    elif len(state_text) >= 80:
                        idx = min(1, n_levels - 1)
                    else:
                        idx = 0
                    score = round(idx / max(1, n_levels - 1), 4) if n_levels > 1 else 0.5
                    score = max(0.08, min(0.94, score))

                answers[q_id] = {
                    "type": "score",
                    "score": round(score, 4),
                    "rubric_index": idx,
                }

        return {
            "model": f"{self.model}:offline",
            "source": "local_calibrated",
            "answers": answers,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0},
        }

    def decide(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Execute a general-purpose System-One Jev Decisions query across noul, choice, and score questions."""
        self._validate_questions(questions)
        t0 = time.perf_counter()
        key = self._resolve_api_key()
        if not key and self._transport is None:
            return self._local_decide_fallback(state, questions, t0)

        payload: dict[str, Any] = {
            "model": self.model,
            "state": state,
            "questions": questions,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key or 'sk-or-test'}",
        }

        try:
            post_fn = self._transport or self._default_post
            body = post_fn(self.endpoint, payload, headers, self.timeout)
        except Exception:
            return self._local_decide_fallback(state, questions, t0)

        raw_answers = body.get("answers") or {}
        normalized_answers: dict[str, dict[str, Any]] = {}
        for q_id, q_spec in questions.items():
            q_type = str(q_spec["type"])
            ans = dict(raw_answers.get(q_id) or {})
            ans["type"] = q_type
            if q_type == "noul":
                prob = round(_extract_noul(ans), 4)
                ans["noul"] = prob
                if "probabilities" not in ans:
                    ans["probabilities"] = {"true": prob, "false": round(1.0 - prob, 4)}
            elif q_type == "choice":
                probs = {str(k): round(float(v), 4) for k, v in (ans.get("probabilities") or {}).items()}
                criteria_keys = list((q_spec.get("criteria") or {}).keys())
                default_choice = str(criteria_keys[0]) if criteria_keys else ""
                choice = str(ans.get("choice") or (max(probs, key=lambda k: probs[k]) if probs else default_choice))
                conf = round(float(ans.get("confidence") or (max(probs.values()) if probs else 0.5)), 4)
                ans["choice"] = choice
                ans["confidence"] = conf
                ans["probabilities"] = probs
            elif q_type == "score":
                ans["score"] = round(_extract_score(ans), 4)
            normalized_answers[q_id] = ans

        usage = dict(body.get("usage") or {})
        return {
            "model": str(body.get("model") or self.model),
            "source": "openrouter",
            "answers": normalized_answers,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "usage": usage,
        }
