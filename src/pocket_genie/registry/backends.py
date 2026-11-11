from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
import time
import urllib.request
from typing import Any, Protocol

from pocket_genie.env_config import ApiLLMConfig, get_ssl_context, resolve_api_llm_config
from pocket_genie.shared import (
    ChatMessage,
    PrecisionTier,
    QuantTelemetrySnapshot,
    ToolCallChunk,
    ToolInvocation,
)

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclasses.dataclass(slots=True)
class GenerationOutput:
    message: ChatMessage
    streamed_chunks: list[ToolCallChunk] = dataclasses.field(default_factory=list)
    telemetry: QuantTelemetrySnapshot = dataclasses.field(
        default_factory=QuantTelemetrySnapshot
    )


class ModelBackend(Protocol):
    name: str
    tier: PrecisionTier

    def generate(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        constrain_schema: bool = False,
    ) -> GenerationOutput: ...


_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sql": ("sql", "select", "table", "ablation", "recipe", "e0", "e1", "e2", "e4s", "metric", "compare", "exact", "footprint"),
    "knowledge_base": ("search", "find", "kurtosis", "smoothquant", "outlier", "doc", "note", "explain", "architecture", "where"),
    "device": ("battery", "cpu", "disk", "system", "telemetry", "phone", "termux", "notify", "notification", "alert", "clipboard"),
    "devops": ("git", "status", "diff", "branch", "pytest", "test", "repo", "commit", "file", "patch"),
}


def _select_tool_for_query(query: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    q_lower = query.lower()
    best_fn = tools[0]["function"]
    best_score = -1

    for t in tools:
        fn = t["function"]
        name_lower = fn["name"].lower()
        desc_lower = fn.get("description", "").lower()
        score = 0

        for tok in q_lower.replace(".", " ").replace("_", " ").split():
            if len(tok) >= 3:
                if tok in name_lower:
                    score += 4
                elif tok in desc_lower:
                    score += 2

        for dom, keywords in _DOMAIN_KEYWORDS.items():
            if dom in name_lower and any(kw in q_lower for kw in keywords):
                score += 6

        if ("notify" in q_lower or "alert" in q_lower) and "send_notification" in name_lower:
            score += 12
        elif ("clipboard" in q_lower or "copy" in q_lower) and "clipboard_set" in name_lower:
            score += 12
        elif ("patch" in q_lower or "write" in q_lower) and "apply_patch" in name_lower:
            score += 12
        elif ("diff" in q_lower) and "git_diff" in name_lower:
            score += 10
        elif ("pytest" in q_lower or "test" in q_lower) and "run_pytest" in name_lower:
            score += 10
        elif ("read" in q_lower or "chunk" in q_lower) and "read_file_chunk" in name_lower:
            score += 10
        elif not any(w in q_lower for w in ("notify", "alert", "clipboard", "patch", "write")):
            if any(m in name_lower for m in ("send_notification", "clipboard_set", "apply_patch", "ack_")):
                score -= 3

        if score > best_score:
            best_score = score
            best_fn = fn

    return best_fn


def parse_tool_calls_from_text(content: str) -> list[ToolInvocation]:
    """Extract tool invocations from <tool_call> tags, markdown fences, or raw JSON objects."""
    invocations: list[ToolInvocation] = []
    for idx, match in enumerate(_TOOL_CALL_BLOCK_RE.finditer(content)):
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict) and "name" in obj:
                args = obj.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args)
                invocations.append(
                    ToolInvocation(id=f"call_tag_{idx}", name=str(obj["name"]), arguments=dict(args))
                )
        except Exception:
            continue

    if invocations:
        return invocations

    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()

    candidates = [stripped]
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        brace_slice = stripped[first_brace : last_brace + 1]
        if brace_slice != stripped:
            candidates.append(brace_slice)

    for cand in candidates:
        if cand.startswith("{") and cand.endswith("}"):
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict) and "name" in obj:
                    args = obj.get("arguments") or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                    invocations.append(
                        ToolInvocation(id="call_json_0", name=str(obj["name"]), arguments=dict(args))
                    )
                    break
            except Exception:
                continue

    return invocations


class DeterministicReplayBackend:
    """Deterministic backend with ReAct synthesis and intent-aware tool routing for offline Termux, CI, and UI testing."""

    def __init__(
        self,
        tier: PrecisionTier = "t1",
        scripted_responses: list[Any] | None = None,
    ) -> None:
        self.name = f"deterministic-{tier}"
        self.tier = tier
        self._queue: list[Any] = list(scripted_responses or [])

    def push_response(self, response: Any) -> None:
        self._queue.append(response)

    def generate(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        constrain_schema: bool = False,
    ) -> GenerationOutput:
        profile_map = {"t1": "P_fast", "t2": "P_mixed", "t3": "P_accurate"}
        act_bits_map = {"t1": 8.0, "t2": 8.24, "t3": 16.0}
        telemetry = QuantTelemetrySnapshot(
            tier=self.tier,
            profile=profile_map[self.tier],  # type: ignore[arg-type]
            weight_bits=4.0,
            effective_act_bits=act_bits_map[self.tier],
            promoted_tokens=0 if self.tier == "t1" else (4 if self.tier == "t2" else 32),
            total_tokens=32,
            min_logit_margin=4.2 if self.tier == "t1" else 6.1,
            mean_entropy=0.12 if self.tier != "t1" else 0.21,
            prefix_kv_hit=True,
            kv_cache_fp16_mb=12.4,
            kv_cache_quant_mb=3.6,
        )

        if self._queue:
            item = self._queue.pop(0)
            if callable(item):
                return item(messages, tools, constrain_schema, telemetry)
            if isinstance(item, GenerationOutput):
                return item
            if isinstance(item, ChatMessage):
                return GenerationOutput(message=item, telemetry=telemetry)

        if messages and messages[-1].role == "tool":
            trailing_obs: list[str] = []
            for m in reversed(messages):
                if m.role != "tool":
                    break
                prefix = f"[{m.name}] " if (m.name and not m.content.startswith("[")) else ""
                trailing_obs.append(f"{prefix}{m.content}")
            trailing_obs.reverse()
            return GenerationOutput(
                message=ChatMessage(
                    role="assistant",
                    content="\n\n".join(trailing_obs),
                    reasoning=f"Synthesized tool observations on {self.tier} ({telemetry.profile}).",
                ),
                telemetry=telemetry,
            )

        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        if tools:
            chosen = _select_tool_for_query(last_user, tools)
            first_tool = chosen["name"]
            args: dict[str, Any] = {}
            props = chosen.get("parameters", {}).get("properties", {})
            if "title" in props and "message" in props:
                args["title"] = "Pocket-Genie Alert"
                args["message"] = last_user or "Task completed"
            elif "query" in props:
                if first_tool == "sql.query_workspace" and not last_user.strip().upper().startswith(("SELECT", "WITH")):
                    args["query"] = (
                        "SELECT recipe_id, avg_bits_linear, footprint_mb, json_valid_rate, values_exact_rate "
                        "FROM ablation_results ORDER BY values_exact_rate DESC"
                    )
                else:
                    args["query"] = last_user or "status"
            elif "task" in props:
                args["task"] = last_user or "inspect workspace"
            elif "name" in props:
                args["name"] = "quant-diagnosis"
            elif "path" in props:
                args["path"] = "README.md"

            raw_json = json.dumps(args)
            mid = max(1, len(raw_json) // 2)
            chunks = [
                ToolCallChunk(index=0, id="call_01", name=first_tool, arguments_fragment=raw_json[:mid]),
                ToolCallChunk(index=0, id=None, name=None, arguments_fragment=raw_json[mid:]),
            ]
            msg = ChatMessage(
                role="assistant",
                content="",
                reasoning=f"Routing to `{first_tool}` on {self.tier} ({telemetry.profile}).",
                tool_calls=[ToolInvocation(id="call_01", name=first_tool, arguments=args)],
            )
            return GenerationOutput(message=msg, streamed_chunks=chunks, telemetry=telemetry)

        return GenerationOutput(
            message=ChatMessage(
                role="assistant",
                content=f"[{self.tier}/{telemetry.profile}] Completed analysis for: {last_user}",
                reasoning="Synthesizing final response from specialist observations.",
            ),
            telemetry=telemetry,
        )


def _post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "pocket-genie/0.3",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout, context=get_ssl_context()) as resp:
        return dict(json.loads(resp.read().decode("utf-8")))


class OpenAICompatibleBackend:
    """HTTP client backend for on-device llama-server (Termux), Ollama, vLLM,
    or OpenAI endpoints with live logprobs routing.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        model: str = "qwen2.5:3b",
        api_key: str = "local",
        tier: PrecisionTier = "t1",
        timeout: float = 60.0,
        backup_model: str | None = None,
        request_logprobs: bool = True,
    ) -> None:
        self.name = f"local-llm:{model}"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.tier = tier
        self.timeout = timeout
        self.backup_model = backup_model if backup_model != model else None
        self.request_logprobs = request_logprobs

    def generate(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        constrain_schema: bool = False,
    ) -> GenerationOutput:
        import urllib.error

        from pocket_genie.quant.enforcer import build_tool_call_schema
        from pocket_genie.quant.entropy_router import TokenPrecisionRouter

        wire_to_orig: dict[str, str] = {}
        orig_to_wire: dict[str, str] = {}
        wire_tools: list[dict[str, Any]] | None = None
        if tools is not None:
            wire_tools = []
            for t in tools:
                tc = copy.deepcopy(t)
                fn = tc.get("function")
                if isinstance(fn, dict) and "name" in fn:
                    orig_name = str(fn["name"])
                    wire_name = re.sub(r"[^a-zA-Z0-9_-]", "__", orig_name)
                    fn["name"] = wire_name
                    wire_to_orig[wire_name] = orig_name
                    orig_to_wire[orig_name] = wire_name
                wire_tools.append(tc)

        out_messages: list[dict[str, Any]] = []
        for m in messages:
            d: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.role == "tool":
                d["tool_call_id"] = m.tool_call_id or "call_0"
                if m.name:
                    d["name"] = orig_to_wire.get(m.name, re.sub(r"[^a-zA-Z0-9_-]", "__", m.name))
            elif m.role == "assistant" and m.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": orig_to_wire.get(
                                tc.name, re.sub(r"[^a-zA-Z0-9_-]", "__", tc.name)
                            ),
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out_messages.append(d)

        is_local = any(h in self.base_url for h in ("127.0.0.1", "localhost", "0.0.0.0"))
        enforce_tool_json = bool(
            wire_tools
            and constrain_schema
            and is_local
            and not any(m.role == "tool" for m in messages)
        )
        if enforce_tool_json:
            schema = build_tool_call_schema(wire_tools)
            out_messages.append(
                {
                    "role": "system",
                    "content": (
                        "Respond ONLY with a valid JSON tool call matching this schema: "
                        + json.dumps(schema)
                    ),
                }
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": out_messages,
            "temperature": 0.0,
        }
        payload["logprobs"] = True
        payload["top_logprobs"] = 5
        if wire_tools:
            payload["tools"] = wire_tools
        if enforce_tool_json:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "tool_call",
                    "schema": build_tool_call_schema(wire_tools),
                },
            }

        endpoint = f"{self.base_url}/chat/completions"
        t0 = time.perf_counter()
        stripped_payload = {
            k: v
            for k, v in payload.items()
            if k not in ("logprobs", "top_logprobs", "response_format")
        }
        try:
            try:
                body = _post_json(endpoint, payload, self.api_key, self.timeout)
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 404, 422, 429, 500, 502, 503, 504):
                    if exc.code in (429, 502, 503, 504) and "PYTEST_CURRENT_TEST" not in os.environ:
                        time.sleep(1.0)
                    body = _post_json(endpoint, stripped_payload, self.api_key, self.timeout)
                else:
                    raise
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if not self.backup_model:
                raise
            backup_payload = {**stripped_payload, "model": self.backup_model}
            body = _post_json(endpoint, backup_payload, self.api_key, self.timeout)
        elapsed = max(1e-3, time.perf_counter() - t0)

        choice_obj = body["choices"][0]
        choice = choice_obj["message"]
        content = choice.get("content") or ""
        invocations: list[ToolInvocation] = []
        for idx, tc in enumerate(choice.get("tool_calls") or []):
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            raw_name = fn.get("name", "")
            orig_name = wire_to_orig.get(raw_name, raw_name.replace("__", "."))
            invocations.append(
                ToolInvocation(
                    id=tc.get("id") or f"call_{idx}",
                    name=orig_name,
                    arguments=parsed_args,
                )
            )
        if not invocations and content:
            raw_invocations = parse_tool_calls_from_text(content)
            invocations = [
                ToolInvocation(
                    id=inv.id,
                    name=wire_to_orig.get(inv.name, inv.name.replace("__", ".")),
                    arguments=inv.arguments,
                )
                for inv in raw_invocations
            ]

        usage = body.get("usage") or {}
        comp_tokens = int(usage.get("completion_tokens") or max(1, len(content) // 4))
        profile_map = {"t1": "P_fast", "t2": "P_mixed", "t3": "P_accurate"}
        act_bits_map = {"t1": 8.0, "t2": 8.24, "t3": 16.0}

        router = TokenPrecisionRouter()
        logprobs_block = choice_obj.get("logprobs") or {}
        token_entries = logprobs_block.get("content") or []
        for step_idx, entry in enumerate(token_entries):
            top_lp = entry.get("top_logprobs") or []
            vals = [float(item["logprob"]) for item in top_lp if "logprob" in item]
            if not vals and "logprob" in entry:
                vals = [float(entry["logprob"]), float(entry["logprob"]) - 2.5]
            if vals:
                router.evaluate_step(step_idx, vals)

        if router.decisions:
            telemetry = router.build_telemetry(prefix_kv_hit=True)
            if self.tier == "t3":
                telemetry.tier = "t3"
                telemetry.profile = "P_accurate"
                telemetry.effective_act_bits = 16.0
            elif self.tier == "t2":
                telemetry.tier = "t2"
                telemetry.profile = "P_mixed"
                telemetry.effective_act_bits = max(8.24, telemetry.effective_act_bits)
        else:
            telemetry = QuantTelemetrySnapshot(
                tier=self.tier,
                profile=profile_map[self.tier],  # type: ignore[arg-type]
                weight_bits=4.0,
                effective_act_bits=act_bits_map[self.tier],
                promoted_tokens=comp_tokens if self.tier != "t1" else 0,
                total_tokens=comp_tokens,
                prefix_kv_hit=True,
            )

        msg = ChatMessage(
            role="assistant",
            content="" if invocations else content,
            reasoning=f"Generated by `{self.model}` ({comp_tokens} tokens in {elapsed:.2f}s).",
            tool_calls=invocations,
        )
        return GenerationOutput(message=msg, telemetry=telemetry)


class BackendRegistry:
    """Manages model backends mapped to precision tiers t1, t2, and t3."""

    def __init__(self) -> None:
        self._backends: dict[PrecisionTier, ModelBackend] = {
            "t1": DeterministicReplayBackend(tier="t1"),
            "t2": DeterministicReplayBackend(tier="t2"),
            "t3": DeterministicReplayBackend(tier="t3"),
        }
        self.hf_runtime: Any | None = None

    def configure_local_llm(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        primary_model: str = "qwen2.5:3b",
        fallback_model: str | None = None,
    ) -> None:
        fb = fallback_model or primary_model
        self._backends["t1"] = OpenAICompatibleBackend(base_url=base_url, model=primary_model, tier="t1")
        self._backends["t2"] = OpenAICompatibleBackend(base_url=base_url, model=primary_model, tier="t2")
        self._backends["t3"] = OpenAICompatibleBackend(base_url=base_url, model=fb, tier="t3")

    def configure_api_llm(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        fast_model: str | None = None,
        accurate_model: str | None = None,
    ) -> ApiLLMConfig:
        cfg = resolve_api_llm_config(
            base_url=base_url,
            fast_model=fast_model,
            accurate_model=accurate_model,
        )
        resolved_key = api_key or cfg.api_key
        self._backends["t1"] = OpenAICompatibleBackend(
            base_url=cfg.base_url,
            model=cfg.fast_model,
            api_key=resolved_key,
            tier="t1",
            timeout=25.0,
            backup_model=cfg.accurate_model,
        )
        self._backends["t2"] = OpenAICompatibleBackend(
            base_url=cfg.base_url,
            model=cfg.fast_model,
            api_key=resolved_key,
            tier="t2",
            timeout=25.0,
            backup_model=cfg.accurate_model,
        )
        self._backends["t3"] = OpenAICompatibleBackend(
            base_url=cfg.base_url,
            model=cfg.accurate_model,
            api_key=resolved_key,
            tier="t3",
            timeout=35.0,
            backup_model=cfg.fast_model,
        )
        return cfg

    def configure_hybrid(
        self,
        local_base_url: str = "http://127.0.0.1:11434/v1",
        local_model: str = "qwen2.5-coder:7b",
        api_base_url: str | None = None,
        api_key: str | None = None,
        cloud_model: str | None = None,
    ) -> ApiLLMConfig:
        cfg = resolve_api_llm_config(
            base_url=api_base_url,
            accurate_model=cloud_model,
        )
        resolved_key = api_key or cfg.api_key
        self._backends["t1"] = OpenAICompatibleBackend(
            base_url=local_base_url,
            model=local_model,
            tier="t1",
        )
        self._backends["t2"] = OpenAICompatibleBackend(
            base_url=local_base_url,
            model=local_model,
            tier="t2",
        )
        self._backends["t3"] = OpenAICompatibleBackend(
            base_url=cfg.base_url,
            model=cloud_model or cfg.accurate_model,
            api_key=resolved_key,
            tier="t3",
            timeout=35.0,
            backup_model=cfg.fast_model,
        )
        return cfg

    def configure_hf_quant(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str | None = None,
        weight_bits: int = 4,
        apply_smoothquant: bool = True,
        max_new_tokens: int = 96,
    ) -> Any:
        from pocket_genie.quant.hf_backend import QuantizedPyTorchBackend, SharedHFQuantRuntime

        runtime = SharedHFQuantRuntime.load(
            model_id=model_id,
            device=device,
            weight_bits=weight_bits,
            apply_smoothquant=apply_smoothquant,
        )
        self.hf_runtime = runtime
        self._backends["t1"] = QuantizedPyTorchBackend(runtime, tier="t1", max_new_tokens=max_new_tokens)
        self._backends["t2"] = QuantizedPyTorchBackend(runtime, tier="t2", max_new_tokens=max_new_tokens)
        self._backends["t3"] = QuantizedPyTorchBackend(runtime, tier="t3", max_new_tokens=max_new_tokens)
        return runtime

    def register(self, tier: PrecisionTier, backend: ModelBackend) -> None:
        self._backends[tier] = backend

    def get(self, tier: PrecisionTier) -> ModelBackend:
        return self._backends[tier]
