from __future__ import annotations

import dataclasses
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Sequence

from pocket_genie.registry.backends import (
    DeterministicReplayBackend,
    ModelBackend,
    OpenAICompatibleBackend,
    parse_tool_calls_from_text,
)
from pocket_genie.registry.tools import ToolRegistry, sanitize_json_schema
from pocket_genie.shared import ChatMessage
from pocket_genie.tools import register_all_builtin_tools


@dataclasses.dataclass(frozen=True, slots=True)
class EvalTask:
    task_id: str
    domain: str
    prompt: str
    expected_tool: str
    required_args: tuple[str, ...] = ()


EVAL_SUITE: tuple[EvalTask, ...] = (
    # DevOps domain (8 tasks)
    EvalTask("devops_01", "devops", "Check the current git repository status.", "devops.git_status", ()),
    EvalTask("devops_02", "devops", "Show unstaged git diff changes in the workspace.", "devops.git_diff", ()),
    EvalTask("devops_03", "devops", "Read lines 1 to 25 from README.md.", "devops.read_file_chunk", ("path",)),
    EvalTask("devops_04", "devops", "Run the pytest unit test suite.", "devops.run_pytest", ()),
    EvalTask("devops_05", "devops", "Inspect git status to see modified files.", "devops.git_status", ()),
    EvalTask("devops_06", "devops", "Read file chunk from pyproject.toml starting at line 1.", "devops.read_file_chunk", ("path",)),
    EvalTask("devops_07", "devops", "Display the git diff for recent edits.", "devops.git_diff", ()),
    EvalTask("devops_08", "devops", "Execute pytest tests to verify nothing is broken.", "devops.run_pytest", ()),
    # SQL Analytics domain (8 tasks)
    EvalTask("sql_01", "sql", "Run a SQL SELECT query on ablation_results to compare json_valid_rate.", "sql.query_workspace", ("query",)),
    EvalTask("sql_02", "sql", "Describe all available SQL tables and their columns.", "sql.describe_tables", ()),
    EvalTask("sql_03", "sql", "SELECT recipe_id, footprint_mb FROM ablation_results ORDER BY footprint_mb ASC", "sql.query_workspace", ("query",)),
    EvalTask("sql_04", "sql", "Query layer_telemetry SQL table for high kurtosis projections.", "sql.query_workspace", ("query",)),
    EvalTask("sql_05", "sql", "List workspace SQL table schemas using describe_tables.", "sql.describe_tables", ()),
    EvalTask("sql_06", "sql", "SELECT recipe_id, corrupted_json FROM failure_taxonomy", "sql.query_workspace", ("query",)),
    EvalTask("sql_07", "sql", "Execute SQL query over ablation_results ordered by values_exact_rate.", "sql.query_workspace", ("query",)),
    EvalTask("sql_08", "sql", "Check which SQL analytics tables exist in the workspace.", "sql.describe_tables", ()),
    # Knowledge Base & Memory domain (7 tasks)
    EvalTask("kb_01", "kb", "Search workspace documentation for SmoothQuant outlier migration.", "kb.search_workspace", ("query",)),
    EvalTask("kb_02", "kb", "Find notes explaining BOS attention sink INT4 KV cache compression.", "kb.search_workspace", ("query",)),
    EvalTask("kb_03", "kb", "Search the knowledge base for 9-stage middleware guardrails.", "kb.search_workspace", ("query",)),
    EvalTask("kb_04", "memory", "Load the quant-diagnosis skill playbook.", "memory.load_skill", ("name",)),
    EvalTask("kb_05", "memory", "Get the current UTC date and time.", "memory.get_current_datetime", ()),
    EvalTask("kb_06", "kb", "Search project files for FastJsonSchemaParser token mask caching.", "kb.search_workspace", ("query",)),
    EvalTask("kb_07", "memory", "Load the repo-ci-triage skill instructions.", "memory.load_skill", ("name",)),
    # Device / Termux domain (7 tasks)
    EvalTask("dev_01", "device", "Check host CPU cores, disk space, and battery telemetry.", "device.system_telemetry", ()),
    EvalTask("dev_02", "device", "Send a notification alert titled Build Green saying All 25 tests passed.", "device.send_notification", ("title", "message")),
    EvalTask("dev_03", "device", "Get system telemetry and battery percentage.", "device.system_telemetry", ()),
    EvalTask("dev_04", "device", "Copy deployment summary text to the device clipboard.", "device.clipboard_set", ("text",)),
    EvalTask("dev_05", "device", "Send phone notification alert that quantization eval finished.", "device.send_notification", ("title", "message")),
    EvalTask("dev_06", "device", "Inspect Mac or Android Termux hardware system telemetry.", "device.system_telemetry", ()),
    EvalTask("dev_07", "device", "Set device clipboard content to v0.3.0-ready.", "device.clipboard_set", ("text",)),
)


@dataclasses.dataclass(slots=True)
class RecipeEvalMetrics:
    recipe_id: str
    mode: str
    weight_bits: float
    avg_act_bits: float
    footprint_mb: float
    total: int
    json_valid: int
    schema_valid: int
    tool_exact: int
    corrupted_json: int
    wrong_tool: int
    invalid_schema: int
    promoted_token_ratio: float
    mean_latency_ms: float
    kv_saved_mb_2k: float

    @property
    def json_valid_rate(self) -> float:
        return round(self.json_valid / max(1, self.total), 4)

    @property
    def schema_valid_rate(self) -> float:
        return round(self.schema_valid / max(1, self.total), 4)

    @property
    def tool_exact_rate(self) -> float:
        return round(self.tool_exact / max(1, self.total), 4)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["json_valid_rate"] = self.json_valid_rate
        d["schema_valid_rate"] = self.schema_valid_rate
        d["tool_exact_rate"] = self.tool_exact_rate
        return d


def _validate_args_against_schema(args: dict[str, Any], schema: dict[str, Any]) -> bool:
    clean = sanitize_json_schema(schema)
    req = clean.get("required") or []
    props = clean.get("properties") or {}
    for k in req:
        if k not in args or args[k] is None:
            return False
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    for k, val in args.items():
        if k in props:
            expected_t = props[k].get("type")
            if expected_t in type_map and not isinstance(val, type_map[expected_t]):
                if expected_t == "integer" and isinstance(val, bool):
                    return False
                return False
    return True


def evaluate_single_recipe(
    *,
    recipe_id: str,
    backend: ModelBackend,
    constrain_schema: bool,
    tasks: Sequence[EvalTask],
    domain_tools: dict[str, list[dict[str, Any]]],
    tool_schemas_by_name: dict[str, dict[str, Any]],
    kv_saved_mb_2k: float,
) -> RecipeEvalMetrics:
    json_valid = 0
    schema_valid = 0
    tool_exact = 0
    corrupted_json = 0
    wrong_tool = 0
    invalid_schema = 0
    promo_ratios: list[float] = []
    latencies_ms: list[float] = []
    act_bits_samples: list[float] = []
    footprint_mb = 315.1
    weight_bits = 4.0

    for task in tasks:
        candidates = domain_tools[task.domain]
        msgs = [ChatMessage(role="user", content=task.prompt)]
        t0 = time.perf_counter()
        out = backend.generate(msgs, tools=candidates, constrain_schema=constrain_schema)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        promo_ratios.append(out.telemetry.promoted_token_ratio)
        act_bits_samples.append(out.telemetry.effective_act_bits)
        footprint_mb = out.telemetry.weight_footprint_mb
        weight_bits = out.telemetry.weight_bits

        invocations = out.message.tool_calls or parse_tool_calls_from_text(out.message.content)
        if not invocations:
            corrupted_json += 1
            continue

        json_valid += 1
        inv = invocations[0]
        target_schema = tool_schemas_by_name.get(inv.name)
        if target_schema is None:
            invalid_schema += 1
            continue

        if not _validate_args_against_schema(inv.arguments, target_schema):
            invalid_schema += 1
            continue

        schema_valid += 1
        if inv.name == task.expected_tool and all(
            req_k in inv.arguments and str(inv.arguments[req_k]).strip() != ""
            for req_k in task.required_args
        ):
            tool_exact += 1
        else:
            wrong_tool += 1

    avg_act = round(sum(act_bits_samples) / max(1, len(act_bits_samples)), 2)
    avg_promo = round(sum(promo_ratios) / max(1, len(promo_ratios)), 4)
    avg_lat = round(sum(latencies_ms) / max(1, len(latencies_ms)), 2)

    return RecipeEvalMetrics(
        recipe_id=recipe_id,
        mode="constrained" if constrain_schema else "unconstrained",
        weight_bits=weight_bits,
        avg_act_bits=avg_act,
        footprint_mb=footprint_mb,
        total=len(tasks),
        json_valid=json_valid,
        schema_valid=schema_valid,
        tool_exact=tool_exact,
        corrupted_json=corrupted_json,
        wrong_tool=wrong_tool,
        invalid_schema=invalid_schema,
        promoted_token_ratio=avg_promo,
        mean_latency_ms=avg_lat,
        kv_saved_mb_2k=round(kv_saved_mb_2k, 3),
    )


def _select_stratified_tasks(suite: Sequence[EvalTask], max_tasks: int | None) -> list[EvalTask]:
    if not max_tasks or max_tasks >= len(suite):
        return list(suite)
    buckets: dict[str, list[EvalTask]] = {}
    for item in suite:
        buckets.setdefault(item.domain, []).append(item)
    selected: list[EvalTask] = []
    round_idx = 0
    while len(selected) < max_tasks:
        added = False
        for dom_items in buckets.values():
            if round_idx < len(dom_items) and len(selected) < max_tasks:
                selected.append(dom_items[round_idx])
                added = True
        if not added:
            break
        round_idx += 1
    return selected


def run_eval_suite(
    *,
    backend_mode: str = "replay",
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    base_url: str = "http://127.0.0.1:11434/v1",
    workspace_dir: Path | None = None,
    max_tasks: int | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Execute the multi-domain tool-calling & quantization ablation suite and optionally save JSON artifacts."""
    ws = (workspace_dir or Path.cwd()).resolve()
    reg = ToolRegistry()
    register_all_builtin_tools(reg, workspace_dir=ws)

    tasks = _select_stratified_tasks(EVAL_SUITE, max_tasks)
    all_specs = reg.list_tools()
    tool_schemas_by_name = {s.name: s.parameters for s in all_specs}

    domain_tools: dict[str, list[dict[str, Any]]] = {}
    for dom in ("devops", "sql", "kb", "memory", "device"):
        dom_specs = reg.list_tools(domain=dom)
        extra_specs = [s for s in all_specs if s.domain != dom][:2]
        domain_tools[dom] = [s.to_openai_tool() for s in (*dom_specs, *extra_specs)]

    results: list[RecipeEvalMetrics] = []
    layer_rows: list[dict[str, Any]] = []

    if backend_mode == "hf-quant":
        from pocket_genie.quant.hf_backend import QuantizedPyTorchBackend, SharedHFQuantRuntime

        runtime = SharedHFQuantRuntime.load(model_id=model_id)
        fp16_mb, comp_mb, _, _, _ = runtime.kv_cache.estimate_memory_mb(2048)
        kv_saved = fp16_mb - comp_mb

        for sens in runtime.sensitivities[:10]:
            parts = sens.layer_name.split(".")
            layer_idx = next((int(p) for p in parts if p.isdigit()), 0)
            proj = ".".join(parts[-2:]) if len(parts) >= 2 else sens.layer_name
            layer_rows.append(
                {
                    "layer_idx": layer_idx,
                    "projection": proj,
                    "kurtosis_all": sens.kurtosis,
                    "mse_w4a8": sens.mse_w4a8,
                    "sensitivity_score": round(sens.sensitivity_score, 3),
                }
            )

        recipes: list[tuple[str, ModelBackend, bool]] = [
            (
                "W4A4_unconstrained",
                QuantizedPyTorchBackend(
                    runtime, tier="t1", max_new_tokens=48, override_act_bits=4, enable_token_promotion=False
                ),
                False,
            ),
            (
                "P_fast_W4A8",
                QuantizedPyTorchBackend(
                    runtime, tier="t1", max_new_tokens=48, enable_token_promotion=False
                ),
                False,
            ),
            (
                "P_mixed_W4A8_16_FSM",
                QuantizedPyTorchBackend(runtime, tier="t2", max_new_tokens=48),
                True,
            ),
            (
                "P_accurate_W4A16_FSM",
                QuantizedPyTorchBackend(runtime, tier="t3", max_new_tokens=48),
                True,
            ),
        ]
        for r_id, b_inst, constrain in recipes:
            results.append(
                evaluate_single_recipe(
                    recipe_id=r_id,
                    backend=b_inst,
                    constrain_schema=constrain,
                    tasks=tasks,
                    domain_tools=domain_tools,
                    tool_schemas_by_name=tool_schemas_by_name,
                    kv_saved_mb_2k=kv_saved,
                )
            )
    elif backend_mode == "local-llm":
        from pocket_genie.quant.kv_cache import BOSSinkINT4KVCache

        fp16_mb, comp_mb, _, _, _ = BOSSinkINT4KVCache().estimate_memory_mb(2048)
        kv_saved = fp16_mb - comp_mb
        for r_id, tier, constrain in (
            ("local_t1_unconstrained", "t1", False),
            ("local_t2_constrained", "t2", True),
        ):
            b_llm = OpenAICompatibleBackend(base_url=base_url, model=model_id, tier=tier)  # type: ignore[arg-type]
            results.append(
                evaluate_single_recipe(
                    recipe_id=r_id,
                    backend=b_llm,
                    constrain_schema=constrain,
                    tasks=tasks,
                    domain_tools=domain_tools,
                    tool_schemas_by_name=tool_schemas_by_name,
                    kv_saved_mb_2k=kv_saved,
                )
            )
    else:
        from pocket_genie.quant.kv_cache import BOSSinkINT4KVCache

        fp16_mb, comp_mb, _, _, _ = BOSSinkINT4KVCache().estimate_memory_mb(2048)
        kv_saved = fp16_mb - comp_mb
        for r_id, tier, constrain in (
            ("replay_P_fast", "t1", False),
            ("replay_P_mixed_FSM", "t2", True),
        ):
            results.append(
                evaluate_single_recipe(
                    recipe_id=r_id,
                    backend=DeterministicReplayBackend(tier=tier),  # type: ignore[arg-type]
                    constrain_schema=constrain,
                    tasks=tasks,
                    domain_tools=domain_tools,
                    tool_schemas_by_name=tool_schemas_by_name,
                    kv_saved_mb_2k=kv_saved,
                )
            )

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backend_mode": backend_mode,
        "model_id": model_id,
        "num_tasks": len(tasks),
        "recipes": [r.to_dict() for r in results],
        "top_sensitive_layers": layer_rows,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
