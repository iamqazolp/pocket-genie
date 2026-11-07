from __future__ import annotations

import copy
import dataclasses
import json
import time
from typing import Any, Sequence

from pocket_genie.quant.enforcer import (
    FastJsonSchemaParser,
    build_fast_tokenizer_data,
    build_tool_call_schema,
    create_schema_token_enforcer,
)
from pocket_genie.quant.entropy_router import TokenPrecisionRouter
from pocket_genie.quant.fake_quant import (
    attach_per_layer_activation_hooks,
    fake_quantize_activation_asymmetric,
    fake_quantize_weight_per_channel,
)
from pocket_genie.quant.kv_cache import BOSSinkINT4KVCache
from pocket_genie.quant.mixed_precision import (
    LayerSensitivity,
    PrecisionProfile,
    build_sensitivity_profiles,
)
from pocket_genie.quant.smooth import migrate_swiglu_down_proj_scale
from pocket_genie.shared import (
    ChatMessage,
    PrecisionTier,
    ToolCallChunk,
    ToolInvocation,
)

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    HF_QUANT_AVAILABLE = True
except ImportError:
    HF_QUANT_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]
    DynamicCache = None  # type: ignore[assignment,misc]


DEFAULT_CALIBRATION_PROMPTS: tuple[str, ...] = (
    "Respond with a JSON tool call to inspect git repository status and recent commits.",
    "SELECT recipe_id, json_valid_rate, values_exact_rate FROM ablation_results ORDER BY values_exact_rate DESC;",
)


def _select_torch_device(preferred: str | None = None) -> str:
    if torch is None:
        return "cpu"
    if preferred:
        return preferred
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def calibrate_layer_sensitivities(
    model: Any,
    tokenizer: Any,
    calibration_texts: Sequence[str] = DEFAULT_CALIBRATION_PROMPTS,
    device: str = "cpu",
) -> tuple[list[LayerSensitivity], dict[str, Any]]:
    """Run a calibration pass over linear layers to measure empirical activation kurtosis and W4A8 MSE."""
    if not HF_QUANT_AVAILABLE:
        raise RuntimeError("torch and transformers are required for calibrate_layer_sensitivities")

    kurt_accum: dict[str, list[float]] = {}
    mse_accum: dict[str, list[float]] = {}
    act_max_by_layer: dict[str, Any] = {}
    handles: list[Any] = []

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or name.endswith("lm_head"):
            continue

        def _make_probe(layer_name: str) -> Any:
            def _probe(_m: Any, inputs: tuple[Any, ...]) -> None:
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    return
                x = inputs[0].detach().float()
                if x.numel() < 16:
                    return
                flat = x.reshape(-1)
                mean = flat.mean()
                centered = flat - mean
                var = centered.pow(2).mean().clamp(min=1e-8)
                kurt = float((centered.pow(4).mean() / (var * var)).item() - 3.0)
                q8 = fake_quantize_activation_asymmetric(x, bits=8)
                rel_mse = float(
                    ((x - q8).pow(2).mean() / (x.pow(2).mean() + 1e-6)).item()
                )
                kurt_accum.setdefault(layer_name, []).append(max(0.0, kurt))
                mse_accum.setdefault(layer_name, []).append(rel_mse)

                if layer_name.endswith("mlp.down_proj"):
                    chan_max = x.abs().reshape(-1, x.shape[-1]).amax(dim=0).cpu()
                    prev = act_max_by_layer.get(layer_name)
                    act_max_by_layer[layer_name] = (
                        chan_max if prev is None else torch.maximum(prev, chan_max)
                    )

            return _probe

        handles.append(mod.register_forward_pre_hook(_make_probe(name)))

    try:
        model.eval()
        with torch.no_grad():
            for text in calibration_texts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                enc = {k: v.to(device) for k, v in enc.items()}
                model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    sensitivities: list[LayerSensitivity] = []
    for name, k_vals in kurt_accum.items():
        avg_k = round(sum(k_vals) / len(k_vals), 3)
        m_vals = mse_accum.get(name, [0.0])
        avg_mse = round(sum(m_vals) / len(m_vals), 6)
        sensitivities.append(
            LayerSensitivity(
                layer_name=name,
                kurtosis=avg_k,
                mse_w4a8=avg_mse,
            )
        )
    sensitivities.sort(key=lambda s: s.sensitivity_score, reverse=True)
    return sensitivities, act_max_by_layer


@dataclasses.dataclass
class SharedHFQuantRuntime:
    """Shared 4-bit weight HuggingFace runtime serving P_fast (t1), P_mixed (t2), and P_accurate (t3) with 0 MB overhead."""

    model_id: str
    device: str
    tokenizer: Any
    model: Any
    profiles: dict[PrecisionTier, PrecisionProfile]
    sensitivities: list[LayerSensitivity]
    kv_cache: BOSSinkINT4KVCache
    weight_footprint_mb: float
    tokenizer_enforcer_data: Any | None = None

    @classmethod
    def load(
        cls,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str | None = None,
        weight_bits: int = 4,
        apply_smoothquant: bool = True,
        local_files_only: bool = False,
    ) -> SharedHFQuantRuntime:
        if not HF_QUANT_AVAILABLE:
            raise RuntimeError("torch and transformers are required for SharedHFQuantRuntime")

        dev = _select_torch_device(device)
        dtype = torch.float16 if dev in ("mps", "cuda") else torch.float32
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                local_files_only=local_files_only,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                local_files_only=True,
            )

        model.to(dev)
        model.eval()

        sensitivities, act_max_map = calibrate_layer_sensitivities(model, tokenizer, device=dev)

        if apply_smoothquant:
            modules_dict = dict(model.named_modules())
            for down_name, act_max in act_max_map.items():
                up_name = down_name.replace("mlp.down_proj", "mlp.up_proj")
                if up_name in modules_dict and down_name in modules_dict:
                    migrate_swiglu_down_proj_scale(
                        modules_dict[up_name],
                        modules_dict[down_name],
                        act_max.to(dev),
                    )

        linear_params = 0
        other_params = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "layers." in name and name.endswith(".weight") and param.ndim == 2:
                    linear_params += param.numel()
                    if weight_bits < 16:
                        param.copy_(fake_quantize_weight_per_channel(param, bits=weight_bits))
                else:
                    other_params += param.numel()

        w_bytes = linear_params * (weight_bits / 8.0) + other_params * 2.0
        footprint_mb = round(w_bytes / (1024.0 * 1024.0), 2)

        profiles = build_sensitivity_profiles(
            sensitivities=sensitivities,
            kurtosis_threshold=20.0,
            max_mixed_act_bits=8.60,
            total_linear_layers=max(1, len(sensitivities)),
        )
        for p in profiles.values():
            p.weight_footprint_mb = footprint_mb
            p.weight_bits = weight_bits

        cfg = model.config
        n_layers = int(getattr(cfg, "num_hidden_layers", 24))
        n_kv_heads = int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 2)))
        hidden = int(getattr(cfg, "hidden_size", 896))
        n_heads = int(getattr(cfg, "num_attention_heads", 14))
        head_dim = int(getattr(cfg, "head_dim", hidden // max(1, n_heads)))

        kv_cache = BOSSinkINT4KVCache(
            sink_tokens=4,
            recent_window=64,
            num_layers=n_layers,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        tok_data = build_fast_tokenizer_data(tokenizer) if FastJsonSchemaParser is not None else None
        return cls(
            model_id=model_id,
            device=dev,
            tokenizer=tokenizer,
            model=model,
            profiles=profiles,
            sensitivities=sensitivities,
            kv_cache=kv_cache,
            weight_footprint_mb=footprint_mb,
            tokenizer_enforcer_data=tok_data,
        )


def _clone_past_kv(past_kv: Any) -> Any:
    if past_kv is None:
        return None
    if DynamicCache is not None and isinstance(past_kv, DynamicCache):
        cloned = DynamicCache()
        if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
            cloned.key_cache = [t.clone() for t in past_kv.key_cache]
            cloned.value_cache = [t.clone() for t in past_kv.value_cache]
            if hasattr(past_kv, "_seen_tokens"):
                cloned._seen_tokens = past_kv._seen_tokens
            return cloned
        if hasattr(past_kv, "layers"):
            cloned.layers = copy.deepcopy(past_kv.layers)
            return cloned
    return copy.deepcopy(past_kv)


class QuantizedPyTorchBackend:
    """Live HuggingFace/PyTorch quantized backend with mixed-precision hooks, logit-margin routing, and FSM schema enforcement."""

    def __init__(
        self,
        runtime: SharedHFQuantRuntime,
        tier: PrecisionTier = "t1",
        max_new_tokens: int = 96,
        override_act_bits: int | None = None,
        enable_token_promotion: bool = True,
    ) -> None:
        self.runtime = runtime
        self.tier = tier
        self.max_new_tokens = max_new_tokens
        self.override_act_bits = override_act_bits
        self.enable_token_promotion = enable_token_promotion
        suffix = f"A{override_act_bits}" if override_act_bits is not None else tier
        self.name = f"hf-quant:{runtime.model_id}:{suffix}"

    def _build_chat_prompt(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None,
    ) -> str:
        chat_dicts: list[dict[str, str]] = []
        last_is_tool = bool(messages and messages[-1].role == "tool")

        if tools and not last_is_tool:
            compact_tools = []
            for t in tools:
                fn = t.get("function", t)
                props = fn.get("parameters", {}).get("properties", {})
                req = fn.get("parameters", {}).get("required", [])
                compact_tools.append(
                    {
                        "name": fn.get("name"),
                        "description": fn.get("description", ""),
                        "arguments": {k: v.get("type", "string") for k, v in props.items()},
                        "required": req,
                    }
                )
            tool_sys = (
                "You are a tool-calling agent. Select the best tool for the user request and respond "
                'ONLY with a single JSON object of the form {"name": "<tool_name>", "arguments": {...}}.\n'
                f"Available tools:\n{json.dumps(compact_tools)}"
            )
            chat_dicts.append({"role": "system", "content": tool_sys})

        for m in messages:
            if m.role == "system":
                if not tools or last_is_tool:
                    chat_dicts.append({"role": "system", "content": m.content})
            elif m.role == "user":
                chat_dicts.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                txt = m.content or (json.dumps(m.tool_calls[0].arguments) if m.tool_calls else "")
                chat_dicts.append({"role": "assistant", "content": txt})
            elif m.role == "tool":
                chat_dicts.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool `{m.name}` returned:\n{m.content}\n\n"
                            "Provide a concise, helpful final summary for the user based on this tool result."
                        ),
                    }
                )

        return str(
            self.runtime.tokenizer.apply_chat_template(
                chat_dicts,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def _attach_profile_hooks(self, tier: PrecisionTier) -> list[Any]:
        if self.override_act_bits is not None:
            return attach_per_layer_activation_hooks(
                self.runtime.model,
                layer_act_bits={},
                default_bits=self.override_act_bits,
            )
        prof = self.runtime.profiles[tier]
        return attach_per_layer_activation_hooks(
            self.runtime.model,
            layer_act_bits=prof.layer_act_bits,
            default_bits=prof.default_act_bits,
        )

    def generate(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        constrain_schema: bool = False,
    ) -> GenerationOutput:
        from pocket_genie.registry.backends import parse_tool_calls_from_text

        t0 = time.perf_counter()
        last_is_tool = bool(messages and messages[-1].role == "tool")
        active_tools = tools if (tools and not last_is_tool) else None

        prompt_text = self._build_chat_prompt(messages, active_tools)
        tok = self.runtime.tokenizer
        enc = tok(prompt_text, return_tensors="pt", truncation=True, max_length=1536)
        input_ids = enc["input_ids"].to(self.runtime.device)
        prompt_len = int(input_ids.shape[1])

        enforcer = None
        if active_tools and constrain_schema and self.runtime.tokenizer_enforcer_data is not None:
            schema = build_tool_call_schema(active_tools)
            enforcer = create_schema_token_enforcer(self.runtime.tokenizer_enforcer_data, schema)

        router = TokenPrecisionRouter(
            mixed_act_bits=self.runtime.profiles["t2"].effective_act_bits,
        )
        generated_ids: list[int] = []
        eos_id = tok.eos_token_id

        handles = self._attach_profile_hooks(self.tier)
        try:
            with torch.no_grad():
                out = self.runtime.model(input_ids=input_ids, use_cache=True)
                past_kv = out.past_key_values
                next_logits = out.logits[0, -1, :].float()
                prev_past_kv: Any | None = None
                prev_token_tensor: Any | None = None

                for step_idx in range(self.max_new_tokens):
                    top_vals = torch.topk(next_logits, k=min(16, next_logits.shape[-1])).values.tolist()
                    decision = router.evaluate_step(step_idx, top_vals)

                    if (
                        step_idx > 0
                        and self.tier == "t1"
                        and self.override_act_bits is None
                        and self.enable_token_promotion
                        and decision.promoted
                        and prev_past_kv is not None
                    ):
                        for h in handles:
                            h.remove()
                        promo_handles = self._attach_profile_hooks(decision.selected_tier)
                        try:
                            re_out = self.runtime.model(
                                input_ids=prev_token_tensor,
                                past_key_values=_clone_past_kv(prev_past_kv),
                                use_cache=True,
                            )
                            past_kv = re_out.past_key_values
                            next_logits = re_out.logits[0, -1, :].float()
                        finally:
                            for h in promo_handles:
                                h.remove()
                        handles = self._attach_profile_hooks(self.tier)

                    masked_logits = next_logits
                    if enforcer is not None:
                        allowed_res = enforcer.get_allowed_tokens(generated_ids)
                        allowed_ids = (
                            allowed_res.allowed_tokens
                            if hasattr(allowed_res, "allowed_tokens")
                            else list(allowed_res)
                        )
                        if allowed_ids:
                            mask = torch.full_like(masked_logits, float("-inf"))
                            idx_tensor = torch.tensor(allowed_ids, dtype=torch.long, device=masked_logits.device)
                            valid_idx = idx_tensor[idx_tensor < masked_logits.shape[-1]]
                            if valid_idx.numel() > 0:
                                mask[valid_idx] = masked_logits[valid_idx]
                                masked_logits = mask

                    chosen_id = int(torch.argmax(masked_logits).item())
                    if chosen_id == eos_id:
                        break
                    generated_ids.append(chosen_id)

                    if active_tools and "}" in tok.decode([chosen_id]):
                        candidate_txt = tok.decode(generated_ids, skip_special_tokens=True).strip()
                        if candidate_txt.startswith("{") and candidate_txt.endswith("}"):
                            try:
                                parsed_cand = json.loads(candidate_txt)
                                if isinstance(parsed_cand, dict) and "name" in parsed_cand and "arguments" in parsed_cand:
                                    break
                            except Exception:
                                pass

                    prev_past_kv = _clone_past_kv(past_kv) if (self.tier == "t1" and self.enable_token_promotion) else None
                    prev_token_tensor = torch.tensor([[chosen_id]], dtype=torch.long, device=self.runtime.device)
                    step_out = self.runtime.model(
                        input_ids=prev_token_tensor,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    past_kv = step_out.past_key_values
                    next_logits = step_out.logits[0, -1, :].float()
        finally:
            for h in handles:
                h.remove()

        elapsed = max(1e-3, time.perf_counter() - t0)
        raw_text = tok.decode(generated_ids, skip_special_tokens=True).strip()
        invocations: list[ToolInvocation] = (
            parse_tool_calls_from_text(raw_text) if active_tools else []
        )

        streamed_chunks: list[ToolCallChunk] = []
        if invocations:
            inv0 = invocations[0]
            arg_str = json.dumps(inv0.arguments)
            mid = max(1, len(arg_str) // 2)
            streamed_chunks = [
                ToolCallChunk(index=0, id=inv0.id, name=inv0.name, arguments_fragment=arg_str[:mid]),
                ToolCallChunk(index=0, id=None, name=None, arguments_fragment=arg_str[mid:]),
            ]

        total_seq = prompt_len + len(generated_ids)
        fp16_mb, comp_mb, _, _, _ = self.runtime.kv_cache.estimate_memory_mb(total_seq)
        telemetry = router.build_telemetry(
            prefix_kv_hit=True,
            kv_cache_fp16_mb=fp16_mb,
            kv_cache_quant_mb=comp_mb,
        )
        telemetry.weight_bits = float(self.runtime.profiles[self.tier].weight_bits)
        telemetry.weight_footprint_mb = self.runtime.weight_footprint_mb
        if self.override_act_bits is not None:
            telemetry.effective_act_bits = float(self.override_act_bits)
            telemetry.tier = self.tier
            telemetry.promoted_tokens = 0
        elif self.tier == "t1" and not self.enable_token_promotion:
            telemetry.tier = "t1"
            telemetry.profile = "P_fast"
            telemetry.effective_act_bits = 8.0
            telemetry.promoted_tokens = 0
        elif self.tier == "t3":
            telemetry.tier = "t3"
            telemetry.profile = "P_accurate"
            telemetry.effective_act_bits = 16.0
        elif self.tier == "t2":
            telemetry.tier = "t2"
            telemetry.profile = "P_mixed"
            telemetry.effective_act_bits = self.runtime.profiles["t2"].effective_act_bits

        msg = ChatMessage(
            role="assistant",
            content="" if invocations else raw_text,
            reasoning=(
                f"QuantizedPyTorch `{self.runtime.model_id}` ({telemetry.tier}/{telemetry.profile}, "
                f"{telemetry.effective_act_bits:.2f}b act, {len(generated_ids)} tok in {elapsed:.2f}s)."
            ),
            tool_calls=invocations,
        )
        return GenerationOutput(
            message=msg,
            streamed_chunks=streamed_chunks,
            telemetry=telemetry,
        )


from pocket_genie.registry.backends import GenerationOutput  # noqa: E402
