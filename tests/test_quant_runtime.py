from __future__ import annotations

from pocket_genie.quant import (
    BOSSinkINT4KVCache,
    LayerSensitivity,
    TokenPrecisionRouter,
    build_sensitivity_profiles,
    build_tool_call_schema,
)


def test_sensitivity_mixed_precision_profiles():
    sens = [
        LayerSensitivity("model.layers.0.mlp.down_proj", kurtosis=542.6, mse_w4a8=0.18),
        LayerSensitivity("model.layers.23.mlp.down_proj", kurtosis=312.4, mse_w4a8=0.11),
        LayerSensitivity("model.layers.5.self_attn.q_proj", kurtosis=4.2, mse_w4a8=0.002),
    ]
    profiles = build_sensitivity_profiles(sens, kurtosis_threshold=100.0, total_linear_layers=168)
    assert profiles["t1"].name == "P_fast"
    assert profiles["t1"].effective_act_bits == 8.0
    assert profiles["t2"].name == "P_mixed"
    assert "model.layers.0.mlp.down_proj" in profiles["t2"].promoted_layers
    assert "model.layers.5.self_attn.q_proj" not in profiles["t2"].promoted_layers
    assert 8.0 < profiles["t2"].effective_act_bits < 8.6
    assert profiles["t3"].effective_act_bits == 16.0


def test_token_precision_router_logit_margin_switching():
    router = TokenPrecisionRouter(margin_threshold_mixed=1.8, margin_threshold_accurate=0.65)
    d0 = router.evaluate_step(0, [8.5, 1.2, 0.1, -1.0])
    d1 = router.evaluate_step(1, [3.1, 2.0, 1.8, 0.5])
    d2 = router.evaluate_step(2, [2.2, 1.9, 1.85, 1.7])

    assert d0.selected_profile == "P_fast"
    assert d0.promoted is False
    assert d1.selected_profile == "P_mixed"
    assert d1.promoted is True
    assert d2.selected_profile == "P_accurate"
    assert d2.promoted is True

    snap = router.build_telemetry(kv_cache_fp16_mb=16.0, kv_cache_quant_mb=5.2)
    assert snap.promoted_tokens == 2
    assert snap.total_tokens == 3
    assert snap.tier == "t3"
    assert snap.kv_memory_saved_mb == 10.8


def test_bos_sink_int4_kv_cache_and_prefix_pinning():
    kv = BOSSinkINT4KVCache(sink_tokens=2, recent_window=4)
    assert kv.pin_prefix("Static system prompt", '{"tools": []}') is False
    assert kv.pin_prefix("Static system prompt", '{"tools": []}') is True

    keys = [[float(t * 2 + c * 10) for c in range(8)] for t in range(16)]
    vals = [[float(t + c * 0.5) for c in range(8)] for t in range(16)]
    k_out, v_out, stats = kv.compress_kv(keys, vals)

    assert stats.sink_tokens == 2
    assert stats.recent_window_tokens == 4
    assert stats.quantized_middle_tokens == 10
    assert k_out[:2] == keys[:2]
    assert k_out[-4:] == keys[-4:]
    assert stats.saved_mb > 0.0
    assert stats.key_mse < 1.0


def test_build_tool_call_schema_anyof():
    schema = build_tool_call_schema(
        [
            {"function": {"name": "devops.git_status", "parameters": {"type": "object", "properties": {}}}},
            {"function": {"name": "sql.query", "parameters": {"type": "object", "properties": {"query": {"type": "str"}}}}},
        ]
    )
    assert "anyOf" in schema
    assert len(schema["anyOf"]) == 2


def test_fast_json_schema_parser_steps_through_freetext_string():
    import pytest

    pytest.importorskip("lmformatenforcer")
    from pocket_genie.quant import FastJsonSchemaParser

    schema = build_tool_call_schema(
        [
            {
                "function": {
                    "name": "kb.search_workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            }
        ]
    )
    parser = FastJsonSchemaParser(schema)
    sample = '{"name": "kb.search_workspace", "arguments": {"query": "SmoothQuant"}}'
    for ch in sample:
        _ = parser.shortcut_key()
        _ = parser.cache_key()
        parser = parser.add_character(ch)
    assert parser.can_end() is True


def test_pytorch_smoothquant_and_selective_layer_hooks():
    import pytest

    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from pocket_genie.quant import (
        attach_per_layer_activation_hooks,
        migrate_swiglu_down_proj_scale,
        quantize_k_per_channel_int4,
        quantize_v_per_token_int4,
    )

    class ToyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up_proj = nn.Linear(8, 8, bias=False)
            self.down_proj = nn.Linear(8, 8, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(self.up_proj(x))

    mlp = ToyMLP()
    x = torch.randn(4, 8)
    x[:, 0] *= 25.0
    with torch.no_grad():
        baseline = mlp(x).clone()
        act_max = mlp.up_proj(x).abs().amax(dim=0)

    alpha = migrate_swiglu_down_proj_scale(mlp.up_proj, mlp.down_proj, act_max, alpha=None)
    assert 0.30 <= alpha <= 0.85
    with torch.no_grad():
        after_smooth = mlp(x)
    assert torch.allclose(baseline, after_smooth, atol=1e-4)

    handles = attach_per_layer_activation_hooks(
        mlp,
        layer_act_bits={"down_proj": 16},
        default_bits=8,
    )
    assert len(handles) == 2
    out_quant = mlp(x)
    assert out_quant.shape == baseline.shape
    for h in handles:
        h.remove()

    k_tensor = torch.randn(12, 8)
    k_deq, k_mse = quantize_k_per_channel_int4(k_tensor)
    v_deq, v_mse = quantize_v_per_token_int4(k_tensor)
    assert len(k_deq) == 12
    assert len(v_deq) == 12
    assert k_mse >= 0.0 and v_mse >= 0.0


def test_quantized_pytorch_backend_kurtosis_calibration_and_generation():
    import pytest

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from pocket_genie.quant import (
        BOSSinkINT4KVCache,
        QuantizedPyTorchBackend,
        SharedHFQuantRuntime,
        build_sensitivity_profiles,
        calibrate_layer_sensitivities,
    )
    from pocket_genie.shared import ChatMessage

    cfg = transformers.Qwen2Config(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    tiny_model = transformers.Qwen2ForCausalLM(cfg).eval()

    class TinyDummyTokenizer:
        eos_token_id = 2
        all_special_ids = [0, 1, 2]

        def __len__(self) -> int:
            return 256

        def __call__(self, text: str, return_tensors: str = "pt", **kwargs):
            ids = [min(255, max(3, ord(ch) % 250)) for ch in text[:32]] or [10, 11, 12]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

        def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True):
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
            return "".join(chr(65 + (int(t) % 26)) for t in token_ids)

    tok = TinyDummyTokenizer()
    sens, act_max = calibrate_layer_sensitivities(
        tiny_model,
        tok,
        calibration_texts=("calibrate one", "calibrate two"),
        device="cpu",
    )
    assert len(sens) > 0
    assert len(act_max) == 2

    profiles = build_sensitivity_profiles(sens, kurtosis_threshold=1.0, total_linear_layers=len(sens))
    kv = BOSSinkINT4KVCache(num_layers=2, num_kv_heads=2, head_dim=8)
    rt = SharedHFQuantRuntime(
        model_id="tiny-qwen2-unit",
        device="cpu",
        tokenizer=tok,
        model=tiny_model,
        profiles=profiles,
        sensitivities=sens,
        kv_cache=kv,
        weight_footprint_mb=0.5,
        tokenizer_enforcer_data=None,
    )
    backend = QuantizedPyTorchBackend(rt, tier="t1", max_new_tokens=6, enable_token_promotion=True)
    out = backend.generate([ChatMessage(role="user", content="Hello tiny model")])
    assert out.telemetry.total_tokens > 0
    assert out.telemetry.weight_footprint_mb == 0.5
    assert len(out.message.content) > 0


