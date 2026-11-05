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
