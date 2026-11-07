from __future__ import annotations

from pocket_genie.quant.enforcer import (
    COMPACT_ALPHABET,
    FastJsonSchemaParser,
    FastSequenceParser,
    build_fast_tokenizer_data,
    build_tool_call_schema,
    create_schema_token_enforcer,
)
from pocket_genie.quant.entropy_router import (
    TokenPrecisionRouter,
    TokenStepDecision,
    compute_margin_and_entropy,
)
from pocket_genie.quant.fake_quant import (
    attach_per_layer_activation_hooks,
    fake_quantize_activation_asymmetric,
    fake_quantize_weight_per_channel,
)
from pocket_genie.quant.hf_backend import (
    HF_QUANT_AVAILABLE,
    QuantizedPyTorchBackend,
    SharedHFQuantRuntime,
    calibrate_layer_sensitivities,
)
from pocket_genie.quant.kv_cache import (
    BOSSinkINT4KVCache,
    KVCompressionStats,
    quantize_k_per_channel_int4,
    quantize_v_per_token_int4,
)
from pocket_genie.quant.mixed_precision import (
    DEFAULT_OUTLIER_DOWN_PROJ_LAYERS,
    LayerSensitivity,
    PrecisionProfile,
    build_sensitivity_profiles,
)
from pocket_genie.quant.smooth import (
    ALPHA_GRID,
    compute_smooth_scale,
    migrate_swiglu_down_proj_scale,
    search_optimal_alpha,
)

__all__ = [
    "ALPHA_GRID",
    "BOSSinkINT4KVCache",
    "COMPACT_ALPHABET",
    "DEFAULT_OUTLIER_DOWN_PROJ_LAYERS",
    "FastJsonSchemaParser",
    "FastSequenceParser",
    "HF_QUANT_AVAILABLE",
    "KVCompressionStats",
    "LayerSensitivity",
    "PrecisionProfile",
    "QuantizedPyTorchBackend",
    "SharedHFQuantRuntime",
    "TokenPrecisionRouter",
    "TokenStepDecision",
    "attach_per_layer_activation_hooks",
    "build_fast_tokenizer_data",
    "build_sensitivity_profiles",
    "build_tool_call_schema",
    "calibrate_layer_sensitivities",
    "compute_margin_and_entropy",
    "compute_smooth_scale",
    "create_schema_token_enforcer",
    "fake_quantize_activation_asymmetric",
    "fake_quantize_weight_per_channel",
    "migrate_swiglu_down_proj_scale",
    "quantize_k_per_channel_int4",
    "quantize_v_per_token_int4",
    "search_optimal_alpha",
]
