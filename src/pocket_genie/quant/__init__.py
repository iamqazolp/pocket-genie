from __future__ import annotations

# Phase 1 quantization primitives: SmoothQuant, fake-quant hooks, and FSM schema parser.
from pocket_genie.quant.enforcer import COMPACT_ALPHABET, FastJsonSchemaParser, FastSequenceParser, build_tool_call_schema
from pocket_genie.quant.fake_quant import attach_per_layer_activation_hooks, fake_quantize_activation_asymmetric, fake_quantize_weight_per_channel
from pocket_genie.quant.smooth import ALPHA_GRID, compute_smooth_scale, migrate_swiglu_down_proj_scale, search_optimal_alpha

QUANT_SUBSYSTEM_VERSION = '0.1.0-alpha'
