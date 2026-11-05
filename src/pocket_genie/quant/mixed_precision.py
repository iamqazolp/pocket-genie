from __future__ import annotations

import dataclasses
from typing import Sequence

from pocket_genie.shared import PrecisionProfileName, PrecisionTier

DEFAULT_OUTLIER_DOWN_PROJ_LAYERS: tuple[str, ...] = (
    "model.layers.0.mlp.down_proj",
    "model.layers.18.mlp.down_proj",
    "model.layers.21.mlp.down_proj",
    "model.layers.22.mlp.down_proj",
    "model.layers.23.mlp.down_proj",
)


@dataclasses.dataclass(slots=True)
class LayerSensitivity:
    layer_name: str
    kurtosis: float
    mse_w4a8: float = 0.0
    mac_weight: float = 1.0

    @property
    def sensitivity_score(self) -> float:
        return self.kurtosis * (1.0 + 10.0 * self.mse_w4a8)


@dataclasses.dataclass(slots=True)
class PrecisionProfile:
    name: PrecisionProfileName
    tier: PrecisionTier
    weight_bits: int
    default_act_bits: int
    layer_act_bits: dict[str, int]
    promoted_layers: list[str]
    effective_act_bits: float
    weight_footprint_mb: float = 315.1
    fallback_overhead_mb: float = 0.0


def build_sensitivity_profiles(
    sensitivities: Sequence[LayerSensitivity] | None = None,
    kurtosis_threshold: float = 100.0,
    max_mixed_act_bits: float = 8.60,
    total_linear_layers: int = 168,
) -> dict[PrecisionTier, PrecisionProfile]:
    """Construct P_fast (t1), P_mixed (t2), and P_accurate (t3) activation profiles sharing 4-bit weights."""
    if sensitivities:
        ranked = sorted(sensitivities, key=lambda s: s.sensitivity_score, reverse=True)
        total_n = max(len(sensitivities), total_linear_layers)
        promoted: list[str] = []
        for item in ranked:
            if item.kurtosis < kurtosis_threshold and promoted:
                break
            candidate_count = len(promoted) + 1
            candidate_avg = (8.0 * (total_n - candidate_count) + 16.0 * candidate_count) / float(total_n)
            if candidate_avg <= max_mixed_act_bits or not promoted:
                promoted.append(item.layer_name)
    else:
        promoted = list(DEFAULT_OUTLIER_DOWN_PROJ_LAYERS)
        total_n = total_linear_layers

    mixed_map = {name: 16 for name in promoted}
    mixed_avg = round((8.0 * (total_n - len(promoted)) + 16.0 * len(promoted)) / float(total_n), 2)

    p_fast = PrecisionProfile(
        name="P_fast",
        tier="t1",
        weight_bits=4,
        default_act_bits=8,
        layer_act_bits={},
        promoted_layers=[],
        effective_act_bits=8.0,
    )
    p_mixed = PrecisionProfile(
        name="P_mixed",
        tier="t2",
        weight_bits=4,
        default_act_bits=8,
        layer_act_bits=mixed_map,
        promoted_layers=promoted,
        effective_act_bits=mixed_avg,
    )
    p_accurate = PrecisionProfile(
        name="P_accurate",
        tier="t3",
        weight_bits=4,
        default_act_bits=16,
        layer_act_bits={},
        promoted_layers=["*all_linear_layers*"],
        effective_act_bits=16.0,
    )
    return {"t1": p_fast, "t2": p_mixed, "t3": p_accurate}
