from __future__ import annotations

import dataclasses
import math
from typing import Sequence

from pocket_genie.shared import PrecisionProfileName, PrecisionTier, QuantTelemetrySnapshot


@dataclasses.dataclass(slots=True)
class TokenStepDecision:
    step_idx: int
    logit_margin: float
    entropy: float
    selected_profile: PrecisionProfileName
    selected_tier: PrecisionTier
    promoted: bool


def compute_margin_and_entropy(logits: Sequence[float], top_k: int = 16) -> tuple[float, float]:
    """Compute top-2 logit margin Delta z = z_(1) - z_(2) and truncated Shannon entropy H(p)."""
    if not logits:
        return 0.0, 0.0
    ordered = sorted((float(x) for x in logits), reverse=True)[: max(2, top_k)]
    if len(ordered) == 1:
        return abs(ordered[0]), 0.0

    margin = ordered[0] - ordered[1]
    max_z = ordered[0]
    exps = [math.exp(max(-60.0, z - max_z)) for z in ordered]
    denom = sum(exps) or 1.0
    probs = [e / denom for e in exps]
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return round(margin, 5), round(entropy, 5)


class TokenPrecisionRouter:
    """Routes individual decoding token steps between P_fast (W4A8), P_mixed, and P_accurate (W4A16)."""

    def __init__(
        self,
        margin_threshold_mixed: float = 1.8,
        margin_threshold_accurate: float = 0.65,
        entropy_threshold_mixed: float = 1.15,
        mixed_act_bits: float = 8.24,
    ) -> None:
        self.margin_threshold_mixed = margin_threshold_mixed
        self.margin_threshold_accurate = margin_threshold_accurate
        self.entropy_threshold_mixed = entropy_threshold_mixed
        self.mixed_act_bits = mixed_act_bits
        self._decisions: list[TokenStepDecision] = []

    @property
    def decisions(self) -> list[TokenStepDecision]:
        return list(self._decisions)

    def reset(self) -> None:
        self._decisions.clear()

    def evaluate_step(self, step_idx: int, logits: Sequence[float]) -> TokenStepDecision:
        margin, entropy = compute_margin_and_entropy(logits)
        if margin < self.margin_threshold_accurate:
            profile: PrecisionProfileName = "P_accurate"
            tier: PrecisionTier = "t3"
            promoted = True
        elif margin < self.margin_threshold_mixed or entropy > self.entropy_threshold_mixed:
            profile = "P_mixed"
            tier = "t2"
            promoted = True
        else:
            profile = "P_fast"
            tier = "t1"
            promoted = False

        decision = TokenStepDecision(
            step_idx=step_idx,
            logit_margin=margin,
            entropy=entropy,
            selected_profile=profile,
            selected_tier=tier,
            promoted=promoted,
        )
        self._decisions.append(decision)
        return decision

    def build_telemetry(
        self,
        prefix_kv_hit: bool = True,
        kv_cache_fp16_mb: float = 0.0,
        kv_cache_quant_mb: float = 0.0,
    ) -> QuantTelemetrySnapshot:
        if not self._decisions:
            return QuantTelemetrySnapshot(
                prefix_kv_hit=prefix_kv_hit,
                kv_cache_fp16_mb=kv_cache_fp16_mb,
                kv_cache_quant_mb=kv_cache_quant_mb,
            )

        total = len(self._decisions)
        promoted = sum(1 for d in self._decisions if d.promoted)
        min_margin = min(d.logit_margin for d in self._decisions)
        mean_ent = sum(d.entropy for d in self._decisions) / float(total)

        bit_sum = 0.0
        highest_tier: PrecisionTier = "t1"
        highest_profile: PrecisionProfileName = "P_fast"
        for d in self._decisions:
            if d.selected_profile == "P_accurate":
                bit_sum += 16.0
                highest_tier = "t3"
                highest_profile = "P_accurate"
            elif d.selected_profile == "P_mixed":
                bit_sum += self.mixed_act_bits
                if highest_tier == "t1":
                    highest_tier = "t2"
                    highest_profile = "P_mixed"
            else:
                bit_sum += 8.0

        return QuantTelemetrySnapshot(
            tier=highest_tier,
            profile=highest_profile,
            weight_bits=4.0,
            effective_act_bits=round(bit_sum / float(total), 3),
            promoted_tokens=promoted,
            total_tokens=total,
            min_logit_margin=round(min_margin, 4),
            mean_entropy=round(mean_ent, 4),
            prefix_kv_hit=prefix_kv_hit,
            kv_cache_fp16_mb=round(kv_cache_fp16_mb, 3),
            kv_cache_quant_mb=round(kv_cache_quant_mb, 3),
        )
