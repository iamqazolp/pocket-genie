from __future__ import annotations

from pocket_genie.quant.kv_cache import BOSSinkINT4KVCache
from pocket_genie.quant.mixed_precision import (
    PrecisionProfile,
    build_sensitivity_profiles,
)
from pocket_genie.registry.backends import BackendRegistry, ModelBackend
from pocket_genie.shared import PrecisionTier


class TieredAgentFactory:
    """Binds quantization profiles (P_fast, P_mixed, P_accurate) and KV cache managers to tiered agent backends."""

    def __init__(
        self,
        backends: BackendRegistry | None = None,
        kv_cache: BOSSinkINT4KVCache | None = None,
    ) -> None:
        self.backends = backends or BackendRegistry()
        self.kv_cache = kv_cache or BOSSinkINT4KVCache()
        self.profiles: dict[PrecisionTier, PrecisionProfile] = build_sensitivity_profiles()

    def get_backend(self, tier: PrecisionTier) -> ModelBackend:
        return self.backends.get(tier)

    def get_profile(self, tier: PrecisionTier) -> PrecisionProfile:
        return self.profiles[tier]
