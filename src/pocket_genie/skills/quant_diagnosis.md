---
name: quant-diagnosis
description: Playbook for analyzing W4A8/W4A16 activation outliers, kurtosis spikes, and schema failure modes.
recommended_tier: t2
tags: [quantization, smoothquant, w4a8, telemetry]
---

# Quantization & Activation Outlier Diagnosis Playbook

1. **Check Layer Kurtosis & SwiGLU Spikes**
   - Inspect `mlp.down_proj` kurtosis across layers `0, 18, 21-23`. Persistent high kurtosis after removing BOS (`token 0`) indicates structural SwiGLU scaling bottlenecks rather than attention sinks.
2. **Compare Failure Taxonomy by Scenario**
   - Use `sql.query_workspace` over experiment summary tables to separate syntax corruption (`corrupted_json`, recoverable via Tier-2 grammar FSM) from value drift (`value_mismatch`, requiring Tier-2/3 outlier `A16` promotion).
3. **Verify Logit Margin & KV Budget**
   - Check `promoted_token_ratio` and `min_logit_margin` in the quantization telemetry trace to confirm `P_mixed` threshold calibration.
