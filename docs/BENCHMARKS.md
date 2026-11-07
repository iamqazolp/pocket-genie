# Empirical Quantization & Tool-Calling Benchmarks (`v0.3.0`)

All numbers below are generated directly by `genie eval` (`src/pocket_genie/eval/harness.py`) and persisted to [`benchmarks/latest_eval.json`](../benchmarks/latest_eval.json).

## 1. Reproduce Locally

```bash
# Stratified 10-task PyTorch MPS/CUDA/CPU ablation on Qwen/Qwen2.5-0.5B-Instruct
genie eval \
  --backend hf-quant \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-tasks 10 \
  --output benchmarks/latest_eval.json

# Full 30-task evaluation against local Ollama (e.g. qwen2.5-coder:7b)
genie eval \
  --backend local-llm \
  --model qwen2.5-coder:7b \
  --max-tasks 30
```

---

## 2. `Qwen/Qwen2.5-0.5B-Instruct` 4-Bit Multi-Domain Tool-Calling Ablation

Evaluated across a stratified multi-domain candidate tool suite (`devops`, `sql`, `kb`, `memory`, `device`) sharing the **exact same 4-bit weight buffer (`430.42 MB`, `0.0 MB` fallback weight overhead)**:

| Recipe | Mode | Weight Bits | Effective Act Bits | Footprint (`+Fallback`) | JSON Valid | Schema Valid | Exact Tool Match | 2K KV Cache Saved |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `W4A4_unconstrained` | Unconstrained | `4.0` | `4.00` | `430.42 MB` (`+0.0 MB`) | `0.0%` | `0.0%` | `0.0%` | `16.71 MB` (`~70%`) |
| `P_fast_W4A8` (`t1`) | Unconstrained | `4.0` | `8.00` | `430.42 MB` (`+0.0 MB`) | `0.0%` | `0.0%` | `0.0%` | `16.71 MB` (`~70%`) |
| **`P_mixed_W4A8_16_FSM` (`t2`)** | **Outlier `A16` + FSM** | **`4.0`** | **`8.57`** | **`430.42 MB` (`+0.0 MB`)** | **`100.0%`** | **`100.0%`** | **`70.0%`** | **`16.71 MB` (`~70%`)** |
| `P_accurate_W4A16_FSM` (`t3`) | In-Place `A16` + FSM | `4.0` | `16.00` | `430.42 MB` (`+0.0 MB`) | `100.0%` | `100.0%` | `60.0%` | `16.71 MB` (`~70%`) |

### Key Takeaways
1. **Why Unconstrained Sub-1B `W4A4` / `W4A8` Fails on Tool Schemas:** Without grammar constraints and outlier protection, `Qwen2.5-0.5B` suffers attention/MLP degeneration into prose repetition loops (`0%` valid JSON).
2. **Selective Outlier Promotion (`8.57` Effective Bits):** Promoting only the top empirically calibrated high-kurtosis `mlp.down_proj` layers to `UINT16` activations (`+0.57` average bits across linear layers, `0.0 MB` extra weight RAM) + cached `FastJsonSchemaParser` (`TokenEnforcer`) restores **`100.0%` JSON & schema validity** and **`70.0%` exact multi-domain tool selection** on a `0.5B` edge model.

---

## 3. Empirical Activation Kurtosis Calibration (`Qwen/Qwen2.5-0.5B-Instruct`)

Measured live by `calibrate_layer_sensitivities` (`src/pocket_genie/quant/hf_backend.py`) prior to SmoothQuant migration:

| Rank | Layer Index | Projection | Excess Kurtosis ($\kappa$) | Relative `W4A8` MSE | Sensitivity Score |
| :---: | :---: | :--- | :---: | :---: | :---: |
| 1 | `Layer 2` | `mlp.down_proj` | **`89,759.90`** | `0.000499` | `90,207.80` |
| 2 | `Layer 21` | `mlp.down_proj` | **`65,319.22`** | `0.002812` | `67,155.99` |
| 3 | `Layer 3` | `mlp.down_proj` | **`52,201.88`** | `0.000810` | `52,624.71` |
| 4 | `Layer 5` | `mlp.down_proj` | **`26,247.31`** | `0.070562` | `44,767.93` |
| 5 | `Layer 3` | `mlp.gate_proj` | `12,983.57` | `0.019219` | `15,478.88` |

SwiGLU gating concentrates extreme channel spikes (`\kappa > 26,000`) almost exclusively inside `mlp.down_proj` inputs—confirming why Boundary-C SmoothQuant migration (`mlp.up_proj` $\leftrightarrow$ `mlp.down_proj`) + selective `mlp.down_proj` `UINT16` activation hooks capture the vast majority of quantization sensitivity at `<4%` layer promotion cost.
