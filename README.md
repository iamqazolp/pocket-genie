# pocket-genie

**Quantization-aware multi-agent runtime built on `LangGraph`, `LangChain`, `Langfuse`, and `TypeSafe Jev` System-One triage—combining selective `W4A{8,16}` mixed-precision execution, cached FSM JSON-schema decoding, BOS-sink `INT4` KV compression, and a mobile Human-in-the-Loop command deck.**

`pocket-genie` pairs a **System-One + System-Two Multi-Agent `StateGraph`** with an **Edge PyTorch / Local-LLM Quantization Runtime**. It runs with full PyTorch MPS/CUDA quantization hooks, LangGraph checkpointing, and OpenTelemetry Langfuse tracing on macOS/Linux, while gracefully degrading to a zero-native-dependency mode on Android Termux (`aarch64`).

---

## Highlights & Empirical Results (`Qwen2.5-0.5B` 4-Bit Edge Ablation)

Evaluated via `genie eval` ([`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), [`benchmarks/latest_eval.json`](benchmarks/latest_eval.json)) across multi-domain tool-calling tasks sharing a single **4-bit weight buffer (`430.42 MB`, `0.0 MB` fallback weight overhead)**:

| Recipe | Mode | Effective Act Bits | JSON & Schema Valid | Exact Tool Match | 2K KV Cache Saved |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `W4A4_unconstrained` | Unconstrained | `4.00` | `0.0%` | `0.0%` | `16.71 MB` (`~70%`) |
| `P_fast_W4A8` (`t1`) | Unconstrained | `8.00` | `0.0%` | `0.0%` | `16.71 MB` (`~70%`) |
| **`P_mixed_W4A8_16_FSM` (`t2`)** | **Outlier `mlp.down_proj` `A16` + FSM** | **`8.57`** | **`100.0%`** | **`70.0%`** | **`16.71 MB` (`~70%`)** |
| `P_accurate_W4A16_FSM` (`t3`) | In-Place `W4A16` + FSM | `16.00` | `100.0%` | `60.0%` | `16.71 MB` (`~70%`) |

- **Empirical Kurtosis Calibration (`quant/hf_backend.py`):** Live forward calibration on `Qwen2.5-0.5B-Instruct` identifies extreme SwiGLU activation spikes in `mlp.down_proj` (`Layer 2` $\kappa = 89,759.9$, `Layer 21` $\kappa = 65,319.2$), promoting only top outlier layers (`8.57` avg act bits) + Boundary-C SmoothQuant scale migration.
- **TypeSafe Jev System-One Triage (`integrations/jev_router.py`):** Parallel `LangGraph` node querying OpenRouter's `/api/alpha/decisions` endpoint (`noul` prompt-injection & mutating-intent predicates, `choice` fast-path specialist routing, and `score` precision-tier pre-escalation) before invoking System-Two LLMs.
- **Compiled `LangGraph` `StateGraph` & Bounded ReAct Delegation (`core/`):** 4-node parallel fan-out (`user_messages_summary`, `recent_conversations`, `load_user_memories`, `jev_system_one_triage`) $\rightarrow$ conditional routing $\rightarrow$ bounded multi-step ReAct specialists (`devops`, `sql`, `kb`, `device`, `red-agent`, and auto-wrapped `MCP` servers) with `InMemorySaver` checkpointing.
- **`LangChain` & `Langfuse` OpenTelemetry Observability (`integrations/`, `observability/`):** Lossless `BaseChatModel` (`PocketGenieChatModel`) and Pydantic-synthesized `StructuredTool` adapters, plus `GenieLangfuseTracer` & `LangfuseTracingMiddleware` exporting `agent`, `generation`, `tool`, and `guardrail` OTel spans with live quantization scores (`effective_act_bits`, `promoted_token_ratio`, `kv_memory_saved_mb`).
- **10-Stage Guardrail Pipeline & Mobile HITL Console (`middlewares/`, `server.py`, `clients/flutter/`):** Orphaned-tool message sanitization, prefix-cache-safe prompts, rolling `RunningSummary` compaction, SQL CTE allowlisting, repeat-call loop breaking, `t1 -> t2 -> t3` fallback escalation, and Web/Flutter Human-in-the-Loop approval cards with post-approval LLM synthesis.

---

## Quick Start

### 1. Installation (macOS / Linux — Full Stack)

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
```

### 2. Android Termux (`aarch64` — Zero Native Dependencies)

```bash
pip install -r requirements-termux.txt
pip install -e . --no-deps
pytest -q
```

### 3. CLI Usage (`genie`)

```bash
# Inspect compiled LangGraph StateGraph Mermaid diagram, specialists, and skills
genie graph
genie specialists
genie skills

# Run live multi-agent ReAct turn via Ollama (with real logprobs token routing)
genie ask \
  --backend local-llm \
  --model qwen2.5-coder:7b \
  --fallback-model llama3.1:8b \
  --prompt "Check git status and summarize recent repository changes"

# Run live PyTorch quantized inference (SmoothQuant + W4A{8,16} + FSM enforcer)
genie ask \
  --backend hf-quant \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --tier t2 \
  --prompt "Check current git repository status"

# Run empirical multi-domain quantization & tool-calling benchmark
genie eval \
  --backend hf-quant \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-tasks 10 \
  --output benchmarks/latest_eval.json

# Launch headless /api/v1 OpenAPI server + Mobile/Web HITL Command Deck
genie serve --backend local-llm --model qwen2.5-coder:7b --host 0.0.0.0 --port 8090
```

### Optional Environment Variables (`.env`)
- `OPENROUTER_API_KEY` / `JEV_MODEL`: Enables live TypeSafe Jev System-One `/api/alpha/decisions` triage (automatically falls back to local calibrated scorer when unset).
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`: Exports OpenTelemetry traces and quantization telemetry scores to Langfuse Cloud or self-hosted Langfuse (`GET /api/v1/traces` always exposes the local trace buffer).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for full architecture diagrams and ablation details.
