# pocket-genie

**Quantization-aware multi-agent runtime with hierarchical specialist delegation, a 9-stage middleware guardrail pipeline, modular Skill/MCP registries, and a mobile Human-in-the-Loop command deck.**

`pocket-genie` combines an **Agentic Supervisor & Specialist Execution Graph** with an **Edge Quantization Runtime** (`P_fast W4A8` $\rightarrow$ `P_mixed` selective outlier-layer `W4A{8,16}` $\rightarrow$ `P_accurate` in-place `W4A16`, token-granular logit-margin routing, and BOS-sink-protected `INT4` KV caching). It runs both with full PyTorch/GPU quantization hooks on macOS/Linux and in zero-native-dependency mode on Android Termux (`aarch64`).

---

## Key Capabilities

- **Hierarchical Agent-as-Tool Delegation (`pocket_genie.core`):**
  - Parallel 3-node state pre-processing (`user_messages_summary_node`, `recent_conversations_node`, `load_user_memories_node`) feeding a multi-route execution graph (`supervisor`, `scoped-demo`, `red-agent`, `run-assistant`, `direct-specialist`).
  - Typed `create_task_<specialist>_t<tier>` delegation tools narrow per-turn schema branching, paired with a stateful `ToolCallChunkAccumulator`, live `subagent_reasoning` streaming, and `[DELEGATION_ERROR]` isolation.
- **9-Stage Middleware Pipeline (`pocket_genie.middlewares`):**
  - `message_sanitizer`, `dynamic_supervisor_prompt` (prefix-cache safe), `summarization` (`RunningSummary`), `project_scoped_tools`, `sql_table_allowlist`, `repeat_tool_call_breaker`, `refusal_detection`, `human_in_the_loop`, `tool_result_truncation`, `tool_error`, and `fallback` (`t1 -> t2 -> t3`).
- **Quantization & KV-Cache Runtime (`pocket_genie.quant`):**
  - **Kurtosis/MSE Mixed-Precision Profiles (`mixed_precision.py`):** `P_fast` (`W4A8`), `P_mixed` (top-$K$ outlier `mlp.down_proj` layers in `UINT16`, `8.24` effective act bits), and `P_accurate` (full in-place `W4A16`, `0.0 MB` extra weight RAM).
  - **Token-Granular Logit-Margin Router (`entropy_router.py`):** Tracks $\Delta z_t = z_{(1),t} - z_{(2),t}$ and token entropy $H(p_t)$ to promote only ambiguous decision tokens mid-decode.
  - **BOS-Sink `INT4` KV Cache (`kv_cache.py`):** Static prefix KV pinning + tri-region KV compression (`Sink FP16` + `Per-Channel K / Per-Token V INT4` + `Recent Window FP16`).
- **Four-Layer Chunking & Hybrid Retrieval (`pocket_genie.retrieval`):**
  - Sequential `chunk_index` document slicing with media marker reconstruction, $\alpha$-blended BM25 + dense vector search with threshold pre-filtering (`SIMILARITY_THRESHOLD_BEFORE_RERANK = 0.10`) and reranking, plus bounded async batch chunking.
- **Zero-Touch Skill & MCP Plug-ins (`pocket_genie.registry`, `pocket_genie.mcp`):**
  - Drop Markdown playbooks with YAML frontmatter into `src/pocket_genie/skills/` or `~/.pocket-genie/skills/`.
  - Bridge `stdio` or `http` MCP servers (`mcp_servers.json`) into isolated `mcp_<server>_agent` specialists automatically protected by schema sanitization and HITL approval gates.

---

## Quick Start

### 1. Installation (macOS / Linux)

```bash
pip install -e ".[dev]"
pytest -q
```

### 2. Android Termux (`aarch64`) Installation

```bash
pip install -r requirements-termux.txt
pip install -e . --no-deps
pytest -q
```

### 3. CLI Usage (`genie`)

```bash
# List registered specialists and tool domains
genie specialists

# List discovered YAML-frontmatter skill playbooks
genie skills

# Run quantization profile, token-router, and INT4 KV cache benchmark
genie bench

# Run a multi-agent turn from the CLI
genie ask --prompt "Check git_status and summarize repository state" --agent-type supervisor

# Launch the 3-Tab SSE Mobile/Web Command Deck
genie serve --host 0.0.0.0 --port 8090
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed dataflow diagrams and module specifications.
