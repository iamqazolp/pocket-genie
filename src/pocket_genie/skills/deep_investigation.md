---
name: deep-investigation
description: Multi-source codebase, experiment SQL, and local notes synthesis protocol.
recommended_tier: t2
tags: [research, rag, sql, synthesis]
---

# Multi-Source Deep Investigation Protocol

1. **Lexical vs. Semantic Retrieval Selection**
   - For exact symbol names, error codes, or file paths, invoke `kb.search` with `alpha=0.2` (BM25-weighted).
   - For architectural or conceptual queries, invoke `kb.search` with `alpha=0.8` (dense-weighted).
2. **Cross-Verify Quantitative Claims**
   - Pair document retrieval with `sql.query_workspace` over structured JSON/CSV artifacts so every metric is backed by exact table aggregations.
3. **Synthesize with Chunk Citations**
   - Reference `file_path#chunk_index` in findings so the supervisor can verify provenance without re-reading raw files.
