from __future__ import annotations

import asyncio

from pocket_genie.retrieval import (
    HybridKnowledgeStore,
    reconstruct_chunk_with_media,
    run_chunked_batches,
    run_chunked_batches_async,
    slice_document_into_chunks,
)


def test_chunk_slicing_and_multimodal_marker_reconstruction():
    doc = (
        "Layer 0 mlp.down_proj exhibits kurtosis 542.6.\n"
        '<image id="kurtosis_01"/>\n\n'
        "Boundary C migrates SwiGLU outlier scale into up_proj.\n"
        "![BOS Sink](bos_sink_hex)"
    )
    chunks = slice_document_into_chunks("doc1", "notes/quant.md", doc, chunk_chars=80, overlap_chars=10)
    assert len(chunks) >= 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert "kurtosis_01" in chunks[0].image_markers

    rendered = reconstruct_chunk_with_media(doc)
    assert "![Image](/api/media/kb-images/kurtosis_01)" in rendered
    assert "![BOS Sink](/api/media/kb-images/bos_sink_hex)" in rendered


def test_hybrid_knowledge_store_alpha_blending_and_rerank():
    store = HybridKnowledgeStore()
    store.add_document(
        "cve_doc",
        "sec/advisory.md",
        "Critical regression in FastJsonSchemaParser token mask cache key CVE-2026-9912.",
    )
    store.add_document(
        "arch_doc",
        "docs/arch.md",
        "Hierarchical supervisor delegates domain tasks to isolated specialist subgraphs.",
    )

    lexical_hits = store.search("CVE-2026-9912 FastJsonSchemaParser", top_k=2, alpha=0.15)
    assert lexical_hits
    assert lexical_hits[0].chunk.doc_id == "cve_doc"

    semantic_hits = store.search("specialist subgraph delegation", top_k=2, alpha=0.80)
    assert semantic_hits
    assert semantic_hits[0].chunk.doc_id == "arch_doc"


def test_batch_chunker_sync_and_async_with_retries():
    items = list(range(29))
    attempts: dict[int, int] = {}

    def flaky_worker(batch: list[int]) -> list[int]:
        key = batch[0]
        attempts[key] = attempts.get(key, 0) + 1
        if key == 12 and attempts[key] == 1:
            raise RuntimeError("Transient rate limit")
        return [x * 2 for x in batch]

    res = run_chunked_batches(items, flaky_worker, chunk_size=12, retry_attempts=2)
    assert res == [x * 2 for x in items]
    assert attempts[12] == 2

    async def async_worker(batch: list[int]) -> list[int]:
        await asyncio.sleep(0.001)
        return [x + 1 for x in batch]

    async_res = asyncio.run(run_chunked_batches_async(items, async_worker, chunk_size=12))
    assert async_res == [x + 1 for x in items]
