from __future__ import annotations

import math
from typing import Any

import pytest

from pocket_genie.retrieval.embeddings import (
    GeminiEmbeddingProvider,
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    build_default_embedder,
)
from pocket_genie.retrieval.hybrid_store import HybridKnowledgeStore


def test_gemini_embedding_provider_model_normalization_and_calls():
    calls: list[tuple[str, dict[str, Any]]] = []

    def mock_transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append((url, payload))
        if ":embedContent" in url:
            return {"embedding": {"values": [3.0, 4.0]}}
        if ":batchEmbedContents" in url:
            return {"embeddings": [{"values": [0.0, 5.0]} for _ in payload.get("requests", [])]}
        return {}

    # Alias normalization & models/ stripping
    p1 = GeminiEmbeddingProvider(api_key="key1", model="gemini-embedding-2", transport=mock_transport)
    assert p1.model == "gemini-embedding-2-preview"
    assert p1.name == "gemini:gemini-embedding-2-preview"

    p2 = GeminiEmbeddingProvider(
        api_key="key1", model="models/gemini-embedding-001", transport=mock_transport
    )
    assert p2.model == "gemini-embedding-001"

    # Single embed_query
    q_vec = p1.embed_query("quantum computing")
    assert math.isclose(math.sqrt(sum(x * x for x in q_vec)), 1.0, abs_tol=1e-5)
    assert q_vec == [0.6, 0.8]
    assert "gemini-embedding-2-preview:embedContent?key=key1" in calls[0][0]
    assert calls[0][1] == {"content": {"parts": [{"text": "quantum computing"}]}}

    # Batched embed_documents
    docs = ["doc 1", "doc 2", "doc 3"]
    doc_vecs = p1.embed_documents(docs, batch_size=2)
    assert len(doc_vecs) == 3
    assert len(calls) == 3  # 1 query + 2 batches (size 2 then 1)
    assert "batchEmbedContents?key=key1" in calls[1][0]
    reqs = calls[1][1]["requests"]
    assert len(reqs) == 2
    assert reqs[0]["model"] == "models/gemini-embedding-2-preview"


def test_gemini_fallback_on_network_error():
    def failing_transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise ConnectionResetError("network down")

    provider = GeminiEmbeddingProvider(api_key="k", transport=failing_transport)
    query_vec = provider.embed_query("test query")
    assert len(query_vec) == 128
    doc_vecs = provider.embed_documents(["doc a", "doc b"])
    assert len(doc_vecs) == 2
    assert len(doc_vecs[0]) == 128


def test_openai_embedding_provider():
    captured: dict[str, Any] = {}

    def mock_transport(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"data": [{"embedding": [1.0, 1.0], "index": 0}]}

    provider = OpenAIEmbeddingProvider(api_key="sk-test", transport=mock_transport)
    assert provider.name == "openai:text-embedding-3-small"
    vec = provider.embed_query("openai search")
    assert len(vec) == 2
    assert math.isclose(vec[0], 1.0 / math.sqrt(2.0), abs_tol=1e-5)
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "text-embedding-3-small"


def test_build_default_embedder(monkeypatch: pytest.MonkeyPatch):
    # Under test run with PYTEST_CURRENT_TEST set
    embedder = build_default_embedder()
    assert isinstance(embedder, HashEmbeddingProvider)

    # When pytest env var is unset
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    embedder = build_default_embedder()
    assert isinstance(embedder, GeminiEmbeddingProvider)
    assert embedder.model == "gemini-embedding-2-preview"


def test_hybrid_store_semantic_retrieval_with_mocked_gemini():
    def mock_transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        def map_text_to_4d(t: str) -> list[float]:
            low = t.lower()
            if "superconductor" in low or "physics" in low:
                return [1.0, 0.0, 0.0, 0.0]
            if "bread" in low or "sourdough" in low:
                return [0.0, 1.0, 0.0, 0.0]
            return [0.0, 0.0, 1.0, 0.0]

        if ":embedContent" in url:
            txt = payload["content"]["parts"][0]["text"]
            return {"embedding": {"values": map_text_to_4d(txt)}}
        reqs = payload.get("requests", [])
        return {
            "embeddings": [
                {"values": map_text_to_4d(r["content"]["parts"][0]["text"])} for r in reqs
            ]
        }

    gemini = GeminiEmbeddingProvider(api_key="test-key", transport=mock_transport)
    store = HybridKnowledgeStore(embedder=gemini)
    assert store.embedder_name == "gemini:gemini-embedding-2-preview"

    store.add_document("doc_phys", "notes/physics.md", "High-Tc superconductor magnetic levitation.")
    store.add_document("doc_bake", "notes/baking.md", "Sourdough bread fermentation and crust.")

    hits = store.search(
        "room temperature superconductor physics", top_k=2, alpha=0.90, similarity_threshold=0.0
    )
    assert len(hits) == 2
    assert hits[0].chunk.doc_id == "doc_phys"
    assert hits[0].dense_score > hits[1].dense_score


