from __future__ import annotations

import collections
import dataclasses
import math
import re
from pathlib import Path
from typing import Any

from pocket_genie.retrieval.chunker import (
    DocumentChunk,
    reconstruct_chunk_with_media,
    slice_document_into_chunks,
)
from pocket_genie.retrieval.embeddings import (
    HashEmbeddingProvider,
    build_default_embedder,
)

SIMILARITY_THRESHOLD_BEFORE_RERANK = 0.10
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_./-]+")


@dataclasses.dataclass(slots=True)
class ScoredChunk:
    chunk: DocumentChunk
    bm25_score: float
    dense_score: float
    blended_score: float
    rerank_score: float
    rendered_text: str


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _cosine(a: list[float], b: list[float]) -> float:
    return max(0.0, sum(x * y for x, y in zip(a, b)))


class HybridKnowledgeStore:
    """Hybrid BM25 + dense vector store with alpha blending, threshold pre-filtering, and reranking."""

    def __init__(self, embedding_dim: int = 128, embedder: Any | None = None) -> None:
        if embedder is not None:
            self._embedder = embedder
        elif embedding_dim != 128:
            self._embedder = HashEmbeddingProvider(dim=embedding_dim)
        else:
            self._embedder = build_default_embedder()

        self._dim = getattr(self._embedder, "dim", embedding_dim)
        self._chunks: list[DocumentChunk] = []
        self._tokens: list[list[str]] = []
        self._embeddings: list[list[float]] = []
        self._doc_freqs: collections.Counter[str] = collections.Counter()
        self._avg_dl: float = 1.0

    @property
    def embedder_name(self) -> str:
        return getattr(self._embedder, "name", str(self._embedder))

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def add_document(
        self,
        doc_id: str,
        source_path: str,
        content: str,
        chunk_chars: int = 800,
        overlap_chars: int = 120,
    ) -> list[DocumentChunk]:
        chunks = slice_document_into_chunks(
            doc_id=doc_id,
            source_path=source_path,
            content=content,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
        )
        if not chunks:
            return []

        embeddings = self._embedder.embed_documents([ch.text for ch in chunks])
        for ch, emb in zip(chunks, embeddings):
            toks = _tokenize(ch.text)
            self._chunks.append(ch)
            self._tokens.append(toks)
            self._embeddings.append(emb)
            for unique_tok in set(toks):
                self._doc_freqs[unique_tok] += 1

        total_len = sum(len(t) for t in self._tokens)
        self._avg_dl = max(1.0, total_len / float(max(1, len(self._tokens))))
        return chunks

    def index_directory(
        self,
        root_dir: Path,
        glob_patterns: tuple[str, ...] = ("*.py", "*.md", "*.yaml", "*.json"),
        max_files: int = 100,
        chunk_chars: int = 800,
        overlap_chars: int = 120,
    ) -> int:
        added = 0
        pending_chunks: list[DocumentChunk] = []
        for pattern in glob_patterns:
            for file_path in sorted(root_dir.rglob(pattern)):
                if added >= max_files:
                    break
                if any(part.startswith(".") or part == "__pycache__" for part in file_path.parts):
                    continue
                try:
                    text = file_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                rel = str(file_path.relative_to(root_dir))
                pending_chunks.extend(
                    slice_document_into_chunks(
                        doc_id=rel,
                        source_path=rel,
                        content=text,
                        chunk_chars=chunk_chars,
                        overlap_chars=overlap_chars,
                    )
                )
                added += 1

        if pending_chunks:
            embeddings = self._embedder.embed_documents([ch.text for ch in pending_chunks])
            for ch, emb in zip(pending_chunks, embeddings):
                toks = _tokenize(ch.text)
                self._chunks.append(ch)
                self._tokens.append(toks)
                self._embeddings.append(emb)
                for unique_tok in set(toks):
                    self._doc_freqs[unique_tok] += 1
            total_len = sum(len(t) for t in self._tokens)
            self._avg_dl = max(1.0, total_len / float(max(1, len(self._tokens))))
        return added

    def _bm25_scores(self, query_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
        n_docs = len(self._chunks)
        if n_docs == 0 or not query_tokens:
            return [0.0] * n_docs

        raw_scores: list[float] = []
        for doc_toks in self._tokens:
            if not doc_toks:
                raw_scores.append(0.0)
                continue
            tf_counter = collections.Counter(doc_toks)
            dl = len(doc_toks)
            score = 0.0
            for q in query_tokens:
                tf = tf_counter.get(q, 0)
                if tf == 0:
                    continue
                df = self._doc_freqs.get(q, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1.0 - b + b * (dl / self._avg_dl))
                score += idf * ((tf * (k1 + 1.0)) / denom)
            raw_scores.append(score)

        max_s = max(raw_scores, default=0.0)
        if max_s > 1e-9:
            return [s / max_s for s in raw_scores]
        return raw_scores

    def _cross_feature_rerank(
        self, query: str, q_tokens: list[str], chunk: DocumentChunk, base_score: float
    ) -> float:
        text_lower = chunk.text.lower()
        q_lower = query.strip().lower()
        exact_phrase_bonus = 0.25 if (len(q_lower) >= 4 and q_lower in text_lower) else 0.0

        c_tokens = set(_tokenize(chunk.text))
        overlap = len(set(q_tokens) & c_tokens) / float(max(1, len(set(q_tokens))))
        return round(0.65 * base_score + 0.35 * overlap + exact_phrase_bonus, 6)

    def search(
        self,
        query: str,
        top_k: int = 4,
        alpha: float = 0.5,
        similarity_threshold: float = SIMILARITY_THRESHOLD_BEFORE_RERANK,
    ) -> list[ScoredChunk]:
        """Search chunks using alpha-blended BM25 + dense retrieval followed by
        threshold filtering and reranking."""
        if not self._chunks:
            return []

        alpha_clamped = max(0.0, min(1.0, float(alpha)))
        q_tokens = _tokenize(query)
        q_emb = self._embedder.embed_query(query)
        if self._embeddings and len(q_emb) != len(self._embeddings[0]):
            q_emb = HashEmbeddingProvider(dim=len(self._embeddings[0])).embed_query(query)
        bm25_list = self._bm25_scores(q_tokens)

        candidates: list[ScoredChunk] = []
        for idx, chunk in enumerate(self._chunks):
            bm25 = bm25_list[idx]
            dense = _cosine(q_emb, self._embeddings[idx])
            blended = (1.0 - alpha_clamped) * bm25 + alpha_clamped * dense
            if blended < similarity_threshold:
                continue
            candidates.append(
                ScoredChunk(
                    chunk=chunk,
                    bm25_score=round(bm25, 4),
                    dense_score=round(dense, 4),
                    blended_score=round(blended, 4),
                    rerank_score=0.0,
                    rendered_text=reconstruct_chunk_with_media(chunk.text),
                )
            )

        candidates.sort(key=lambda c: c.blended_score, reverse=True)
        pre_rerank_pool = candidates[: max(1, top_k * 3)]

        for item in pre_rerank_pool:
            item.rerank_score = self._cross_feature_rerank(
                query=query,
                q_tokens=q_tokens,
                chunk=item.chunk,
                base_score=item.blended_score,
            )

        pre_rerank_pool.sort(key=lambda c: c.rerank_score, reverse=True)
        return pre_rerank_pool[:top_k]
