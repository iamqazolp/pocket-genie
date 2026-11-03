"""Initial lexical-only BM25 index before adding dense vector blending."""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class LexicalStore:
    def __init__(self) -> None:
        self.docs: list[tuple[str, list[str]]] = []

    def add(self, doc_id: str, text: str) -> None:
        self.docs.append((doc_id, tokenize(text)))

    def search(self, query: str, top_k: int = 4) -> list[tuple[str, float]]:
        q_toks = tokenize(query)
        n_docs = max(1, len(self.docs))
        scores: list[tuple[str, float]] = []
        for doc_id, toks in self.docs:
            tf = Counter(toks)
            s = sum(math.log(1.0 + tf[t]) * math.log(1.0 + n_docs) for t in q_toks if t in tf)
            if s > 0:
                scores.append((doc_id, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
