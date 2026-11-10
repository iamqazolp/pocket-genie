from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_./-]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 1e-9 else vec


def _get_ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _text_sha256(text: str) -> str:
    return hashlib.sha256((text or " ").encode("utf-8")).hexdigest()


class EmbeddingDiskCache:
    """Content-addressed SQLite cache for cloud embedding vectors."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embedding_cache ("
            "model TEXT NOT NULL, "
            "text_hash TEXT NOT NULL, "
            "vector_json TEXT NOT NULL, "
            "PRIMARY KEY (model, text_hash))"
        )
        return conn

    def get_many(self, model: str, hashes: list[str]) -> dict[str, list[float]]:
        if not hashes:
            return {}
        out: dict[str, list[float]] = {}
        try:
            unique = list(dict.fromkeys(hashes))
            with self._connect() as conn:
                for i in range(0, len(unique), 400):
                    chunk = unique[i : i + 400]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT text_hash, vector_json FROM embedding_cache "
                        f"WHERE model = ? AND text_hash IN ({placeholders})",
                        [model, *chunk],
                    ).fetchall()
                    for th, vj in rows:
                        out[str(th)] = json.loads(vj)
        except Exception:
            return {}
        return out

    def put_many(self, model: str, items: list[tuple[str, list[float]]]) -> None:
        if not items:
            return
        try:
            with self._connect() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO embedding_cache (model, text_hash, vector_json) "
                    "VALUES (?, ?, ?)",
                    [(model, th, json.dumps(vec)) for th, vec in items],
                )
        except Exception:
            pass


def _default_cache(
    cache_path: Path | None, transport: Callable[..., dict[str, Any]] | None
) -> EmbeddingDiskCache | None:
    if cache_path is not None:
        return EmbeddingDiskCache(cache_path)
    if transport is not None or "PYTEST_CURRENT_TEST" in os.environ:
        return None
    return EmbeddingDiskCache(Path.home() / ".pocket_genie" / "embeddings_cache.db")


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    transport: Callable[..., dict[str, Any]] | None,
) -> dict[str, Any]:
    if transport is not None:
        try:
            return transport(url, payload, headers)
        except TypeError:
            return transport(url, payload)
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", **headers}
    ctx = _get_ssl_context()
    for attempt in range(2):
        req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if attempt == 0 and exc.code in (429, 502, 503, 504):
                time.sleep(1.5)
                continue
            raise
    return {}


class HashEmbeddingProvider:
    """Deterministic character-trigram and token feature projection for dense similarity."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    @property
    def name(self) -> str:
        return f"hash-{self.dim}"

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        norm_text = " ".join(_tokenize(text))
        if not norm_text:
            return vec

        padded = f"  {norm_text}  "
        for i in range(len(padded) - 2):
            digest = hashlib.md5(padded[i : i + 3].encode("utf-8")).digest()
            vec[digest[0] % self.dim] += 1.0 if (digest[1] & 1) == 0 else -1.0
        for tok in _tokenize(norm_text):
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            vec[digest[2] % self.dim] += 2.0 * (1.0 if (digest[3] & 1) == 0 else -1.0)
        return _l2_normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


class GeminiEmbeddingProvider:
    """Gemini embedding provider using generativelanguage v1beta REST endpoints."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2",
        timeout: float = 30.0,
        transport: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        cache_path: Path | None = None,
    ) -> None:
        normalized = model[len("models/") :] if model.startswith("models/") else model
        if normalized == "gemini-embedding-2":
            normalized = "gemini-embedding-2-preview"
        self.api_key = api_key
        self.model = normalized
        self.timeout = timeout
        self._transport = transport
        self._fallback = HashEmbeddingProvider()
        self._cache = None

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    def embed_query(self, text: str) -> list[float]:
        th = _text_sha256(text)
        if self._cache is not None:
            hit = self._cache.get_many(self.name, [th]).get(th)
            if hit:
                return hit
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:embedContent?key={self.api_key}"
        )
        try:
            resp = _post_json(
                url,
                {"content": {"parts": [{"text": text or " "}]}},
                {},
                self.timeout,
                self._transport,
            )
            values = resp.get("embedding", {}).get("values", [])
            if not values:
                return self._fallback.embed_query(text)
            normed = _l2_normalize(values)
            if self._cache is not None:
                self._cache.put_many(self.name, [(th, normed)])
            return normed
        except Exception:
            return self._fallback.embed_query(text)

    def embed_documents(self, texts: list[str], batch_size: int = 50) -> list[list[float]]:
        if not texts:
            return []
        hashes = [_text_sha256(t) for t in texts]
        cached = self._cache.get_many(self.name, hashes) if self._cache is not None else {}

        missing_hashes: list[str] = []
        missing_texts: list[str] = []
        seen_missing: set[str] = set()
        for t, th in zip(texts, hashes):
            if th not in cached and th not in seen_missing:
                seen_missing.add(th)
                missing_hashes.append(th)
                missing_texts.append(t)

        if missing_texts:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model}:batchEmbedContents?key={self.api_key}"
            )
            try:
                for i in range(0, len(missing_texts), batch_size):
                    b_texts = missing_texts[i : i + batch_size]
                    b_hashes = missing_hashes[i : i + batch_size]
                    payload = {
                        "requests": [
                            {
                                "model": f"models/{self.model}",
                                "content": {"parts": [{"text": t or " "}]},
                            }
                            for t in b_texts
                        ]
                    }
                    resp = _post_json(url, payload, {}, self.timeout, self._transport)
                    raw = resp.get("embeddings", [])
                    if len(raw) != len(b_texts):
                        return self._fallback.embed_documents(texts)
                    batch_pairs: list[tuple[str, list[float]]] = []
                    for th, item in zip(b_hashes, raw):
                        vals = item.get("values", [])
                        if not vals:
                            return self._fallback.embed_documents(texts)
                        normed = _l2_normalize(vals)
                        cached[th] = normed
                        batch_pairs.append((th, normed))
                    if self._cache is not None:
                        self._cache.put_many(self.name, batch_pairs)
            except Exception:
                return self._fallback.embed_documents(texts)

        return [cached[th] for th in hashes]


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider for /v1/embeddings compatible endpoints."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 15.0,
        transport: Callable[..., dict[str, Any]] | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport
        self._fallback = HashEmbeddingProvider()
        self._cache = None

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def embed_query(self, text: str) -> list[float]:
        res = self.embed_documents([text])
        return res[0] if res else self._fallback.embed_query(text)

    def embed_documents(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        if not texts:
            return []
        hashes = [_text_sha256(t) for t in texts]
        cached = self._cache.get_many(self.name, hashes) if self._cache is not None else {}

        missing_hashes: list[str] = []
        missing_texts: list[str] = []
        seen_missing: set[str] = set()
        for t, th in zip(texts, hashes):
            if th not in cached and th not in seen_missing:
                seen_missing.add(th)
                missing_hashes.append(th)
                missing_texts.append(t)

        if missing_texts:
            url = f"{self.base_url}/embeddings"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            try:
                for i in range(0, len(missing_texts), batch_size):
                    b_texts = [t or " " for t in missing_texts[i : i + batch_size]]
                    b_hashes = missing_hashes[i : i + batch_size]
                    resp = _post_json(
                        url,
                        {"model": self.model, "input": b_texts},
                        headers,
                        self.timeout,
                        self._transport,
                    )
                    items = sorted(resp.get("data", []), key=lambda x: x.get("index", 0))
                    if len(items) != len(b_texts):
                        return self._fallback.embed_documents(texts)
                    batch_pairs: list[tuple[str, list[float]]] = []
                    for th, item in zip(b_hashes, items):
                        vals = item.get("embedding", [])
                        if not vals:
                            return self._fallback.embed_documents(texts)
                        normed = _l2_normalize(vals)
                        cached[th] = normed
                        batch_pairs.append((th, normed))
                    if self._cache is not None:
                        self._cache.put_many(self.name, batch_pairs)
            except Exception:
                return self._fallback.embed_documents(texts)

        return [cached[th] for th in hashes]


def build_default_embedder() -> Any:
    if "PYTEST_CURRENT_TEST" in os.environ:
        return HashEmbeddingProvider()
    gemini_key = os.environ.get("GOOGLE_AI_STUDIO_KEY") or os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        model = os.environ.get("GENIE_EMBEDDING_MODEL", "gemini-embedding-2")
        return GeminiEmbeddingProvider(api_key=gemini_key, model=model)
    return HashEmbeddingProvider()
