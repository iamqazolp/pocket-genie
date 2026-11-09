from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import sqlite3
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

from pocket_genie.integrations.jev_router import JevDecisionClient
from pocket_genie.registry.tools import ToolRegistry
from pocket_genie.retrieval.hybrid_store import HybridKnowledgeStore

_BLOCKED_HOST_PREFIXES: tuple[str, ...] = (
    "127.",
    "0.0.0.0",
    "10.",
    "192.168.",
    "169.254.",
)


class _HTMLToMarkdownParser(HTMLParser):
    """Minimal stdlib HTML parser that extracts clean Markdown while stripping noisy tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._pieces: list[str] = []
        self._skip_depth: int = 0
        self._in_pre: bool = False
        self._skip_tags: frozenset[str] = frozenset({"script", "style", "nav", "footer"})

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in self._skip_tags:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return

        if tag_lower in ("h1", "h2", "h3"):
            level = int(tag_lower[1])
            self._ensure_newline(count=2)
            self._pieces.append(f"{'#' * level} ")
        elif tag_lower == "p":
            self._ensure_newline(count=2)
        elif tag_lower == "br":
            self._pieces.append("\n")
        elif tag_lower == "li":
            self._ensure_newline(count=1)
            self._pieces.append("- ")
        elif tag_lower == "pre":
            self._in_pre = True
            self._ensure_newline(count=2)
            self._pieces.append("```\n")
        elif tag_lower == "code":
            if not self._in_pre:
                self._pieces.append("`")
        elif tag_lower in ("tr", "div", "blockquote"):
            self._ensure_newline(count=1)

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth > 0:
            return

        if tag_lower in ("h1", "h2", "h3", "p"):
            self._pieces.append("\n\n")
        elif tag_lower == "li":
            self._pieces.append("\n")
        elif tag_lower == "pre":
            self._in_pre = False
            self._ensure_newline(count=1)
            self._pieces.append("```\n\n")
        elif tag_lower == "code":
            if not self._in_pre:
                self._pieces.append("`")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        self._pieces.append(data)

    def _ensure_newline(self, count: int = 1) -> None:
        if not self._pieces:
            return
        trailing = 0
        for piece in reversed(self._pieces):
            if not piece:
                continue
            for char in reversed(piece):
                if char == "\n":
                    trailing += 1
                else:
                    break
            else:
                continue
            break
        needed = count - trailing
        if needed > 0:
            self._pieces.append("\n" * needed)

    def get_markdown(self) -> str:
        raw = "".join(self._pieces)
        lines = [line.rstrip() for line in raw.splitlines()]
        cleaned = "\n".join(lines)
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def html_to_clean_markdown(html_text: str) -> str:
    """Convert HTML string to clean Markdown, stripping scripts, styles, nav, and footers."""
    if not html_text:
        return ""
    parser = _HTMLToMarkdownParser()
    parser.feed(html_text)
    parser.close()
    return parser.get_markdown()


def validate_public_http_url(url: str) -> str:
    """Validate that the given URL uses http/https and does not target localhost or private IPs."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    cleaned = url.strip()
    try:
        parsed = urllib.parse.urlsplit(cleaned)
    except Exception as exc:
        raise ValueError(f"Invalid URL structure: {exc}") from exc

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"Disallowed URL scheme '{parsed.scheme}'. Only http and https are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must contain a valid hostname.")

    host_lower = hostname.lower().strip("[]")
    if host_lower == "localhost" or host_lower.endswith(".localhost"):
        raise ValueError(f"Access to localhost is blocked: {hostname}")

    if host_lower == "::1" or host_lower.endswith("::1"):
        raise ValueError(f"Blocked loopback IPv6 address: {hostname}")

    for prefix in _BLOCKED_HOST_PREFIXES:
        if host_lower == prefix or host_lower.startswith(prefix):
            raise ValueError(f"Blocked private/internal IP address: {hostname}")

    try:
        ip = ipaddress.ip_address(host_lower)
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified:
            raise ValueError(f"Blocked private/loopback IP address: {hostname}")
    except ValueError as ve:
        if "Blocked" in str(ve):
            raise

    return cleaned


def _default_http_get(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PocketGenie/0.3.0 (ResearchStudyBot; +https://github.com)"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        data = resp.read()
        return data.decode(charset, errors="replace")


def _parse_arxiv_atom(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    entries: list[dict[str, Any]] = []

    def local_tag(elem: ET.Element) -> str:
        return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

    for child in root:
        if local_tag(child) != "entry":
            continue

        raw_id = ""
        title = ""
        summary = ""
        published = ""
        authors: list[str] = []
        pdf_url = ""

        for sub in child:
            ltag = local_tag(sub)
            if ltag == "id":
                raw_id = (sub.text or "").strip()
            elif ltag == "title":
                title = re.sub(r"\s+", " ", sub.text or "").strip()
            elif ltag == "summary":
                summary = re.sub(r"\s+", " ", sub.text or "").strip()
            elif ltag == "published":
                published = (sub.text or "").strip()
            elif ltag == "author":
                for a_child in sub:
                    if local_tag(a_child) == "name" and a_child.text:
                        authors.append(a_child.text.strip())
            elif ltag == "link":
                attrib = sub.attrib
                href = attrib.get("href", "")
                if (
                    attrib.get("title") == "pdf"
                    or attrib.get("type") == "application/pdf"
                    or href.endswith(".pdf")
                ):
                    pdf_url = href

        arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        entries.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "published": published,
                "summary": summary,
                "pdf_url": pdf_url,
            }
        )
    return entries



def register_research_tools(
    registry: ToolRegistry,
    workspace_dir: Path,
    kb_store: HybridKnowledgeStore | None = None,
) -> None:
    kb = kb_store
    @registry.register(
        name="research.search_arxiv",
        description=(
            "Search arXiv scientific preprints via Atom API and return structured paper metadata "
            "(arxiv_id, title, authors, published, summary, pdf_url)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords, topic, or authors to search for in arXiv.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of preprints to retrieve (default: 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        domain="research",
        mutating=False,
    )
    def search_arxiv(query: str, max_results: int = 5) -> str:
        if not query.strip():
            raise ValueError("query cannot be empty.")
        encoded_query = urllib.parse.quote_plus(query.strip())
        url = (
            f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}"
            f"&max_results={int(max_results)}"
        )
        xml_text = fetcher(url)
        entries = _parse_arxiv_atom(xml_text)
        return json.dumps(entries, indent=2)

    @registry.register(
        name="research.fetch_url",
        description=(
            "Fetch an HTTP/HTTPS webpage, convert HTML to clean Markdown, and optionally index "
            "it into the hybrid knowledge store."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Public HTTP/HTTPS URL to fetch.",
                },
                "index_into_kb": {
                    "type": "boolean",
                    "description": "Whether to index converted Markdown into HybridKnowledgeStore.",
                    "default": True,
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters of Markdown to return and index.",
                    "default": 6000,
                },
            },
            "required": ["url"],
        },
        domain="research",
        mutating=False,
    )
    def fetch_url(url: str, index_into_kb: bool = True, max_chars: int = 6000) -> str:
        validated_url = validate_public_http_url(url)
        raw_html = fetcher(validated_url)
        md = html_to_clean_markdown(raw_html)
        if max_chars > 0 and len(md) > max_chars:
            md = md[:max_chars]

        indexed_chunks = 0
        if index_into_kb and kb_store is not None:
            doc_id = f"web_{hashlib.sha256(validated_url.encode('utf-8')).hexdigest()[:12]}"
            chunks = kb_store.add_document(doc_id=doc_id, source_path=validated_url, content=md)
            indexed_chunks = len(chunks)

        return json.dumps(
            {
                "url": validated_url,
                "indexed_chunks": indexed_chunks,
                "markdown": md,
            },
            indent=2,
        )

