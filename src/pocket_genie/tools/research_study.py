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


class StudyDeckStore:
    """SQLite flashcard store implementing SuperMemo SM-2 spaced repetition."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS flashcards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                front TEXT NOT NULL,
                back TEXT NOT NULL,
                repetitions INTEGER NOT NULL DEFAULT 0,
                easiness_factor REAL NOT NULL DEFAULT 2.5,
                interval_days INTEGER NOT NULL DEFAULT 0,
                due_at TEXT NOT NULL,
                last_reviewed_at TEXT
            )
            """
        )
        self._conn.commit()

    def add_flashcard(self, topic: str, front: str, back: str) -> dict[str, Any]:
        if not front.strip():
            raise ValueError("Flashcard front cannot be empty.")
        if not back.strip():
            raise ValueError("Flashcard back cannot be empty.")

        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO flashcards (
                topic, front, back, repetitions, easiness_factor,
                interval_days, due_at, last_reviewed_at
            ) VALUES (?, ?, ?, 0, 2.5, 0, ?, NULL)
            """,
            (topic.strip(), front.strip(), back.strip(), now),
        )
        self._conn.commit()
        card_id = cur.lastrowid
        card = self.get_flashcard(card_id)
        if card is None:
            raise RuntimeError("Failed to retrieve created flashcard.")
        return card

    def get_flashcard(self, card_id: int) -> dict[str, Any] | None:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT id, topic, front, back, repetitions, easiness_factor,
                   interval_days, due_at, last_reviewed_at
            FROM flashcards
            WHERE id = ?
            """,
            (int(card_id),),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_due_flashcards(
        self,
        topic: str = "",
        limit: int = 10,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        cutoff = as_of or datetime.now(timezone.utc).isoformat()
        cur = self._conn.cursor()
        max_limit = max(1, int(limit))

        if topic.strip():
            cur.execute(
                """
                SELECT id, topic, front, back, repetitions, easiness_factor,
                       interval_days, due_at, last_reviewed_at
                FROM flashcards
                WHERE due_at <= ? AND lower(topic) = lower(?)
                ORDER BY due_at ASC
                LIMIT ?
                """,
                (cutoff, topic.strip(), max_limit),
            )
        else:
            cur.execute(
                """
                SELECT id, topic, front, back, repetitions, easiness_factor,
                       interval_days, due_at, last_reviewed_at
                FROM flashcards
                WHERE due_at <= ?
                ORDER BY due_at ASC
                LIMIT ?
                """,
                (cutoff, max_limit),
            )
        return [dict(row) for row in cur.fetchall()]

    def review_flashcard(
        self,
        card_id: int,
        quality: int,
        reviewed_at: datetime | None = None,
    ) -> dict[str, Any]:
        card = self.get_flashcard(card_id)
        if card is None:
            raise KeyError(f"Flashcard with id {card_id} not found.")

        q = int(quality)
        if q < 0 or q > 5:
            raise ValueError(f"Quality must be an integer between 0 and 5, got {quality}.")

        ef = float(card["easiness_factor"])
        rep = int(card["repetitions"])
        interval = int(card["interval_days"])

        new_ef = max(1.3, ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))
        new_ef = round(new_ef, 4)

        if q < 3:
            new_rep = 0
            new_interval = 1
        else:
            if rep == 0:
                new_interval = 1
            elif rep == 1:
                new_interval = 6
            else:
                new_interval = max(1, int(round(interval * new_ef)))
            new_rep = rep + 1

        now = reviewed_at or datetime.now(timezone.utc)
        last_reviewed_at = now.isoformat()
        due_at = (now + timedelta(days=new_interval)).isoformat()

        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE flashcards
            SET repetitions = ?, easiness_factor = ?, interval_days = ?,
                due_at = ?, last_reviewed_at = ?
            WHERE id = ?
            """,
            (new_rep, new_ef, new_interval, due_at, last_reviewed_at, int(card_id)),
        )
        self._conn.commit()
        updated = self.get_flashcard(card_id)
        if updated is None:
            raise RuntimeError(f"Failed to fetch updated flashcard {card_id}.")
        return updated

    def count(self, topic: str = "") -> int:
        cur = self._conn.cursor()
        if topic.strip():
            cur.execute(
                "SELECT count(*) FROM flashcards WHERE lower(topic) = lower(?)",
                (topic.strip(),),
            )
        else:
            cur.execute("SELECT count(*) FROM flashcards")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


def register_research_and_study_tools(
    registry: ToolRegistry,
    workspace_dir: Path,
    kb_store: HybridKnowledgeStore | None = None,
    study_store: StudyDeckStore | None = None,
    jev_client: JevDecisionClient | None = None,
    http_get: Callable[[str], str] | None = None,
) -> StudyDeckStore:
    """Register research and active-recall study tools with ToolRegistry."""
    fetcher = http_get or _default_http_get
    jev = jev_client or JevDecisionClient()

    if study_store is None:
        if "PYTEST_CURRENT_TEST" in os.environ:
            db_path: str | Path = ":memory:"
        else:
            db_path = workspace_dir / ".genie_study.db"
        study_deck = StudyDeckStore(db_path=db_path)
    else:
        study_deck = study_store

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

    @registry.register(
        name="study.add_flashcard",
        description="Add a new flashcard to the active-recall study deck.",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Subject or topic category."},
                "front": {"type": "string", "description": "Flashcard question or cue."},
                "back": {"type": "string", "description": "Flashcard answer or solution."},
            },
            "required": ["topic", "front", "back"],
        },
        domain="study",
        mutating=False,
    )
    def add_flashcard(topic: str, front: str, back: str) -> str:
        card = study_deck.add_flashcard(topic=topic, front=front, back=back)
        return json.dumps(card, indent=2)

    @registry.register(
        name="study.due_flashcards",
        description="Retrieve flashcards due for review under SuperMemo SM-2 spaced repetition.",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Optional topic filter.", "default": ""},
                "limit": {
                    "type": "integer",
                    "description": "Maximum flashcards to return.",
                    "default": 10,
                },
                "as_of": {
                    "type": "string",
                    "description": "Optional ISO 8601 timestamp cutoff.",
                    "default": "",
                },
            },
        },
        domain="study",
        mutating=False,
    )
    def due_flashcards(topic: str = "", limit: int = 10, as_of: str = "") -> str:
        cards = study_deck.get_due_flashcards(
            topic=topic,
            limit=limit,
            as_of=as_of if as_of.strip() else None,
        )
        return json.dumps(cards, indent=2)

    @registry.register(
        name="study.review_flashcard",
        description="Record a review grade (0..5) for a flashcard and advance its SM-2 interval.",
        parameters={
            "type": "object",
            "properties": {
                "card_id": {"type": "integer", "description": "ID of the flashcard reviewed."},
                "quality": {
                    "type": "integer",
                    "description": "SM-2 grade from 0 (blackout) to 5 (perfect recall).",
                },
            },
            "required": ["card_id", "quality"],
        },
        domain="study",
        mutating=False,
    )
    def review_flashcard(card_id: int, quality: int) -> str:
        card = study_deck.review_flashcard(card_id=int(card_id), quality=int(quality))
        return json.dumps(card, indent=2)

    @registry.register(
        name="study.grade_answer",
        description=(
            "Grade a user's active recall response using TypeSafe Jev Decisions and optionally "
            "advance the flashcard's SM-2 interval."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Flashcard question or cue."},
                "reference_answer": {
                    "type": "string",
                    "description": "Ground truth reference answer.",
                },
                "user_answer": {"type": "string", "description": "User's submitted answer."},
                "card_id": {
                    "type": "integer",
                    "description": "Optional flashcard ID to automatically review and advance.",
                    "default": 0,
                },
            },
            "required": ["question", "reference_answer", "user_answer"],
        },
        domain="study",
        mutating=False,
    )
    def grade_answer(
        question: str,
        reference_answer: str,
        user_answer: str,
        card_id: int = 0,
    ) -> str:
        state = (
            f"Question: {question.strip()}\n"
            f"Reference Answer: {reference_answer.strip()}\n"
            f"User Answer: {user_answer.strip()}"
        )
        questions = {
            "factual_accuracy": {
                "type": "noul",
                "instructions": (
                    "The user answer is factually consistent and accurate with respect to the "
                    "reference answer."
                ),
            },
            "missing_key_concept": {
                "type": "noul",
                "instructions": (
                    "The user answer is missing essential domain concepts or key technical terms "
                    "from the reference."
                ),
            },
            "mastery_score": {
                "type": "score",
                "criteria": [
                    "Completely incorrect, irrelevant, or hallucinatory response (0.0).",
                    (
                        "Partially correct answer but missing core concepts "
                        "or contains factual errors (0.33)."
                    ),
                    "Mostly correct answer covering essential ideas with minor omissions (0.67).",
                    (
                        "Exemplary and comprehensive answer demonstrating "
                        "full conceptual mastery (1.0)."
                    ),
                ],
            },
        }

        decision = jev.decide(state=state, questions=questions)
        answers = decision.get("answers") or {}

        raw_mastery = (answers.get("mastery_score") or {}).get("score", 0.5)
        factual_prob = (answers.get("factual_accuracy") or {}).get("noul", 0.5)
        missing_prob = (answers.get("missing_key_concept") or {}).get("noul", 0.5)

        score_val = float(raw_mastery)
        if score_val <= 1.0:
            quality = max(0, min(5, int(score_val * 5.0 + 0.5)))
        else:
            quality = max(0, min(5, int(score_val + 0.5)))

        card_result = None
        if int(card_id) > 0:
            card_result = study_deck.review_flashcard(card_id=int(card_id), quality=quality)

        feedback = {
            "quality": quality,
            "mastery_score": round(score_val, 4),
            "factual_accuracy": round(float(factual_prob), 4),
            "missing_key_concept": round(float(missing_prob), 4),
            "card_id": int(card_id),
            "card": card_result,
            "decision": decision,
        }
        return json.dumps(feedback, indent=2)

    return study_deck


__all__ = [
    "StudyDeckStore",
    "html_to_clean_markdown",
    "register_research_and_study_tools",
    "validate_public_http_url",
]
