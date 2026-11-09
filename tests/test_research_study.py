from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pocket_genie.integrations.jev_router import JevDecisionClient
from pocket_genie.registry.tools import ToolRegistry
from pocket_genie.retrieval.hybrid_store import HybridKnowledgeStore
from pocket_genie.tools.research_study import (
    StudyDeckStore,
    html_to_clean_markdown,
    register_research_and_study_tools,
    validate_public_http_url,
)

SAMPLE_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2104.08653v1</id>
    <published>2021-04-18T14:48:42Z</published>
    <title>Diffusion Models Beat GANs on Image Synthesis</title>
    <summary>We show that classifier guidance allows diffusion models to beat GANs.</summary>
    <author><name>Prafulla Dhariwal</name></author>
    <author><name>Alexander Nichol</name></author>
    <link title="pdf"
          href="http://arxiv.org/pdf/2104.08653v1"
          rel="related"
          type="application/pdf"/>
  </entry>
</feed>
"""

SAMPLE_HTML = """
<html>
  <head><style>body { color: red; }</style></head>
  <body>
    <nav><a href="/">Home</a></nav>
    <h1>Quantization Guide</h1>
    <p>Quantization compresses weights. Use <code>int8</code> for fast matrix math.</p>
    <pre><code>def quantize(x):
    return round(x)
</code></pre>
    <ul>
      <li>Preserves accuracy</li>
      <li>Reduces VRAM</li>
    </ul>
    <footer>Copyright 2026</footer>
    <script>console.log("tracker");</script>
  </body>
</html>
"""


def test_html_to_clean_markdown() -> None:
    md = html_to_clean_markdown(SAMPLE_HTML)
    assert "# Quantization Guide" in md
    assert "`int8`" in md
    assert "```\ndef quantize(x):\n    return round(x)\n```" in md
    assert "- Preserves accuracy" in md
    assert "- Reduces VRAM" in md
    assert "color: red" not in md
    assert "Home" not in md
    assert "Copyright 2026" not in md
    assert "tracker" not in md


def test_validate_public_http_url_and_ssrf_block() -> None:
    valid = "https://export.arxiv.org/abs/2104.08653"
    assert validate_public_http_url(valid) == valid

    blocked = [
        "ftp://example.com/file.txt",
        "file:///etc/passwd",
        "http://localhost/admin",
        "http://localhost:8080/metrics",
        "http://127.0.0.1:8000/internal",
        "http://0.0.0.0:9000",
        "http://10.0.0.5/api",
        "http://192.168.1.1/router",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]:8080/status",
    ]
    for url in blocked:
        with pytest.raises(ValueError):
            validate_public_http_url(url)


def test_arxiv_search_and_url_fetch_kb_indexing(tmp_path: Path) -> None:
    registry = ToolRegistry()
    kb = HybridKnowledgeStore()

    def fake_http_get(url: str) -> str:
        if "arxiv.org" in url:
            return SAMPLE_ATOM_XML
        return SAMPLE_HTML

    register_research_and_study_tools(
        registry=registry,
        workspace_dir=tmp_path,
        kb_store=kb,
        http_get=fake_http_get,
    )

    tool = registry.get("research.search_arxiv")
    assert tool is not None
    assert tool.domain == "research"
    assert tool.mutating is False
    search_fn = tool.handler
    arxiv_raw = search_fn(query="diffusion models", max_results=1)
    arxiv_list = json.loads(arxiv_raw)
    assert len(arxiv_list) == 1
    assert arxiv_list[0]["arxiv_id"] == "2104.08653v1"
    assert arxiv_list[0]["title"] == "Diffusion Models Beat GANs on Image Synthesis"
    assert arxiv_list[0]["authors"] == ["Prafulla Dhariwal", "Alexander Nichol"]
    assert "classifier guidance" in arxiv_list[0]["summary"]
    assert arxiv_list[0]["pdf_url"] == "http://arxiv.org/pdf/2104.08653v1"

    fetch_tool = registry.get("research.fetch_url")
    assert fetch_tool is not None
    fetch_fn = fetch_tool.handler
    with pytest.raises(ValueError):
        fetch_fn(url="http://127.0.0.1:8080/secret")

    fetch_raw = fetch_fn(url="https://docs.example.com/quant", index_into_kb=True)
    fetch_data = json.loads(fetch_raw)
    assert fetch_data["indexed_chunks"] >= 1
    assert "# Quantization Guide" in fetch_data["markdown"]
    assert kb.num_chunks >= 1


def test_sm2_interval_progression_in_study_store() -> None:
    store = StudyDeckStore(":memory:")
    card = store.add_flashcard(
        topic="ML",
        front="What does SmoothQuant migrate?",
        back="Migrates activation outliers into weights via per-channel scaling.",
    )
    card_id = card["id"]
    assert card["repetitions"] == 0
    assert card["interval_days"] == 0
    assert card["easiness_factor"] == 2.5

    # Review 1: perfect recall (q=5) -> rep=1, interval=1, ef=2.6
    r1 = store.review_flashcard(card_id=card_id, quality=5)
    assert r1["repetitions"] == 1
    assert r1["interval_days"] == 1
    assert r1["easiness_factor"] == 2.6

    # Review 2: good recall (q=4) -> rep=2, interval=6, ef=2.6
    r2 = store.review_flashcard(card_id=card_id, quality=4)
    assert r2["repetitions"] == 2
    assert r2["interval_days"] == 6
    assert r2["easiness_factor"] == 2.6

    # Review 3: good recall (q=4) -> rep=3, interval=round(6 * 2.6)=16, ef=2.6
    r3 = store.review_flashcard(card_id=card_id, quality=4)
    assert r3["repetitions"] == 3
    assert r3["interval_days"] == 16
    assert r3["easiness_factor"] == 2.6

    # Review 4: failed recall (q=2) -> rep=0, interval=1, ef drops to 2.28
    r4 = store.review_flashcard(card_id=card_id, quality=2)
    assert r4["repetitions"] == 0
    assert r4["interval_days"] == 1
    assert r4["easiness_factor"] == 2.28

    # Multiple failures verify EF floor of 1.3
    for _ in range(10):
        r_fail = store.review_flashcard(card_id=card_id, quality=0)
    assert r_fail["easiness_factor"] == 1.3


def test_jev_answer_grading_and_flashcard_update(tmp_path: Path) -> None:
    registry = ToolRegistry()
    store = StudyDeckStore(":memory:")
    card = store.add_flashcard(
        topic="quant",
        front="What is AWQ?",
        back="Activation-aware Weight Quantization protecting salient weight channels.",
    )
    card_id = card["id"]

    def fake_transport(
        _url: str,
        _payload: dict[str, Any],
        _headers: dict[str, str],
        _timeout: float,
    ) -> dict[str, Any]:
        return {
            "model": "typesafe/jev-latest",
            "answers": {
                "factual_accuracy": {"type": "noul", "noul": 0.94},
                "missing_key_concept": {"type": "noul", "noul": 0.06},
                "mastery_score": {"type": "score", "score": 0.90},
            },
        }

    jev = JevDecisionClient(api_key="sk-test", transport=fake_transport)
    register_research_and_study_tools(
        registry=registry,
        workspace_dir=tmp_path,
        study_store=store,
        jev_client=jev,
    )

    grade_tool = registry.get("study.grade_answer")
    assert grade_tool is not None
    assert grade_tool.domain == "study"
    assert grade_tool.mutating is False
    grade_fn = grade_tool.handler
    res_raw = grade_fn(
        question="What is AWQ?",
        reference_answer=(
            "Activation-aware Weight Quantization protecting salient weight channels."
        ),
        user_answer="It protects salient weight channels using activation distribution stats.",
        card_id=card_id,
    )
    res = json.loads(res_raw)
    assert res["quality"] == 5
    assert res["mastery_score"] == 0.9
    assert res["factual_accuracy"] == 0.94
    assert res["card"] is not None
    assert res["card"]["repetitions"] == 1
    assert res["card"]["interval_days"] == 1

    due_tool = registry.get("study.due_flashcards")
    assert due_tool is not None
    due_fn = due_tool.handler
    # Card is scheduled for tomorrow so not due now
    due_raw = due_fn(topic="quant")
    assert len(json.loads(due_raw)) == 0

    # Due in 5 days
    due_future = due_fn(topic="quant", as_of="2099-01-01T00:00:00+00:00")
    assert len(json.loads(due_future)) == 1
