from __future__ import annotations

import datetime as dt
import json

from pocket_genie.core.memory import SessionMemoryStore
from pocket_genie.registry.skills import SkillRegistry
from pocket_genie.registry.tools import ToolRegistry
from pocket_genie.retrieval.hybrid_store import HybridKnowledgeStore


def register_memory_and_kb_tools(
    registry: ToolRegistry,
    memory_store: SessionMemoryStore,
    skill_registry: SkillRegistry,
    kb_store: HybridKnowledgeStore,
) -> None:
    @registry.register(
        name="memory.get_current_datetime",
        description="Retrieve current UTC wall-clock timestamp on demand (preserving static system prompt caching).",
        parameters={"type": "object", "properties": {}},
        domain="memory",
        mutating=False,
    )
    def get_current_datetime() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    @registry.register(
        name="memory.save_preference",
        description="Persist a user preference or workspace fact across sessions.",
        parameters={
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "Concise preference or project fact to remember."},
            },
            "required": ["fact"],
        },
        domain="memory",
        mutating=False,
    )
    def save_preference(fact: str) -> str:
        saved = memory_store.add_memory(fact)
        return f"Saved memory: {saved}"

    @registry.register(
        name="memory.load_skill",
        description="Load the full Markdown instructions for a named skill playbook.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill playbook identifier (e.g. repo-ci-triage)."},
            },
            "required": ["name"],
        },
        domain="memory",
        mutating=False,
    )
    def load_skill(name: str) -> str:
        spec = skill_registry.get(name)
        if spec is None:
            available = ", ".join(s.name for s in skill_registry.list_skills())
            return f"Skill {name!r} not found. Available: {available}"
        return f"# Skill: {spec.name} (recommended_tier={spec.recommended_tier})\n\n{spec.body}"

    @registry.register(
        name="kb.search_workspace",
        description="Hybrid BM25 + dense vector search over indexed workspace files, docs, and notes.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "alpha": {
                    "type": "number",
                    "description": "Blending factor: 0.0-0.3 for lexical/exact symbols, 0.7-1.0 for semantic.",
                },
                "top_k": {"type": "integer", "description": "Maximum number of chunks to return."},
            },
            "required": ["query"],
        },
        domain="kb",
        mutating=False,
    )
    def search_workspace(query: str, alpha: float = 0.5, top_k: int = 4) -> str:
        hits = kb_store.search(query=query, top_k=int(top_k), alpha=float(alpha))
        payload = [
            {
                "citation": h.chunk.citation,
                "rerank_score": h.rerank_score,
                "bm25_score": h.bm25_score,
                "dense_score": h.dense_score,
                "text": h.rendered_text,
            }
            for h in hits
        ]
        return json.dumps({"query": query, "alpha": alpha, "results": payload}, indent=2)
