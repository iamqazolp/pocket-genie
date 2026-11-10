from __future__ import annotations

import dataclasses
from typing import Sequence

from pocket_genie.shared import PrecisionTier


@dataclasses.dataclass(slots=True)
class SpecialistConfig:
    agent_id: str
    display_name: str
    description: str
    tool_domains: tuple[str, ...]
    default_tier: PrecisionTier = "t1"
    fallback_tiers: tuple[PrecisionTier, ...] = ("t1", "t2", "t3")
    system_prompt: str = ""


SPECIALIST_ROSTER: dict[str, SpecialistConfig] = {
    "devops_agent": SpecialistConfig(
        agent_id="devops_agent",
        display_name="DevOps & Repository Specialist",
        description="Lists files, searches code, inspects git status/diffs/logs, reads chunks, runs pytest, and applies patches.",
        tool_domains=("devops", "jev"),
        default_tier="t1",
        system_prompt=(
            "You are the DevOps & Repository Specialist. Search code, list files, and inspect git status/diffs/logs "
            "before proposing minimal patches or running TypeSafe Jev patch audits."
        ),
    ),
    "knowledge_base_agent": SpecialistConfig(
        agent_id="knowledge_base_agent",
        display_name="Knowledge Base & RAG Specialist",
        description="Performs alpha-blended BM25 + dense vector search over project files, docs, and notes.",
        tool_domains=("kb",),
        default_tier="t1",
        system_prompt=(
            "You are the Knowledge Base Specialist. Retrieve relevant document chunks using alpha=0.2 for exact "
            "identifiers and alpha=0.8 for conceptual queries, citing file_path#chunk=index."
        ),
    ),
    "sql_agent": SpecialistConfig(
        agent_id="sql_agent",
        display_name="SQL & Experiment Analytics Specialist",
        description="Loads workspace CSV/JSONL/SQLite datasets and runs read-only allowlisted SQL queries and charts.",
        tool_domains=("sql", "visualization"),
        default_tier="t2",
        system_prompt=(
            "You are the SQL Analytics Specialist. Load workspace datasets via `sql.load_dataset` or execute read-only "
            "SELECT queries against allowlisted tables and summarize numeric findings."
        ),
    ),
    "device_agent": SpecialistConfig(
        agent_id="device_agent",
        display_name="Edge Device & OS Bridge Specialist",
        description="Queries macOS/Linux/Android Termux battery, RAM/CPU telemetry, top processes, clipboard, and notifications.",
        tool_domains=("device",),
        default_tier="t1",
        system_prompt=(
            "You are the Edge Device & OS Bridge Specialist. Monitor host/mobile telemetry, inspect top processes, "
            "and manage clipboard or notifications safely."
        ),
    ),
    "red_teaming_agent": SpecialistConfig(
        agent_id="red_teaming_agent",
        display_name="Red-Teaming & Quantization Fault Specialist",
        description="Executes adversarial schema probes, Jev structured decisions, activation outlier stress tests, and fallback verification.",
        tool_domains=("sql", "kb", "jev"),
        default_tier="t2",
        system_prompt=(
            "You are the Red-Teaming & Quantization Fault Specialist. Stress-test tool schemas, outlier-sensitive "
            "layers, TypeSafe Jev structured decisions, and fallback tier transitions."
        ),
    ),
    "run_assistant_agent": SpecialistConfig(
        agent_id="run_assistant_agent",
        display_name="Workflow & Run Assistant",
        description="Monitors quantization runs, GitHub CI pipelines, Antigravity/Claude Code sessions, and Databricks jobs.",
        tool_domains=("devops", "github", "agent_ops", "databricks", "sql", "device", "jev"),
        default_tier="t1",
        system_prompt=(
            "You are the Run Assistant. Inspect GitHub PRs/CI, Antigravity and Claude Code sessions, Databricks "
            "job runs, experiment metrics, and system health to report actionable run status."
        ),
    ),
    "personal_agent": SpecialistConfig(
        agent_id="personal_agent",
        display_name="Personal Hub & Triage Specialist",
        description="Manages personal deadlines, countdowns, daily schedule/Calendar bridge, and Jev-scored message replies.",
        tool_domains=("personal", "device"),
        default_tier="t1",
        system_prompt=(
            "You are the Personal Hub Specialist. Track deadlines with live countdowns, coordinate daily agendas "
            "and native reminders, and triage incoming messages with Jev urgency/tone/priority scoring."
        ),
    ),
    "work_ops_agent": SpecialistConfig(
        agent_id="work_ops_agent",
        display_name="GitHub, Databricks & Agent-Ops Specialist",
        description="Operates across GitHub PRs/issues/CI, Databricks Unity Catalog SQL & job runs, Antigravity, and Claude Code.",
        tool_domains=("github", "databricks", "agent_ops", "devops", "jev"),
        default_tier="t2",
        system_prompt=(
            "You are the Work Ops Specialist. Inspect GitHub PRs/issues/CI, query Databricks jobs/clusters and "
            "read-only Unity Catalog SQL, and recap Antigravity and Claude Code coding sessions."
        ),
    ),
    "research_study_agent": SpecialistConfig(
        agent_id="research_study_agent",
        display_name="Research & Active-Recall Study Specialist",
        description="Searches arXiv papers, ingests web docs into local RAG, manages SM-2 flashcards, and grades study answers.",
        tool_domains=("research", "study", "kb", "jev"),
        default_tier="t2",
        system_prompt=(
            "You are the Research & Study Specialist. Search arXiv preprints, fetch and index technical docs into "
            "the knowledge base, schedule SM-2 flashcards, and grade active-recall answers with TypeSafe Jev."
        ),
    ),
}


def get_specialist_config(agent_id: str) -> SpecialistConfig:
    if agent_id not in SPECIALIST_ROSTER:
        raise KeyError(f"Unknown specialist config {agent_id!r}")
    return SPECIALIST_ROSTER[agent_id]


def list_specialist_configs(ids: Sequence[str] | None = None) -> list[SpecialistConfig]:
    if ids is None:
        return list(SPECIALIST_ROSTER.values())
    return [SPECIALIST_ROSTER[i] for i in ids if i in SPECIALIST_ROSTER]
