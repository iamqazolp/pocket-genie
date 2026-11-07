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
        description="Inspects git status/diffs, reads code chunks, runs pytest suites, and proposes patches.",
        tool_domains=("devops",),
        default_tier="t1",
        system_prompt=(
            "You are the DevOps & Repository Specialist. Inspect git status, diffs, and targeted file chunks "
            "before proposing minimal patches."
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
        description="Runs read-only allowlist-validated SQL queries over quantization tables, taxonomies, and CSVs.",
        tool_domains=("sql", "visualization"),
        default_tier="t2",
        system_prompt=(
            "You are the SQL Analytics Specialist. Execute read-only SELECT queries against allowlisted tables "
            "(ablation_results, failure_taxonomy, layer_telemetry, session_events) and summarize numeric findings."
        ),
    ),
    "device_agent": SpecialistConfig(
        agent_id="device_agent",
        display_name="Edge Device & Termux Specialist",
        description="Queries Android Termux hardware APIs (battery, notifications, clipboard, TTS) and host telemetry.",
        tool_domains=("device",),
        default_tier="t1",
        system_prompt=(
            "You are the Edge Device & Termux Specialist. Monitor host/mobile telemetry and dispatch notifications "
            "or clipboard updates safely."
        ),
    ),
    "red_teaming_agent": SpecialistConfig(
        agent_id="red_teaming_agent",
        display_name="Red-Teaming & Quantization Fault Specialist",
        description="Executes adversarial schema probes, activation outlier stress tests, and fallback verification.",
        tool_domains=("sql", "kb"),
        default_tier="t2",
        system_prompt=(
            "You are the Red-Teaming & Quantization Fault Specialist. Stress-test tool schemas, outlier-sensitive "
            "layers, and fallback tier transitions."
        ),
    ),
    "run_assistant_agent": SpecialistConfig(
        agent_id="run_assistant_agent",
        display_name="Workflow & Run Assistant",
        description="Monitors quantization runs, CI pipelines, and benchmark quality gates.",
        tool_domains=("devops", "sql", "device"),
        default_tier="t1",
        system_prompt=(
            "You are the Run Assistant. Inspect experiment metrics, test suites, and system health to report "
            "actionable run status."
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
