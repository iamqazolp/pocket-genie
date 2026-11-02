from __future__ import annotations

from pocket_genie.middlewares.base import BaseMiddleware, ModelCallContext
from pocket_genie.prompts.supervisor_prompt import render_supervisor_system_prompt
from pocket_genie.registry.skills import SkillRegistry
from pocket_genie.shared import ChatMessage


class DynamicSupervisorPromptMiddleware(BaseMiddleware):
    """Injects prefix-cacheable system prompt with pinned session metadata and skill catalog."""

    name = "dynamic_supervisor_prompt"

    def __init__(self, skill_registry: SkillRegistry | None = None) -> None:
        self._skills = skill_registry or SkillRegistry()

    def before_model(self, ctx: ModelCallContext) -> None:
        if ctx.messages and ctx.messages[0].role == "system":
            return
        sys_prompt = render_supervisor_system_prompt(
            session_created_at=ctx.shared.session_created_at,
            project_id=ctx.shared.project_id,
            pinned_memories=ctx.state.pinned_memories,
            skill_catalog=self._skills.render_catalog(),
            running_summary=ctx.state.running_summary,
        )
        ctx.messages.insert(
            0,
            ChatMessage(
                role="system",
                content=sys_prompt,
                metadata={"prefix_cacheable": True},
            ),
        )
        ctx.state.prefix_kv_pinned = True
