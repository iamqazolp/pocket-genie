from __future__ import annotations

import uuid
from typing import Callable

from pocket_genie.core.state import HITLApprovalRequest
from pocket_genie.middlewares.base import BaseMiddleware, ToolCallContext


class HumanInTheLoopMiddleware(BaseMiddleware):
    """Intercepts mutating tool invocations and queues approval cards for the mobile/web console."""

    name = "human_in_the_loop"

    def __init__(self) -> None:
        self._decisions: dict[str, str] = {}

    def record_decision(self, request_id: str, approved: bool) -> None:
        self._decisions[request_id] = "approved" if approved else "rejected"

    def wrap_tool(
        self,
        ctx: ToolCallContext,
        call_next: Callable[[ToolCallContext], str],
    ) -> str:
        is_mutating = bool(ctx.spec and ctx.spec.mutating)
        if not is_mutating or ctx.shared.auto_approve_hitl:
            return call_next(ctx)

        req_id = ctx.invocation.id or f"hitl_{uuid.uuid4().hex[:8]}"
        decision = self._decisions.get(req_id)
        if decision == "approved":
            for item in ctx.state.pending_hitl:
                if item.request_id == req_id:
                    item.status = "approved"
            return call_next(ctx)
        if decision == "rejected":
            for item in ctx.state.pending_hitl:
                if item.request_id == req_id:
                    item.status = "rejected"
            return f"[HITL_REJECTED] User rejected execution of mutating tool `{ctx.invocation.name}`."

        req = HITLApprovalRequest(
            request_id=req_id,
            tool_name=ctx.invocation.name,
            arguments=dict(ctx.invocation.arguments),
            specialist=ctx.specialist,
            tier=ctx.state.active_tier,
            reason=f"Tool `{ctx.invocation.name}` modifies workspace or device state.",
            status="pending",
        )
        ctx.state.pending_hitl.append(req)
        ctx.writer.emit("hitl_request", **req.to_dict())
        ctx.state.middleware_trace.append(
            {"middleware": self.name, "request_id": req_id, "tool": ctx.invocation.name}
        )
        return (
            f"[HITL_PENDING:{req_id}] Execution of `{ctx.invocation.name}` paused awaiting "
            "user approval on the mobile console."
        )
