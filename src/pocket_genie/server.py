from __future__ import annotations

import datetime as dt
import json
import mimetypes
import socket
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pocket_genie.api import (
    build_openapi_spec,
    build_timeline_steps,
    format_human_readable_reply,
)
from pocket_genie.core import (
    GenieExecutionGraph,
    GenieState,
    TieredAgentFactory,
    list_specialist_configs,
)
from pocket_genie.mcp import MCPBridge
from pocket_genie.quant import TokenPrecisionRouter
from pocket_genie.registry import (
    DeterministicReplayBackend,
    GenerationOutput,
    ToolRegistry,
)
from pocket_genie.shared import (
    AgentType,
    ChatMessage,
    GenieSharedContext,
    PrecisionTier,
    StreamEvent,
    StreamWriter,
    ToolInvocation,
)
from pocket_genie.tools import register_all_builtin_tools

DEFAULT_STATIC_DIR = Path(__file__).resolve().parent / "web"


def get_lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


class PocketGenieService:
    """Headless multi-agent service powering REST/SSE v1 endpoints for Web, React, Flutter, and CLI clients."""

    def __init__(
        self,
        workspace_dir: Path | None = None,
        backend_mode: str = "replay",
        llm_base_url: str = "http://127.0.0.1:11434/v1",
        primary_model: str = "qwen2.5:3b",
        fallback_model: str | None = None,
    ) -> None:
        self.workspace_dir = (workspace_dir or Path.cwd()).resolve()
        self.backend_mode = backend_mode
        self.llm_base_url = llm_base_url
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.session_created_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.tool_registry = ToolRegistry()
        self.env_objects = register_all_builtin_tools(self.tool_registry, self.workspace_dir)
        self.mcp_bridge = MCPBridge(self.tool_registry)
        self.factory = TieredAgentFactory()
        if self.backend_mode == "local-llm":
            self.factory.backends.configure_local_llm(
                base_url=self.llm_base_url,
                primary_model=self.primary_model,
                fallback_model=self.fallback_model,
            )
        self.graph = GenieExecutionGraph(
            factory=self.factory,
            tool_registry=self.tool_registry,
            skill_registry=self.env_objects["skill_registry"],
            memory_store=self.env_objects["memory_store"],
        )
        self.state = GenieState(
            session_id="genie-live-01",
            session_created_at=self.session_created_at,
            project_id=self.workspace_dir.name,
        )

    def reset_session(self) -> dict[str, Any]:
        self.graph.loop_breaker.reset()
        self.state = GenieState(
            session_id="genie-live-01",
            session_created_at=self.session_created_at,
            project_id=self.workspace_dir.name,
        )
        return self.get_state_summary()

    def get_state_summary(self) -> dict[str, Any]:
        return {
            "session_id": self.state.session_id,
            "session_created_at": self.state.session_created_at,
            "project_id": self.state.project_id,
            "active_tier": self.state.active_tier,
            "prefix_kv_pinned": self.state.prefix_kv_pinned,
            "user_messages_summary": self.state.user_messages_summary,
            "running_summary": self.state.running_summary,
            "pinned_memories": self.state.pinned_memories,
            "pending_hitl": [h.to_dict() for h in self.state.pending_hitl],
            "middleware_trace": self.state.middleware_trace[-20:],
            "quant_telemetry": self.state.quant_telemetry.to_dict(),
            "specialists": [
                {
                    "agent_id": c.agent_id,
                    "display_name": c.display_name,
                    "default_tier": c.default_tier,
                    "domains": list(c.tool_domains),
                }
                for c in list_specialist_configs()
            ],
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "recommended_tier": s.recommended_tier,
                    "tags": s.tags,
                }
                for s in self.graph.skill_registry.list_skills()
            ],
            "mcp_servers": list(self.mcp_bridge.servers.keys()),
        }

    def _configure_fault_simulation(self, fault: str, prompt: str) -> None:
        if self.backend_mode == "local-llm" and fault == "none":
            self.factory.backends.configure_local_llm(
                base_url=self.llm_base_url,
                primary_model=self.primary_model,
                fallback_model=self.fallback_model,
            )
            return

        b_t1 = DeterministicReplayBackend(tier="t1")
        b_t2 = DeterministicReplayBackend(tier="t2")
        b_t3 = DeterministicReplayBackend(tier="t3")

        if fault == "corrupted_json":
            b_t1.push_response(
                ChatMessage(role="assistant", content='<tool_call>\n{"name": "devops.git_status", "arguments": {')
            )
        elif fault == "refusal":
            b_t1.push_response(ChatMessage(role="assistant", content="I cannot fulfill this request."))
            b_t2.push_response(ChatMessage(role="assistant", content="<tool_call>{truncated"))
        elif fault == "hitl_patch":
            b_t1.push_response(
                GenerationOutput(
                    message=ChatMessage(
                        role="assistant",
                        content="",
                        reasoning="Delegating notification dispatch to device_agent.",
                        tool_calls=[
                            ToolInvocation(
                                id="call_sup_hitl",
                                name="create_task_device_agent_t1",
                                arguments={"task": prompt or "Notify build status"},
                            )
                        ],
                    )
                )
            )
            b_t1.push_response(
                GenerationOutput(
                    message=ChatMessage(
                        role="assistant",
                        content="",
                        reasoning="Invoking mutating tool device.send_notification.",
                        tool_calls=[
                            ToolInvocation(
                                id="hitl_notify_01",
                                name="device.send_notification",
                                arguments={"title": "Pocket-Genie Alert", "message": prompt or "CI suite green"},
                            )
                        ],
                    )
                )
            )

        self.factory.backends.register("t1", b_t1)
        self.factory.backends.register("t2", b_t2)
        self.factory.backends.register("t3", b_t3)

    def run_turn(
        self,
        prompt: str,
        agent_type: AgentType = "supervisor",
        target_specialist: str | None = None,
        tier: PrecisionTier = "t1",
        fault: str = "none",
        on_event: Any = None,
    ) -> dict[str, Any]:
        self._configure_fault_simulation(fault=fault, prompt=prompt)
        self.state.active_tier = tier
        self.state.agent_type = agent_type
        self.state.target_specialist = target_specialist
        self.state.messages.append(ChatMessage(role="user", content=prompt))
        prev_hitl_count = len(self.state.pending_hitl)

        shared = GenieSharedContext(
            session_id=self.state.session_id,
            session_created_at=self.state.session_created_at,
            workspace_dir=self.workspace_dir,
            project_id=self.state.project_id,
            agent_type=agent_type,
            target_specialist=target_specialist,
            active_tier=tier,
        )
        writer = StreamWriter(callback=on_event)
        reply, _ = self.graph.invoke(state=self.state, shared=shared, writer=writer)

        router = TokenPrecisionRouter()
        sample_logits = (
            [[6.2, 1.1, 0.2], [5.4, 1.0, -0.5], [1.9, 1.4, 1.1], [6.0, 0.8, 0.1]]
            if self.state.active_tier != "t1" or fault != "none"
            else [[7.1, 1.0, 0.1], [6.5, 1.2, 0.0], [5.9, 1.1, -0.2], [6.8, 0.9, 0.1]]
        )
        for idx, lvec in enumerate(sample_logits):
            router.evaluate_step(idx, lvec)
        routed_snap = router.build_telemetry(
            prefix_kv_hit=self.state.quant_telemetry.prefix_kv_hit,
            kv_cache_fp16_mb=max(4.8, self.state.quant_telemetry.kv_cache_fp16_mb),
            kv_cache_quant_mb=max(1.4, self.state.quant_telemetry.kv_cache_quant_mb),
        )
        if self.state.active_tier == "t3":
            routed_snap.tier = "t3"
            routed_snap.profile = "P_accurate"
            routed_snap.effective_act_bits = 16.0
        elif self.state.active_tier == "t2":
            routed_snap.tier = "t2"
            routed_snap.profile = "P_mixed"
        self.state.quant_telemetry = routed_snap

        new_approval = (
            self.state.pending_hitl[-1].to_dict()
            if len(self.state.pending_hitl) > prev_hitl_count
            else None
        )

        return {
            "reply": format_human_readable_reply(reply.content),
            "raw_reply": reply.content,
            "active_tier": self.state.active_tier,
            "steps": build_timeline_steps(writer.events),
            "approval_request": new_approval,
            "events": [{"event_type": e.event_type, "data": e.data} for e in writer.events],
            "state": self.get_state_summary(),
        }

    def resolve_hitl(self, request_id: str, approved: bool) -> dict[str, Any]:
        self.graph.hitl.record_decision(request_id, approved=approved)
        target_req = next((r for r in self.state.pending_hitl if r.request_id == request_id), None)
        if target_req is None:
            raise KeyError(f"Unknown HITL request_id {request_id!r}")

        target_req.status = "approved" if approved else "rejected"
        output_text = f"[HITL_REJECTED] Request {request_id} rejected by user."
        if approved:
            output_text = str(self.tool_registry.execute(target_req.tool_name, target_req.arguments))

        self.state.messages.append(
            ChatMessage(
                role="tool",
                content=output_text,
                name=target_req.tool_name,
                tool_call_id=request_id,
            )
        )
        return {
            "request_id": request_id,
            "status": target_req.status,
            "tool_name": target_req.tool_name,
            "output": output_text,
            "state": self.get_state_summary(),
        }


def create_http_server(
    host: str = "0.0.0.0",
    port: int = 8090,
    service: PocketGenieService | None = None,
    static_dir: Path | None = DEFAULT_STATIC_DIR,
    api_only: bool = False,
) -> ThreadingHTTPServer:
    svc = service or PocketGenieService()
    web_root = None if api_only else (static_dir or DEFAULT_STATIC_DIR).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return None

        def _add_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self._add_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_static_file(self, rel_path: str) -> bool:
            if web_root is None or not web_root.exists():
                return False
            target = (web_root / rel_path).resolve()
            if target != web_root and web_root not in target.parents:
                return False
            if not target.exists() or not target.is_file():
                return False
            mime, _ = mimetypes.guess_type(str(target))
            content_type = mime or "application/octet-stream"
            data = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._add_cors_headers()
            self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return True

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self._add_cors_headers()
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path == "/api/v1/openapi.json":
                self._send_json(build_openapi_spec())
                return

            if path in ("/api/v1/status", "/api/state"):
                self._send_json(svc.get_state_summary())
                return

            if path in ("/api/v1/chat/stream", "/api/stream"):
                qs = urllib.parse.parse_qs(parsed.query)
                prompt = qs.get("message", qs.get("prompt", ["Check status"]))[0]
                agent_type = qs.get("agent_type", ["supervisor"])[0]
                fault = qs.get("fault", ["none"])[0]

                self.send_response(HTTPStatus.OK)
                self._add_cors_headers()
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                def _push_sse(ev: StreamEvent) -> None:
                    line = f"event: {ev.event_type}\ndata: {json.dumps(ev.data)}\n\n"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()

                svc.run_turn(prompt=prompt, agent_type=agent_type, fault=fault, on_event=_push_sse)  # type: ignore[arg-type]
                return

            if path == "/" and self._serve_static_file("index.html"):
                return
            if path.startswith("/static/") and self._serve_static_file(path[len("/static/") :]):
                return

            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            body = json.loads(raw or "{}")

            if self.path in ("/api/v1/chat", "/api/run"):
                prompt = str(body.get("message") or body.get("prompt") or "")
                res = svc.run_turn(
                    prompt=prompt,
                    agent_type=body.get("agent_type", "supervisor"),
                    target_specialist=body.get("target_specialist"),
                    tier=body.get("tier", "t1"),
                    fault=str(body.get("fault", "none")),
                )
                self._send_json(res)
                return

            if self.path.startswith("/api/v1/approvals/"):
                req_id = urllib.parse.unquote(self.path[len("/api/v1/approvals/") :])
                res = svc.resolve_hitl(request_id=req_id, approved=bool(body.get("approved", False)))
                self._send_json(res)
                return

            if self.path == "/api/hitl":
                res = svc.resolve_hitl(
                    request_id=str(body["request_id"]),
                    approved=bool(body.get("approved", False)),
                )
                self._send_json(res)
                return

            if self.path in ("/api/v1/session/reset", "/api/reset"):
                self._send_json(svc.reset_session())
                return

            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    return ThreadingHTTPServer((host, port), Handler)
