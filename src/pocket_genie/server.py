from __future__ import annotations

import datetime as dt
import json
import socket
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

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
    """Stateful multi-agent service powering the SSE mobile console, HITL approval inbox, and CLI."""

    def __init__(self, workspace_dir: Path | None = None) -> None:
        self.workspace_dir = (workspace_dir or Path.cwd()).resolve()
        self.session_created_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.tool_registry = ToolRegistry()
        self.env_objects = register_all_builtin_tools(self.tool_registry, self.workspace_dir)
        self.mcp_bridge = MCPBridge(self.tool_registry)
        self.factory = TieredAgentFactory()
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

        return {
            "reply": reply.content,
            "active_tier": self.state.active_tier,
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


_MOBILE_CONSOLE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1"/>
<title>Pocket-Genie Edge Command Deck</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #131b2e;
    --border: #23304d;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --ok: #34d399;
    --warn: #fbbf24;
    --err: #f87171;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); padding: 14px; max-width: 780px; margin-inline: auto;
  }
  header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  h1 { font-size: 1.15rem; margin: 0; letter-spacing: -0.02em; }
  .badge { font-size: 0.75rem; padding: 3px 8px; border-radius: 999px; background: #1e293b; color: var(--accent); border: 1px solid var(--border); }
  .tabs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 12px; }
  .tab-btn {
    background: var(--panel); color: var(--muted); border: 1px solid var(--border);
    padding: 9px 6px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; cursor: pointer;
  }
  .tab-btn.active { color: var(--text); border-color: var(--accent); background: #172542; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px; margin-bottom: 12px; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
  .chip {
    background: #1e293b; color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 9px; font-size: 0.78rem; cursor: pointer;
  }
  textarea, select {
    width: 100%; background: #090d16; color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 9px; font-size: 0.88rem; margin-bottom: 8px;
  }
  .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
  .btn-row { display: flex; gap: 8px; }
  .btn {
    flex: 1; padding: 10px; border-radius: 8px; border: none; font-weight: 600;
    background: var(--accent); color: #090d16; cursor: pointer; font-size: 0.88rem;
  }
  .btn.secondary { background: #1e293b; color: var(--text); border: 1px solid var(--border); flex: 0.35; }
  .hud-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
  .metric { background: #090d16; border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
  .metric .lbl { font-size: 0.72rem; color: var(--muted); }
  .metric .val { font-size: 1.02rem; font-weight: 700; color: var(--ok); margin-top: 2px; }
  pre {
    background: #090d16; border: 1px solid var(--border); border-radius: 8px;
    padding: 10px; overflow-x: auto; font-size: 0.78rem; white-space: pre-wrap; margin: 6px 0 0;
  }
  .hidden { display: none; }
  .hitl-item { border-left: 3px solid var(--warn); padding-left: 10px; margin-bottom: 10px; }
</style>
</head>
<body>
<header>
  <h1>Pocket-Genie Command Deck</h1>
  <span id="tier-badge" class="badge">t1 · P_fast (W4A8)</span>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('stream')">1. Genie Stream</button>
  <button class="tab-btn" onclick="switchTab('hitl')">2. HITL Inbox (<span id="hitl-count">0</span>)</button>
  <button class="tab-btn" onclick="switchTab('hud')">3. Quant & MCP HUD</button>
</div>

<div id="tab-stream">
  <div class="card">
    <div class="chips">
      <button class="chip" onclick="setPreset('Check git_status and summarize repository state', 'supervisor', 'none')">Repo Git Status</button>
      <button class="chip" onclick="setPreset('SELECT recipe_id, footprint_mb, values_exact_rate FROM ablation_results', 'direct-specialist', 'none', 'sql_agent')">SQL Ablation Query</button>
      <button class="chip" onclick="setPreset('Search workspace for kurtosis 542.6 and SmoothQuant', 'direct-specialist', 'none', 'knowledge_base_agent')">Hybrid RAG Search</button>
      <button class="chip" onclick="setPreset('Dispatch build alert notification to phone', 'supervisor', 'hitl_patch')">Trigger Phone HITL Gate</button>
      <button class="chip" onclick="setPreset('Audit repository under W4A8 JSON syntax fault', 'supervisor', 'corrupted_json')">Simulate W4A8 -> P_mixed Recovery</button>
    </div>
    <textarea id="prompt" rows="2" placeholder="Ask Supervisor or Specialist...">Check git_status and summarize repository state</textarea>
    <div class="controls">
      <select id="agent_type">
        <option value="supervisor">Mode: Supervisor (Hierarchical)</option>
        <option value="scoped-demo">Mode: Scoped Demo (Read-Only)</option>
        <option value="direct-specialist">Mode: Direct Specialist</option>
        <option value="red-agent">Mode: Red-Team Fault Agent</option>
        <option value="run-assistant">Mode: Run Assistant</option>
      </select>
      <select id="fault">
        <option value="none">Fault: None (Clean P_fast)</option>
        <option value="corrupted_json">Fault: Corrupted JSON (t1 -> t2)</option>
        <option value="refusal">Fault: Refusal + Corrupt (t1 -> t3)</option>
        <option value="hitl_patch">Fault: Mutating Tool (HITL Gate)</option>
      </select>
    </div>
    <div class="btn-row">
      <button class="btn" onclick="runGenie()">Run Turn</button>
      <button class="btn secondary" onclick="resetSession()">Reset</button>
    </div>
  </div>

  <div class="card">
    <strong>Response & Specialist Observations</strong>
    <pre id="reply-box">Ready.</pre>
    <strong style="display:block;margin-top:10px;">Live Event Stream (Reasoning · Chunks · Middlewares)</strong>
    <pre id="events-box">[]</pre>
  </div>
</div>

<div id="tab-hitl" class="hidden">
  <div class="card">
    <strong>Pending Human-in-the-Loop Approvals</strong>
    <div id="hitl-list" style="margin-top:8px;color:var(--muted);">No pending mutating tool requests.</div>
  </div>
</div>

<div id="tab-hud" class="hidden">
  <div class="card">
    <strong>Quantization & KV-Cache Telemetry</strong>
    <div class="hud-grid" style="margin-top:8px;">
      <div class="metric"><div class="lbl">Active Tier / Profile</div><div class="val" id="m-profile">t1 / P_fast</div></div>
      <div class="metric"><div class="lbl">Effective Act Bits</div><div class="val" id="m-bits">8.00 bits</div></div>
      <div class="metric"><div class="lbl">Promoted Token Ratio</div><div class="val" id="m-promo">0.0%</div></div>
      <div class="metric"><div class="lbl">BOS-Sink INT4 KV Saved</div><div class="val" id="m-kv">3.40 MB</div></div>
    </div>
    <strong style="display:block;margin-top:12px;">Registered Skills & Specialists</strong>
    <pre id="hud-box">Loading...</pre>
  </div>
</div>

<script>
let currentSpecialist = null;
function switchTab(name) {
  ['stream', 'hitl', 'hud'].forEach((t, i) => {
    document.getElementById('tab-' + t).classList.toggle('hidden', t !== name);
    document.querySelectorAll('.tab-btn')[i].classList.toggle('active', t === name);
  });
}
function setPreset(p, mode, fault, spec=null) {
  document.getElementById('prompt').value = p;
  document.getElementById('agent_type').value = mode;
  document.getElementById('fault').value = fault;
  currentSpecialist = spec;
}
function renderState(st) {
  const q = st.quant_telemetry || {};
  document.getElementById('tier-badge').textContent = `${st.active_tier} · ${q.profile || 'P_fast'} (${q.effective_act_bits || 8}b)`;
  document.getElementById('m-profile').textContent = `${st.active_tier} / ${q.profile || 'P_fast'}`;
  document.getElementById('m-bits').textContent = `${q.effective_act_bits || 8.0} bits`;
  document.getElementById('m-promo').textContent = `${((q.promoted_token_ratio || 0) * 100).toFixed(1)}%`;
  document.getElementById('m-kv').textContent = `${q.kv_memory_saved_mb || 0.0} MB`;
  const pending = (st.pending_hitl || []).filter(h => h.status === 'pending');
  document.getElementById('hitl-count').textContent = pending.length;
  const listEl = document.getElementById('hitl-list');
  if (!pending.length) {
    listEl.innerHTML = 'No pending mutating tool requests.';
  } else {
    listEl.innerHTML = pending.map(h => `
      <div class="hitl-item">
        <div><strong>${h.tool_name}</strong> (${h.specialist}, tier=${h.tier})</div>
        <pre>${JSON.stringify(h.arguments, null, 2)}</pre>
        <div class="btn-row" style="margin-top:6px;">
          <button class="btn" onclick="decideHitl('${h.request_id}', true)">Approve & Execute</button>
          <button class="btn secondary" onclick="decideHitl('${h.request_id}', false)">Reject</button>
        </div>
      </div>
    `).join('');
  }
  document.getElementById('hud-box').textContent = JSON.stringify({
    specialists: st.specialists,
    skills: st.skills,
    middleware_trace: st.middleware_trace
  }, null, 2);
}
async function runGenie() {
  const payload = {
    prompt: document.getElementById('prompt').value,
    agent_type: document.getElementById('agent_type').value,
    target_specialist: currentSpecialist,
    fault: document.getElementById('fault').value
  };
  const r = await fetch('/api/run', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  const data = await r.json();
  document.getElementById('reply-box').textContent = data.reply;
  document.getElementById('events-box').textContent = JSON.stringify(data.events, null, 2);
  renderState(data.state);
}
async function decideHitl(id, approved) {
  const r = await fetch('/api/hitl', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({request_id: id, approved})});
  const data = await r.json();
  document.getElementById('reply-box').textContent = data.output;
  renderState(data.state);
  switchTab('stream');
}
async function resetSession() {
  const r = await fetch('/api/reset', {method: 'POST'});
  const st = await r.json();
  document.getElementById('reply-box').textContent = 'Session reset.';
  document.getElementById('events-box').textContent = '[]';
  renderState(st);
}
fetch('/api/state').then(r => r.json()).then(renderState);
</script>
</body>
</html>
"""


def create_http_server(
    host: str = "0.0.0.0",
    port: int = 8090,
    service: PocketGenieService | None = None,
) -> ThreadingHTTPServer:
    svc = service or PocketGenieService()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return None

        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                body = _MOBILE_CONSOLE_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/state":
                self._send_json(svc.get_state_summary())
                return

            if parsed.path == "/api/stream":
                qs = urllib.parse.parse_qs(parsed.query)
                prompt = qs.get("prompt", ["Check status"])[0]
                agent_type = qs.get("agent_type", ["supervisor"])[0]
                fault = qs.get("fault", ["none"])[0]

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                def _push_sse(ev: StreamEvent) -> None:
                    line = f"event: {ev.event_type}\ndata: {json.dumps(ev.data)}\n\n"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()

                svc.run_turn(prompt=prompt, agent_type=agent_type, fault=fault, on_event=_push_sse)  # type: ignore[arg-type]
                return

            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            body = json.loads(raw or "{}")

            if self.path == "/api/run":
                res = svc.run_turn(
                    prompt=str(body.get("prompt", "")),
                    agent_type=body.get("agent_type", "supervisor"),
                    target_specialist=body.get("target_specialist"),
                    tier=body.get("tier", "t1"),
                    fault=str(body.get("fault", "none")),
                )
                self._send_json(res)
                return

            if self.path == "/api/hitl":
                res = svc.resolve_hitl(
                    request_id=str(body["request_id"]),
                    approved=bool(body.get("approved", False)),
                )
                self._send_json(res)
                return

            if self.path == "/api/reset":
                self._send_json(svc.reset_session())
                return

            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    return ThreadingHTTPServer((host, port), Handler)
