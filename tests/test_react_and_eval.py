from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pocket_genie.cli import main
from pocket_genie.core import GenieExecutionGraph, GenieState, TieredAgentFactory
from pocket_genie.registry import (
    DeterministicReplayBackend,
    OpenAICompatibleBackend,
    ToolRegistry,
)
from pocket_genie.server import PocketGenieService
from pocket_genie.shared import ChatMessage, GenieSharedContext, ToolInvocation
from pocket_genie.tools.sql_workspace import WorkspaceSQLStore


def test_multistep_react_specialist_chains_tools_and_synthesizes():
    reg = ToolRegistry()

    @reg.register(
        name="devops.git_status",
        description="Check git status",
        parameters={"type": "object", "properties": {}},
        domain="devops",
    )
    def git_status() -> str:
        return "modified: src/app.py"

    @reg.register(
        name="devops.git_diff",
        description="Inspect git diff",
        parameters={"type": "object", "properties": {}},
        domain="devops",
    )
    def git_diff() -> str:
        return "+added_line = True"

    b_t1 = DeterministicReplayBackend(
        tier="t1",
        scripted_responses=[
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[ToolInvocation(id="c1", name="devops.git_status", arguments={})],
            ),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[ToolInvocation(id="c2", name="devops.git_diff", arguments={})],
            ),
            ChatMessage(
                role="assistant",
                content="Chained summary: src/app.py modified with +added_line = True.",
            ),
        ],
    )
    factory = TieredAgentFactory()
    factory.backends.register("t1", b_t1)

    graph = GenieExecutionGraph(factory=factory, tool_registry=reg)
    state = GenieState(
        session_id="react-01",
        session_created_at="2026-11-10T10:15:00Z",
        messages=[ChatMessage(role="user", content="Check status then diff")],
    )
    shared = GenieSharedContext(
        session_id="react-01",
        session_created_at="2026-11-10T10:15:00Z",
        workspace_dir=Path("."),
        project_id="pocket-genie",
        agent_type="direct-specialist",
        target_specialist="devops_agent",
    )
    reply, stream = graph.invoke(state, shared)
    assert "Chained summary: src/app.py modified with +added_line = True." in reply.content
    tool_events = [e for e in stream.events if e.event_type == "tool_message"]
    assert len(tool_events) == 2


def test_openai_compatible_backend_parses_real_logprobs_for_token_router():
    class MockLLMHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return None

        def do_POST(self):
            payload = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Real logprob answer"},
                        "logprobs": {
                            "content": [
                                {"token": "Real", "top_logprobs": [{"logprob": -0.02}, {"logprob": -4.5}]},
                                {"token": " logprob", "top_logprobs": [{"logprob": -0.60}, {"logprob": -0.95}]},
                                {"token": " answer", "top_logprobs": [{"logprob": -0.05}, {"logprob": -3.8}]},
                            ]
                        },
                    }
                ],
                "usage": {"completion_tokens": 3},
            }
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockLLMHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        backend = OpenAICompatibleBackend(base_url=f"http://127.0.0.1:{port}/v1", model="mock-qwen", tier="t1")
        out = backend.generate([ChatMessage(role="user", content="Hi")])
        assert out.message.content == "Real logprob answer"
        assert out.telemetry.total_tokens == 3
        assert out.telemetry.promoted_tokens == 1
        assert abs(out.telemetry.min_logit_margin - 0.35) < 1e-4
    finally:
        srv.shutdown()
        srv.server_close()


def test_eval_cli_and_sql_artifact_ingestion(tmp_path: Path, capsys):
    out_file = tmp_path / "benchmarks" / "latest_eval.json"
    rc = main(
        [
            "eval",
            "--backend",
            "replay",
            "--max-tasks",
            "6",
            "--workspace-dir",
            str(tmp_path),
            "--output",
            str(out_file),
        ]
    )
    assert rc == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["num_tasks"] == 6
    assert out_file.exists()

    sql_store = WorkspaceSQLStore(workspace_dir=tmp_path)
    rows = sql_store.execute_select(
        "SELECT recipe_id, json_valid_rate FROM ablation_results WHERE recipe_id = 'replay_P_mixed_FSM'"
    )
    assert len(rows) == 1
    assert rows[0]["json_valid_rate"] == 1.0

    assert main(["graph"]) == 0
    mermaid_out = capsys.readouterr().out
    assert "route_dispatch" in mermaid_out

    svc = PocketGenieService(workspace_dir=tmp_path)
    turn = svc.run_turn(prompt="Alert phone", fault="hitl_patch")
    assert turn["approval_request"] is not None
    res = svc.resolve_hitl("hitl_notify_01", approved=True)
    assert "assistant_reply" in res
    assert res["state"]["observability"]["trace_count"] >= 1
