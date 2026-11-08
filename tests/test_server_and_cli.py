from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

from pocket_genie.cli import main
from pocket_genie.server import PocketGenieService, create_http_server


def test_http_server_v1_api_cors_openapi_and_hitl(tmp_path: Path):
    svc = PocketGenieService(workspace_dir=tmp_path)
    server = create_http_server(host="127.0.0.1", port=0, service=svc)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base}/") as resp:
            html = resp.read().decode("utf-8")
            cors = resp.headers.get("Access-Control-Allow-Origin")
        assert "Pocket-Genie" in html
        assert cors == "*"

        with urllib.request.urlopen(f"{base}/static/app.js") as resp:
            js_code = resp.read().decode("utf-8")
        assert "class PocketGenieClient" in js_code

        with urllib.request.urlopen(f"{base}/api/v1/openapi.json") as resp:
            openapi = json.loads(resp.read().decode("utf-8"))
        assert openapi["openapi"] == "3.1.0"
        assert "/api/v1/chat" in openapi["paths"]

        req_sql = urllib.request.Request(
            f"{base}/api/v1/chat",
            data=json.dumps({"message": "Compare quantization recipes from ablation_results"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_sql) as resp:
            sql_turn = json.loads(resp.read().decode("utf-8"))
        assert "E4s" in sql_turn["reply"]
        assert len(sql_turn["steps"]) >= 2

        req_hitl = urllib.request.Request(
            f"{base}/api/v1/chat",
            data=json.dumps({"message": "Alert phone", "fault": "hitl_patch"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_hitl) as resp:
            run_data = json.loads(resp.read().decode("utf-8"))
        assert run_data["approval_request"] is not None
        assert run_data["approval_request"]["request_id"] == "hitl_notify_01"

        req_approve = urllib.request.Request(
            f"{base}/api/v1/approvals/hitl_notify_01",
            data=json.dumps({"approved": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_approve) as resp:
            hitl_data = json.loads(resp.read().decode("utf-8"))
        assert hitl_data["status"] == "approved"
        assert "Notification dispatched" in hitl_data["output"]

        with urllib.request.urlopen(f"{base}/api/v1/chat/stream?message=git_status&fault=corrupted_json") as resp:
            sse_text = resp.read().decode("utf-8")
        assert "event: middleware_event" in sse_text
        assert "event: final" in sse_text
    finally:
        server.shutdown()
        server.server_close()


def test_cli_commands(capsys):
    assert main(["skills"]) == 0
    out_skills = capsys.readouterr().out
    assert "repo-ci-triage" in out_skills

    assert main(["specialists"]) == 0
    out_specs = capsys.readouterr().out
    assert "devops_agent" in out_specs

    assert main(["bench"]) == 0
    out_bench = json.loads(capsys.readouterr().out)
    assert out_bench["profiles"]["t2"]["name"] == "P_mixed"
    assert out_bench["kv_cache_2048_tokens"]["saved_mb"] > 0
