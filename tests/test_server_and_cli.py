from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

from pocket_genie.cli import main
from pocket_genie.server import PocketGenieService, create_http_server


def test_http_server_endpoints_sse_and_hitl_approval(tmp_path: Path):
    svc = PocketGenieService(workspace_dir=tmp_path)
    server = create_http_server(host="127.0.0.1", port=0, service=svc)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base}/") as resp:
            html = resp.read().decode("utf-8")
        assert "Pocket-Genie Command Deck" in html

        req_hitl = urllib.request.Request(
            f"{base}/api/run",
            data=json.dumps({"prompt": "Alert phone", "fault": "hitl_patch"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_hitl) as resp:
            run_data = json.loads(resp.read().decode("utf-8"))
        assert "[HITL_PENDING:hitl_notify_01]" in run_data["reply"]
        assert len(run_data["state"]["pending_hitl"]) == 1

        req_approve = urllib.request.Request(
            f"{base}/api/hitl",
            data=json.dumps({"request_id": "hitl_notify_01", "approved": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_approve) as resp:
            hitl_data = json.loads(resp.read().decode("utf-8"))
        assert hitl_data["status"] == "approved"
        assert "Notification dispatched" in hitl_data["output"]

        with urllib.request.urlopen(f"{base}/api/stream?prompt=git_status&fault=corrupted_json") as resp:
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
