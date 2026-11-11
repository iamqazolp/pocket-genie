from __future__ import annotations

import argparse
import json
from pathlib import Path

from pocket_genie.cli_daily import (
    build_daily_briefing,
    format_briefing_markdown,
    run_vibe_repl,
)
from pocket_genie.core import GenieExecutionGraph, list_specialist_configs
from pocket_genie.eval import run_eval_suite
from pocket_genie.mcp import MCPBridge
from pocket_genie.quant import (
    BOSSinkINT4KVCache,
    TokenPrecisionRouter,
    build_sensitivity_profiles,
)
from pocket_genie.registry import SkillRegistry, ToolRegistry
from pocket_genie.server import PocketGenieService, create_http_server, get_lan_ip


def cmd_serve(args: argparse.Namespace) -> int:
    ws = Path(args.workspace_dir).resolve() if args.workspace_dir else Path.cwd()
    svc = PocketGenieService(
        workspace_dir=ws,
        backend_mode=args.backend,
        llm_base_url=args.base_url,
        primary_model=args.model,
        fallback_model=args.fallback_model,
    )
    if args.mcp_config:
        svc.mcp_bridge.load_config_file(Path(args.mcp_config))
    static_dir = Path(args.static_dir).resolve() if args.static_dir else None
    server = create_http_server(
        host=args.host,
        port=args.port,
        service=svc,
        static_dir=static_dir,
        api_only=args.api_only,
    )
    lan_ip = get_lan_ip()
    print(f"Pocket-Genie Server ({args.backend}: {svc.primary_model}) listening on:")
    print(f"  Local UI:     http://127.0.0.1:{args.port}")
    print(f"  Phone UI:     http://{lan_ip}:{args.port}")
    print(f"  Headless API: http://{lan_ip}:{args.port}/api/v1/openapi.json")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    ws = Path(args.workspace_dir).resolve() if args.workspace_dir else Path.cwd()
    svc = PocketGenieService(
        workspace_dir=ws,
        backend_mode=args.backend,
        llm_base_url=args.base_url,
        primary_model=args.model,
        fallback_model=args.fallback_model,
    )
    result = svc.run_turn(
        prompt=args.prompt,
        agent_type=args.agent_type,
        target_specialist=args.specialist,
        tier=args.tier,
        fault=args.fault,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_skills(_args: argparse.Namespace) -> int:
    reg = SkillRegistry()
    for s in reg.list_skills():
        print(f"{s.name:<22} | tier={s.recommended_tier} | {s.description}")
    return 0


def cmd_specialists(_args: argparse.Namespace) -> int:
    for cfg in list_specialist_configs():
        domains = ",".join(cfg.tool_domains)
        print(f"{cfg.agent_id:<24} | default={cfg.default_tier} | domains=[{domains}] | {cfg.description}")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    reg = ToolRegistry()
    bridge = MCPBridge(reg)
    loaded = bridge.load_config_file(Path(args.config))
    print(json.dumps([{"name": s.name, "transport": s.transport} for s in loaded], indent=2))
    return 0


def cmd_bench(_args: argparse.Namespace) -> int:
    profiles = build_sensitivity_profiles()
    kv = BOSSinkINT4KVCache(sink_tokens=4, recent_window=64)
    fp16_mb, comp_mb, sink_n, mid_n, rec_n = kv.estimate_memory_mb(seq_len=2048)

    router = TokenPrecisionRouter()
    logits_trace = [
        [7.2, 1.0, 0.2],
        [6.8, 1.1, 0.1],
        [1.9, 1.4, 1.1],
        [6.4, 0.9, 0.0],
        [1.5, 1.2, 0.9],
        [7.0, 0.8, 0.1],
        [6.9, 1.0, 0.2],
        [6.1, 1.1, 0.3],
    ]
    for idx, lvec in enumerate(logits_trace):
        router.evaluate_step(idx, lvec)
    telemetry = router.build_telemetry(
        prefix_kv_hit=True,
        kv_cache_fp16_mb=fp16_mb,
        kv_cache_quant_mb=comp_mb,
    )

    report = {
        "profiles": {
            tier: {
                "name": p.name,
                "weight_bits": p.weight_bits,
                "effective_act_bits": p.effective_act_bits,
                "promoted_layers_count": len(p.promoted_layers),
            }
            for tier, p in profiles.items()
        },
        "kv_cache_2048_tokens": {
            "sink_fp16_tokens": sink_n,
            "middle_int4_tokens": mid_n,
            "recent_fp16_tokens": rec_n,
            "fp16_mb": fp16_mb,
            "compressed_mb": comp_mb,
            "saved_mb": round(fp16_mb - comp_mb, 4),
        },
        "token_router_telemetry": telemetry.to_dict(),
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    ws = Path(args.workspace_dir).resolve() if args.workspace_dir else Path.cwd()
    out_path = Path(args.output).resolve() if args.output else None
    report = run_eval_suite(
        backend_mode=args.backend,
        model_id=args.model,
        base_url=args.base_url,
        workspace_dir=ws,
        max_tasks=args.max_tasks,
        output_path=out_path,
    )
    print(json.dumps(report, indent=2))
    return 0


def cmd_graph(_args: argparse.Namespace) -> int:
    graph = GenieExecutionGraph()
    print(graph.get_mermaid_graph())
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    ws = Path(args.workspace_dir).resolve() if args.workspace_dir else Path.cwd()
    svc = PocketGenieService(workspace_dir=ws, backend_mode="replay")
    brief = build_daily_briefing(svc)
    if args.json:
        print(json.dumps(brief, indent=2, default=str))
    else:
        print(format_briefing_markdown(brief))
    return 0


def cmd_vibe(args: argparse.Namespace) -> int:
    ws = Path(args.workspace_dir).resolve() if args.workspace_dir else Path.cwd()
    svc = PocketGenieService(
        workspace_dir=ws,
        backend_mode=args.backend,
        llm_base_url=args.base_url,
        primary_model=args.model,
        fallback_model=args.fallback_model,
    )
    return run_vibe_repl(svc, initial_tier=args.tier)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genie",
        description="Pocket-Genie CLI: quantization-aware LangGraph/LangChain supervisor, Langfuse tracing, and mobile HITL console.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_srv = sub.add_parser("serve", help="Start the headless /api/v1 server and static web client.")
    p_srv.add_argument("--host", type=str, default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=8090)
    p_srv.add_argument("--workspace-dir", type=str, default=None)
    p_srv.add_argument("--mcp-config", type=str, default=None)
    p_srv.add_argument("--static-dir", type=str, default=None, help="Custom frontend build directory (e.g. React/Vite dist)")
    p_srv.add_argument("--api-only", action="store_true", help="Disable static UI serving (headless mode for Flutter/mobile)")
    p_srv.add_argument(
        "--backend",
        type=str,
        default="replay",
        choices=["replay", "local-llm", "api-llm", "hybrid", "hf-quant"],
        help="Model execution backend",
    )
    p_srv.add_argument("--base-url", type=str, default="http://127.0.0.1:11434/v1", help="OpenAI-compatible endpoint (llama-server, ollama, or cloud API)")
    p_srv.add_argument("--model", type=str, default="qwen2.5-coder:7b", help="Primary model identifier (defaults to gpt-6-luna under --backend api-llm)")
    p_srv.add_argument("--fallback-model", type=str, default=None, help="Fallback model for t3 escalation (defaults to gpt-6.1-terra under api-llm/hybrid)")
    p_srv.set_defaults(func=cmd_serve)

    p_ask = sub.add_parser("ask", help="Execute a multi-agent turn from the CLI.")
    p_ask.add_argument("--prompt", type=str, required=True)
    p_ask.add_argument(
        "--agent-type",
        type=str,
        default="supervisor",
        choices=["supervisor", "scoped-demo", "red-agent", "run-assistant", "direct-specialist"],
    )
    p_ask.add_argument("--specialist", type=str, default=None)
    p_ask.add_argument("--tier", type=str, default="t1", choices=["t1", "t2", "t3"])
    p_ask.add_argument(
        "--fault",
        type=str,
        default="none",
        choices=["none", "corrupted_json", "refusal", "hitl_patch"],
    )
    p_ask.add_argument("--workspace-dir", type=str, default=None)
    p_ask.add_argument(
        "--backend",
        type=str,
        default="replay",
        choices=["replay", "local-llm", "api-llm", "hybrid", "hf-quant"],
    )
    p_ask.add_argument("--base-url", type=str, default="http://127.0.0.1:11434/v1")
    p_ask.add_argument("--model", type=str, default="qwen2.5-coder:7b")
    p_ask.add_argument("--fallback-model", type=str, default=None)
    p_ask.set_defaults(func=cmd_ask)

    p_sk = sub.add_parser("skills", help="List discovered markdown skill playbooks.")
    p_sk.set_defaults(func=cmd_skills)

    p_sp = sub.add_parser("specialists", help="List configured specialist agents and tool scopes.")
    p_sp.set_defaults(func=cmd_specialists)

    p_mcp = sub.add_parser("mcp", help="Inspect MCP server configuration file.")
    p_mcp.add_argument("--config", type=str, required=True)
    p_mcp.set_defaults(func=cmd_mcp)

    p_bn = sub.add_parser("bench", help="Run quantization profile, token-router, and INT4 KV cache benchmark.")
    p_bn.set_defaults(func=cmd_bench)

    p_ev = sub.add_parser("eval", help="Run empirical 30-task multi-domain tool-calling & quantization benchmark.")
    p_ev.add_argument(
        "--backend",
        type=str,
        default="hf-quant",
        choices=["replay", "local-llm", "api-llm", "hybrid", "hf-quant"],
    )
    p_ev.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p_ev.add_argument("--base-url", type=str, default="http://127.0.0.1:11434/v1")
    p_ev.add_argument("--workspace-dir", type=str, default=None)
    p_ev.add_argument("--max-tasks", type=int, default=None)
    p_ev.add_argument("--output", type=str, default=None, help="Optional JSON path to persist benchmark report")
    p_ev.set_defaults(func=cmd_eval)

    p_gr = sub.add_parser("graph", help="Print compiled LangGraph StateGraph topology in Mermaid format.")
    p_gr.set_defaults(func=cmd_graph)

    p_br = sub.add_parser(
        "brief",
        help="Print morning daily dashboard across deadlines, messages, Git, Agent Ops, Databricks, and study cards.",
    )
    p_br.add_argument("--workspace-dir", type=str, default=None)
    p_br.add_argument("--json", action="store_true", help="Output raw JSON instead of Markdown")
    p_br.set_defaults(func=cmd_brief)

    p_vb = sub.add_parser(
        "vibe",
        help="Start interactive multi-turn CLI vibe-coding & personal-ops REPL with inline HITL prompts.",
    )
    p_vb.add_argument("--workspace-dir", type=str, default=None)
    p_vb.add_argument("--tier", type=str, default="t1", choices=["t1", "t2", "t3"])
    p_vb.add_argument(
        "--backend",
        type=str,
        default="api-llm",
        choices=["replay", "local-llm", "api-llm", "hybrid", "hf-quant"],
    )
    p_vb.add_argument("--base-url", type=str, default="http://127.0.0.1:11434/v1")
    p_vb.add_argument("--model", type=str, default="qwen2.5-coder:7b")
    p_vb.add_argument("--fallback-model", type=str, default=None)
    p_vb.set_defaults(func=cmd_vibe)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
