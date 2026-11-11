from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from pocket_genie.env_config import (
    get_ssl_context,
    load_dotenv_file,
    resolve_api_llm_config,
)
from pocket_genie.registry import (
    BackendRegistry,
    OpenAICompatibleBackend,
)
from pocket_genie.shared import ChatMessage, ToolInvocation


def test_get_ssl_context():
    ctx = get_ssl_context()
    assert ctx is not None


def test_load_dotenv_file_and_override(tmp_path: Path):
    env_file = tmp_path / ".env.mock"
    env_file.write_text(
        "# Mock environment file\n"
        "\n"
        'export OPENAI_BASE_URL="https://api.openai.com/v1"\n'
        "OPENAI_KEY='sk-test-openai-key'\n"
        "GENIE_FAST_MODEL=gpt-6-luna\n"
        'GENIE_ACCURATE_MODEL="gpt-6.1-terra"\n'
        "IGNORED_LINE_WITHOUT_EQUALS\n",
        encoding="utf-8",
    )
    loaded = load_dotenv_file(env_file, override=True)
    assert loaded["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
    assert loaded["OPENAI_KEY"] == "sk-test-openai-key"
    assert loaded["GENIE_FAST_MODEL"] == "gpt-6-luna"
    assert loaded["GENIE_ACCURATE_MODEL"] == "gpt-6.1-terra"

    missing = load_dotenv_file(tmp_path / "nonexistent.env")
    assert missing == {}


def test_resolve_api_llm_config_google_and_openrouter():
    env = {
        "GOOGLE_AI_STUDIO_KEY": "studio-ai-key",
        "OPENAI_BASE_URL": "https://custom-gateway.openai.com/v1/",
    }
    cfg = resolve_api_llm_config(env=env)
    assert cfg.api_key == "studio-ai-key"
    assert cfg.google_api_key == "studio-ai-key"
    assert cfg.base_url == "https://custom-gateway.openai.com/v1"
    assert cfg.fast_model == "gpt-6-luna"
    assert cfg.accurate_model == "gpt-6.1-terra"
    assert cfg.embedding_model == "gemini-embedding-2"

    openrouter_env = {"OPENROUTER_API_KEY": "sk-or-v1-abc"}
    cfg_or = resolve_api_llm_config(env=openrouter_env)
    assert cfg_or.base_url == "https://openrouter.ai/api/v1"
    assert cfg_or.api_key == "sk-or-v1-abc"


def test_wire_sanitization_and_tool_mapping():
    backend = OpenAICompatibleBackend(
        base_url="https://api.openai.com/v1",
        model="gpt-6.1-terra",
        api_key="secret-key",
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "devops.git_status",
                "description": "Get git status",
                "parameters": {"type": "object", "properties": {"repo": {"type": "string"}}},
            },
        }
    ]
    messages = [
        ChatMessage(role="user", content="check repo status"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolInvocation(
                    id="call_prev",
                    name="devops.git_status",
                    arguments={"repo": "."},
                )
            ],
        ),
        ChatMessage(
            role="tool",
            tool_call_id="call_prev",
            name="devops.git_status",
            content="clean",
        ),
    ]

    captured_payload: dict = {}

    def fake_post_json(url, payload, api_key, timeout):
        nonlocal captured_payload
        captured_payload = payload
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "devops__git_status",
                                    "arguments": '{"repo": "main"}',
                                },
                            }
                        ],
                    },
                    "logprobs": None,
                }
            ],
            "usage": {"completion_tokens": 15},
        }

    with patch("pocket_genie.registry.backends._post_json", side_effect=fake_post_json):
        res = backend.generate(messages, tools=tools)

    assert captured_payload["tools"][0]["function"]["name"] == "devops__git_status"
    assert (
        captured_payload["messages"][1]["tool_calls"][0]["function"]["name"]
        == "devops__git_status"
    )
    assert captured_payload["messages"][2]["name"] == "devops__git_status"

    assert len(res.message.tool_calls) == 1
    call = res.message.tool_calls[0]
    assert call.id == "call_123"
    assert call.name == "devops.git_status"
    assert call.arguments == {"repo": "main"}


def test_http_error_fallback_without_logprobs():
    import urllib.error

    backend = OpenAICompatibleBackend(
        base_url="https://api.openai.com/v1",
        model="gpt-6.1-terra",
    )
    calls: list[dict] = []

    def mock_post_json(url, payload, api_key, timeout):
        calls.append(payload)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                url=url, code=400, msg="Logprobs not supported", hdrs=MagicMock(), fp=None
            )
        return {
            "choices": [{"message": {"role": "assistant", "content": "Fallback OK"}}],
            "usage": {"completion_tokens": 5},
        }

    with patch("pocket_genie.registry.backends._post_json", side_effect=mock_post_json):
        res = backend.generate([ChatMessage(role="user", content="hello")])

    assert len(calls) == 2
    assert "logprobs" in calls[0]
    assert "logprobs" not in calls[1]
    assert res.message.content == "Fallback OK"


def test_registry_configure_api_llm_and_hybrid():
    reg = BackendRegistry()
    cfg = reg.configure_api_llm(
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        fast_model="gpt-6-luna",
        accurate_model="gpt-6.1-terra",
    )
    assert cfg.fast_model == "gpt-6-luna"
    assert cfg.accurate_model == "gpt-6.1-terra"

    t1 = reg.get("t1")
    t2 = reg.get("t2")
    t3 = reg.get("t3")
    assert isinstance(t1, OpenAICompatibleBackend)
    assert t1.model == "gpt-6-luna"
    assert t1.tier == "t1"
    assert isinstance(t2, OpenAICompatibleBackend)
    assert t2.model == "gpt-6-luna"
    assert t2.tier == "t2"
    assert isinstance(t3, OpenAICompatibleBackend)
    assert t3.model == "gpt-6.1-terra"
    assert t3.tier == "t3"

    hybrid_cfg = reg.configure_hybrid(
        local_base_url="http://127.0.0.1:11434/v1",
        local_model="qwen2.5-coder:7b",
        api_base_url="https://api.openai.com/v1",
        api_key="sk-hybrid-test",
        cloud_model="gpt-6.1-terra",
    )
    assert hybrid_cfg.base_url == "https://api.openai.com/v1"

    ht1 = reg.get("t1")
    ht2 = reg.get("t2")
    ht3 = reg.get("t3")
    assert isinstance(ht1, OpenAICompatibleBackend)
    assert ht1.model == "qwen2.5-coder:7b"
    assert ht1.base_url == "http://127.0.0.1:11434/v1"
    assert isinstance(ht2, OpenAICompatibleBackend)
    assert ht2.model == "qwen2.5-coder:7b"
    assert isinstance(ht3, OpenAICompatibleBackend)
    assert ht3.model == "gpt-6.1-terra"
    assert ht3.api_key == "sk-hybrid-test"


