from __future__ import annotations

import dataclasses
import os
import ssl
from collections.abc import Mapping
from pathlib import Path


def get_ssl_context() -> ssl.SSLContext:
    """Return an SSLContext configured with certifi CAs if available, falling back to default."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def load_dotenv_file(dotenv_path: Path | str, override: bool = False) -> dict[str, str]:
    """Parse KEY=VALUE lines from a dotenv file and apply to os.environ."""
    p = Path(dotenv_path)
    if not p.is_file():
        return {}

    loaded: dict[str, str] = {}
    content = p.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            if len(val) >= 2:
                val = val[1:-1]
        loaded[key] = val
        if override:
            os.environ[key] = val
        else:
            os.environ.setdefault(key, val)

    return loaded


@dataclasses.dataclass(slots=True)
class ApiLLMConfig:
    base_url: str
    api_key: str
    fast_model: str = "gpt-6-luna"
    accurate_model: str = "gpt-6.1-terra"
    embedding_model: str = "gemini-embedding-2"
    google_api_key: str = ""


def resolve_api_llm_config(
    env: Mapping[str, str] | None = None,
    fast_model: str | None = None,
    accurate_model: str | None = None,
    base_url: str | None = None,
) -> ApiLLMConfig:
    """Resolve cloud API LLM configuration from environment variables or explicit parameters."""
    env_map = os.environ if env is None else env

    api_key = (
        env_map.get("OPENAI_API_KEY")
        or env_map.get("OPENAI_KEY")
        or env_map.get("OPENROUTER_API_KEY")
        or env_map.get("GOOGLE_AI_STUDIO_KEY")
        or env_map.get("GEMINI_API_KEY")
        or ""
    )

    if base_url:
        resolved_base = base_url
    elif env_map.get("OPENAI_BASE_URL"):
        resolved_base = env_map["OPENAI_BASE_URL"]
    elif env_map.get("LLM_BASE_URL"):
        resolved_base = env_map["LLM_BASE_URL"]
    elif env_map.get("OPENROUTER_API_KEY"):
        resolved_base = "https://openrouter.ai/api/v1"
    else:
        resolved_base = "https://api.openai.com/v1"
    resolved_base = resolved_base.rstrip("/")

    fast = fast_model or env_map.get("GENIE_FAST_MODEL") or "gpt-6-luna"
    accurate = accurate_model or env_map.get("GENIE_ACCURATE_MODEL") or "gpt-6.1-terra"
    emb_model = env_map.get("GENIE_EMBEDDING_MODEL") or "gemini-embedding-2"
    google_key = env_map.get("GOOGLE_AI_STUDIO_KEY") or env_map.get("GEMINI_API_KEY") or ""

    return ApiLLMConfig(
        base_url=resolved_base,
        api_key=api_key,
        fast_model=fast,
        accurate_model=accurate,
        embedding_model=emb_model,
        google_api_key=google_key,
    )
