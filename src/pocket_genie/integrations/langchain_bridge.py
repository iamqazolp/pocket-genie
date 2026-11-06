from __future__ import annotations

import re
from typing import Any, Sequence

from pocket_genie.registry.tools import ToolSpec, sanitize_json_schema
from pocket_genie.shared import ChatMessage, ToolInvocation

try:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import BaseTool, StructuredTool
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from pydantic import BaseModel, ConfigDict, Field, create_model

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    CallbackManagerForLLMRun = Any  # type: ignore[assignment,misc]
    BaseChatModel = object  # type: ignore[assignment,misc]
    AIMessage = Any  # type: ignore[assignment,misc]
    BaseMessage = Any  # type: ignore[assignment,misc]
    HumanMessage = Any  # type: ignore[assignment,misc]
    SystemMessage = Any  # type: ignore[assignment,misc]
    ToolMessage = Any  # type: ignore[assignment,misc]
    ChatGeneration = Any  # type: ignore[assignment,misc]
    ChatResult = Any  # type: ignore[assignment,misc]
    BaseTool = Any  # type: ignore[assignment,misc]
    StructuredTool = Any  # type: ignore[assignment,misc]
    convert_to_openai_tool = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment,misc]
    ConfigDict = dict  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment]
    create_model = None  # type: ignore[assignment]


_JSON_TO_PY_TYPE: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _require_langchain() -> None:
    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError("langchain-core is required for LangChain integration features.")


def chat_messages_to_langchain(messages: Sequence[ChatMessage]) -> list[Any]:
    """Convert Pocket-Genie ChatMessage objects into LangChain BaseMessage instances."""
    _require_langchain()
    lc_msgs: list[Any] = []
    for m in messages:
        extra: dict[str, Any] = {}
        if m.reasoning:
            extra["reasoning"] = m.reasoning
        if m.metadata:
            extra["genie_metadata"] = dict(m.metadata)

        if m.role == "system":
            lc_msgs.append(SystemMessage(content=m.content, additional_kwargs=extra))
        elif m.role == "user":
            lc_msgs.append(HumanMessage(content=m.content, additional_kwargs=extra))
        elif m.role == "tool":
            lc_msgs.append(
                ToolMessage(
                    content=m.content,
                    tool_call_id=m.tool_call_id or "call_0",
                    name=m.name,
                    additional_kwargs=extra,
                )
            )
        else:
            tcs = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "args": dict(tc.arguments),
                    "type": "tool_call",
                }
                for tc in m.tool_calls
            ]
            lc_msgs.append(AIMessage(content=m.content, tool_calls=tcs, additional_kwargs=extra))
    return lc_msgs


def langchain_to_chat_messages(messages: Sequence[Any]) -> list[ChatMessage]:
    """Convert LangChain BaseMessage instances back into Pocket-Genie ChatMessage objects."""
    _require_langchain()
    out: list[ChatMessage] = []
    for m in messages:
        extra = getattr(m, "additional_kwargs", {}) or {}
        reasoning = extra.get("reasoning")
        meta = dict(extra.get("genie_metadata") or {})
        content = m.content if isinstance(m.content, str) else str(m.content)

        if isinstance(m, SystemMessage):
            out.append(ChatMessage(role="system", content=content, reasoning=reasoning, metadata=meta))
        elif isinstance(m, HumanMessage):
            out.append(ChatMessage(role="user", content=content, reasoning=reasoning, metadata=meta))
        elif isinstance(m, ToolMessage):
            out.append(
                ChatMessage(
                    role="tool",
                    content=content,
                    name=getattr(m, "name", None),
                    tool_call_id=getattr(m, "tool_call_id", None),
                    reasoning=reasoning,
                    metadata=meta,
                )
            )
        elif isinstance(m, AIMessage):
            invocations: list[ToolInvocation] = []
            for idx, tc in enumerate(getattr(m, "tool_calls", None) or []):
                invocations.append(
                    ToolInvocation(
                        id=str(tc.get("id") or f"call_{idx}"),
                        name=str(tc.get("name") or ""),
                        arguments=dict(tc.get("args") or {}),
                    )
                )
            out.append(
                ChatMessage(
                    role="assistant",
                    content=content,
                    tool_calls=invocations,
                    reasoning=reasoning,
                    metadata=meta,
                )
            )
        else:
            out.append(ChatMessage(role="user", content=content, reasoning=reasoning, metadata=meta))
    return out


def json_schema_to_pydantic_model(name: str, schema: dict[str, Any]) -> Any:
    """Synthesize a Pydantic v2 BaseModel from a sanitized JSON Schema dictionary."""
    _require_langchain()
    clean = sanitize_json_schema(schema)
    props = clean.get("properties") or {}
    required = set(clean.get("required") or [])
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name) + "Input"

    field_defs: dict[str, tuple[Any, Any]] = {}
    for prop_name, prop_schema in props.items():
        jtype = str(prop_schema.get("type", "string"))
        py_type = _JSON_TO_PY_TYPE.get(jtype, Any)
        desc = str(prop_schema.get("description", ""))
        if prop_name in required:
            field_defs[prop_name] = (py_type, Field(..., description=desc))
        else:
            default_val = prop_schema.get("default", None)
            field_defs[prop_name] = (py_type | None, Field(default=default_val, description=desc))

    return create_model(safe_name, **field_defs)


def tool_spec_to_structured_tool(spec: ToolSpec) -> Any:
    """Wrap a Pocket-Genie ToolSpec into a LangChain StructuredTool with typed Pydantic schema."""
    _require_langchain()
    args_model = json_schema_to_pydantic_model(spec.name, spec.parameters)

    def _wrapped(**kwargs: Any) -> Any:
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        return spec.handler(**filtered)

    return StructuredTool.from_function(
        func=_wrapped,
        name=spec.name,
        description=spec.description,
        args_schema=args_model,
        metadata={
            "domain": spec.domain,
            "mutating": spec.mutating,
            "project_scoped": spec.project_scoped,
            "source": spec.source,
        },
    )


if LANGCHAIN_AVAILABLE:

    class PocketGenieChatModel(BaseChatModel):
        """LangChain BaseChatModel adapter backed by any Pocket-Genie tiered ModelBackend."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        backend: Any
        constrain_schema: bool = False
        bound_tools: list[dict[str, Any]] = Field(default_factory=list)

        @property
        def _llm_type(self) -> str:
            return f"pocket-genie-{getattr(self.backend, 'name', 'backend')}"

        @property
        def _identifying_params(self) -> dict[str, Any]:
            return {
                "backend_name": getattr(self.backend, "name", "unknown"),
                "tier": getattr(self.backend, "tier", "t1"),
                "constrain_schema": self.constrain_schema,
            }

        def bind_tools(
            self,
            tools: Sequence[dict[str, Any] | type | Any],
            **kwargs: Any,
        ) -> PocketGenieChatModel:
            normalized: list[dict[str, Any]] = []
            for t in tools:
                if isinstance(t, ToolSpec):
                    normalized.append(t.to_openai_tool())
                elif isinstance(t, dict) and "function" in t:
                    normalized.append(t)
                else:
                    normalized.append(convert_to_openai_tool(t))
            return self.model_copy(update={"bound_tools": normalized, **kwargs})

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            genie_msgs = langchain_to_chat_messages(messages)
            tools = kwargs.get("tools") or self.bound_tools or None
            constrain = bool(kwargs.get("constrain_schema", self.constrain_schema))
            out = self.backend.generate(
                genie_msgs,
                tools=tools,
                constrain_schema=constrain,
            )
            lc_ai = chat_messages_to_langchain([out.message])[0]
            telemetry_dict = out.telemetry.to_dict()
            prompt_est = max(1, sum(len(m.content) for m in genie_msgs) // 4)
            comp_toks = max(1, int(out.telemetry.total_tokens or max(1, len(out.message.content) // 4)))
            lc_ai.usage_metadata = {
                "input_tokens": prompt_est,
                "output_tokens": comp_toks,
                "total_tokens": prompt_est + comp_toks,
            }
            lc_ai.response_metadata = {
                "model_name": getattr(self.backend, "name", "pocket-genie"),
                "tier": out.telemetry.tier,
                "quant_telemetry": telemetry_dict,
            }
            return ChatResult(
                generations=[ChatGeneration(message=lc_ai)],
                llm_output={"quant_telemetry": telemetry_dict, "tier": out.telemetry.tier},
            )

else:

    class PocketGenieChatModel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_langchain()
