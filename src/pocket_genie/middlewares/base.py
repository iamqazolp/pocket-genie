from __future__ import annotations

import dataclasses
from typing import Any, Callable

from pocket_genie.core.state import GenieState
from pocket_genie.registry.backends import GenerationOutput
from pocket_genie.registry.tools import ToolSpec
from pocket_genie.shared import ChatMessage, GenieSharedContext, StreamWriter, ToolInvocation


@dataclasses.dataclass(slots=True)
class ModelCallContext:
    state: GenieState
    shared: GenieSharedContext
    messages: list[ChatMessage]
    tools: list[ToolSpec]
    writer: StreamWriter
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True)
class ToolCallContext:
    state: GenieState
    shared: GenieSharedContext
    invocation: ToolInvocation
    spec: ToolSpec | None
    specialist: str
    writer: StreamWriter
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class BaseMiddleware:
    """Base interceptor hook contract for model and tool execution pipelines."""

    name: str = "base"

    def before_model(self, ctx: ModelCallContext) -> None:
        return None

    def wrap_model(
        self,
        ctx: ModelCallContext,
        call_next: Callable[[ModelCallContext], GenerationOutput],
    ) -> GenerationOutput:
        return call_next(ctx)

    def after_model(self, ctx: ModelCallContext, output: GenerationOutput) -> GenerationOutput:
        return output

    def wrap_tool(
        self,
        ctx: ToolCallContext,
        call_next: Callable[[ToolCallContext], str],
    ) -> str:
        return call_next(ctx)


class MiddlewarePipeline:
    """Executes an ordered sequence of middlewares around model and tool invocations."""

    def __init__(self, middlewares: list[BaseMiddleware] | None = None) -> None:
        self._middlewares: list[BaseMiddleware] = list(middlewares or [])

    def add(self, middleware: BaseMiddleware) -> None:
        self._middlewares.append(middleware)

    @property
    def middlewares(self) -> list[BaseMiddleware]:
        return list(self._middlewares)

    def run_model(
        self,
        ctx: ModelCallContext,
        handler: Callable[[ModelCallContext], GenerationOutput],
    ) -> GenerationOutput:
        for mw in self._middlewares:
            mw.before_model(ctx)

        def build_chain(
            index: int,
        ) -> Callable[[ModelCallContext], GenerationOutput]:
            if index >= len(self._middlewares):
                return handler
            current = self._middlewares[index]
            next_fn = build_chain(index + 1)
            return lambda c: current.wrap_model(c, next_fn)

        chain = build_chain(0)
        output = chain(ctx)

        for mw in reversed(self._middlewares):
            output = mw.after_model(ctx, output)
        return output

    def run_tool(
        self,
        ctx: ToolCallContext,
        handler: Callable[[ToolCallContext], str],
    ) -> str:
        def build_chain(index: int) -> Callable[[ToolCallContext], str]:
            if index >= len(self._middlewares):
                return handler
            current = self._middlewares[index]
            next_fn = build_chain(index + 1)
            return lambda c: current.wrap_tool(c, next_fn)

        chain = build_chain(0)
        return chain(ctx)
