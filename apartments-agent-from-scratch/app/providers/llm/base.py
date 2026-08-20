"""Provider-agnostic LLM interface for the agent loop.

Supports OpenAI-style function/tool calling since agent_loop.py needs the
model to be able to request a tool call and receive the tool's structured
result back in a follow-up turn. Every concrete provider (openai_llm.py,
anthropic_llm.py, mock_llm.py) implements `chat()` against this same
shape, so agent_loop.py, app/main.py, and app/cli.py never import a
vendor SDK or know which vendor is active.

Swap providers via LLM_PROVIDER in .env — see app/config.py,
app/providers/llm/__init__.py, and README "Swapping a provider"."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    provider: str
    model: str
    raw: Any = field(default=None, repr=False)


class LLMProvider(Protocol):
    name: str
    model: str

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse: ...
