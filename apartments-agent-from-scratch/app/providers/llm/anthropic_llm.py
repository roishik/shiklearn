"""Anthropic LLM provider — the swap target for LLM_PROVIDER=anthropic.

Implemented against the documented Messages API tool-use shape (system as
a top-level field, assistant tool requests as `tool_use` content blocks,
tool results sent back as a user message with a `tool_result` content
block) — a real implementation, not a stub. NOTE: verified in this build
only via the OpenAI provider; this file has not been exercised against a
live Anthropic key. See README "What was verified vs not" and DECISIONS.md.

Swapping LLM_PROVIDER=openai -> anthropic should not require touching
agent_loop.py, app/main.py, or app/cli.py. If it does, the abstraction
has a leak — that's worth noticing and fixing, and worth saying out loud
in a design-doc defence."""
from __future__ import annotations

from typing import Any

import httpx

from app.config import ANTHROPIC_API_KEY, ANTHROPIC_LLM_MODEL
from app.providers.llm.base import LLMResponse, ToolCall

_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicLLMProvider:
    name = "anthropic"

    def __init__(self, model: str = ANTHROPIC_LLM_MODEL):
        self.model = model
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY (or lowercase anthropic_api_key) is not set — "
                "cannot construct AnthropicLLMProvider. Use LLM_PROVIDER=mock "
                "to run without a key."
            )
        self._api_key: str = ANTHROPIC_API_KEY

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        system, anthropic_messages = _to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "system": system,
            "messages": anthropic_messages,
        }
        if tools:
            payload["tools"] = [_to_anthropic_tool(t) for t in tools]

        resp = httpx.post(
            _MESSAGES_URL,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block["text"])
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(id=block["id"], name=block["name"], arguments=block.get("input", {})))

        return LLMResponse(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tuple(tool_calls),
            provider=self.name,
            model=self.model,
            raw=data,
        )


def _to_anthropic_tool(openai_tool: dict[str, Any]) -> dict[str, Any]:
    """Anthropic's tool schema is flatter than OpenAI's — no nested
    'function' wrapper, and 'input_schema' instead of 'parameters'."""
    fn = openai_tool["function"]
    return {"name": fn["name"], "description": fn.get("description", ""), "input_schema": fn["parameters"]}


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Anthropic has no 'system' or 'tool' role in the messages array:
    system prompt is a separate top-level field, and tool results are
    sent back as a `user` message containing a `tool_result` block."""
    system = ""
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system = (system + "\n" + m["content"]).strip()
        elif role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            if m.get("tool_calls"):
                content: list[dict[str, Any]] = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
                out.append({"role": "assistant", "content": content})
            else:
                out.append({"role": "assistant", "content": m.get("content") or ""})
        elif role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                    ],
                }
            )
    return system, out
