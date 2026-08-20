"""OpenAI LLM provider with function/tool calling, via the plain REST
Chat Completions endpoint (httpx), not the `openai` SDK — keeps the
vendor surface area to one file and one dependency (httpx).
Non-streaming: the agent loop needs the full tool_calls list before it can
decide what to do next, so there's no latency win from streaming here (see
README "Why non-streaming for tool-calling turns")."""
from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import OPENAI_API_KEY, OPENAI_LLM_MODEL
from app.providers.llm.base import LLMResponse, ToolCall

_CHAT_URL = "https://api.openai.com/v1/chat/completions"


class OpenAILLMProvider:
    name = "openai"

    def __init__(self, model: str = OPENAI_LLM_MODEL):
        self.model = model
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY (or lowercase openai_api_key) is not set — "
                "cannot construct OpenAILLMProvider. Use LLM_PROVIDER=mock "
                "to run without a key."
            )
        self._api_key: str = OPENAI_API_KEY
        # Instance attribute, not just the module-level _CHAT_URL, so
        # groq_llm.py's GroqLLMProvider can subclass this and point the
        # exact same request/response plumbing at Groq's wire-compatible
        # endpoint by overriding one attribute instead of copying chat().
        self._chat_url: str = _CHAT_URL
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def reset_usage(self) -> None:
        """Zero the usage counters. The caller decides what a measurement
        window is — the eval runner treats one trial as one window — so
        this provider accumulates and never resets itself. Resetting
        inside chat() would silently make multi-tool turns look like
        single calls."""
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        resp = httpx.post(
            self._chat_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]["message"]

        # Token usage from the API response, ACCUMULATED across calls
        # rather than overwritten. One agent turn is several chat() calls
        # (one per tool round-trip), so last-call-wins would undercount a
        # multi-tool task — exactly the expensive ones worth measuring.
        # Callers reset this via reset_usage() at a boundary they define;
        # the eval runner does so per trial.
        usage = data.get("usage") or {}
        self.last_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        self.last_usage["completion_tokens"] += int(usage.get("completion_tokens", 0))
        self.last_usage["calls"] += 1

        tool_calls = tuple(
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"]["arguments"] or "{}"),
            )
            for tc in (choice.get("tool_calls") or [])
        )
        return LLMResponse(
            content=choice.get("content"),
            tool_calls=tool_calls,
            provider=self.name,
            model=self.model,
            raw=data,
        )


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reformat the project's internal message shape (plain dicts, with
    assistant tool_calls as {id, name, arguments}) into the OpenAI wire
    shape (tool_calls as {id, type, function: {name, arguments: <json str>}})."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content"),
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                        }
                        for tc in m["tool_calls"]
                    ],
                }
            )
        else:
            out.append(m)
    return out
