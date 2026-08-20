"""
agent_loop.py — the hand-rolled agent loop.

No LangGraph, no LlamaIndex, no framework of any kind: this is a plain
Python while-loop over one provider-agnostic `chat()` call, with tool
dispatch, guardrail wrapping, and turn-limiting done by hand in ~70 lines.
See README "Why no framework" for the full justification: control,
debuggability, latency, and the fact that at this size a framework's
abstractions cost more to reason about than the ~70 lines they replace.

Loop shape:
  1. Send the running message list + tool schemas to the LLM provider.
  2. If the response has tool_calls, execute each one against
     tool_registry, wrap every result as untrusted data
     (guardrails.wrap_untrusted) before it goes back into the
     conversation, append it as a 'tool' message, and loop.
  3. If the response has content and no tool_calls, that's the final
     answer — return it.
  4. Stop after max_turns to GUARANTEE termination even if a provider
     keeps requesting tools forever (a real failure mode worth defending
     against, not a hypothetical).

Every tool call and its result is recorded in `tool_log`, in call order —
that list is what the chat UI's live tool-call log renders (app/main.py,
app/cli.py, static/index.html), and it doubles as the reasoning trace —
full visibility into what the agent did, which tools it called, and what
came back. This agent is expected to "explain its reasoning clearly";
this log is that requirement made literal rather than asserted.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from app.guardrails import wrap_untrusted
from app.providers.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger("agent_loop")


@dataclass
class ToolLogEntry:
    """One row of the live tool-call log the chat UI displays."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any
    error: str | None = None


@dataclass
class AgentResult:
    final_text: str
    messages: list[dict[str, Any]]
    tool_log: list[ToolLogEntry] = field(default_factory=list)
    turns_used: int = 0


class MaxTurnsExceeded(RuntimeError):
    """Raised when the provider keeps requesting tool calls past
    max_turns without ever returning a final answer. A hand-rolled loop
    has to enforce this itself — nothing does it for you.

    Carries the work done so far. Hitting the ceiling means the agent ran
    out of turns, NOT that it learned nothing: it may have made five
    successful tool calls and simply never written them up. Throwing that
    away would turn a partial answer into a blank screen, so the caller
    gets the transcript and the tool log and can present what was found
    while saying plainly that the answer is incomplete.
    """

    def __init__(
        self,
        message: str,
        messages: list[dict[str, Any]] | None = None,
        tool_log: list[ToolLogEntry] | None = None,
        turns_used: int = 0,
    ):
        super().__init__(message)
        self.messages = messages or []
        self.tool_log = tool_log or []
        self.turns_used = turns_used


def run_agent(
    user_message: str,
    history: list[dict[str, Any]],
    provider: LLMProvider,
    tool_schemas: list[dict[str, Any]],
    tool_registry: dict[str, Callable[[dict[str, Any]], Any]],
    system_prompt: str,
    max_turns: int = 6,
    on_tool_call: Callable[[ToolLogEntry], None] | None = None,
) -> AgentResult:
    """`on_tool_call`, if given, fires synchronously the instant each tool
    call finishes (success or error) — BEFORE the loop goes back to the
    provider for the next turn. This is what lets app/main.py's SSE
    endpoint push a live tool-call log to the browser turn-by-turn on a
    multi-tool question, instead of the whole log appearing at once after
    the entire loop finishes. Purely an observation hook: it cannot
    affect control flow, retry a call, or see anything the tool_log list
    doesn't already carry — the loop's behavior is identical whether or
    not a caller passes one."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    tool_log: list[ToolLogEntry] = []

    for turn in range(1, max_turns + 1):
        response: LLMResponse = provider.chat(messages, tools=tool_schemas)

        if not response.tool_calls:
            final_text = response.content or ""
            messages.append({"role": "assistant", "content": final_text})
            return AgentResult(final_text=final_text, messages=messages, tool_log=tool_log, turns_used=turn)

        # Record the assistant's tool-call REQUEST in the transcript (the
        # provider-specific wire format is reconstructed inside each
        # provider's chat() caller; this internal shape is provider-neutral).
        messages.append(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls
                ],
            }
        )

        for tc in response.tool_calls:
            error: str | None = None
            fn = tool_registry.get(tc.name)
            if fn is None:
                error = f"unknown tool {tc.name!r}"
                result: Any = {"error": error}
            else:
                try:
                    result = fn(tc.arguments)
                except Exception as exc:  # tool errors are DATA the model sees, not a crash
                    error = str(exc)
                    result = {"error": error}
                    logger.warning("tool %s raised: %s", tc.name, exc)

            entry = ToolLogEntry(tool_name=tc.name, arguments=dict(tc.arguments), result=result, error=error)
            tool_log.append(entry)
            if on_tool_call is not None:
                on_tool_call(entry)

            # Guardrail: EVERY tool result is treated as untrusted data,
            # no exceptions, before it re-enters the conversation. This is
            # the one line that matters most in this whole file — see
            # app/guardrails.py and tests/test_guardrails.py.
            raw_text = json.dumps(result, default=str)
            wrapped = wrap_untrusted(raw_text, source=f"tool:{tc.name}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": wrapped,
                }
            )

    raise MaxTurnsExceeded(
        f"agent did not produce a final answer within {max_turns} turns",
        messages=messages,
        tool_log=tool_log,
        turns_used=max_turns,
    )
