"""
runner.py — executes one Trial in isolation and turns it into a graded
TrialResult.

Isolation guarantee: `run_trial()` builds a brand-new `messages` list
(via app.agent_loop.run_agent, called with a fresh `history=list(...)`
copy every time) and a brand-new tool-registry dict (base TOOL_REGISTRY
merged with the task's extra tools, never mutated) on every call. Two
trials — of the same task or different tasks — never share mutable
state. The only thing genuinely shared across trials is the `provider`
object, which is safe: every concrete LLMProvider.chat() call
(mock/openai/anthropic) is a pure function of the messages it's given,
not of any per-instance state (see app/providers/llm/*.py).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from app.agent_loop import AgentResult, MaxTurnsExceeded, run_agent
from app.providers.llm.base import LLMProvider
from app.system_prompt import BASE_SYSTEM_PROMPT
from app.tools import TOOL_REGISTRY, TOOL_SCHEMAS
from evals.graders.deterministic import ToolCallGrader, _extract_stated_numbers
from evals.types import Outcome, Task, Trace, Trial, TrialResult

# Shares its extraction with NoFabricatedNumbersGrader rather than
# keeping a second, looser regex here. Two independent number-extractors
# on the same text is how this field went unnoticed as dead: it was
# populated by this simpler pattern and read by nothing, while the real
# fabrication check used its own copy in evals/graders/deterministic.py.
# One source of truth now — Outcome.numbers_in_text and the grader's own
# extraction can no longer silently disagree.


# Deliberately separate from _extract_stated_numbers (which this module
# also uses, below): that one parses MODEL PROSE, where "36,497,303.0"
# and "4.68%" are real formatting a human-facing sentence uses and must
# be normalized. This one parses json.dumps() of a tool's own return
# value, which is never comma-grouped or percent-suffixed — it is
# whatever repr Python's json encoder produces — so a plain decimal
# pattern is both sufficient and correct here; reusing the prose pattern
# would be reusing normalization logic this text doesn't need.
_JSON_DECIMAL_RE = re.compile(r"-?\d+\.\d+")


def _numbers_in(obj: Any) -> tuple[float, ...]:
    """Extract every decimal number appearing in the JSON-serialized form
    of `obj` — used to build the ground-truth pool NoFabricatedNumbersGrader
    checks the model's stated numbers against."""
    try:
        text = json.dumps(obj, default=str)
    except TypeError:
        text = str(obj)
    return tuple(float(m) for m in _JSON_DECIMAL_RE.findall(text))


def _run_agent_task(task: Task, provider: LLMProvider) -> Trace:
    tool_registry = dict(TOOL_REGISTRY)
    if task.extra_tool_registry:
        tool_registry.update(task.extra_tool_registry)
    tool_schemas = list(TOOL_SCHEMAS) + list(task.extra_tool_schemas)

    try:
        result: AgentResult = run_agent(
            user_message=task.user_message or "",
            history=list(task.history),  # fresh copy — isolation
            provider=provider,
            tool_schemas=tool_schemas,
            tool_registry=tool_registry,
            system_prompt=BASE_SYSTEM_PROMPT,
        )
        return Trace(
            final_text=result.final_text,
            messages=tuple(result.messages),
            tool_log=tuple(result.tool_log),
            turns_used=result.turns_used,
            raised=None,
        )
    except MaxTurnsExceeded:
        return Trace(final_text="", messages=(), tool_log=(), turns_used=0, raised="MaxTurnsExceeded")
    except Exception as exc:  # a trial crashing must not take the whole suite down
        return Trace(final_text="", messages=(), tool_log=(), turns_used=0, raised=f"{type(exc).__name__}: {exc}")


def _outcome_from_trace(trace: Trace) -> Outcome:
    tools_called = tuple(e.tool_name for e in trace.tool_log)
    tool_args_by_call = tuple((e.tool_name, dict(e.arguments)) for e in trace.tool_log)
    numbers_in_text = tuple(_extract_stated_numbers(trace.final_text))
    numbers_in_tool_outputs: list[float] = []
    for e in trace.tool_log:
        if e.error is None:
            numbers_in_tool_outputs.extend(_numbers_in(e.result))
    return Outcome(
        trace=trace,
        tools_called=tools_called,
        tool_args_by_call=tool_args_by_call,
        numbers_in_text=numbers_in_text,
        numbers_in_tool_outputs=tuple(numbers_in_tool_outputs),
    )


def _run_direct_task(task: Task) -> Outcome:
    assert task.run_direct is not None
    empty_trace = Trace(final_text="", messages=(), tool_log=(), turns_used=0, raised=None)
    try:
        value = task.run_direct()
        return Outcome(
            trace=empty_trace,
            tools_called=(),
            tool_args_by_call=(),
            numbers_in_text=(),
            numbers_in_tool_outputs=(),
            direct_result=value,
            direct_error=None,
        )
    except Exception as exc:
        return Outcome(
            trace=empty_trace,
            tools_called=(),
            tool_args_by_call=(),
            numbers_in_text=(),
            numbers_in_tool_outputs=(),
            direct_result=None,
            direct_error=f"{type(exc).__name__}: {exc}",
        )


def run_trial(task: Task, trial_index: int, provider: LLMProvider) -> TrialResult:
    """Run exactly one isolated Trial of `task` and grade it.

    Graders run are `task.graders` PLUS, if set, one auto-generated
    ToolCallGrader per `task.expected_tool` / `task.forbidden_tools` —
    this is the one place the suite grades the tool-call PATH rather
    than the outcome, and it's opt-in per task via those two fields (see
    evals/types.py:Task docstring, "grade outcomes, not paths")."""
    trial = Trial(task_id=task.id, trial_index=trial_index, provider_name=provider.name, model_name=provider.model)

    # The agent run is timed on its own. Grading is timed separately below
    # because an LLM-judge grader makes its own API call — folding that in
    # would make a slow JUDGE look like a slow AGENT, and latency numbers
    # exist to answer "how long does a user wait", which the judge is not
    # part of.
    # One trial is one measurement window — reset before, read after, so a
    # multi-tool turn's several chat() calls are summed rather than the
    # last one winning.
    reset = getattr(provider, "reset_usage", None)
    if callable(reset):
        reset()

    started = time.perf_counter()
    if task.run_direct is not None:
        outcome = _run_direct_task(task)
    else:
        trace = _run_agent_task(task, provider)
        outcome = _outcome_from_trace(trace)
    latency_seconds = time.perf_counter() - started

    path_graders: list = []
    if task.expected_tool:
        path_graders.append(ToolCallGrader(task.expected_tool, should_be_called=True))
    for forbidden in task.forbidden_tools:
        path_graders.append(ToolCallGrader(forbidden, should_be_called=False))

    grading_started = time.perf_counter()
    grades = [grader.grade(task, outcome) for grader in (*path_graders, *task.graders)]
    grading_seconds = time.perf_counter() - grading_started

    prompt_tokens, completion_tokens, cost_usd = _usage_since(provider, task)

    return TrialResult(
        trial=trial,
        outcome=outcome,
        grades=grades,
        error=None,
        latency_seconds=latency_seconds,
        grading_seconds=grading_seconds,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )


# Per-1M-token USD prices. Deliberately a plain table rather than a live
# lookup: pricing changes, and a run's recorded cost should be
# reproducible from the report rather than depending on what an API
# returned that day. Stale entries make cost WRONG, not missing, so the
# table is small on purpose and unknown models return None (unknown),
# never 0.0 (free).
_MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def _usage_since(provider: LLMProvider, task: Task) -> tuple[int, int, float | None]:
    """Read token usage off the provider if it tracks any, and price it.

    Returns (0, 0, None) for the mock provider and for run_direct tasks:
    no API call was made, so cost is NOT APPLICABLE rather than zero. The
    None is load-bearing — a 0.0 would aggregate into a total that reads
    as "this suite run was free."
    """
    if task.run_direct is not None:
        return 0, 0, None

    usage = getattr(provider, "last_usage", None)
    if not usage:
        return 0, 0, None

    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))

    pricing = _MODEL_PRICING_PER_1M.get(provider.model)
    if pricing is None:
        return prompt_tokens, completion_tokens, None

    prompt_price, completion_price = pricing
    cost = (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000
    return prompt_tokens, completion_tokens, cost
