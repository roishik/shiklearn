"""
types.py — the named anatomy: Task, Trial, Trace, Outcome, Grader, Suite.

These are real dataclasses/ABCs, not dicts with string keys, so the
vocabulary is enforced by the type system rather than by convention:
Task (inputs + success criteria), Trial (one execution), Trace (what
happened), Outcome (what it produced), Grader (how it's judged), Suite
(the collection). The decomposition follows Anthropic's published
agent-eval anatomy.

Grade outcomes, not paths — with one documented exception
-----------------------------------------------------------
The default and expected way to grade a Task is against its Outcome: the
final answer text, whether a particular tool's numbers are traceable, etc.
A trial that reaches the right answer via an unexpected but valid tool
sequence should still pass.

The ONE deliberate exception is `Task.expected_tool` /
`Task.forbidden_tools`. These exist because a handful of tasks in this
suite specifically test tool-selection behavior itself (e.g. "does the
agent call compare_items when asked to compare two named items" or "does
the agent avoid calling any tool for an off-topic question"). That is a
distinct, explicit, separately-documented check — not the default
grading mode. See ToolCalledGrader / ToolNotCalledGrader in
evals/graders/deterministic.py, and evals/tasks/seed_tasks.py for which
tasks use it and why.

Isolation
---------
A Task is immutable data. Nothing in this module or in evals/runner.py
mutates a Task between trials, and every trial builds its own fresh
`messages` list and its own fresh tool registry dict (see
evals/runner.py:run_trial). Two trials of the same task, or two
different tasks, never share conversation state.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable

from app.agent_loop import ToolLogEntry


# ─────────────────────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Task:
    """One eval task: an input to the agent (or, for pure-code tasks, a
    zero-argument callable) plus the graders that decide whether the
    outcome was correct.

    Two shapes of task are supported by the same type, on purpose — a
    real eval suite needs both "does the agent behave correctly"
    (agent-level, uses `user_message`) and "is the deterministic core
    correct" (code-level, uses `run_direct`) tasks side by side. See
    seed_tasks.py tasks tagged category="scoring-direct" for the latter.

    Fields
    ------
    id                 unique slug, used as the row key in reports
    category           the realistic-failure-mode bucket this task
                        exercises (ambiguous | tool-selection |
                        self-computation | missing-data | injection |
                        explanation-quality | scoring-direct | robustness)
    description        one sentence, human-readable, shown in the report
    user_message       the input sent to run_agent(); None for
                        run_direct tasks
    history            prior turns for multi-turn tasks (tuple of the
                        project's plain-dict message shape); empty for
                        single-turn tasks
    extra_tool_registry / extra_tool_schemas
                        merged on top of app.tools.TOOL_REGISTRY /
                        TOOL_SCHEMAS for this task only — lets a task
                        exercise a tool that doesn't exist in the base
                        app (e.g. the injection-bearing
                        get_airport_advisory_note fixture) without mutating the
                        shared app.tools module
    expected_tool       see module docstring "grade outcomes, not paths"
    forbidden_tools      ditto
    graders              the Grader instances that score this task's
                        outcome; combined by simple average in
                        TrialResult.score (see below)
    num_trials           how many independent trials to run (>1 exists
                        to surface non-determinism; providers other than
                        the deterministic mock will vary run to run)
    run_direct           for scoring-direct / pure-code tasks: a
                        zero-arg callable executed instead of the agent
                        loop. Mutually exclusive with user_message.
    notes                free-text: why this task exists / what real
                        failure it's modeled on
    """

    id: str
    category: str
    description: str
    user_message: str | None = None
    history: tuple[dict[str, Any], ...] = ()
    extra_tool_registry: dict[str, Callable[[dict[str, Any]], Any]] | None = None
    extra_tool_schemas: tuple[dict[str, Any], ...] = ()
    expected_tool: str | None = None
    forbidden_tools: tuple[str, ...] = ()
    graders: tuple["Grader", ...] = ()
    num_trials: int = 1
    run_direct: Callable[[], Any] | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.run_direct is None and self.user_message is None:
            raise ValueError(f"task {self.id!r}: must set either user_message or run_direct")
        if self.run_direct is not None and self.user_message is not None:
            raise ValueError(f"task {self.id!r}: user_message and run_direct are mutually exclusive")
        if not self.graders:
            raise ValueError(f"task {self.id!r}: must have at least one grader")


# ─────────────────────────────────────────────────────────────────────────
# Trial
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Trial:
    """One attempt at a Task. Agents are non-deterministic (even the
    scripted mock provider is a stand-in for that non-determinism), so a
    Task with num_trials > 1 produces multiple Trials, each isolated from
    the others — see evals/runner.py:run_trial."""

    task_id: str
    trial_index: int  # 0-based
    provider_name: str
    model_name: str


# ─────────────────────────────────────────────────────────────────────────
# Trace
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Trace:
    """The transcript of what actually happened: every tool call, its
    arguments and result, the full message history, and the final
    answer. This is app.agent_loop.AgentResult, renamed/wrapped at the
    eval-harness boundary so this module doesn't have to import
    agent_loop's AgentResult name directly everywhere and so pure-code
    (run_direct) tasks can produce a degenerate Trace with an empty tool
    log instead of a special-cased type.

    `raised`, when set, is the exception class name a trial raised
    instead of returning normally (e.g. "MaxTurnsExceeded") — captured
    rather than propagated, because a suite run must survive one bad
    trial (see evals/runner.py)."""

    final_text: str
    messages: tuple[dict[str, Any], ...]
    tool_log: tuple[ToolLogEntry, ...]
    turns_used: int
    raised: str | None = None


# ─────────────────────────────────────────────────────────────────────────
# Outcome
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Outcome:
    """The final state a Grader actually looks at — derived from the
    Trace, not the Trace itself. This split exists because "grade
    outcomes, not paths" needs a place to put the outcome: tools_called
    is state (what ended up happening), not a path (the order/reasoning
    that got there). Graders that DO need the raw transcript (e.g. the
    injection-flagged-in-trace check) can still reach into
    `outcome.trace`.

    For run_direct (pure-code) tasks, `trace` is a degenerate/empty
    Trace and `direct_result` / `direct_error` carry the real payload.
    """

    trace: Trace
    tools_called: tuple[str, ...]
    tool_args_by_call: tuple[tuple[str, dict[str, Any]], ...]
    numbers_in_text: tuple[float, ...]
    numbers_in_tool_outputs: tuple[float, ...]
    direct_result: Any = None
    direct_error: str | None = None


# ─────────────────────────────────────────────────────────────────────────
# Grader
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GradeResult:
    """One grader's verdict on one outcome. `score` is partial credit in
    [0, 1] — NOT collapsed to a bool internally; `passed` is a
    threshold-derived convenience field a grader sets explicitly (most
    graders use score >= 0.5 or score >= 1.0 depending on how strict the
    check is; a few use a different threshold and say so in `rationale`).
    """

    grader_name: str
    score: float  # 0..1, partial credit
    passed: bool
    rationale: str
    details: dict[str, Any] = field(default_factory=dict)


class Grader(abc.ABC):
    """A grader is a named, callable assertion over (Task, Outcome).
    Two concrete families exist:
      - deterministic (evals/graders/deterministic.py): code-based, fast,
        cheap, reproducible, and — by construction — brittle to phrasing
        it wasn't written for. Good for exact-match/tolerance/tool-call
        checks.
      - LLM-as-judge (evals/graders/llm_judge.py): model-based, needed
        for open-ended text quality (does the explanation cite the right
        reasoning), nuanced but non-deterministic and requires
        calibration — see evals/judge_validation.py for why an
        unvalidated judge grader is not trustworthy on its own.
    """

    name: str

    @abc.abstractmethod
    def grade(self, task: Task, outcome: Outcome) -> GradeResult: ...


# ─────────────────────────────────────────────────────────────────────────
# Trial / Task / Suite results (aggregation, still part of the anatomy —
# these are what a report.py renders)
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class TrialResult:
    trial: Trial
    outcome: Outcome | None
    grades: list[GradeResult]
    error: str | None = None  # set if the trial itself crashed (not a grader failure)
    # Wall-clock seconds for the trial's agent run only — graders (which
    # may themselves call an LLM judge) are timed separately and NOT
    # included, or a slow judge would masquerade as a slow agent.
    latency_seconds: float = 0.0
    grading_seconds: float = 0.0
    # Token usage, when the provider reports it. Left at 0 for the mock
    # provider and for run_direct tasks, which make no API call at all —
    # 0 here means "not applicable", not "free", and cost_usd stays None
    # rather than 0.0 so the distinction survives into the report.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None

    @property
    def score(self) -> float:
        """Mean partial credit across this trial's graders. A trial with
        zero completed graders (e.g. it crashed before grading) scores 0."""
        if self.error is not None or not self.grades:
            return 0.0
        return sum(g.score for g in self.grades) / len(self.grades)

    @property
    def passed(self) -> bool:
        return self.error is None and all(g.passed for g in self.grades)


@dataclass
class TaskResult:
    task: Task
    trial_results: list[TrialResult]

    @property
    def pass_rate(self) -> float:
        if not self.trial_results:
            return 0.0
        return sum(1 for t in self.trial_results if t.passed) / len(self.trial_results)

    @property
    def avg_score(self) -> float:
        if not self.trial_results:
            return 0.0
        return sum(t.score for t in self.trial_results) / len(self.trial_results)


@dataclass
class SuiteResult:
    task_results: list[TaskResult]
    provider_name: str
    model_name: str

    @property
    def total_trials(self) -> int:
        return sum(len(tr.trial_results) for tr in self.task_results)

    @property
    def overall_pass_rate(self) -> float:
        all_trials = [t for tr in self.task_results for t in tr.trial_results]
        if not all_trials:
            return 0.0
        return sum(1 for t in all_trials if t.passed) / len(all_trials)

    @property
    def overall_avg_score(self) -> float:
        all_trials = [t for tr in self.task_results for t in tr.trial_results]
        if not all_trials:
            return 0.0
        return sum(t.score for t in all_trials) / len(all_trials)

    @property
    def total_cost_usd(self) -> float | None:
        """Summed cost across trials that reported one. None when NO trial
        reported cost (mock provider, or run_direct-only runs) — reporting
        $0.00 there would read as "this run was free" rather than "cost was
        never measured", and those are different claims."""
        costs = [
            t.cost_usd
            for tr in self.task_results
            for t in tr.trial_results
            if t.cost_usd is not None
        ]
        return sum(costs) if costs else None

    @property
    def total_latency_seconds(self) -> float:
        return sum(t.latency_seconds for tr in self.task_results for t in tr.trial_results)

    @property
    def p50_latency_seconds(self) -> float:
        return self._latency_percentile(0.50)

    @property
    def p95_latency_seconds(self) -> float:
        """Tail latency. Reported alongside the mean because an agent loop's
        cost distribution is long-tailed by construction — one task that
        burns every one of its max_turns dominates the tail while barely
        moving the average, and the tail is what a user actually feels."""
        return self._latency_percentile(0.95)

    def _latency_percentile(self, q: float) -> float:
        values = sorted(
            t.latency_seconds for tr in self.task_results for t in tr.trial_results if t.latency_seconds > 0
        )
        if not values:
            return 0.0
        # Nearest-rank; exact interpolation is false precision at these n.
        index = min(len(values) - 1, int(round(q * (len(values) - 1))))
        return values[index]

    def by_category(self) -> dict[str, list[TaskResult]]:
        out: dict[str, list[TaskResult]] = {}
        for tr in self.task_results:
            out.setdefault(tr.task.category, []).append(tr)
        return out
