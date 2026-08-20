"""
report.py — renders a SuiteResult two ways:

  1. `render_markdown()` — the test-matrix table format the brief asks
     for: input -> expected tool -> expected behavior -> pass/fail,
     readable directly, one row per task (the exact shape
     evaluation_plan.md §4 already uses, generated instead of hand-filled).
  2. `render_json()` — full machine-readable dump (every trial, every
     grader's score/rationale) for programmatic inspection or diffing
     between runs.

How to read the report: see evals/README.md "Reading a suite's report".
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from evals.types import SuiteResult, TaskResult


def _expected_tool_cell(task_result: TaskResult) -> str:
    task = task_result.task
    if task.expected_tool:
        return f"`{task.expected_tool}` (required)"
    if task.forbidden_tools:
        return f"NOT {', '.join(f'`{t}`' for t in task.forbidden_tools)}"
    tools_seen: set[str] = set()
    for tr in task_result.trial_results:
        if tr.outcome is not None:
            tools_seen.update(tr.outcome.tools_called)
    return ", ".join(f"`{t}`" for t in sorted(tools_seen)) or "(none / direct)"

def _pass_fail_cell(task_result: TaskResult) -> str:
    n = len(task_result.trial_results)
    n_pass = sum(1 for t in task_result.trial_results if t.passed)
    if n <= 1:
        return "PASS" if n_pass == n else "FAIL"
    return f"{n_pass}/{n} PASS ({task_result.pass_rate:.0%})"


def render_markdown(result: SuiteResult, title: str = "Eval suite results") -> str:
    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Provider: `{result.provider_name}` / `{result.model_name}` — generated {ts}")
    lines.append("")
    lines.append(
        f"**Overall: {result.overall_pass_rate:.0%} pass rate, "
        f"{result.overall_avg_score:.2f} avg partial-credit score, "
        f"{result.total_trials} trials across {len(result.task_results)} tasks.**"
    )
    lines.append("")
    _cost = result.total_cost_usd
    # "not measured" and NOT "no API calls billed" — that phrasing was
    # false for exactly the run most likely to use it. LLM_PROVIDER=mock
    # only stops the AGENT from calling a real model; every judge-graded
    # task still builds its own OpenAILLMProvider (see
    # evals/graders/llm_judge.py get_judge_provider) and bills it
    # whenever an API key is on disk, which is the normal configured
    # state — this run's cost figure never counted those tokens even when
    # they were real and billed.
    lines.append(
        f"**Agent cost: {'not measured (mock provider)' if _cost is None else f'${_cost:.4f}'} — "
        f"latency p50 {result.p50_latency_seconds:.2f}s, p95 {result.p95_latency_seconds:.2f}s, "
        f"total {result.total_latency_seconds:.1f}s.** "
        "(Agent time and cost only. LLM-judge grading calls a SEPARATE provider instance and is "
        "neither timed nor cost-tracked here — if an OpenAI key is configured, judge-graded tasks "
        "make real, billed API calls regardless of --provider, including under mock.)"
    )
    lines.append("")

    lines.append("## Test matrix")
    lines.append("")
    lines.append("| # | Task ID | Category | Input | Expected tool | Expected behavior | Score | Pass/Fail |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, tr in enumerate(result.task_results, start=1):
        task = tr.task
        user_input = (task.user_message or "(direct code call — no agent input)").replace("|", "\\|")
        expected = task.description.replace("|", "\\|")
        lines.append(
            f"| {i} | `{task.id}` | {task.category} | {user_input} | {_expected_tool_cell(tr)} | {expected} "
            f"| {tr.avg_score:.2f} | {_pass_fail_cell(tr)} |"
        )
    lines.append("")

    lines.append("## By category")
    lines.append("")
    lines.append("| Category | Tasks | Pass rate | Avg score |")
    lines.append("|---|---|---|---|")
    for category, task_results in sorted(result.by_category().items()):
        trials = [t for tr in task_results for t in tr.trial_results]
        pass_rate = sum(1 for t in trials if t.passed) / len(trials) if trials else 0.0
        avg_score = sum(t.score for t in trials) / len(trials) if trials else 0.0
        lines.append(f"| {category} | {len(task_results)} | {pass_rate:.0%} | {avg_score:.2f} |")
    lines.append("")

    lines.append("## Per-task grader detail")
    lines.append("")
    for tr in result.task_results:
        task = tr.task
        lines.append(f"### `{task.id}` ({task.category})")
        if task.notes:
            lines.append(f"> {task.notes}")
        lines.append("")
        for trial_result in tr.trial_results:
            trial_label = f"trial {trial_result.trial.trial_index}"
            lines.append(f"- **{trial_label}** — {'PASS' if trial_result.passed else 'FAIL'} (score {trial_result.score:.2f})")
            for g in trial_result.grades:
                mark = "pass" if g.passed else "FAIL"
                lines.append(f"  - `{g.grader_name}` [{mark}, {g.score:.2f}]: {g.rationale}")
        lines.append("")

    return "\n".join(lines)


def render_json(result: SuiteResult) -> str:
    def trial_to_dict(t):
        d = {
            "trial": asdict(t.trial),
            "error": t.error,
            "score": t.score,
            "passed": t.passed,
            "grades": [asdict(g) for g in t.grades],
        }
        d["latency_seconds"] = round(t.latency_seconds, 4)
        d["grading_seconds"] = round(t.grading_seconds, 4)
        d["prompt_tokens"] = t.prompt_tokens
        d["completion_tokens"] = t.completion_tokens
        d["cost_usd"] = t.cost_usd  # None means "not measured", never "free"
        if t.outcome is not None:
            o = t.outcome
            d["outcome"] = {
                "final_text": o.trace.final_text,
                "tools_called": list(o.tools_called),
                "turns_used": o.trace.turns_used,
                "raised": o.trace.raised,
                "direct_result": repr(o.direct_result) if o.direct_result is not None else None,
                "direct_error": o.direct_error,
            }
        return d

    payload = {
        "provider": result.provider_name,
        "model": result.model_name,
        "overall_pass_rate": result.overall_pass_rate,
        "overall_avg_score": result.overall_avg_score,
        "total_trials": result.total_trials,
        "total_cost_usd": result.total_cost_usd,
        "total_latency_seconds": round(result.total_latency_seconds, 3),
        "p50_latency_seconds": round(result.p50_latency_seconds, 3),
        "p95_latency_seconds": round(result.p95_latency_seconds, 3),
        "tasks": [
            {
                "id": tr.task.id,
                "category": tr.task.category,
                "description": tr.task.description,
                "pass_rate": tr.pass_rate,
                "avg_score": tr.avg_score,
                "trials": [trial_to_dict(t) for t in tr.trial_results],
            }
            for tr in result.task_results
        ],
    }
    return json.dumps(payload, indent=2, default=str)


def render_comparison(current: SuiteResult, prior_path: str) -> str:
    """Diff this run against a prior run's JSON report.

    The missing structural piece for "how would you tell if v2 of your
    prompt is better than v1?" — a single pass rate can't answer that.
    Two runs at 82% are not necessarily the same 82%: a change that fixes
    two injection tasks and breaks two ambiguity tasks moves nothing in
    aggregate while being an obviously important regression. This reports
    PER-TASK transitions, so a swap is visible rather than netted out.

    Compares by task id, and reports ids present in only one run rather
    than dropping them — a task that disappeared between runs is usually
    a filter mistake, and silently omitting it makes the comparison lie.
    """
    with open(prior_path) as fh:
        prior = json.load(fh)

    prior_tasks = {t["id"]: t for t in prior.get("tasks", [])}
    current_tasks = {tr.task.id: tr for tr in current.task_results}

    lines: list[str] = []
    lines.append(f"# Run comparison — {prior_path}  ->  current run\n")
    lines.append(
        f"- Prior:   provider={prior.get('provider')} model={prior.get('model')} "
        f"pass={prior.get('overall_pass_rate', 0):.0%} avg_score={prior.get('overall_avg_score', 0):.2f}"
    )
    lines.append(
        f"- Current: provider={current.provider_name} model={current.model_name} "
        f"pass={current.overall_pass_rate:.0%} avg_score={current.overall_avg_score:.2f}"
    )
    if prior.get("provider") != current.provider_name or prior.get("model") != current.model_name:
        lines.append(
            "\n> **Different provider/model between runs** — differences below mix the "
            "change you made with the change in model. Compare like with like before "
            "drawing a conclusion."
        )

    delta_pass = current.overall_pass_rate - prior.get("overall_pass_rate", 0.0)
    lines.append(f"\n**Overall pass-rate delta: {delta_pass:+.1%}**\n")

    regressed: list[str] = []
    fixed: list[str] = []
    changed_score: list[tuple[str, float, float]] = []

    for task_id, tr in sorted(current_tasks.items()):
        if task_id not in prior_tasks:
            continue
        before = prior_tasks[task_id]
        before_rate, after_rate = before.get("pass_rate", 0.0), tr.pass_rate
        if before_rate >= 1.0 > after_rate:
            regressed.append(task_id)
        elif after_rate >= 1.0 > before_rate:
            fixed.append(task_id)
        before_score, after_score = before.get("avg_score", 0.0), tr.avg_score
        if abs(after_score - before_score) >= 0.05:
            changed_score.append((task_id, before_score, after_score))

    lines.append(f"## Regressed ({len(regressed)}) — passed before, fails now")
    lines.extend([f"- `{t}`" for t in regressed] or ["- none"])
    lines.append(f"\n## Fixed ({len(fixed)}) — failed before, passes now")
    lines.extend([f"- `{t}`" for t in fixed] or ["- none"])

    lines.append("\n## Score moved by >= 0.05")
    if changed_score:
        lines.append("\n| Task | Before | After | Delta |")
        lines.append("|---|---|---|---|")
        for task_id, b, a in sorted(changed_score, key=lambda x: x[2] - x[1]):
            lines.append(f"| `{task_id}` | {b:.2f} | {a:.2f} | {a - b:+.2f} |")
    else:
        lines.append("- none")

    only_prior = sorted(set(prior_tasks) - set(current_tasks))
    only_current = sorted(set(current_tasks) - set(prior_tasks))
    if only_prior or only_current:
        lines.append("\n## Task-set mismatch — the two runs did not cover the same tasks")
        if only_prior:
            lines.append(f"- Only in prior run: {', '.join(f'`{t}`' for t in only_prior)}")
        if only_current:
            lines.append(f"- Only in current run: {', '.join(f'`{t}`' for t in only_current)}")

    prior_cost, current_cost = prior.get("total_cost_usd"), current.total_cost_usd
    if prior_cost is not None or current_cost is not None:
        lines.append("\n## Cost and latency")
        lines.append(
            f"- Cost: {_fmt_cost(prior_cost)} -> {_fmt_cost(current_cost)}"
            + (
                f"  ({current_cost - prior_cost:+.4f} USD)"
                if prior_cost is not None and current_cost is not None
                else ""
            )
        )
        lines.append(
            f"- p50 latency: {prior.get('p50_latency_seconds', 0):.2f}s -> {current.p50_latency_seconds:.2f}s"
        )
        lines.append(
            f"- p95 latency: {prior.get('p95_latency_seconds', 0):.2f}s -> {current.p95_latency_seconds:.2f}s"
        )

    return "\n".join(lines) + "\n"


def _fmt_cost(value: float | None) -> str:
    """'not measured' rather than '$0.0000' when nothing reported cost —
    a mock run is not a free run, it's an unmeasured one."""
    return "not measured" if value is None else f"${value:.4f}"
