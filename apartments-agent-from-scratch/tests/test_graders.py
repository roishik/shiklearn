"""Tests for the eval graders themselves.

A grader nobody tests is an unvalidated measuring instrument — it will
happily report confident nonsense, and every number downstream of it
inherits the error. These specifically pin the fix for the
ScoringMatchesGroundTruthGrader gap documented in evals/README.md.
"""
from __future__ import annotations

import pytest

from evals.graders.deterministic import (
    ScoringMatchesGroundTruthGrader,
    _looks_like_a_clarifying_question,
)
from evals.types import Outcome, Task, Trace


def _outcome(final_text: str, tool_log=()) -> Outcome:
    """An Outcome with no successful compare_items call, unless one is
    supplied — the state the fixed branch of the grader handles."""
    trace = Trace(final_text=final_text, messages=(), tool_log=tuple(tool_log), turns_used=1)
    return Outcome(
        trace=trace,
        tools_called=tuple(e.tool_name for e in tool_log),
        tool_args_by_call=(),
        numbers_in_text=(),
        numbers_in_tool_outputs=(),
    )


# Task validates that it carries at least one grader, so supply the one
# under test. It's unused by these assertions — every check here calls
# grader.grade() directly rather than going through the runner.
_TASK = Task(
    id="t",
    category="self-computation",
    description="d",
    user_message="m",
    graders=(ScoringMatchesGroundTruthGrader(),),
)


# ── the clarifying-question detector ────────────────────────────────────
@pytest.mark.parametrize(
    "text",
    [
        "Which items would you like me to compare?",
        "Could you specify which options you mean?",
        "I can help with that — please let me know which ones to rank.",
        "Happy to compare these. Can you clarify which items?",
    ],
)
def test_detects_a_clarifying_question(text):
    assert _looks_like_a_clarifying_question(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "I compared which options were cheapest and option_a won.",  # phrase, no question mark
        "Option A is the best choice overall.",  # neither
        "Isn't that interesting?",  # question mark, no relevant phrasing
        "",
    ],
)
def test_does_not_mistake_prose_for_a_clarifying_question(text):
    """A statement-shaped phrasing without a question mark must not
    count, or a non-answer would score as correct."""
    assert _looks_like_a_clarifying_question(text) is False


def test_explicit_request_counts_without_a_question_mark():
    """The two-tier rule. Models routinely phrase a request as a polite
    imperative; requiring '?' would score that as a non-answer."""
    assert _looks_like_a_clarifying_question("Please specify which items to compare.") is True


def test_ambiguous_phrasing_still_requires_a_question_mark():
    """The other tier — the same words inside a statement must not count,
    which is what stops the detector rewarding prose that answers
    nothing."""
    assert _looks_like_a_clarifying_question("I ranked which options were cheapest.") is False
    assert _looks_like_a_clarifying_question("Which options should I rank?") is True


# ── the actual bug: "no tool called" is two opposite behaviors ──────────
def test_no_tool_but_asked_for_specifics_scores_full_marks():
    """THE FIX. Previously scored 0.0, punishing the model for doing
    exactly what NEVER_COMPUTE_RULE demands: refusing to guess and asking
    which items to compare."""
    grader = ScoringMatchesGroundTruthGrader()
    result = grader.grade(
        _TASK,
        _outcome("I can't estimate scores myself. Which items would you like me to compare?"),
    )
    assert result.score == 1.0
    assert result.passed is True


def test_no_tool_but_stated_numbers_still_fails():
    """The other half of the same branch — silently guessing must remain
    a hard failure. If the fix scored this leniently it would have traded
    one blind spot for a worse one."""
    grader = ScoringMatchesGroundTruthGrader()
    result = grader.grade(
        _TASK,
        _outcome("Option A scores about 0.72 and option B about 0.65, so A wins."),
    )
    assert result.score == 0.0
    assert result.passed is False


def test_no_tool_no_numbers_no_question_scores_partial():
    """Neither fabrication nor a useful response — deliberately in
    between, so it can't be confused with either."""
    grader = ScoringMatchesGroundTruthGrader()
    result = grader.grade(_TASK, _outcome("I am unable to help with that."))
    assert result.score == pytest.approx(0.5)
    assert result.passed is False


def test_politely_phrased_fabrication_is_still_a_failure():
    """Deliberately conservative: numbers present means failure REGARDLESS
    of how the answer is phrased. A fabrication wrapped in a question must
    not sneak through the clarifying-question path."""
    grader = ScoringMatchesGroundTruthGrader()
    result = grader.grade(
        _TASK,
        _outcome("I'd estimate option A at 0.72 — which items did you want me to compare?"),
    )
    assert result.score == 0.0
    assert result.passed is False


def test_extract_stated_numbers_handles_commas_and_percent():
    """Two grader gaps found live in P4, both invisible in the old mock
    domain: comma-formatted large numbers (36,497,303.0 -> was read as
    303.0) and percent-formatted rates (traffic_growth 0.0468 written as
    '4.68%', which the grader used to compare digit-for-digit against
    the un-scaled pool and flag as a fabricated 100x-too-large number)."""
    from evals.graders.deterministic import _extract_stated_numbers

    out = _extract_stated_numbers(
        "SFO scores 9,124,325.75 with traffic growth of 4.68% and a plain 0.3584 total."
    )
    assert out == pytest.approx([9124325.75, 0.0468, 0.3584])


# ── judge_text's response parser ─────────────────────────────────────────
# Every judge score in the project flows through this. Untested before —
# a parse failure returns score=0.0/passed=False, indistinguishable in an
# aggregate pass rate from the model genuinely failing the rubric.
def test_judge_text_parses_the_plain_instructed_format():
    from evals.graders.llm_judge import judge_text

    class FakeProvider:
        def chat(self, messages, tools=None):
            class R:
                content = "SCORE: 8\nRATIONALE: solid answer, minor phrasing issue."
            return R()

    score, rationale = judge_text(FakeProvider(), "prompt")
    assert score == 8
    assert "solid answer" in rationale


def test_judge_text_survives_markdown_wrapping_around_score():
    """Observed in practice: a judge model wraps its own instructed
    format in bold. The strict `SCORE:\\s*(\\d+)` form returns None for
    both of these, which folds a parse failure into a 0.0 grade
    indistinguishable from the judge actually failing the rubric."""
    from evals.graders.llm_judge import judge_text

    for wrapped in ("**SCORE:** 8\nRATIONALE: ok.", "SCORE: **8**\nRATIONALE: ok."):
        class FakeProvider:
            def chat(self, messages, tools=None, _text=wrapped):
                class R:
                    content = _text
                return R()

        score, _ = judge_text(FakeProvider(), "prompt")
        assert score == 8, f"failed to parse: {wrapped!r}"


def test_judge_text_returns_none_score_on_a_genuinely_unparseable_response():
    from evals.graders.llm_judge import judge_text

    class FakeProvider:
        def chat(self, messages, tools=None):
            class R:
                content = "I think this is pretty good overall."
            return R()

    score, rationale = judge_text(FakeProvider(), "prompt")
    assert score is None
    assert rationale  # falls back to the raw text rather than being empty
