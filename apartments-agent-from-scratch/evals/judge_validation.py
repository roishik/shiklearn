"""
judge_validation.py — judge-vs-human agreement script.

An unvalidated LLM judge is a random number generator with good PR. This
script is the answer: it runs the SAME rubric prompt and SAME parsing
logic
(evals.graders.llm_judge.judge_text — literally the function
LLMJudgeGrader.grade() calls at eval time, not a lookalike) against
evals/judge_calibration_data.py's 10 hand-labeled examples, and reports
real agreement between the judge's verdict and the human's.

Two agreement numbers are reported:
  - Binary pass/fail agreement (score>=7 threshold on both sides — same
    threshold LLMJudgeGrader uses by default)
  - Mean absolute score difference (1-10 scale)

Verdict threshold (from the brief's "intern test", §1.3 "Rubric-writing
detail worth stealing"): >=80% binary agreement means the rubric is
specific enough to automate. Below that, the judge should not be trusted
without human spot-checks, and this script says so explicitly rather
than a bare percentage.

Usage:
    python -m evals.judge_validation

Requires OPENAI_API_KEY (the judge always uses a real LLM — see
evals/graders/llm_judge.py's module docstring for why). Prints a
plain-text table and an explicit VALIDATED / NOT YET RELIABLE verdict,
and writes evals/results/judge_validation_<timestamp>.md.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402

from evals.graders.llm_judge import (  # noqa: E402
    RUBRIC_EXPLANATION_CITES_REASONING,
    get_judge_provider,
    judge_text,
)
from evals.judge_calibration_data import CALIBRATION_SET  # noqa: E402

AGREEMENT_THRESHOLD = 0.80  # the "intern test" threshold from the research brief


def main() -> None:
    provider = get_judge_provider()
    if provider is None:
        print(
            "No OPENAI_API_KEY configured (see app/config.py) — cannot run the judge. "
            "This script requires a real LLM; there is no mock judge (see evals/graders/llm_judge.py).",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = []
    agree_count = 0
    abs_diffs = []
    for ex in CALIBRATION_SET:
        prompt = RUBRIC_EXPLANATION_CITES_REASONING.format(
            tool_context=json.dumps(ex.tool_context, indent=2), final_text=ex.final_text
        )
        judge_score, judge_rationale = judge_text(provider, prompt)
        judge_pass = (judge_score or 0) >= 7
        agree = judge_pass == ex.human_pass
        agree_count += int(agree)
        if judge_score is not None:
            abs_diffs.append(abs(judge_score - ex.human_score))
        rows.append(
            {
                "id": ex.id,
                "human_score": ex.human_score,
                "human_pass": ex.human_pass,
                "judge_score": judge_score,
                "judge_pass": judge_pass,
                "agree": agree,
                "judge_rationale": judge_rationale,
                "human_rationale": ex.human_rationale,
            }
        )

    n = len(CALIBRATION_SET)
    binary_agreement = agree_count / n
    mean_abs_diff = sum(abs_diffs) / len(abs_diffs) if abs_diffs else float("nan")

    lines: list[str] = []
    lines.append(f"# Judge-vs-human agreement — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append(f"Rubric: RUBRIC_EXPLANATION_CITES_REASONING. Calibration set: {n} hand-labeled examples "
                  f"(evals/judge_calibration_data.py). Pass threshold: score >= 7/10 on both sides.")
    lines.append("")
    lines.append(f"**Binary pass/fail agreement: {agree_count}/{n} = {binary_agreement:.0%}**")
    lines.append(f"**Mean absolute score difference: {mean_abs_diff:.2f} (1-10 scale)**")
    lines.append("")
    if binary_agreement >= AGREEMENT_THRESHOLD:
        verdict = (
            f"VALIDATED — {binary_agreement:.0%} >= the {AGREEMENT_THRESHOLD:.0%} 'intern test' threshold. "
            "This judge/rubric pair is specific enough to automate for RUBRIC_EXPLANATION_CITES_REASONING-shaped "
            "tasks. Still spot-check transcripts by hand periodically — see README 'Read transcripts by hand.'"
        )
    else:
        verdict = (
            f"NOT YET RELIABLE — {binary_agreement:.0%} < the {AGREEMENT_THRESHOLD:.0%} threshold. Do not trust "
            "this judge's grades on their own; the disagreements below are the concrete cases to look at, "
            "tighten the rubric anchors against, and re-run."
        )
    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    lines.append("| Example | Human score | Human pass | Judge score | Judge pass | Agree? |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| `{r['id']}` | {r['human_score']} | {r['human_pass']} | {r['judge_score']} | {r['judge_pass']} "
            f"| {'yes' if r['agree'] else 'NO'} |"
        )
    lines.append("")
    lines.append("## Disagreements — read these first")
    lines.append("")
    disagreements = [r for r in rows if not r["agree"]]
    if not disagreements:
        lines.append("(none)")
    for r in disagreements:
        lines.append(f"### `{r['id']}`")
        lines.append(f"- Human ({r['human_score']}/10): {r['human_rationale']}")
        lines.append(f"- Judge ({r['judge_score']}/10): {r['judge_rationale']}")
        lines.append("")

    lines.append("## All rationales")
    lines.append("")
    for r in rows:
        lines.append(f"### `{r['id']}`")
        lines.append(f"- Human ({r['human_score']}/10): {r['human_rationale']}")
        lines.append(f"- Judge ({r['judge_score']}/10): {r['judge_rationale']}")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"judge_validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.md"
    out_path.write_text(report)
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
