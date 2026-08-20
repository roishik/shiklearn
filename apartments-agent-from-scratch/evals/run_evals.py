"""
run_evals.py — CLI entrypoint. Runs the seeded Suite for real against a
configured LLMProvider and writes a markdown + JSON report.

Usage (from the repo root, with the venv active or via
.venv/bin/python):

    python -m evals.run_evals                       # LLM_PROVIDER=mock (default, zero setup)
    python -m evals.run_evals --provider openai      # real OpenAI key from .env
    python -m evals.run_evals --category injection   # only run one category
    python -m evals.run_evals --trials 3             # override every task's num_trials

Writes evals/results/<provider>_<timestamp>.md and .json, and prints the
overall pass rate / avg score to stdout.

IMPORTANT ordering note (bit us once, worth keeping the comment): the
--provider flag works by setting the LLM_PROVIDER env var BEFORE
anything under app/ or evals/ is imported. app/config.py reads
LLM_PROVIDER from the environment exactly once, at module-import time
(see its module docstring), and app/providers/llm/__init__.py imports
that frozen value at ITS import time too — so if any app.*/evals.*
import happens before argparse runs, --provider silently has no effect.
That's why argparse runs first, in `main()`, before the deferred
imports below it.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `python evals/run_evals.py` also works


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["mock", "openai", "anthropic"], default=None, help="Sets LLM_PROVIDER.")
    parser.add_argument("--category", default=None, help="Only run tasks in this category.")
    parser.add_argument("--id-contains", default=None, help="Only run tasks whose id contains this substring.")
    parser.add_argument("--trials", type=int, default=None, help="Override every task's num_trials.")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log the exact rendered prompt sent to the LLM judge for every LLMJudgeGrader call "
        "(evals/graders/llm_judge.py's logger.debug line) to stderr.",
    )
    parser.add_argument(
        "--compare",
        default=None,
        metavar="PRIOR_RUN.json",
        help="Diff this run against a prior run's JSON report: per-task regressions, "
        "fixes, score moves, and cost/latency deltas. A single pass rate can't tell "
        "you whether v2 of a prompt is better than v1 — two runs at 82%% can be "
        "different 82%%s.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

    # MUST happen before any app.*/evals.* import — see module docstring.
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    from app.providers.llm import get_llm_provider  # noqa: E402  (deliberately deferred, see above)
    from evals.report import render_json, render_markdown  # noqa: E402
    from evals.suite import Suite  # noqa: E402
    from evals.tasks.seed_tasks import TASKS  # noqa: E402

    provider = get_llm_provider()
    if args.provider and provider.name != args.provider:
        print(f"WARNING: requested --provider {args.provider!r} but got {provider.name!r} — check env/config.", file=sys.stderr)

    suite = Suite(tasks=list(TASKS)).filter(category=args.category, id_substring=args.id_contains)
    if not suite.tasks:
        print("No tasks matched the given filters.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(suite.tasks)} task(s) against provider={provider.name} model={provider.model} ...")
    result = suite.run(provider, trials_override=args.trials)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path = out_dir / f"{provider.name}_{ts}.md"
    json_path = out_dir / f"{provider.name}_{ts}.json"
    md_path.write_text(render_markdown(result, title=f"airport-agent eval suite — provider={provider.name}"))
    json_path.write_text(render_json(result))

    print()
    print(f"Overall: {result.overall_pass_rate:.0%} pass rate, {result.overall_avg_score:.2f} avg score, "
          f"{result.total_trials} trials across {len(result.task_results)} tasks.")
    cost = result.total_cost_usd
    print(
        f"Cost: {'not measured' if cost is None else f'${cost:.4f}'} | "
        f"latency p50 {result.p50_latency_seconds:.2f}s, p95 {result.p95_latency_seconds:.2f}s"
    )
    print(f"Report written to:\n  {md_path}\n  {json_path}")

    if args.compare:
        from evals.report import render_comparison  # noqa: E402

        try:
            comparison = render_comparison(result, args.compare)
        except FileNotFoundError:
            print(f"\n--compare: no such prior report: {args.compare}", file=sys.stderr)
            sys.exit(1)
        cmp_path = out_dir / f"compare_{provider.name}_{ts}.md"
        cmp_path.write_text(comparison)
        print()
        print(comparison)
        print(f"Comparison written to:\n  {cmp_path}")


if __name__ == "__main__":
    main()
