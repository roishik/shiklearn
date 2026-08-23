"""
sensitivity_report.py -- how much do the DEFAULT_CRITERIA weights actually
matter to the ranking, honestly measured against the real data.

ONE-OFF ANALYSIS UTILITY. Not part of the pytest suite (pytest.ini's
testpaths is "tests" only, so this is never collected), and not meant to
be. Run it by hand:

    PYTHONPATH=. python scripts/sensitivity_report.py

It answers, with real numbers computed by app/tools.py + app/scoring.py
against the real 104 eligible localities -- never fabricated, never
hand-typed:

  1. The full default ranking (compare_items on every eligible locality).
  2. For each of the 5 criteria: the smallest weight-multiplier change
     that flips the #1 spot, and who it flips to (app.tools.
     weight_robustness_report finds the flip factor; app.tools.
     analyze_weight_sensitivity, called at that exact factor, names the
     new winner and the resulting Kendall's tau).
  3. Rank stability at a FIXED, comparable perturbation (+-50% on each
     criterion's weight in turn) -- the flip-point search above finds
     different-sized changes for different criteria, which makes them
     hard to compare to each other; +-50% is the same probe applied to
     all five, so the resulting tau values ARE comparable across criteria.
  4. Which localities at the top of the default ranking sit within
     DECISIVE_SCORE_GAP (app.tools' own "too close to call" threshold,
     currently 0.005 = 0.5% of total_score) of the #1 spot.
  5. A plain-English summary generated FROM the numbers above, not
     written in advance of running them. If a run of this script
     produces different numbers (a data refresh, a weight change), the
     summary changes with it -- it is templated from the measured
     values, not a fixed paragraph with numbers dropped in.

Every number below came from calling the real, tested functions in
app/tools.py (compare_items, weight_robustness_report,
analyze_weight_sensitivity) against app/dataset.py's real 104 eligible
localities -- nothing here reimplements scoring, sweeping, or Kendall's
tau; app/scoring.py already has all three (score_item/rank_items,
find_weight_flip_point, and the sensitivity_analysis kendall_tau field),
and app/tools.py already exposes them as tools an LLM can call. This
script is a batch driver over those same entry points, not a parallel
implementation that could silently disagree with what the agent itself
would report.

Writes the same report (as markdown) to
evals/results/sensitivity_<UTC timestamp>.md so it can be cited by exact
filename from the docs.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import dataset  # noqa: E402
from app.tools import (  # noqa: E402
    DECISIVE_SCORE_GAP,
    DEFAULT_CRITERIA,
    analyze_weight_sensitivity,
    compare_items,
    weight_robustness_report,
)

ALL_ELIGIBLE_IDS: list[str] = list(dataset.ELIGIBLE_IDS)

# A fixed, comparable probe applied to EVERY criterion in turn, in
# addition to each criterion's own flip point (which is a different-sized
# change per criterion and therefore not comparable across criteria).
# +50% is a large-but-plausible "what if this weight is wrong by half"
# question -- big enough to show real churn if the ranking has any, not
# so big it is a strawman.
FIXED_PROBE_FACTOR_UP = 1.5
FIXED_PROBE_FACTOR_DOWN = 0.5


def _name(item_id: str) -> str:
    row = dataset.LOCALITIES.get(item_id, {})
    return f"{row.get('name_en', item_id)} ({row.get('name_he', '')})"


def _fmt_row(r: dict) -> str:
    return (
        f"{r['rank']:>4}  {r['item_id']:<6}  {r['name_en'] or '':<28}  "
        f"{r['total_score']:.4f}  covered={r['covered_weight']:.2f}"
    )


def section_default_ranking(lines: list[str]) -> dict:
    result = compare_items(item_ids=ALL_ELIGIBLE_IDS)
    ranking = result["ranking"]
    excluded = result.get("excluded", [])

    lines.append("## 1. Full default ranking\n")
    lines.append(
        f"All {len(ALL_ELIGIBLE_IDS)} eligible localities scored on DEFAULT_CRITERIA "
        f"(price_level 30, accessibility 20, price_momentum_vs_country 20, "
        f"socioeconomic_level 15, rental_yield 15). {len(ranking)} ranked, "
        f"{len(excluded)} excluded for insufficient criterion coverage.\n"
    )
    lines.append("```")
    lines.append(f"{'rank':>4}  {'id':<6}  {'name_en':<28}  score   coverage")
    for r in ranking:
        lines.append(_fmt_row(r))
    lines.append("```")
    if excluded:
        lines.append("\nExcluded (covered_weight below the 0.5 coverage floor):\n")
        for e in excluded:
            lines.append(f"- {e['item_id']}: covered_weight={e['covered_weight']:.2f}, {e['reason']}")
    lines.append("")
    return result


def section_flip_points(lines: list[str]) -> tuple[dict, dict[str, dict]]:
    robustness = weight_robustness_report(item_ids=ALL_ELIGIBLE_IDS)
    per_criterion: dict[str, dict] = {}

    lines.append("## 2. Weight flip points -- how big a change flips #1, and to whom\n")
    lines.append(
        "For each criterion: the smallest multiplier on that ONE criterion's weight "
        "(all others held fixed) that changes the #1-ranked locality, found by a linear "
        "sweep up to 10x in either direction (app.scoring.find_weight_flip_point, "
        "resolution 0.05x). `None` means no change up to 10x in either direction actually "
        "changes the winner within the searched range.\n"
    )
    lines.append(
        f"Baseline #1: **{robustness['baseline_top']}** ({_name(robustness['baseline_top'])})\n"
    )
    lines.append("```")
    lines.append(f"{'criterion':<28} {'weight':>7} {'flip x':>8}  flips to")
    for finding in robustness["criteria"]:
        name = finding["criterion"]
        flip = finding["flip_factor"]
        if flip is None:
            per_criterion[name] = {"flip_factor": None, "flip_result": None}
            lines.append(f"{name:<28} {finding['current_weight']:>7} {'none':>8}  (never flips within 10x)")
            continue
        detail = analyze_weight_sensitivity(item_ids=ALL_ELIGIBLE_IDS, criterion=name, factor=flip)
        per_criterion[name] = {"flip_factor": flip, "flip_result": detail}
        new_top = detail["top_item"]["after"]
        pct = (flip - 1.0) * 100
        lines.append(
            f"{name:<28} {finding['current_weight']:>7} {flip:>7.3f}x  "
            f"-> {new_top} ({dataset.LOCALITIES.get(new_top, {}).get('name_en', '?')})  "
            f"[{pct:+.0f}% weight change, tau={detail['kendall_tau']:.3f}]"
        )
    lines.append("```\n")
    return robustness, per_criterion


def section_fixed_probe(lines: list[str]) -> dict[str, dict]:
    lines.append("## 3. Rank stability at a FIXED +-50% weight change (comparable across criteria)\n")
    lines.append(
        "Flip points above are different sizes per criterion, so they cannot be compared "
        "to each other directly (a criterion that flips at 1.4x is not necessarily "
        "'more fragile' than one that flips at -0.7x if the first is a 40% change and the "
        f"second a 30% one -- but they ARE directionally comparable). This section applies "
        f"the SAME {FIXED_PROBE_FACTOR_DOWN}x/{FIXED_PROBE_FACTOR_UP}x probe to every "
        "criterion, so the resulting Kendall's tau values are apples-to-apples: how much "
        "does the WHOLE ranking (not just #1) churn under an equal-sized nudge to each "
        "criterion in turn.\n"
    )
    lines.append("```")
    lines.append(f"{'criterion':<28} {'-50% top':<10} {'tau':>7}   {'+50% top':<10} {'tau':>7}")
    results: dict[str, dict] = {}
    for criterion in DEFAULT_CRITERIA:
        name = criterion.name
        down = analyze_weight_sensitivity(
            item_ids=ALL_ELIGIBLE_IDS, criterion=name, factor=FIXED_PROBE_FACTOR_DOWN
        )
        up = analyze_weight_sensitivity(
            item_ids=ALL_ELIGIBLE_IDS, criterion=name, factor=FIXED_PROBE_FACTOR_UP
        )
        results[name] = {"down": down, "up": up}
        lines.append(
            f"{name:<28} {down['top_item']['after']:<10} {down['kendall_tau']:>7.3f}   "
            f"{up['top_item']['after']:<10} {up['kendall_tau']:>7.3f}"
        )
    lines.append("```\n")
    lines.append(
        "tau = +1.000 means the ORDER is completely untouched; lower means real churn. "
        "This is over the whole ranked list, not just whether #1 changed -- a criterion can "
        "leave #1 alone and still reshuffle the rest, or vice versa.\n"
    )
    return results


def section_near_ties(lines: list[str], baseline_ranking: list[dict]) -> list[dict]:
    lines.append(f"## 4. Near-ties at the top (within DECISIVE_SCORE_GAP = {DECISIVE_SCORE_GAP})\n")
    if not baseline_ranking:
        lines.append("No ranked items.\n")
        return []
    top_score = baseline_ranking[0]["total_score"]
    tied = [r for r in baseline_ranking if top_score - r["total_score"] <= DECISIVE_SCORE_GAP]
    if len(tied) < 2:
        lines.append(
            f"Nobody else is within {DECISIVE_SCORE_GAP} of #1 "
            f"({baseline_ranking[0]['item_id']}, score {top_score:.4f}). The #1 spot is a "
            "clean leader by this measure, whatever the flip-point analysis above says about "
            "how much a WEIGHT change could move it.\n"
        )
        return []
    lines.append(
        f"{len(tied)} localities sit within {DECISIVE_SCORE_GAP} of the #1 score "
        f"({top_score:.4f}) -- a gap this small is noise relative to the judgement calls "
        "in the weights themselves, not a real difference in value-for-money:\n"
    )
    lines.append("```")
    for r in tied:
        lines.append(_fmt_row(r))
    lines.append("```\n")
    return tied


def build_summary(
    baseline_ranking: list[dict],
    robustness: dict,
    per_criterion_flip: dict[str, dict],
    fixed_probe: dict[str, dict],
    tied_at_top: list[dict],
) -> list[str]:
    lines = ["## 5. Plain-English summary\n"]
    if not baseline_ranking:
        lines.append("No localities were ranked at all -- nothing to summarize.\n")
        return lines

    leader = baseline_ranking[0]
    lines.append(
        f"The default ranking's #1 locality is **{_name(leader['item_id'])}** "
        f"(id {leader['item_id']}), total_score {leader['total_score']:.4f}.\n"
    )

    # Which criteria are load-bearing (flip within a plausible range) vs not.
    load_bearing = [
        (name, info["flip_factor"])
        for name, info in per_criterion_flip.items()
        if info["flip_factor"] is not None and abs(info["flip_factor"] - 1.0) <= 0.5
    ]
    not_load_bearing = [name for name, info in per_criterion_flip.items() if info["flip_factor"] is None]

    if load_bearing:
        parts = ", ".join(f"{name} ({factor:.2f}x)" for name, factor in load_bearing)
        lines.append(
            f"**The ranking is fragile with respect to {len(load_bearing)} of 5 criteria**: "
            f"{parts} each flip the #1 spot with a weight change of 50% or less. A ranking "
            "that changes its answer on a weight nudge this small should be presented as "
            "close, not as a confident recommendation.\n"
        )
    else:
        lines.append(
            "No criterion flips the #1 spot with a weight change of 50% or less -- by that "
            "bar, the #1 result is NOT hanging on any single weight's exact value.\n"
        )

    if not_load_bearing:
        lines.append(
            f"**{len(not_load_bearing)} criteria never flip the #1 spot even at a 10x weight "
            f"change**: {', '.join(not_load_bearing)}. Their precise weight is not "
            "load-bearing for who wins first place (it can still matter further down the "
            "ranking -- see the tau values above, which are computed over the WHOLE order, "
            "not just #1).\n"
        )

    # weight_robustness_report's OWN verdict, quoted rather than
    # recomputed -- app.tools already picks out the single tightest flip
    # point (most_sensitive_criterion) and states it in one sentence
    # (summary). Restating that in the model's own words here, instead of
    # only in the per-criterion table above, is the same discipline this
    # whole project applies to the agent itself: the tool computes the
    # verdict, prose quotes it, nothing upstream re-derives it a second,
    # possibly-diverging way.
    lines.append(f"By `weight_robustness_report`'s own account: {robustness['summary']}\n")

    # Average tau across the fixed +-50% probe, both directions, all criteria.
    all_tau = [v["down"]["kendall_tau"] for v in fixed_probe.values()] + [
        v["up"]["kendall_tau"] for v in fixed_probe.values()
    ]
    min_tau = min(all_tau)
    avg_tau = sum(all_tau) / len(all_tau)
    lines.append(
        f"Across a fixed +-50% probe applied to each of the 5 criteria in turn (10 "
        f"reweighted rankings total), Kendall's tau against the default ranking ranges "
        f"down to {min_tau:.3f} and averages {avg_tau:.3f}. "
        + (
            "Tau this low means a plausible weight error does not just nudge the ranking, "
            "it meaningfully reshuffles it."
            if min_tau < 0.7
            else "Even the most disruptive +-50% probe leaves the order mostly intact."
        )
        + "\n"
    )

    if tied_at_top:
        ids = ", ".join(_name(r["item_id"]) for r in tied_at_top)
        lines.append(
            f"**{len(tied_at_top)} localities are within DECISIVE_SCORE_GAP of #1**: {ids}. "
            "A gap this small (0.5% of total_score) is not a real difference given the "
            "judgement calls baked into the weights -- these should be reported as "
            "effectively tied, not as a single clear winner.\n"
        )
    else:
        lines.append(
            "No locality sits within DECISIVE_SCORE_GAP of #1 -- on the score alone (as "
            "opposed to the weight-sensitivity question above), the #1 spot is not a "
            "near-tie.\n"
        )

    # Overall verdict, computed from the two independent signals above rather
    # than asserted: fragile-on-weights AND/OR a real near-tie both count.
    fragile = bool(load_bearing) or bool(tied_at_top) or min_tau < 0.7
    lines.append(
        "\n**Overall**: "
        + (
            "the ranking is FRAGILE -- either the #1 spot flips on a plausible weight "
            "change, real localities sit within noise of #1's score, or a +-50% probe "
            "meaningfully reorders the list. Report the top result as 'close' rather than "
            "decisive, and lead with the flip-point evidence above when asked to defend it."
            if fragile
            else "the ranking is ROBUST by every measure checked here -- no criterion flips "
            "#1 within 50%, nobody sits within DECISIVE_SCORE_GAP of #1, and the +-50% "
            "probe leaves the order largely intact."
        )
        + "\n"
    )
    return lines


def main() -> int:
    lines: list[str] = []
    lines.append("# Sensitivity report -- DEFAULT_CRITERIA weights vs. the real 104-locality dataset\n")
    lines.append(
        f"Generated {_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} "
        "by scripts/sensitivity_report.py, against the committed "
        "data/processed_data/localities.json (104 eligible localities) and the live "
        "DEFAULT_CRITERIA in app/tools.py. Every number below came from calling "
        "app.tools.compare_items / weight_robustness_report / analyze_weight_sensitivity "
        "directly -- nothing here is a separate, possibly-disagreeing reimplementation.\n"
    )

    default_result = section_default_ranking(lines)
    baseline_ranking = default_result["ranking"]

    robustness, per_criterion_flip = section_flip_points(lines)
    fixed_probe = section_fixed_probe(lines)
    tied_at_top = section_near_ties(lines, baseline_ranking)
    lines.extend(build_summary(baseline_ranking, robustness, per_criterion_flip, fixed_probe, tied_at_top))

    report = "\n".join(lines)
    print(report)

    results_dir = Path(__file__).resolve().parent.parent / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = results_dir / f"sensitivity_{stamp}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nwrote {out_path.relative_to(Path(__file__).resolve().parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
