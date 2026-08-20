"""
calibrate_resolver.py — pick MIN_RELEVANCE from data, not from feel.

Scores two hand-labelled sets of (query, candidate_text) pairs — ones
that SHOULD match and ones that should NOT — and reports the separation
between them. Exits non-zero if the classes overlap, i.e. if no single
threshold can cleanly divide them.

Run it after touching any weight or metric in app/entity_resolution.py:

    PYTHONPATH=. python scripts/calibrate_resolver.py

This exists because the first blend FAILED it: whole-string Jaro-Winkler
at weight 0.6 scored 'the weather in paris' ~ 'the alpha option' at
0.5420, above the genuine match 'alpha' ~ 'the alpha option' at 0.4825 —
negative separation, no workable threshold. The fix was a new signal
(length-weighted token containment), not a tuned constant. Keep this
script honest: when this is pointed at a real dataset, replace these pairs
with real ones from that dataset before trusting the number.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the script runnable as `python scripts/calibrate_resolver.py` from
# the project root, not only via `PYTHONPATH=. python -m ...`. A
# verification script that only works when invoked one exact way will
# eventually be reported as "passing" by whoever invokes it the other way
# and sees no output — silence is not success.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.entity_resolution import MIN_RELEVANCE, score_pair  # noqa: E402

SHOULD_MATCH: list[tuple[str, str]] = [
    ("option_a", "Option A"),
    ("Alpha", "Alpha"),
    ("alpha", "the alpha option"),
    ("Charlie", "Charlie"),
    ("charlie", "the charlie option"),
    ("Optn B", "Option B"),  # typo
    ("Bravo", "Bravo"),
    ("brovo", "Bravo"),  # phonetic misspelling
    ("opshun c", "Option C"),  # heavy phonetic misspelling
    ("OPTION_A", "option_a"),  # case only
    ("option a", "option-a"),  # punctuation only
]

SHOULD_NOT_MATCH: list[tuple[str, str]] = [
    ("quantum tunnelling diode", "the charlie option"),
    ("quantum tunnelling diode", "Option A"),
    ("zzzzz", "Alpha"),
    ("the weather in paris", "the alpha option"),  # shared stopword — the original failure
    ("the price of tea", "the bravo option"),  # ditto
    ("database migration", "Option B"),
    ("helicopter", "Charlie"),
    ("supercalifragilistic", "the bravo option"),
    ("12345", "Option C"),
    ("what is the weather", "the alpha option"),
]


def main() -> int:
    print(f"{'score':>8}  {'query':<28}  candidate")
    print("-" * 72)

    match_scores: list[float] = []
    print("SHOULD MATCH")
    for query, candidate in SHOULD_MATCH:
        score, _ = score_pair(query, candidate)
        match_scores.append(score)
        print(f"{score:>8.4f}  {query:<28}  {candidate}")

    non_match_scores: list[float] = []
    print("\nSHOULD NOT MATCH")
    for query, candidate in SHOULD_NOT_MATCH:
        score, _ = score_pair(query, candidate)
        non_match_scores.append(score)
        print(f"{score:>8.4f}  {query:<28}  {candidate}")

    weakest_true = min(match_scores)
    strongest_false = max(non_match_scores)
    separation = weakest_true - strongest_false

    print("\n" + "=" * 72)
    print(f"weakest true match      : {weakest_true:.4f}")
    print(f"strongest false match   : {strongest_false:.4f}")
    print(f"separation              : {separation:+.4f}")
    print(f"MIN_RELEVANCE currently : {MIN_RELEVANCE:.4f}")

    if separation <= 0:
        print("\nFAIL: the classes overlap — no threshold separates them.")
        print("Fix the SIGNALS, not the constant: a threshold can't undo a metric")
        print("that scores a non-match above a real match.")
        return 1

    midpoint = (weakest_true + strongest_false) / 2
    print(f"\nmidpoint of the gap     : {midpoint:.4f}  <- the defensible threshold")
    if not strongest_false < MIN_RELEVANCE < weakest_true:
        print(f"\nFAIL: MIN_RELEVANCE={MIN_RELEVANCE} is outside the separating band")
        print(f"      ({strongest_false:.4f}, {weakest_true:.4f}). Set it to ~{midpoint:.2f}.")
        return 1

    print("\nPASS: MIN_RELEVANCE sits inside the separating band.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
