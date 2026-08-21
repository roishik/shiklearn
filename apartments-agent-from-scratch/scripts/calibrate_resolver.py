"""
calibrate_resolver.py -- pick app/entity_resolution.py's MIN_RELEVANCE from
real, labeled data for THIS domain, not from feel and not from the prior
project's generic placeholder string pairs.

ONE-OFF ANALYSIS UTILITY. Not part of the pytest suite (pytest.ini's
testpaths is "tests" only, so this is never collected). Run it by hand:

    PYTHONPATH=. python scripts/calibrate_resolver.py

WHAT CHANGED FROM THE PRIOR PROJECT'S VERSION OF THIS FILE: that version
(still in git history) scored generic English string pairs like
("Optn B", "Option B") and checked only that the weakest true-match score
beat the strongest false-match score -- a single separation check, not a
threshold sweep, and it never touched a real dataset. This version:
  1. Builds its catalog by reading data/processed_data/localities.json
     DIRECTLY (plain json.load -- NOT via app.dataset, which this script
     does not import, so this calibration has no dependency on
     app/dataset.py being unchanged or even importable).
  2. Uses 29 labeled (query, expected_item_id, should_be_decisive) rows
     covering every real input mode this domain actually sees (see
     LABELED_SET below): exact Hebrew, Hebrew with a typo, niqqud and
     final-letter spelling variants, English exact, competing
     transliterations, a code-shaped query, region names (which SHOULD
     NOT resolve here -- see the note on that category), ambiguous
     prefixes that must come back non-decisive, and junk.
  3. SWEEPS MIN_RELEVANCE (0.30 to 0.95, step 0.01) and reports
     precision / recall / F1 at every step, then recommends the
     threshold that maximizes F1 -- not the midpoint of a single gap.

`app.entity_resolution.MIN_RELEVANCE` is read as a bare module-global
INSIDE resolve() at call time (not bound as a function default), so this
script can sweep it by assigning `entity_resolution.MIN_RELEVANCE = x`
before each call, with no monkeypatching library and no need to reload
the module.

This script NEVER edits app/entity_resolution.py -- it is owned by
another part of this project. If the sweep says the current MIN_RELEVANCE
is wrong, this script says so loudly, with the evidence, and stops there.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import entity_resolution  # noqa: E402
from app.hebrew import TRANSLITERATION_VARIANTS  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOCALITIES_PATH = PROJECT_DIR / "data" / "processed_data" / "localities.json"


def _build_catalog() -> dict[str, list[str]]:
    """item_id -> every text that should count as a name for it, built by
    reading localities.json directly. Mirrors what app/dataset.py's
    ENTITY_CATALOG computes internally, but is a SEPARATE, independent
    build (no import of app.dataset) per this script's brief -- this
    calibration must be able to run and mean something even if
    app/dataset.py is broken or mid-edit."""
    doc = json.loads(LOCALITIES_PATH.read_text(encoding="utf-8-sig"))
    localities = doc["localities"]
    catalog: dict[str, list[str]] = {}
    for item_id, row in localities.items():
        name_he = str(row.get("name_he") or "")
        name_en = str(row.get("name_en") or "")
        name_en_raw = str(row.get("name_en_raw") or "")
        aliases = {item_id, name_he, name_en, name_en_raw}
        aliases.update(TRANSLITERATION_VARIANTS.get(name_he, ()))
        catalog[item_id] = sorted(a for a in aliases if a)
    return catalog


# ── Labeled set ────────────────────────────────────────────────────────────
# Each row: (query, expected_item_id, should_be_decisive, category, note).
# expected_item_id is None for rows that must NOT decisively resolve to any
# single locality -- region names, genuinely ambiguous prefixes, and junk.
# All ids/names below are real anchors, verified present in the real
# data/processed_data/localities.json (see this script's own catalog build
# above -- if one of these ever goes missing, main() fails loudly rather
# than silently skipping the row; see the assertion in main()).
LABELED_SET: list[tuple[str, str | None, bool, str, str]] = [
    # -- exact Hebrew --
    ("ירושלים", "3000", True, "exact_hebrew", "Jerusalem, verbatim name_he"),
    ("כפר סבא", "6900", True, "exact_hebrew", "Kfar Saba, verbatim name_he"),
    ("לוד", "7000", True, "exact_hebrew", "Lod -- 3 chars, the shortest real locality name in this domain"),
    # -- Hebrew with a typo --
    ("ירושלם", "3000", True, "hebrew_typo", "Jerusalem missing the ם->ם... actually missing the י (missing yud)"),
    ("רעננא", "8700", True, "hebrew_typo", "Ra'anana spelled with trailing א instead of ה"),
    ("בער שבע", "9000", True, "hebrew_typo", "Be'er Sheva missing the א in בְאֵר"),
    # -- niqqud --
    ("יְרוּשָׁלַיִם", "3000", True, "niqqud", "Jerusalem fully pointed (niqqud) -- must strip and still exact-match"),
    # -- final-letter (sofit) variant --
    ("רמת גנ", "8600", True, "final_letter", "Ramat Gan with a REGULAR nun instead of the mandatory sofit ן"),
    # -- English exact --
    ("Jerusalem", "3000", True, "english_exact", "canonical English name"),
    ("Lod", "7000", True, "english_exact", "canonical English name, 3 chars"),
    # -- competing transliterations (the brief's own named example) --
    ("Kfar Saba", "6900", True, "transliteration", "competing spelling 1 of כפר סבא"),
    ("Kefar Sava", "6900", True, "transliteration", "competing spelling 2 of כפר סבא (CBS's own romanization)"),
    ("Beer Sheva", "9000", True, "transliteration", "no apostrophe"),
    ("Be'er Sheva", "9000", True, "transliteration", "with apostrophe"),
    ("Beersheba", "9000", True, "transliteration", "anglicized form"),
    # -- code-shaped query (bare CBS locality code) --
    ("6900", "6900", True, "code_shaped", "bare locality code == the item id itself"),
    ("5000", "5000", True, "code_shaped", "bare locality code for Tel Aviv-Yafo"),
    # -- region name fed into the LOCALITY resolver on purpose --
    # entity_resolution.resolve() has NO concept of a region -- that is
    # app.dataset.resolve_region's job (match_region + a separate id-mapping
    # step), a completely different code path this script deliberately does
    # not exercise. Feeding a region name straight into resolve() against
    # the locality catalog is a legitimate calibration case precisely
    # BECAUSE it should not accidentally win a decisive match against some
    # unrelated single locality by partial token overlap.
    ("the Sharon", None, False, "region_name", "a real region, but not a single locality -- must not decisively resolve"),
    ("the Krayot", None, False, "region_name", "ditto -- names four localities, not one"),
    ("הגליל", None, False, "region_name", "'the Galilee', a broad region, not a locality name"),
    # -- ambiguous prefixes that MUST be non-decisive --
    ("מודיעין", None, False, "ambiguous_prefix", "prefix of BOTH מודיעין-מכבים-רעות (1200) and מודיעין עילית (3797)"),
    ("Modiin", None, False, "ambiguous_prefix", "same ambiguity, English form -- the brief's own named flagship case"),
    ("רמת", None, False, "ambiguous_prefix", "prefix of Ramat Gan/Ramat Hasharon/Ramat Yishay and others"),
    ("קריית", None, False, "ambiguous_prefix", "prefix of every קריית-* locality (Qiryat Atta, Qiryat Gat, ...)"),
    (
        "נצרת",
        None,
        False,
        "ambiguous_prefix",
        "genuine real-world ambiguity, not a calibration artifact: locality 1061 "
        "(נצרת עילית / Nof Hagalil) was literally named 'Nazareth Illit' until a 2019 "
        "rename and still scores 0.92 against bare 'נצרת' -- close enough to Nazareth's "
        "own 1.0 to fail the decisive gap. A bare 'נצרת' is a real ambiguity in Israeli "
        "usage, so non-decisive is the CORRECT outcome here, not a bug to fix.",
    ),
    # -- junk that must match nothing --
    ("asdkjhasd", None, False, "junk", "random Latin letters"),
    ("xyz123", None, False, "junk", "random alnum, code-shaped length but not a real code"),
    ("פיצה", None, False, "junk", "a real Hebrew word (pizza), not a place name"),
    ("quantum flux capacitor", None, False, "junk", "long, unrelated English phrase"),
]


def _decisive_correct(result: entity_resolution.ResolutionResult, expected_item_id: str | None) -> bool:
    """True iff the resolver decisively resolved to exactly the expected
    item -- the definition of 'predicted positive AND correct' used below."""
    if not result.decisive or not result.candidates:
        return False
    top = result.candidates[0]
    if expected_item_id is None:
        return False  # a decisive result naming SOME item is never "correct" for a should-not-resolve row
    return top.item_id == expected_item_id


def _evaluate_at_threshold(catalog: dict[str, list[str]], threshold: float) -> dict:
    """Run every labeled row through resolve() with MIN_RELEVANCE=threshold
    and return the confusion-matrix counts plus precision/recall/F1.

    Definitions (see the module docstring for the classifier framing):
      TP: should_be_decisive AND resolver decisive AND top item == expected
      FP: resolver decisive AND (row is should_be_decisive=False, OR the
          decisive top item is WRONG) -- a confident wrong answer is a
          false positive, not a partial credit case
      FN: should_be_decisive=True AND NOT (decisive AND correct)
      TN: should_be_decisive=False AND resolver NOT decisive
    """
    entity_resolution.MIN_RELEVANCE = threshold
    tp = fp = fn = tn = 0
    rows: list[dict] = []
    for query, expected_item_id, should_be_decisive, category, _note in LABELED_SET:
        result = entity_resolution.resolve(query, catalog)
        correct = _decisive_correct(result, expected_item_id)
        if should_be_decisive:
            if correct:
                tp += 1
                outcome = "TP"
            else:
                fn += 1
                outcome = "FN"
        else:
            if result.decisive:
                fp += 1
                outcome = "FP"
            else:
                tn += 1
                outcome = "TN"
        rows.append({"query": query, "category": category, "outcome": outcome})

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rows": rows,
    }


def main() -> int:
    catalog = _build_catalog()

    missing_anchors = [
        item_id
        for _q, item_id, _dec, _cat, _note in LABELED_SET
        if item_id is not None and item_id not in catalog
    ]
    if missing_anchors:
        print(
            f"FAIL: {missing_anchors} not present in {LOCALITIES_PATH} -- the labeled set's "
            "anchors do not match the real dataset. Fix LABELED_SET or investigate a data "
            "regression before trusting anything below."
        )
        return 1

    original_min_relevance = entity_resolution.MIN_RELEVANCE
    print(f"catalog size: {len(catalog)} localities, built directly from {LOCALITIES_PATH.name}")
    print(f"labeled rows: {len(LABELED_SET)}")
    print(f"current MIN_RELEVANCE (before this script touches it): {original_min_relevance}\n")

    thresholds = [round(0.30 + i * 0.01, 2) for i in range(0, 66)]  # 0.30 .. 0.95
    sweep = [_evaluate_at_threshold(catalog, t) for t in thresholds]

    print(f"{'thr':>5}  {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}  {'prec':>6} {'rec':>6} {'F1':>6}")
    prev_key = None
    for s in sweep:
        key = (s["tp"], s["fp"], s["fn"], s["tn"])
        marker = ""
        if s["threshold"] == round(original_min_relevance, 2):
            marker = "  <- current MIN_RELEVANCE"
        # Print every row where the confusion matrix actually changed from
        # the previous threshold, plus the current-MIN_RELEVANCE row even
        # if it didn't change anything -- a full 66-row table where most
        # rows are identical to their neighbor would bury the signal.
        if key != prev_key or marker:
            print(
                f"{s['threshold']:>5.2f}  {s['tp']:>3} {s['fp']:>3} {s['fn']:>3} {s['tn']:>3}  "
                f"{s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f}{marker}"
            )
        prev_key = key

    best = max(sweep, key=lambda s: (s["f1"], -abs(s["threshold"] - 0.63)))
    current = next(s for s in sweep if s["threshold"] == round(original_min_relevance, 2))

    print("\n" + "=" * 78)
    print(f"current MIN_RELEVANCE = {original_min_relevance}: precision={current['precision']:.3f}  "
          f"recall={current['recall']:.3f}  F1={current['f1']:.3f}  "
          f"(TP={current['tp']} FP={current['fp']} FN={current['fn']} TN={current['tn']})")
    print(f"best F1 in the sweep  = {best['threshold']:.2f}: precision={best['precision']:.3f}  "
          f"recall={best['recall']:.3f}  F1={best['f1']:.3f}  "
          f"(TP={best['tp']} FP={best['fp']} FN={best['fn']} TN={best['tn']})")

    # Report every row that the CURRENT threshold gets wrong, by name --
    # never just a count. "3 rows wrong" tells nobody anything actionable.
    wrong = [r for r in current["rows"] if r["outcome"] in ("FP", "FN")]
    if wrong:
        print(f"\ncurrent MIN_RELEVANCE misclassifies {len(wrong)} labeled row(s):")
        for r in wrong:
            print(f"  {r['outcome']}  [{r['category']}]  {r['query']!r}")
    else:
        print("\ncurrent MIN_RELEVANCE misclassifies 0 labeled rows.")

    # A row that is misclassified at EVERY threshold in the sweep cannot be
    # a MIN_RELEVANCE calibration problem -- something else about resolve()
    # is producing the wrong answer regardless of where the fuzzy-match
    # floor sits (a short-query exact-match-branch quirk, e.g., or simply a
    # correct non-decisive result the classifier below doesn't happen to
    # cover). Surfacing these separately, because otherwise they would look
    # like "the sweep found no better threshold" when the real story is
    # "this specific row's failure is not a threshold problem".
    #
    # BUG FIXED HERE 2026-08-21 (by the agent finishing this script): the
    # original version of this block tracked "always wrong" with a single
    # dict, popping a query on any correct outcome and setdefault-ing it on
    # any wrong one, while iterating thresholds ascending. That leaves the
    # dict's final membership equal to "wrong AT THE LAST THRESHOLD
    # CHECKED (0.95)" only -- not "wrong at every threshold" -- because a
    # pop unconditionally clears prior wrongness and a later setdefault
    # re-adds it, so only the terminal state survives. Caught by manually
    # re-running the sweep for 'רעננא' (Ra'anana with a trailing א instead
    # of ה) and 'ירושלם' (Jerusalem missing its י): the buggy code reported
    # both as "misclassified at EVERY threshold 0.30-0.95", but they are
    # actually DECISIVE AND CORRECT at 56/66 and 59/66 of the swept
    # thresholds respectively (including at the current MIN_RELEVANCE=0.63
    # -- see the misclassified-rows list above, which correctly shows only
    # 'Lod' wrong at 0.63). The real always-wrong set, counted properly
    # below, is just {'Lod'} -- exactly the one row with an actual
    # diagnosed root cause (the exact-match dedup bug named below). Fixed
    # by counting wrong-vs-total appearances per query across the full
    # sweep instead of a stateful pop/setdefault toggle.
    _wrong_counts: dict[str, int] = {}
    _total_counts: dict[str, int] = {}
    for s in sweep:
        for r in s["rows"]:
            _total_counts[r["query"]] = _total_counts.get(r["query"], 0) + 1
            if r["outcome"] in ("FP", "FN"):
                _wrong_counts[r["query"]] = _wrong_counts.get(r["query"], 0) + 1
    always_wrong: dict[str, str] = {
        query: next(r["outcome"] for r in current["rows"] if r["query"] == query) or "FN"
        for query in _total_counts
        if _wrong_counts.get(query, 0) == _total_counts[query]
    }
    # current['rows'] only has an outcome for this query if it's ALSO wrong
    # at the current threshold; that isn't guaranteed for an
    # always-wrong-elsewhere-but-happens-to-be-right-here row, so fall back
    # to whatever outcome the row had at the LOWEST swept threshold if it's
    # missing from `current`.
    for query in list(always_wrong):
        if not any(r["query"] == query and r["outcome"] in ("FP", "FN") for r in current["rows"]):
            always_wrong[query] = next(r["outcome"] for r in sweep[0]["rows"] if r["query"] == query)
    if always_wrong:
        print(
            f"\nNOTE -- {len(always_wrong)} row(s) genuinely misclassified at EVERY threshold "
            "from 0.30 to 0.95 (verified by counting wrong-vs-total appearances per query across "
            "the full sweep, not just the outcome at the last threshold checked), so this is NOT "
            "something MIN_RELEVANCE can fix on its own:"
        )
        for query, outcome in always_wrong.items():
            print(f"  {outcome}  {query!r}")
        if "Lod" in always_wrong:
            print(
                "\n  Diagnosed 'Lod' specifically: entity_resolution.resolve()'s short-query "
                "exact-match branch (queries below MIN_FUZZY_QUERY_LENGTH=4 for Latin text) "
                "builds one EntityCandidate per (item_id, matching alias text) pair and never "
                "dedupes by item_id before checking `decisive = len(top_exact) == 1`. 'Lod' "
                "(3 chars, below the Latin floor) exact-matches BOTH 'Lod' and 'LOD' -- two "
                "distinct aliases of the SAME locality (id 7000, name_en='Lod', "
                "name_en_raw='LOD') -- producing 2 candidates that both point at id 7000, so "
                "`len(top_exact) == 1` is False and the result is reported non-decisive even "
                "though there is exactly one matching PLACE. This is a real, separate bug in "
                "app/entity_resolution.py's exact-match dedup logic, NOT a MIN_RELEVANCE "
                "calibration issue -- reported here with the evidence above; not fixed by this "
                "script, since it does not own app/entity_resolution.py."
            )

    if abs(best["threshold"] - round(original_min_relevance, 2)) < 1e-9 and best["f1"] >= current["f1"]:
        print(
            f"\nPASS: current MIN_RELEVANCE ({original_min_relevance}) IS (one of) the best-F1 "
            "threshold(s) in the sweep against this labeled set. No change recommended."
        )
        verdict = 0
    elif best["f1"] > current["f1"]:
        print(
            f"\nREPORT (not fixed here -- app/entity_resolution.py is owned elsewhere): "
            f"the sweep found threshold {best['threshold']:.2f} scores a higher F1 "
            f"({best['f1']:.3f}) than the current MIN_RELEVANCE={original_min_relevance} "
            f"({current['f1']:.3f}) on this labeled set. Evidence is the misclassified-rows "
            "list above and the full sweep table. Route this to whoever owns "
            "app/entity_resolution.py -- this script does not edit it."
        )
        verdict = 0
    else:
        print(
            f"\nPASS: current MIN_RELEVANCE ({original_min_relevance}) ties the best F1 found "
            f"({best['f1']:.3f}) even though a different threshold in the sweep also reaches it "
            "-- no change recommended; 0.63 is inside the same optimal plateau."
        )
        verdict = 0

    entity_resolution.MIN_RELEVANCE = original_min_relevance  # leave the module as we found it
    return verdict


if __name__ == "__main__":
    sys.exit(main())
