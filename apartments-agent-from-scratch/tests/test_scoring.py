"""Unit tests for app/scoring.py — the pure, zero-I/O, zero-LLM scoring
module. These tests should pass with no network access, no API key, and
no other part of the app imported."""
from __future__ import annotations

import pytest

from app import dataset
from app.tools import DEFAULT_CRITERIA
from app.scoring import (
    Criterion,
    InsufficientCoverageError,
    apply_priority_emphasis,
    apply_weight_overrides,
    find_weight_flip_point,
    rank_items,
    scale_criterion_weight,
    score_item,
    sensitivity_analysis,
)


def test_normalize_clamps_below_lower_bound():
    c = Criterion(name="x", weight=1.0, lower_bound=10, upper_bound=20)
    assert c.normalize(0) == 0.0


def test_normalize_clamps_above_upper_bound():
    c = Criterion(name="x", weight=1.0, lower_bound=10, upper_bound=20)
    assert c.normalize(1000) == 1.0


def test_normalize_midpoint():
    c = Criterion(name="x", weight=1.0, lower_bound=0, upper_bound=10)
    assert c.normalize(5) == pytest.approx(0.5)


def test_normalize_degenerate_bounds_returns_half():
    c = Criterion(name="x", weight=1.0, lower_bound=5, upper_bound=5)
    assert c.normalize(5) == 0.5
    assert c.normalize(999) == 0.5  # no divide-by-zero crash


def test_normalize_lower_is_better_inverts():
    c = Criterion(name="cost", weight=1.0, lower_bound=0, upper_bound=100, higher_is_better=False)
    assert c.normalize(0) == 1.0
    assert c.normalize(100) == 0.0
    assert c.normalize(25) == pytest.approx(0.75)


def test_score_item_single_criterion_perfect_score():
    criteria = [Criterion(name="quality", weight=1.0, lower_bound=0, upper_bound=10)]
    result = score_item("a", {"quality": 10}, criteria)
    assert result.total_score == pytest.approx(1.0)
    assert result.components[0].contribution == pytest.approx(1.0)


def test_score_item_weights_normalize_regardless_of_scale():
    """Weights of [1, 1] and [50, 50] must produce identical scores —
    only relative weight matters, not absolute magnitude."""
    criteria_small = [
        Criterion(name="a", weight=1.0, lower_bound=0, upper_bound=10),
        Criterion(name="b", weight=1.0, lower_bound=0, upper_bound=10),
    ]
    criteria_large = [
        Criterion(name="a", weight=50.0, lower_bound=0, upper_bound=10),
        Criterion(name="b", weight=50.0, lower_bound=0, upper_bound=10),
    ]
    raw = {"a": 4, "b": 8}
    r1 = score_item("x", raw, criteria_small)
    r2 = score_item("x", raw, criteria_large)
    assert r1.total_score == pytest.approx(r2.total_score)


def test_score_item_component_weights_sum_to_one():
    criteria = [
        Criterion(name="a", weight=3.0, lower_bound=0, upper_bound=10),
        Criterion(name="b", weight=1.0, lower_bound=0, upper_bound=10),
    ]
    result = score_item("x", {"a": 5, "b": 5}, criteria)
    assert sum(c.weight for c in result.components) == pytest.approx(1.0)


def test_score_item_all_criteria_missing_raises_insufficient_coverage():
    """Zero coverage is not special-cased — it's just the extreme case of
    'below the floor', same exception as any other under-coverage."""
    criteria = [Criterion(name="quality", weight=1.0, lower_bound=0, upper_bound=10)]
    with pytest.raises(InsufficientCoverageError):
        score_item("a", {"other": 5}, criteria)


def test_score_item_below_coverage_threshold_raises():
    criteria = [
        Criterion(name="a", weight=1.0, lower_bound=0, upper_bound=10),
        Criterion(name="b", weight=1.0, lower_bound=0, upper_bound=10),
        Criterion(name="c", weight=2.0, lower_bound=0, upper_bound=10),
    ]
    # only "a" present -> covered_weight = 1/4 = 0.25, well under the default threshold of 0.5
    with pytest.raises(InsufficientCoverageError):
        score_item("x", {"a": 5}, criteria)


def test_score_item_drops_missing_criterion_and_renormalizes_when_above_floor():
    criteria = [
        Criterion(name="a", weight=2.0, lower_bound=0, upper_bound=10),
        Criterion(name="b", weight=1.0, lower_bound=0, upper_bound=10),
        Criterion(name="c", weight=1.0, lower_bound=0, upper_bound=10),
    ]
    # "c" missing -> covered_weight = 3/4 = 0.75, comfortably above the threshold
    result = score_item("x", {"a": 10, "b": 10}, criteria)
    assert result.covered_weight == pytest.approx(0.75)
    assert result.missing_criteria == ("c",)
    assert len(result.components) == 2
    # weights renormalized over what's PRESENT: 2/3 and 1/3, not 2/4 and 1/4
    assert result.components[0].weight == pytest.approx(2 / 3)
    assert result.components[1].weight == pytest.approx(1 / 3)
    assert sum(c.weight for c in result.components) == pytest.approx(1.0)
    assert result.total_score == pytest.approx(1.0)


def test_score_item_exactly_at_floor_is_excluded_not_scored():
    """The boundary is EXCLUSIVE and deliberately so: "enough evidence"
    means MOST of the evidence, and exactly half is not most. Reachable in
    practice whenever one criterion carries half the weight — here the
    weight-2 criterion of a 1/1/2 set. Pinned by a test because it's one
    character in scoring.py and nothing else would catch a silent flip."""
    criteria = [
        Criterion(name="a", weight=1.0, lower_bound=0, upper_bound=10),
        Criterion(name="b", weight=1.0, lower_bound=0, upper_bound=10),
        Criterion(name="c", weight=2.0, lower_bound=0, upper_bound=10),
    ]
    # "c" missing -> covered_weight = 2/4 = exactly the default threshold of 0.5
    with pytest.raises(InsufficientCoverageError):
        score_item("x", {"a": 5, "b": 5}, criteria)


def test_score_item_just_above_floor_is_scored():
    """The other side of the same boundary — coverage a hair above 0.5
    scores normally, proving the exclusion above is the floor doing its
    job and not something else failing."""
    criteria = [
        Criterion(name="a", weight=1.01, lower_bound=0, upper_bound=10),
        Criterion(name="b", weight=1.0, lower_bound=0, upper_bound=10),
    ]
    result = score_item("x", {"a": 10}, criteria)
    assert result.covered_weight > 0.5
    assert result.missing_criteria == ("b",)


def test_score_item_full_coverage_has_no_missing_criteria():
    criteria = [Criterion(name="quality", weight=1.0, lower_bound=0, upper_bound=10)]
    result = score_item("a", {"quality": 5}, criteria)
    assert result.covered_weight == pytest.approx(1.0)
    assert result.missing_criteria == ()


def test_score_item_custom_coverage_threshold_is_stricter():
    criteria = [
        Criterion(name="a", weight=3.0, lower_bound=0, upper_bound=10),
        Criterion(name="b", weight=1.0, lower_bound=0, upper_bound=10),
        Criterion(name="c", weight=1.0, lower_bound=0, upper_bound=10),
    ]
    # "c" missing -> covered_weight = 4/5 = 0.8: passes the default 0.5
    # threshold but fails a stricter, caller-supplied 0.85.
    result = score_item("x", {"a": 5, "b": 5}, criteria)
    assert result.covered_weight == pytest.approx(0.8)
    with pytest.raises(InsufficientCoverageError):
        score_item("x", {"a": 5, "b": 5}, criteria, coverage_threshold=0.85)


def test_score_item_zero_available_weight_raises_coverage_error_not_zerodivision():
    """Regression, and the reason the second guard in score_item exists at
    all. A NEGATIVE coverage_threshold ("rank absolutely everything") is
    the one input the coverage check lets through with zero available
    weight — and the renormalization loop then divided by zero. A raw
    ZeroDivisionError is not InsufficientCoverageError, so rank_items would
    NOT catch it and one degenerate item would kill the entire batch."""
    criteria = [
        Criterion(name="a", weight=0.0, lower_bound=0, upper_bound=10),
        Criterion(name="b", weight=1.0, lower_bound=0, upper_bound=10),
    ]
    with pytest.raises(InsufficientCoverageError):
        score_item("x", {"a": 5}, criteria, coverage_threshold=-1.0)


def test_score_item_no_criteria_available_never_returns_a_fake_zero():
    """Zero coverage must not come back as total_score=0.0 with empty
    components — that reads as 'scored zero' when it means 'never scored'.
    Same negative-threshold route as above; with any coverage_threshold
    >= 0 the ordinary coverage check catches this first."""
    criteria = [
        Criterion(name="a", weight=1.0, lower_bound=0, upper_bound=10),
        Criterion(name="b", weight=1.0, lower_bound=0, upper_bound=10),
    ]
    with pytest.raises(InsufficientCoverageError):
        score_item("y", {"unrelated": 5}, criteria, coverage_threshold=-1.0)


def test_rank_items_zero_available_weight_is_excluded_not_a_crash():
    """The batch-level consequence of the two regressions above: a
    degenerate item lands in `excluded` alongside every other coverage
    failure, and the good items still rank."""
    criteria = [
        Criterion(name="a", weight=0.0, lower_bound=0, upper_bound=10),
        Criterion(name="b", weight=1.0, lower_bound=0, upper_bound=10),
    ]
    items = {"good": {"a": 5, "b": 9}, "degenerate": {"a": 5}}
    result = rank_items(items, criteria, coverage_threshold=-1.0)
    assert [r.item_id for r in result.ranked] == ["good"]
    assert [e.item_id for e in result.excluded] == ["degenerate"]


def test_score_item_empty_criteria_raises_valueerror():
    with pytest.raises(ValueError):
        score_item("a", {"quality": 5}, [])


def test_score_item_zero_total_weight_raises_valueerror():
    criteria = [Criterion(name="quality", weight=0.0, lower_bound=0, upper_bound=10)]
    with pytest.raises(ValueError):
        score_item("a", {"quality": 5}, criteria)


def test_rank_items_orders_by_total_score_desc():
    criteria = [Criterion(name="quality", weight=1.0, lower_bound=0, upper_bound=10)]
    items = {
        "low": {"quality": 2},
        "high": {"quality": 9},
        "mid": {"quality": 5},
    }
    result = rank_items(items, criteria)
    assert [r.item_id for r in result.ranked] == ["high", "mid", "low"]
    assert [r.rank for r in result.ranked] == [1, 2, 3]
    assert result.excluded == ()


def test_rank_items_tie_break_is_deterministic_by_item_id():
    criteria = [Criterion(name="quality", weight=1.0, lower_bound=0, upper_bound=10)]
    items = {
        "zebra": {"quality": 5},
        "apple": {"quality": 5},
        "mango": {"quality": 5},
    }
    result = rank_items(items, criteria)
    assert [r.item_id for r in result.ranked] == ["apple", "mango", "zebra"]


def test_rank_items_is_repeatable():
    """Same input, called twice, must produce identical output — no
    hidden randomness or dict-order dependency."""
    criteria = [
        Criterion(name="cost", weight=1.0, lower_bound=0, upper_bound=200, higher_is_better=False),
        Criterion(name="quality", weight=2.0, lower_bound=0, upper_bound=10),
    ]
    items = {
        "option_a": {"cost": 120, "quality": 8.5},
        "option_b": {"cost": 90, "quality": 6.0},
        "option_c": {"cost": 150, "quality": 9.2},
    }
    r1 = rank_items(items, criteria)
    r2 = rank_items(items, criteria)
    assert [(r.item_id, r.total_score) for r in r1.ranked] == [(r.item_id, r.total_score) for r in r2.ranked]


def test_rank_items_excludes_items_below_coverage_floor():
    criteria = [
        Criterion(name="a", weight=1.0, lower_bound=0, upper_bound=10),
        Criterion(name="b", weight=1.0, lower_bound=0, upper_bound=10),
        Criterion(name="c", weight=2.0, lower_bound=0, upper_bound=10),
    ]
    items = {
        "full": {"a": 5, "b": 5, "c": 5},
        "sparse": {"a": 5},  # covered_weight = 1/4 = 0.25, below default 0.5 floor
    }
    result = rank_items(items, criteria)
    assert [r.item_id for r in result.ranked] == ["full"]
    assert [e.item_id for e in result.excluded] == ["sparse"]
    assert result.excluded[0].covered_weight == pytest.approx(0.25)
    assert result.excluded[0].missing_criteria == ("b", "c")


def test_rank_items_a_caller_bug_still_propagates():
    """Empty criteria list is a caller mistake, not a coverage problem —
    it must still raise, not get silently absorbed into `excluded`."""
    with pytest.raises(ValueError):
        rank_items({"x": {"a": 1}}, [])


def test_higher_is_better_monotonicity():
    """Increasing a higher-is-better raw value (all else equal) must not
    decrease the total score — a basic sanity property of the ranking."""
    criteria = [
        Criterion(name="quality", weight=1.0, lower_bound=0, upper_bound=10),
        Criterion(name="cost", weight=1.0, lower_bound=0, upper_bound=200, higher_is_better=False),
    ]
    worse = score_item("x", {"quality": 4, "cost": 100}, criteria)
    better = score_item("x", {"quality": 8, "cost": 100}, criteria)
    assert better.total_score > worse.total_score


def test_lower_is_better_monotonicity():
    criteria = [Criterion(name="cost", weight=1.0, lower_bound=0, upper_bound=200, higher_is_better=False)]
    cheaper = score_item("x", {"cost": 50}, criteria)
    pricier = score_item("x", {"cost": 150}, criteria)
    assert cheaper.total_score > pricier.total_score


# ─────────────────────────────────────────────────────────────────────────
# B1 — sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────
SENS_CRITERIA = [
    Criterion(name="cost", weight=1.0, lower_bound=80, upper_bound=160, higher_is_better=False),
    Criterion(name="quality", weight=2.0, lower_bound=0, upper_bound=10, higher_is_better=True),
]
# Chosen so the baseline winner is unambiguous AND the flip point is known
# analytically, rather than trusted from the implementation under test:
# with cost weight fixed at 1 and quality weight w, the two scores are
# equal when 0.75 + 0.3w == 0.375 + 0.95w, i.e. w = 0.375/0.65 = 0.5769.
# Baseline w is 2.0, so the true flip factor is 0.5769/2 = 0.2885.
SENS_ITEMS = {
    "cheap_ok": {"cost": 100.0, "quality": 3.0},  # normalized: cost 0.75, quality 0.30
    "pricey_great": {"cost": 130.0, "quality": 9.5},  # normalized: cost 0.375, quality 0.95
}
TRUE_FLIP_FACTOR = (0.375 / 0.65) / 2.0


def test_apply_weight_overrides_does_not_mutate_the_original():
    """The baseline must survive being compared against — a mutated input
    would silently make every perturbation look like 'no change'."""
    original = [c.weight for c in SENS_CRITERIA]
    apply_weight_overrides(SENS_CRITERIA, {"cost": 99.0})
    assert [c.weight for c in SENS_CRITERIA] == original


def test_apply_weight_overrides_rejects_unknown_criterion():
    """A typo'd name would otherwise report 'nothing moved', which reads
    as reassuring stability — the exact opposite of the truth."""
    with pytest.raises(ValueError):
        apply_weight_overrides(SENS_CRITERIA, {"qulaity": 1.0})


def test_scale_criterion_weight_halves():
    scaled = scale_criterion_weight(SENS_CRITERIA, "quality", 0.5)
    assert {c.name: c.weight for c in scaled} == {"cost": 1.0, "quality": 1.0}


def test_scale_criterion_weight_rejects_unknown_criterion():
    with pytest.raises(ValueError):
        scale_criterion_weight(SENS_CRITERIA, "nonexistent", 2.0)


def test_scaling_all_weights_equally_is_a_no_op():
    """Weights are RELATIVE and renormalized internally, so a uniform
    rescale must change nothing. If this ever fails, normalization broke."""
    doubled = [
        Criterion(name=c.name, weight=c.weight * 2, lower_bound=c.lower_bound,
                  upper_bound=c.upper_bound, higher_is_better=c.higher_is_better)
        for c in SENS_CRITERIA
    ]
    a = rank_items(SENS_ITEMS, SENS_CRITERIA)
    b = rank_items(SENS_ITEMS, doubled)
    assert [(r.item_id, r.total_score) for r in a.ranked] == [
        (r.item_id, r.total_score) for r in b.ranked
    ]


def test_sensitivity_detects_a_winner_flip():
    """Quality is weighted 2x, so pricey_great wins. Drop quality's weight
    far enough and the cheap option takes over."""
    result = sensitivity_analysis(SENS_ITEMS, SENS_CRITERIA, {"quality": 0.2})
    assert result.baseline_top == "pricey_great"
    assert result.perturbed_top == "cheap_ok"
    assert result.top_changed is True


def test_sensitivity_no_change_reports_tau_one_and_zero_movement():
    result = sensitivity_analysis(SENS_ITEMS, SENS_CRITERIA, {"quality": 2.0})  # same value
    assert result.top_changed is False
    assert result.kendall_tau == pytest.approx(1.0)
    assert result.items_moved == 0
    assert result.max_rank_movement == 0


def test_sensitivity_full_reversal_gives_tau_minus_one():
    result = sensitivity_analysis(SENS_ITEMS, SENS_CRITERIA, {"quality": 0.2})
    assert result.kendall_tau == pytest.approx(-1.0)


def test_sensitivity_reports_per_item_rank_and_score_deltas():
    result = sensitivity_analysis(SENS_ITEMS, SENS_CRITERIA, {"quality": 0.2})
    by_id = {c.item_id: c for c in result.changes}
    assert by_id["cheap_ok"].baseline_rank == 2
    assert by_id["cheap_ok"].perturbed_rank == 1
    assert by_id["cheap_ok"].rank_delta == -1  # negative = moved UP
    assert by_id["pricey_great"].rank_delta == 1


def test_sensitivity_tracks_items_leaving_the_ranking_via_coverage():
    """A weight change alters covered_weight, so raising the weight of a
    criterion an item is MISSING can push it under the threshold and out
    of the ranking entirely. Reporting that as a rank of 0, or dropping it
    silently, would hide a real consequence of the weight choice."""
    criteria = [
        Criterion(name="a", weight=1.0, lower_bound=0, upper_bound=10),
        Criterion(name="b", weight=1.0, lower_bound=0, upper_bound=10),
        Criterion(name="c", weight=1.0, lower_bound=0, upper_bound=10),
    ]
    items = {
        "complete": {"a": 5, "b": 5, "c": 5},
        "missing_c": {"a": 9, "b": 9},  # covered 2/3 = 0.667 > 0.5 -> ranked
    }
    baseline = rank_items(items, criteria)
    assert {r.item_id for r in baseline.ranked} == {"complete", "missing_c"}

    # Raise c's weight to 4 -> total 6, missing_c now covers 2/6 = 0.333 < 0.5
    result = sensitivity_analysis(items, criteria, {"c": 4.0})
    assert "missing_c" in result.items_left
    by_id = {ch.item_id: ch for ch in result.changes}
    assert by_id["missing_c"].perturbed_rank is None
    assert by_id["missing_c"].rank_delta is None  # entered/left, not moved


def test_find_weight_flip_point_finds_a_flip():
    flip = find_weight_flip_point(SENS_ITEMS, SENS_CRITERIA, "quality")
    assert flip is not None
    assert flip < 1.0  # lowering quality's weight is what flips this one


def test_find_weight_flip_point_returns_none_when_ranking_is_robust():
    """One item dominating on every criterion cannot be dethroned by
    reweighting — no multiplier should flip it."""
    dominant = {
        "best": {"cost": 80.0, "quality": 10.0},
        "worst": {"cost": 160.0, "quality": 0.0},
    }
    assert find_weight_flip_point(dominant, SENS_CRITERIA, "quality") is None
    assert find_weight_flip_point(dominant, SENS_CRITERIA, "cost") is None


def test_find_weight_flip_point_searches_both_directions():
    """A weight can be too HIGH as easily as too low. Only sweeping upward
    would report 'robust' for a ranking that flips when halved."""
    flip = find_weight_flip_point(SENS_ITEMS, SENS_CRITERIA, "quality", max_factor=10.0)
    assert flip is not None and flip < 1.0


def test_find_weight_flip_point_returns_the_smallest_perturbation():
    """Interleaving outward from 1.0 must find the NEAREST flip, not just
    any flip — the defence claim is 'it takes this much to change the
    answer', which is only true if it's the minimum."""
    flip = find_weight_flip_point(SENS_ITEMS, SENS_CRITERIA, "quality", steps=200)
    assert flip is not None
    # nothing closer to 1.0 should also flip
    closer = 1.0 - (1.0 - flip) / 2
    perturbed = rank_items(
        SENS_ITEMS, apply_weight_overrides(SENS_CRITERIA, {"quality": 2.0 * closer})
    )
    assert perturbed.ranked[0].item_id == "pricey_great"  # still the baseline winner


def test_find_weight_flip_point_matches_the_analytic_answer():
    """Checked against arithmetic done by hand (see TRUE_FLIP_FACTOR), not
    against the implementation's own output — otherwise the test only
    proves the code is self-consistent, which it would be even if the
    sweep were wrong. Tolerance is the sweep's own resolution,
    max_factor/steps, since a fixed-step sweep cannot do better than that."""
    resolution = 10.0 / 200
    flip = find_weight_flip_point(SENS_ITEMS, SENS_CRITERIA, "quality", max_factor=10.0, steps=200)
    assert flip is not None
    assert abs(flip - TRUE_FLIP_FACTOR) <= resolution


def test_find_weight_flip_point_resolution_is_honest():
    """A coarser sweep must still bracket the true answer within its own
    (worse) resolution — the documented limitation, asserted rather than
    assumed."""
    coarse = find_weight_flip_point(SENS_ITEMS, SENS_CRITERIA, "quality", max_factor=10.0, steps=20)
    assert coarse is not None
    assert abs(coarse - TRUE_FLIP_FACTOR) <= 10.0 / 20


def test_sensitivity_is_repeatable():
    a = sensitivity_analysis(SENS_ITEMS, SENS_CRITERIA, {"quality": 0.5})
    b = sensitivity_analysis(SENS_ITEMS, SENS_CRITERIA, {"quality": 0.5})
    assert a == b


# ─────────────────────────────────────────────────────────────────────────
# Priority emphasis — honoring stated user preferences, safely
# ─────────────────────────────────────────────────────────────────────────
def test_emphasis_doubles_and_deemphasis_halves():
    adjusted = apply_priority_emphasis(SENS_CRITERIA, emphasize=["cost"], deemphasize=["quality"])
    weights = {c.name: c.weight for c in adjusted}
    assert weights["cost"] == pytest.approx(2.0)  # 1.0 * 2
    assert weights["quality"] == pytest.approx(1.0)  # 2.0 / 2


def test_emphasis_does_not_mutate_the_original():
    before = [c.weight for c in SENS_CRITERIA]
    apply_priority_emphasis(SENS_CRITERIA, emphasize=["cost"])
    assert [c.weight for c in SENS_CRITERIA] == before


def test_emphasis_rejects_unknown_criterion():
    """A criterion name the model invented must fail loudly. Silently
    ignoring it would apply NO reweighting while reporting that the
    user's priorities were honored."""
    with pytest.raises(ValueError):
        apply_priority_emphasis(SENS_CRITERIA, emphasize=["speed"])


def test_emphasis_rejects_contradictory_priorities():
    """Naming a criterion as both more and less important is a confused
    request, not a no-op — cancelling silently would hide it."""
    with pytest.raises(ValueError):
        apply_priority_emphasis(SENS_CRITERIA, emphasize=["cost"], deemphasize=["cost"])


def test_emphasis_cannot_zero_out_a_criterion():
    """The safety property: the caller gets a verb, not a value. A
    criterion can at most double or halve, so no amount of stated
    'priority' can switch one off and manufacture a desired winner."""
    adjusted = apply_priority_emphasis(SENS_CRITERIA, deemphasize=["quality", "cost"])
    assert all(c.weight > 0 for c in adjusted)
    ratio = max(c.weight for c in adjusted) / min(c.weight for c in adjusted)
    baseline_ratio = max(c.weight for c in SENS_CRITERIA) / min(c.weight for c in SENS_CRITERIA)
    assert ratio == pytest.approx(baseline_ratio)  # uniform deemphasis is a no-op after renormalization


def test_emphasis_with_no_priorities_is_identity():
    adjusted = apply_priority_emphasis(SENS_CRITERIA)
    assert [c.weight for c in adjusted] == [c.weight for c in SENS_CRITERIA]


def test_emphasis_actually_changes_the_ranking_when_it_should():
    """End-to-end: 'I care about cost' must be able to flip a ranking that
    quality-weighting produced, or the feature is decorative."""
    default = rank_items(SENS_ITEMS, SENS_CRITERIA)
    cost_first = rank_items(
        SENS_ITEMS, apply_priority_emphasis(SENS_CRITERIA, emphasize=["cost"], deemphasize=["quality"])
    )
    assert default.ranked[0].item_id == "pricey_great"
    assert cost_first.ranked[0].item_id == "cheap_ok"


# ── The arithmetic itself ───────────────────────────────────────────────
# These exist because of a hole found by mutation testing: changing
# `contribution = normalized_score * weight` to `normalized_score ** 2 *
# weight` — which silently changes every item's total score — left the
# entire suite green.
#
# The reason is subtle and worth stating, because it is a trap any
# numeric test suite can fall into. Every numeric assertion in this file
# used raw values that normalize to exactly 0.0, 0.5, or 1.0, and those
# are precisely the fixed points where x**2 == x. Everything else
# asserted on ORDERING, and squaring is monotone, so the order never
# moved either. The suite tested the shape of the formula thoroughly and
# its value not at all.
#
# The fix is to pin at least one case whose normalized scores are none of
# 0, 0.5, or 1, with the arithmetic written out longhand so the expected
# number is derived here rather than copied from a run.


def test_weighted_score_matches_arithmetic_done_by_hand():
    """One item, two criteria, no missing data, nothing degenerate.

    a = 3 on a 0..10 scale  -> normalized 0.30
    b = 70 on a 0..100 scale -> normalized 0.70
    weights 2 and 1         -> normalized 2/3 and 1/3
    total = 0.30*(2/3) + 0.70*(1/3) = 0.20 + 0.2333... = 0.43333...

    Deliberately chosen so no normalized value is 0, 0.5, or 1 — the
    three points at which a squared-contribution bug is invisible.
    """
    criteria = [
        Criterion(name="a", weight=2, lower_bound=0.0, upper_bound=10.0),
        Criterion(name="b", weight=1, lower_bound=0.0, upper_bound=100.0),
    ]
    result = rank_items({"x": {"a": 3.0, "b": 70.0}}, criteria)
    scored = result.ranked[0]

    assert scored.total_score == pytest.approx(0.43333333, abs=1e-8)

    by_name = {c.criterion: c for c in scored.components}
    assert by_name["a"].normalized_score == pytest.approx(0.30, abs=1e-9)
    assert by_name["a"].weight == pytest.approx(2 / 3, abs=1e-9)
    assert by_name["a"].contribution == pytest.approx(0.20, abs=1e-9)
    assert by_name["b"].normalized_score == pytest.approx(0.70, abs=1e-9)
    assert by_name["b"].weight == pytest.approx(1 / 3, abs=1e-9)
    assert by_name["b"].contribution == pytest.approx(0.23333333, abs=1e-8)

    # The whole is exactly the sum of its parts. This is the property the
    # explanations depend on: the model reconstructs the total from the
    # breakdown, so if they ever disagree the explanation is fiction.
    assert sum(c.contribution for c in scored.components) == pytest.approx(
        scored.total_score, abs=1e-12
    )


def test_contribution_is_linear_in_the_normalized_score():
    """Doubling one criterion's normalized input doubles its contribution.

    This is what fails under any exponent other than 1, at any value —
    it does not depend on choosing a lucky test point.
    """
    criteria = [Criterion(name="a", weight=1, lower_bound=0.0, upper_bound=10.0)]
    low = rank_items({"x": {"a": 2.0}}, criteria).ranked[0]
    high = rank_items({"x": {"a": 4.0}}, criteria).ranked[0]
    assert high.components[0].contribution == pytest.approx(
        2 * low.components[0].contribution, abs=1e-12
    )


# ─────────────────────────────────────────────────────────────────────────
# Real-dataset pin — the regression check an offline test suite needs
# ─────────────────────────────────────────────────────────────────────────
# Everything above this line tests scoring.py against small, hand-built
# fixtures chosen to isolate one property at a time. None of them would
# notice if a NORMALIZATION BOUND drifted — e.g. price_level's upper_bound
# silently changing from 3,467,262.5 to some other number after a data
# refresh — because a hand-built fixture's bounds are redefined alongside
# it in the same test. The only thing that catches a bound moving under
# real data is scoring the REAL data and pinning the result.
#
# This is not a hypothetical risk: the airport-era version of this exact
# test (against LAX/SNA) caught a silent drift in a normalization floor
# that had moved a real ranking, with nobody having changed anything on
# purpose. That is what this test is for, and why it must be a PIN —
# literal expected numbers written into the test — rather than a
# tautology like `assert score == rank_items(...)`, which would just
# restate whatever the code currently does and catch nothing.
#
# Five localities, picked to span the actual ranking rather than
# clustering near the top: Kefar Sava (the real #1 at these weights —
# see app/tools.py's DECISIVE_SCORE_GAP comment), Tel Aviv-Yafo and
# Ra'annana (expensive, high-socioeconomic, opposite ends of rental
# yield), Be'er Sheva (cheap, peripheral), and Jerusalem (expensive AND
# lower socioeconomic-cluster-for-its-price than Tel Aviv, so it lands
# LAST among these five despite being a major city — a genuinely
# discriminating result, not a foregone one). All five have full 5/5
# criterion coverage (covered_weight == 1.0), so this test is purely
# about the scoring arithmetic and the frozen bounds/weights, with no
# renormalization-for-missing-data interaction to also account for.
#
# Values below were produced by running, on 2026-08-21, against the
# dataset committed at that date:
#
#     items = {i: dataset.METRICS[i] for i in _PIN_IDS}
#     rank_items(items, DEFAULT_CRITERIA)
#
# and reading off result.ranked[*].total_score to 16 significant digits.
# They are NOT re-derived at test time from anything the test imports —
# copying the runtime value into the assertion would make this a
# tautology, exactly the failure mode this comment warns against two
# paragraphs up.
#
# IF THIS TEST FAILS: it means either (a) data/refresh_data.py was rerun
# and the underlying nadlan.gov.il/CBS figures for one of these five
# localities changed (expected, requires no code change — re-pin after a
# human confirms the new numbers look sane), or (b) DEFAULT_CRITERIA's
# weights/bounds in app/tools.py moved, or scoring.py's arithmetic
# changed (NOT expected from a routine data refresh — this is the
# scenario the airport-era version of this test actually caught, and it
# needs a human to look at *why* a bound or the formula moved before
# re-pinning). A diff here is a prompt to find out which of the two
# happened, never something to silence by regenerating the numbers.
_PIN_IDS: tuple[str, ...] = ("6900", "5000", "8700", "9000", "3000")
_PINNED_TOTAL_SCORES: dict[str, float] = {
    "6900": 0.5797013993974158,  # Kefar Sava (Kfar Saba) — the real #1
    "5000": 0.5088020762326761,  # Tel Aviv-Yafo
    "8700": 0.470069013617954,  # Ra'annana
    "9000": 0.44405899520052206,  # Be'er Sheva
    "3000": 0.37901548248367817,  # Jerusalem — last of these five
}


def test_real_dataset_scores_are_pinned_to_known_values():
    """Score five real localities with the REAL DEFAULT_CRITERIA against
    the REAL committed dataset, and check the total_score of each against
    a value written into this test by hand (see the comment block above
    for how it was produced and why a failure here is never something to
    silence by re-running and copying the new number in).
    """
    items = {item_id: dataset.METRICS[item_id] for item_id in _PIN_IDS}
    result = rank_items(items, DEFAULT_CRITERIA)

    # All five must actually have scored (not landed in `excluded`) —
    # a coverage-threshold regression that silently dropped one of them
    # would otherwise make the loop below vacuously pass on four items
    # instead of failing loudly on the missing fifth.
    assert result.excluded == ()
    scored = {r.item_id: r for r in result.ranked}
    assert set(scored) == set(_PIN_IDS)

    for item_id, expected_score in _PINNED_TOTAL_SCORES.items():
        # Every pinned locality has full 5/5 criterion coverage in the
        # real dataset (see the comment above) — asserted here so a
        # future data gap for one of these specific five (e.g. a future
        # rental_yield join failure) fails as "coverage changed", a much
        # clearer signal than a silently-renormalized score that merely
        # LOOKS like a formula drift.
        assert scored[item_id].covered_weight == pytest.approx(1.0), (
            f"{item_id} no longer has full criterion coverage in the real dataset — "
            "this test's premise (a pure arithmetic/bounds check, with no "
            "renormalization involved) no longer holds for this locality."
        )
        assert scored[item_id].total_score == pytest.approx(expected_score, abs=1e-9), (
            f"total_score for {item_id!r} drifted from the pinned value {expected_score}. "
            "This means either the underlying nadlan.gov.il/CBS data for this locality "
            "changed (re-pin after confirming the new number looks sane) or a "
            "DEFAULT_CRITERIA bound/weight or scoring.py's formula moved (find out why "
            "before re-pinning) — see the comment above this test for the full decision "
            "tree. Never silence this by copying in whatever the code now produces."
        )

    # The relative ORDER is itself part of what a bounds/formula drift can
    # break even when no single score moves outside its own tolerance —
    # e.g. two scores swapping without either one crossing 1e-9 relative
    # to its OWN old value is not possible here, but a bound shift that
    # nudges several scores in different directions could still preserve
    # each one's approx equality while reshuffling the order. Pinning the
    # order too closes that gap.
    assert [r.item_id for r in result.ranked] == list(_PIN_IDS)
