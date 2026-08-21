"""Tests for the app/tools.py entry points not already exercised by
tests/test_tools_domain.py: aggregate_records, estimate_derived_metric,
find_items, get_item_metrics, rank_by_priorities,
analyze_weight_sensitivity, weight_robustness_report, and
get_current_mortgage_rates (mocked — see its own section below for why).

Same discipline as test_tools_domain.py: real ids, real committed
dataset, values pinned by actually running the code on 2026-08-21 and
reading off the result — see _working/agent-logs/tool-tests.md for the
verification trail. Nothing here should need a synthetic fixture; the
real 104-locality dataset already contains every edge case this file
needs (a locality with zero neighborhoods tracked, one with
neighborhoods but none priced, one missing income data, and so on) —
using the real gaps instead of fabricating one means these tests can't
silently drift out of sync with what the data pipeline actually
produces.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app import dataset
from app.tools import (
    DEFAULT_CRITERIA,
    UnknownItemError,
    aggregate_records,
    analyze_weight_sensitivity,
    estimate_derived_metric,
    find_items,
    get_current_mortgage_rates,
    get_item_metrics,
    rank_by_priorities,
    weight_robustness_report,
)

KEFAR_SAVA = "6900"
RAANANA = "8700"
BEER_SHEVA = "9000"
TEL_AVIV = "5000"
JERUSALEM = "3000"


# ─────────────────────────────────────────────────────────────────────────
# aggregate_records — single-locality neighborhood statistics
# ─────────────────────────────────────────────────────────────────────────
def test_aggregate_share_above_median_tel_aviv_pinned():
    """Pinned against the real nadlan.gov.il neighborhood breakdown for
    Tel Aviv-Yafo: 72 neighborhoods tracked, 41 of them priced, 13 of
    those 41 above the city's own median 4-room price — a real,
    hand-verified 0.317073 share. This is the exact example question
    from PROGRESS.md's 'four question shapes' table ('what share of Tel
    Aviv's 72 neighborhoods are above the city median price?')."""
    result = aggregate_records(TEL_AVIV, "share", "above_median")
    assert result["defined"] is True
    assert result["value"] == pytest.approx(0.317073, abs=1e-6)
    assert result["total_neighborhoods_tracked"] == 72
    assert result["neighborhoods_with_price_data"] == 41
    assert result["neighborhoods_without_price_data"] == 31
    assert result["matching_records"] == 13
    assert result["total_records"] == 41
    assert result["unknown_category"] is False


def test_aggregate_unknown_category_is_not_a_zero_result():
    """A category that doesn't exist in the dataset must come back
    defined=False and unknown_category=True, carrying the REAL known
    categories — never as value=0.0 or 'none', which would read as a
    measured answer to a question that was never actually computed. This
    is the exact distinction aggregate_records' own docstring calls out:
    'no such category' and 'a real category with zero rows' are
    different facts."""
    result = aggregate_records(TEL_AVIV, "share", "waterfront")
    assert result["defined"] is False
    assert result["unknown_category"] is True
    assert result["value"] is None
    assert result["known_categories"] == ["above_median", "at_or_below_median"]
    assert "category_semantics" in result
    assert result["category_semantics"] is not None
    # The real denominator situation must still be reported even on an
    # unknown-category call — this is discovery-mode data the model needs
    # to formulate the NEXT, correct call.
    assert result["total_neighborhoods_tracked"] == 72


def test_aggregate_share_with_no_category_is_a_discovery_call_not_a_fake_100_percent():
    """A bare 'share' with no category is, by definition, 100% of
    everything — not a useful number, and not what a caller who forgot
    to supply a category meant. aggregate_records must recognize this and
    return an undefined discovery response (with the real
    known_categories) instead of confidently reporting 1.0."""
    result = aggregate_records(TEL_AVIV, "share")
    assert result["defined"] is False
    assert result["value"] is None
    assert result["known_categories"] == ["above_median", "at_or_below_median"]


def test_aggregate_locality_with_zero_neighborhoods_tracked_is_distinct_from_zero_priced():
    """Atlit (locality id 53) is a real locality nadlan.gov.il tracks
    ZERO neighborhoods for at all — not 'zero of some tracked
    neighborhoods have price data', a fundamentally different situation
    that must not be reported as a share of 0%."""
    result = aggregate_records("53", "share", "above_median")
    assert result["defined"] is False
    assert result["value"] is None
    assert result["total_neighborhoods_tracked"] == 0
    assert result["neighborhoods_with_price_data"] == 0
    assert result["known_categories"] == []
    assert result["unknown_category"] is False  # this is a data-absence failure, not a bad category
    assert "zero" in result["guidance"].lower()


def test_aggregate_locality_with_neighborhoods_but_none_priced():
    """Zur Hadassa (locality id 1113) is the other real half of the same
    distinction: 4 neighborhoods ARE tracked, but none of them carry
    price data (every summary/price field nadlan publishes for them is
    null). This must be reported as 'tracked but unpriced', not
    conflated with the zero-tracked case above."""
    result = aggregate_records("1113", "share", "above_median")
    assert result["defined"] is False
    assert result["value"] is None
    assert result["total_neighborhoods_tracked"] == 4
    assert result["neighborhoods_with_price_data"] == 0
    assert result["neighborhoods_without_price_data"] == 4


def test_aggregate_records_neighborhood_id_raises_the_neighborhood_specific_error():
    """A neighborhood id handed to aggregate_records — which takes a
    LOCALITY id — must raise the same _neighborhood_not_rankable error
    the ranking tools raise via fetch_item_metrics, not a generic
    UnknownItemError. This is the mirror case of the eligibility gate:
    aggregate_records has no gate to bypass (it doesn't rank), but it
    still must not silently treat a neighborhood as if it were its own
    locality."""
    with pytest.raises(UnknownItemError) as excinfo:
        aggregate_records("28:65211075", "share", "above_median")
    assert "NEIGHBORHOOD" in str(excinfo.value)


def test_aggregate_records_unknown_locality_id_raises():
    with pytest.raises(UnknownItemError):
        aggregate_records("not-a-real-locality", "share", "above_median")


def test_aggregate_count_operation():
    """A different SUPPORTED_OPERATIONS value than 'share', to confirm
    aggregate_records passes `operation` through rather than hard-coding
    share — pinned against the same real Tel Aviv breakdown as the share
    test above (13 of 41 priced neighborhoods are above_median)."""
    result = aggregate_records(TEL_AVIV, "count", "above_median")
    assert result["defined"] is True
    assert result["value"] == pytest.approx(13.0)
    assert result["operation"] == "count"


# ─────────────────────────────────────────────────────────────────────────
# estimate_derived_metric — the modelled required-income figure
# ─────────────────────────────────────────────────────────────────────────
def test_estimate_derived_metric_beer_sheva_pinned_value_and_vintage_caveat():
    """Pinned against the real median_price_4room for Be'er Sheva
    (1,203,200 ₪) run through the standard BoI Directive 329 first-home
    mortgage model. The caveat is asserted on CONTENT, not just
    non-emptiness — the 2021-income/2026-price vintage mismatch is
    exactly the kind of disclosure that is easy to accidentally drop in
    a refactor while leaving the string non-empty, so a bare
    `assert result['caveat']` would not actually catch its loss."""
    result = estimate_derived_metric(BEER_SHEVA)
    assert result["value"] == pytest.approx(14532.83, abs=0.01)
    assert result["confidence"] == "medium"
    assert result["unit"] == "₪/month (household)"

    caveat = result["caveat"].lower()
    assert "modelled" in caveat
    assert "2021" in caveat  # names the income survey's vintage
    assert "2026" in caveat  # names the price data's vintage
    assert "understate" in caveat or "overstate" in caveat  # states the DIRECTION of the bias

    # The two factors must sum EXACTLY to the total (build_derived_metric's
    # own invariant — see analytics.py), checked here through the tool's
    # rounding, same discipline as the compare_items reconstruction test.
    assert sum(f["magnitude"] for f in result["factors"]) == pytest.approx(result["value"], abs=0.02)
    assert result["missing_inputs"] == []
    assert result["actual_monthly_household_income"] == pytest.approx(16953.0)


def test_estimate_derived_metric_locality_missing_income_data_drops_to_low_confidence():
    """Atlit (locality id 53) is one of the real 2/104 eligible
    localities the CBS socioeconomic survey does not cover, so it has no
    income_per_household_monthly at all. The required-income figure must
    still compute (median_price_4room IS present), but confidence must
    drop to 'low' and missing_inputs must name the gap explicitly —
    silently falling back to some default income would misrepresent a
    genuinely unmeasured quantity as a real comparison."""
    result = estimate_derived_metric("53")
    assert result["confidence"] == "low"
    assert result["missing_inputs"] == ["income_per_household_monthly"]
    assert result["actual_monthly_household_income"] is None
    assert result["income_gap_monthly"] is None
    assert result["years_of_income_to_buy_at_actual_income"] is None
    # The value itself is still a real number — median_price_4room is
    # present for every eligible locality (see LOCALITIES_META's
    # eligibility_rule) — only the comparison against actual income drops.
    assert result["value"] is not None
    assert result["value"] > 0


def test_estimate_derived_metric_neighborhood_id_raises_neighborhood_specific_error():
    with pytest.raises(UnknownItemError) as excinfo:
        estimate_derived_metric("28:65211075")
    assert "NEIGHBORHOOD" in str(excinfo.value)


def test_estimate_derived_metric_unknown_id_raises():
    with pytest.raises(UnknownItemError):
        estimate_derived_metric("not-a-real-locality")


# ─────────────────────────────────────────────────────────────────────────
# find_items — attribute filtering, and the "unknown key" vs "known key,
# unknown value" distinction
# ─────────────────────────────────────────────────────────────────────────
def test_find_items_empty_filters_matches_every_eligible_locality():
    result = find_items({})
    assert result["match_count"] == len(dataset.ELIGIBLE_IDS)
    assert set(result["item_ids"]) == set(dataset.ELIGIBLE_IDS)
    assert result["unknown_filter_keys"] == []


def test_find_items_unknown_filter_key_is_distinguished_from_zero_matches():
    """Filtering on a field that doesn't exist at all must report
    unknown_filter_keys non-empty — a caller checking that field first
    can tell 'you asked about something that isn't in this dataset' from
    'nothing in this dataset matches your value'."""
    result = find_items({"walkability_score": "high"})
    assert result["item_ids"] == []
    assert result["unknown_filter_keys"] == ["walkability_score"]


def test_find_items_known_key_but_wrong_value_reports_actual_values_not_a_bare_zero():
    """'district' holds the six official CBS districts; 'Sharon' is a
    real SUBDISTRICT name, not a district — a real, documented trap (see
    find_items' own docstring). Filtering district='Sharon' must not
    read as 'no localities in the Sharon exist' — it must surface the
    real values district actually takes, so the model can self-correct
    to subdistrict_name instead."""
    result = find_items({"district": "Sharon"})
    assert result["item_ids"] == []
    assert result["unknown_filter_keys"] == []  # 'district' IS a real key
    assert "known_values_for_filtered_keys" in result
    actual_district_values = result["known_values_for_filtered_keys"]["district"]["values"]
    assert "Sharon" not in actual_district_values
    assert "Center" in actual_district_values  # a real CBS district that IS a valid value


def test_find_items_real_filter_matches_a_known_subset():
    """A real, positive case: district=Center matches a genuine nonempty
    subset of the 104 eligible localities (29, measured against the
    committed dataset) — not the whole set and not zero."""
    result = find_items({"district": "Center"})
    assert 0 < result["match_count"] < len(dataset.ELIGIBLE_IDS)
    for item_id in result["item_ids"]:
        assert dataset.ATTRIBUTES[item_id]["district"] == "Center"


# ─────────────────────────────────────────────────────────────────────────
# get_item_metrics — single-locality raw metrics, no scoring
# ─────────────────────────────────────────────────────────────────────────
def test_get_item_metrics_returns_raw_unscored_values():
    result = get_item_metrics(KEFAR_SAVA)
    assert result["name_en"] == "Kefar Sava"
    assert result["metrics"]["price_level"] == pytest.approx(2753850.0)
    assert result["metrics"]["socioeconomic_level"] == pytest.approx(8.0)
    # Raw, not normalized — a normalized value would be inside [0, 1];
    # a real ₪ price is not, and this test would catch scoring logic
    # accidentally leaking into what is meant to be a pass-through.
    assert result["metrics"]["price_level"] > 1.0


def test_get_item_metrics_unknown_id_raises():
    with pytest.raises(UnknownItemError):
        get_item_metrics("not-a-real-locality")


# ─────────────────────────────────────────────────────────────────────────
# rank_by_priorities — reweighting to a user's stated priorities
# ─────────────────────────────────────────────────────────────────────────
def test_rank_by_priorities_emphasizing_price_flips_a_real_winner():
    """Real, checkable end-to-end case: at the DEFAULT weights, Kefar
    Sava beats Be'er Sheva despite being far more expensive (its
    accessibility/socioeconomic/momentum/yield advantages outweigh
    price_level's 30-point weight). Doubling price_level's weight (the
    documented PRIORITY_EMPHASIS_FACTOR) is enough to flip the winner to
    Be'er Sheva, the far cheaper of the two — a real demonstration that
    'I care more about price' actually changes the answer, not a
    decorative reweighting that never moves anything."""
    result = rank_by_priorities([KEFAR_SAVA, RAANANA, BEER_SHEVA], emphasize=["price_level"])

    assert result["default_ranking"][0]["item_id"] == KEFAR_SAVA
    assert result["adjusted_ranking"][0]["item_id"] == BEER_SHEVA
    assert result["priorities_changed_the_winner"] is True
    assert result["adjusted_weights"]["price_level"] == pytest.approx(60.0)  # 30 * 2.0
    assert result["default_weights"]["price_level"] == pytest.approx(30.0)
    assert "price_level" in result["assumption_to_state"]


def test_rank_by_priorities_returns_both_rankings_not_only_the_adjusted_one():
    """Handing back only the reweighted list would let a reweighting
    change the answer invisibly — the tool's own docstring requires both
    the default AND adjusted rankings to be present so the effect of the
    stated preference is legible."""
    result = rank_by_priorities([KEFAR_SAVA, RAANANA], emphasize=["accessibility"])
    assert len(result["default_ranking"]) == 2
    assert len(result["adjusted_ranking"]) == 2
    assert result["default_ranking"] != result["adjusted_ranking"]  # different score fields at least


def test_rank_by_priorities_deemphasize_alone_is_accepted():
    result = rank_by_priorities([KEFAR_SAVA, RAANANA], deemphasize=["rental_yield"])
    assert result["deemphasized"] == ["rental_yield"]
    assert result["adjusted_weights"]["rental_yield"] == pytest.approx(7.5)  # 15 / 2.0


# ─────────────────────────────────────────────────────────────────────────
# analyze_weight_sensitivity
# ─────────────────────────────────────────────────────────────────────────
def test_analyze_weight_sensitivity_reports_kendall_tau_and_top_item_change():
    result = analyze_weight_sensitivity([KEFAR_SAVA, RAANANA], "price_level", 0.1)
    assert result["criterion"] == "price_level"
    assert result["factor"] == pytest.approx(0.1)
    assert result["perturbed_weights"]["price_level"] == pytest.approx(3.0)  # 30 * 0.1
    assert result["baseline_weights"]["price_level"] == pytest.approx(30.0)
    assert "kendall_tau" in result
    assert -1.0 <= result["kendall_tau"] <= 1.0
    assert result["top_item"]["before"] == KEFAR_SAVA


def test_analyze_weight_sensitivity_rejects_a_non_positive_factor():
    """A zero or negative factor produces a zero/negative weight, which
    breaks scoring.py's 0..1 total_score guarantee — this must fail
    loudly with a ValueError, not silently produce a nonsense score."""
    with pytest.raises(ValueError):
        analyze_weight_sensitivity([KEFAR_SAVA], "price_level", 0.0)
    with pytest.raises(ValueError):
        analyze_weight_sensitivity([KEFAR_SAVA], "price_level", -2.0)


# ─────────────────────────────────────────────────────────────────────────
# weight_robustness_report
# ─────────────────────────────────────────────────────────────────────────
def test_weight_robustness_report_finds_a_flip_factor_for_every_criterion_or_none():
    """Every criterion gets an entry; flip_factor is either a real number
    (a multiplier that changes the winner) or None (robust up to the
    search range) — never missing, never a bare crash on a criterion
    with no flip at all (a real, reachable outcome, exercised on the
    real 104-locality set)."""
    result = weight_robustness_report(list(dataset.ELIGIBLE_IDS))
    assert {f["criterion"] for f in result["criteria"]} == {c.name for c in DEFAULT_CRITERIA}
    for finding in result["criteria"]:
        assert finding["flip_factor"] is None or finding["flip_factor"] > 0
    assert result["baseline_top"] is not None


def test_weight_robustness_report_names_the_most_sensitive_criterion():
    """most_sensitive_criterion must be one of the five real criterion
    names when the ranking has at least one flip — a bare None here
    while flips exist would silently drop the report's headline finding."""
    result = weight_robustness_report([KEFAR_SAVA, RAANANA, BEER_SHEVA, TEL_AVIV, JERUSALEM])
    if any(f["flip_factor"] is not None for f in result["criteria"]):
        assert result["most_sensitive_criterion"] in {c.name for c in DEFAULT_CRITERIA}


# ─────────────────────────────────────────────────────────────────────────
# get_current_mortgage_rates — the one live call, MOCKED here
# ─────────────────────────────────────────────────────────────────────────
# Deliberately mocked rather than hitting the real Bank of Israel API:
# a pytest suite that depends on a live third-party endpoint being up is
# flaky by construction (see the standing project rule on live-vs-mocked
# I/O), and — more importantly — mocking is the only way to deterministically
# exercise BOTH of get_current_mortgage_rates' two documented code paths
# (the live success path AND the "feed unreachable, fall back to the
# stated assumption" path) in the same run. Hitting the real endpoint
# would only ever test whichever path happens to be live right now.
def test_get_current_mortgage_rates_success_path_computes_the_prime_spread():
    fake_payload = b'{"currentInterest": 3.5, "lastPublishedDate": "2026-07-12", "nextInterestDate": "2026-09-01"}'

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return fake_payload

    with patch("app.tools.urllib.request.urlopen", return_value=_FakeResponse()):
        result = get_current_mortgage_rates()

    assert result["available"] is True
    assert result["is_stated_assumption"] is False
    assert result["boi_known_rate"] == pytest.approx(0.035)
    # estimated_prime_mortgage_rate = boi_known_rate + PRIME_MORTGAGE_SPREAD (0.015)
    assert result["estimated_prime_mortgage_rate"] == pytest.approx(0.05)
    assert "not part of the value-for-money score" in result["note"].lower()


def test_get_current_mortgage_rates_falls_back_when_the_live_feed_is_unreachable():
    """The failure path must NEVER raise — this tool is decoration on an
    answer that must still work without it (see its own docstring) — and
    must return a clearly labelled STATED ASSUMPTION default rather than
    silently pretending to be live data."""
    import urllib.error

    with patch(
        "app.tools.urllib.request.urlopen",
        side_effect=urllib.error.URLError("simulated network failure"),
    ):
        result = get_current_mortgage_rates()

    assert result["available"] is False
    assert result["is_stated_assumption"] is True
    assert result["boi_known_rate"] > 0  # a real, usable fallback number, not None/0
    assert result["estimated_prime_mortgage_rate"] > result["boi_known_rate"]
    assert "not current" in result["note"].lower() or "stated" in result["note"].lower()


def test_get_current_mortgage_rates_falls_back_on_an_unexpected_response_shape():
    """A live response that returns 200 but the wrong JSON shape (a
    KeyError on 'currentInterest') must degrade the same way a network
    failure does, not propagate an opaque KeyError up through the tool
    layer to the agent loop."""

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"unexpected_shape": true}'

    with patch("app.tools.urllib.request.urlopen", return_value=_FakeResponse()):
        result = get_current_mortgage_rates()

    assert result["available"] is False
    assert result["is_stated_assumption"] is True
