"""Unit tests for app/analytics.py — the non-ranking query shapes
(filter, aggregate, derived metric). Pure, no network, no key, no LLM."""
from __future__ import annotations

import math

import pytest

from app.analytics import (
    FactorInput,
    UnsupportedOperationError,
    aggregate,
    build_derived_metric,
    filter_items,
    known_attribute_keys,
    known_attribute_values,
)

ATTRS = {
    "option_a": {"region": "north", "tier": "premium", "status": "active"},
    "option_b": {"region": "south", "tier": "standard", "status": "active"},
    "option_c": {"region": "north", "tier": "premium", "status": "retired"},
}

RECORDS = [
    {"category": "new_build", "units": 40.0},
    {"category": "resale", "units": 60.0},
    {"category": "new_build", "units": 20.0},
]


# ── A3: filter_items ────────────────────────────────────────────────────
def test_filter_single_attribute():
    assert filter_items(ATTRS, {"region": "north"}) == ("option_a", "option_c")


def test_filter_multiple_attributes_is_AND_not_OR():
    """Two filters means BOTH must hold. Unioning would return a superset
    that looks authoritative and answers a different question."""
    result = filter_items(ATTRS, {"region": "north", "status": "active"})
    assert result == ("option_a",)


def test_filter_is_case_insensitive_on_key_and_value():
    assert filter_items(ATTRS, {"REGION": "NORTH"}) == ("option_a", "option_c")


def test_filter_no_matches_returns_empty():
    assert filter_items(ATTRS, {"region": "atlantis"}) == ()


def test_filter_unknown_key_matches_nothing_rather_than_being_ignored():
    """Ignoring an unrecognised filter would silently broaden the query —
    answering a different question while looking successful."""
    assert filter_items(ATTRS, {"colour": "blue"}) == ()
    # ...and specifically it must not fall back to "everything"
    assert filter_items(ATTRS, {"region": "north", "colour": "blue"}) == ()


def test_filter_empty_filters_returns_everything_sorted():
    assert filter_items(ATTRS, {}) == ("option_a", "option_b", "option_c")


def test_filter_result_is_sorted_deterministically():
    assert list(filter_items(ATTRS, {})) == sorted(filter_items(ATTRS, {}))


def test_known_attribute_keys_lists_every_field():
    assert known_attribute_keys(ATTRS) == ("region", "status", "tier")


def test_known_attribute_values_lists_the_values_a_key_actually_takes():
    """The distinction known_attribute_keys can't draw: `region` exists,
    but it never equals 'atlantis'. Without this a caller cannot tell
    "no such field" from "real field, value you invented"."""
    result = known_attribute_values(ATTRS, ["region"])
    assert result["region"]["values"] == ["north", "south"]
    assert result["region"]["distinct_count"] == 2
    assert result["region"]["truncated"] is False


def test_known_attribute_values_skips_keys_that_do_not_exist():
    """An unknown key is already reported via unknown_filter_keys; echoing
    it here with an empty value list would imply the field exists."""
    assert known_attribute_values(ATTRS, ["colour"]) == {}


def test_known_attribute_values_truncates_high_cardinality_keys():
    """locality name has hundreds of values in the real dataset. The
    result is a hint to the model, not a dump of the dataset — but the
    SHAPE stays uniform so a reader never branches on which form it
    got."""
    wide = {f"i{n}": {"city": f"city_{n:03d}"} for n in range(50)}
    result = known_attribute_values(wide, ["city"], max_listed=5)
    assert result["city"]["distinct_count"] == 50
    assert len(result["city"]["values"]) == 5
    assert result["city"]["truncated"] is True


def test_known_attribute_values_matches_keys_case_insensitively():
    """filter_items folds case on keys, so this must agree with it or a
    caller passing 'Region' gets an empty diagnostic on a real field."""
    assert known_attribute_values(ATTRS, ["REGION"])["region"]["values"] == [
        "north",
        "south",
    ]


# ── A4: aggregate ───────────────────────────────────────────────────────
def test_aggregate_share_is_by_units_not_record_count():
    """The documented ambiguity: 2 of 3 RECORDS are new_build (0.667) but
    they carry 60 of 120 UNITS (0.5). This asserts which reading `share`
    implements, so a silent flip can't happen."""
    result = aggregate(RECORDS, "share", group_by_field="category", group_value="new_build")
    assert result.value == pytest.approx(0.5)
    assert result.matching_records == 2
    assert result.total_records == 3


def test_aggregate_count_counts_records():
    result = aggregate(RECORDS, "count", group_by_field="category", group_value="new_build")
    assert result.value == 2.0


def test_aggregate_sum_sums_units():
    result = aggregate(RECORDS, "sum", group_by_field="category", group_value="new_build")
    assert result.value == pytest.approx(60.0)


def test_aggregate_mean_averages_units_over_matching_records():
    result = aggregate(RECORDS, "mean", group_by_field="category", group_value="new_build")
    assert result.value == pytest.approx(30.0)


def test_aggregate_without_group_covers_all_records():
    result = aggregate(RECORDS, "sum")
    assert result.value == pytest.approx(120.0)
    assert result.matching_records == 3


def test_aggregate_is_case_insensitive_on_category():
    result = aggregate(RECORDS, "count", group_by_field="category", group_value="NEW_BUILD")
    assert result.value == 2.0


def test_aggregate_mean_of_empty_subset_is_nan_not_zero():
    """0.0 would be a real-looking number the model would report as fact.
    NaN forces the caller to handle 'undefined' explicitly."""
    result = aggregate(RECORDS, "mean", group_by_field="category", group_value="nonexistent")
    assert math.isnan(result.value)


def test_aggregate_share_of_zero_total_is_nan_not_zero():
    result = aggregate([], "share", group_by_field="category", group_value="new_build")
    assert math.isnan(result.value)


def test_aggregate_unsupported_operation_raises_named_error():
    with pytest.raises(UnsupportedOperationError):
        aggregate(RECORDS, "median")


def test_aggregate_reports_both_numerator_and_denominator():
    """The model must be able to say '60 of 120 units', not just '50%'."""
    result = aggregate(RECORDS, "share", group_by_field="category", group_value="new_build")
    assert result.matching_units == pytest.approx(60.0)
    assert result.total_units == pytest.approx(120.0)


# ── A5: build_derived_metric — the generic CONTRACT ─────────────────────
# The domain model that uses this (an affordability estimate) lives in
# app/tools.py and is tested in tests/test_tools_domain.py. What's
# asserted here is only what every modelled metric must satisfy, whatever
# the domain.
def _valid_contributions():
    return [
        FactorInput(name="a", magnitude=30.0, source_field="field_a", explanation="because a"),
        FactorInput(name="b", magnitude=10.0, source_field="field_b", explanation="because b"),
    ]


def test_build_derived_metric_value_is_the_sum_of_factors():
    """The load-bearing invariant: the total is COMPUTED from the
    contributions, never passed in beside them, so an explanation cannot
    drift from the number it claims to explain."""
    result = build_derived_metric(
        metric="m", unit="units", contributions=_valid_contributions(),
        assumptions=["an assumption"], confidence="medium", caveat="a caveat",
    )
    assert result.value == pytest.approx(40.0)
    assert sum(f.magnitude for f in result.factors) == pytest.approx(result.value)


def test_build_derived_metric_shares_sum_to_one():
    result = build_derived_metric(
        metric="m", unit="units", contributions=_valid_contributions(),
        assumptions=["an assumption"], confidence="medium", caveat="a caveat",
    )
    assert sum(f.share_of_total for f in result.factors) == pytest.approx(1.0)


def test_build_derived_metric_rejects_missing_assumptions():
    """A modelled number presented without assumptions is
    indistinguishable from a measured one — refuse to build it."""
    with pytest.raises(ValueError):
        build_derived_metric(
            metric="m", unit="units", contributions=_valid_contributions(),
            assumptions=[], confidence="medium", caveat="a caveat",
        )


def test_build_derived_metric_rejects_empty_caveat():
    with pytest.raises(ValueError):
        build_derived_metric(
            metric="m", unit="units", contributions=_valid_contributions(),
            assumptions=["an assumption"], confidence="medium", caveat="   ",
        )


def test_build_derived_metric_rejects_unknown_confidence_level():
    with pytest.raises(ValueError):
        build_derived_metric(
            metric="m", unit="units", contributions=_valid_contributions(),
            assumptions=["an assumption"], confidence="pretty sure", caveat="a caveat",
        )


def test_build_derived_metric_zero_magnitudes_do_not_divide_by_zero():
    result = build_derived_metric(
        metric="m", unit="units",
        contributions=[FactorInput(name="a", magnitude=0.0, source_field="f", explanation="e")],
        assumptions=["an assumption"], confidence="low", caveat="a caveat",
    )
    assert result.value == 0.0
    assert result.factors[0].share_of_total == 0.0


def test_build_derived_metric_offsetting_terms_keep_shares_interpretable():
    """Shares use ABSOLUTE magnitudes, so a model with a negative
    offsetting term still reports 'how much of the explanation is this',
    rather than exploding as the net approaches zero."""
    result = build_derived_metric(
        metric="m", unit="units",
        contributions=[
            FactorInput(name="up", magnitude=10.0, source_field="f1", explanation="e"),
            FactorInput(name="down", magnitude=-9.0, source_field="f2", explanation="e"),
        ],
        assumptions=["an assumption"], confidence="low", caveat="a caveat",
    )
    assert result.value == pytest.approx(1.0)
    assert sum(f.share_of_total for f in result.factors) == pytest.approx(1.0)
    assert all(0.0 <= f.share_of_total <= 1.0 for f in result.factors)
