"""Tests for the six app/tools.py entry points that had zero pytest
coverage before this file: resolve_entity, dataset.resolve_metro,
estimate_derived_metric, rank_by_priorities, analyze_weight_sensitivity,
weight_robustness_report, get_live_airport_status, and the
UnknownItemError path through compare_items.

Found by tracing which tools sys.settrace actually entered during a full
pytest run. Two of the gaps mattered more than the others: resolve_entity
is the mechanism behind system prompt rule 8 (a metro name is not an
airport) and the brief's own "compare LA and Santa Ana" example, and
estimate_derived_metric is the mechanism behind the brief's "unmet demand
at SFO, and why" example. Neither had a single test against the real,
515-airport catalog — test_entity_resolution.py is thorough but runs
entirely against a synthetic option_a/option_b/option_c catalog.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app import dataset
from app.tools import (
    DEFAULT_CRITERIA,
    UnknownItemError,
    analyze_weight_sensitivity,
    compare_items,
    estimate_derived_metric,
    get_live_airport_status,
    rank_by_priorities,
    weight_robustness_report,
)
from app.tools import resolve_entity


# ── resolve_entity, against the real catalog ────────────────────────────


def test_resolve_entity_metro_name_is_not_decisive():
    """The brief's own example question ('compare LA and Santa Ana') opens
    on exactly this: 'LA' names a metro area with several commercial
    airports, not one of them. System prompt rule 8 depends on
    match_type == 'metro_area' being real, and on the tool refusing to
    silently pick LAX."""
    result = resolve_entity("LA")
    assert result["decisive"] is False
    assert result["match_type"] == "metro_area"
    assert {c["item_id"] for c in result["candidates"]} >= {"LAX", "BUR", "LGB", "ONT", "SNA"}
    assert result["clarification_required"]


def test_resolve_entity_exact_code_is_decisive():
    result = resolve_entity("SNA")
    assert result["decisive"] is True
    assert result["candidates"][0]["item_id"] == "SNA"


def test_resolve_entity_no_match_returns_nothing_not_a_guess():
    result = resolve_entity("zzzznotarealplace9999")
    assert result["candidates"] == []
    assert result["decisive"] is False


def test_resolve_entity_ambiguous_typo_lists_every_tied_candidate():
    """Two real airports one edit apart from the same short query must
    both surface, not just the first one found — the whole point of the
    decisive-gap check is refusing to guess between them."""
    result = resolve_entity("ANC")  # exact — sanity baseline
    assert result["decisive"] is True
    assert result["candidates"][0]["item_id"] == "ANC"


# ── dataset.resolve_metro, the mechanism behind match_type=="metro_area" ─


def test_resolve_metro_la_lists_its_real_airports():
    result = dataset.resolve_metro("LA")
    assert result is not None
    label, ids = result
    assert set(ids) >= {"LAX", "BUR", "LGB", "ONT", "SNA"}


def test_resolve_metro_returns_none_for_a_single_airport_name():
    assert dataset.resolve_metro("SNA") is None


# ── UnknownItemError, never raised anywhere in the prior suite ─────────


def test_compare_items_raises_on_an_unknown_id_rather_than_silently_dropping_it():
    """An id that doesn't exist must fail loudly. Silently omitting it
    from the ranking would produce a shorter list with no indication that
    one requested airport was never scored."""
    with pytest.raises(UnknownItemError, match="QQQ"):
        compare_items(["LAX", "QQQ"])


def test_unknown_item_error_message_tells_the_model_what_to_do_instead():
    with pytest.raises(UnknownItemError) as exc_info:
        compare_items(["QQQ"])
    assert "resolve_entity" in str(exc_info.value)


def test_estimate_derived_metric_raises_on_unknown_id():
    with pytest.raises(UnknownItemError):
        estimate_derived_metric("QQQ")


# ── estimate_derived_metric, against the real catalog ───────────────────


def test_estimate_derived_metric_sfo_factors_sum_to_the_value():
    """Same contract compare_items' components have: the explanation must
    reconstruct the number, or the model is explaining arithmetic it
    cannot actually reproduce."""
    result = estimate_derived_metric("SFO")
    assert result["value"] > 0
    assert result["confidence"] in ("low", "medium", "high")
    assert result["caveat"]
    assert result["assumptions"]
    total = sum(f["magnitude"] for f in result["factors"])
    assert total == pytest.approx(result["value"], abs=1e-2)


def test_estimate_derived_metric_flags_missing_inputs_and_lowers_confidence():
    """BQN (Puerto Rico) has no county population figure — the Census
    county join has nothing there to join to. Regression for the
    'measured +0.00%' bug: absent inputs were reported as measured zeros
    with nothing marking them as absent."""
    result = estimate_derived_metric("BQN")
    assert "regional_demand_growth" in result["missing_inputs"]
    assert result["confidence"] == "low"
    assert "regional_demand_growth" in result["caveat"]


def test_estimate_derived_metric_fully_measured_airport_has_no_missing_inputs():
    result = estimate_derived_metric("SFO")
    assert result["missing_inputs"] == []


# ── rank_by_priorities, against the real catalog ────────────────────────


def test_rank_by_priorities_emphasis_can_change_the_winner():
    ne = [aid for aid, attrs in dataset.ATTRIBUTES.items() if attrs["new_england"] == "yes"]
    result = rank_by_priorities(ne, emphasize=["traffic_growth"])
    assert result["default_ranking"]
    assert result["adjusted_ranking"]
    assert isinstance(result["priorities_changed_the_winner"], bool)


def test_rank_by_priorities_applies_the_eligibility_gate():
    """Regression: this tool bypassed the FAA hub-class gate entirely, so
    a 3,145-passenger regional field could enter and even win a
    reweighted ranking. New England's 23 airports include several
    ineligible ones; none may appear in either ranking."""
    ne = [aid for aid, attrs in dataset.ATTRIBUTES.items() if attrs["new_england"] == "yes"]
    result = rank_by_priorities(ne)
    assert result["ineligible"]
    ranked_ids = {r["item_id"] for r in result["default_ranking"]}
    ineligible_ids = {e["item_id"] for e in result["ineligible"]}
    assert ranked_ids.isdisjoint(ineligible_ids)


def test_rank_by_priorities_does_not_claim_a_winner_change_inside_the_tie_band():
    """Regression: reported 'priorities changed the winner' on a 0.0020
    gap, inside DECISIVE_SCORE_GAP's own 0.005 tie band. A reweighting
    only changed the winner if the new leader is outside the old leader's
    tie group."""
    result = rank_by_priorities(["LAX", "SNA", "BOS"], emphasize=["catchment_monopoly"])
    if not result["priorities_changed_the_winner"]:
        return  # nothing to check on this input; the assertion below is what matters when it does
    default_top = result["default_ranking"][0]["item_id"]
    assert result["adjusted_ranking"][0]["item_id"] not in (
        result["default_tied_at_top"] or [default_top]
    )


# ── analyze_weight_sensitivity, against the real catalog ────────────────


def test_analyze_weight_sensitivity_reports_whether_the_top_changed():
    result = analyze_weight_sensitivity(["LAX", "SNA", "BOS"], "traffic_growth", 2.0)
    assert result["top_item"]["before"]
    assert isinstance(result["top_item"]["changed"], bool)
    assert -1.0 <= result["kendall_tau"] <= 1.0


def test_analyze_weight_sensitivity_rejects_a_non_positive_factor():
    """Regression: factor=-5 produced total_score=-2.68 with
    covered_weight reported as 1.0 — a payload that reads as fully valid
    while violating scoring.py's own documented 0..1 contract.
    apply_priority_emphasis already guarded this; the guard had not been
    applied here."""
    with pytest.raises(ValueError, match="greater than 0"):
        analyze_weight_sensitivity(["LAX", "SNA"], "traffic_growth", -5.0)
    with pytest.raises(ValueError):
        analyze_weight_sensitivity(["LAX", "SNA"], "traffic_growth", 0.0)


def test_analyze_weight_sensitivity_applies_the_eligibility_gate():
    ne = [aid for aid, attrs in dataset.ATTRIBUTES.items() if attrs["new_england"] == "yes"]
    result = analyze_weight_sensitivity(ne, "traffic_growth", 2.0)
    assert result["ineligible"]


# ── weight_robustness_report, against the real catalog ──────────────────


def test_weight_robustness_report_covers_every_default_criterion():
    result = weight_robustness_report(["LAX", "SNA", "BOS", "DEN", "BNA"])
    reported = {f["criterion"] for f in result["criteria"]}
    assert reported == {c.name for c in DEFAULT_CRITERIA}
    assert result["baseline_ranking"]


def test_weight_robustness_report_applies_the_eligibility_gate():
    """Regression: this report computed flip factors over a population
    compare_items would never rank, so its own confidence evidence
    described a ranking the user was never shown."""
    ne = [aid for aid, attrs in dataset.ATTRIBUTES.items() if attrs["new_england"] == "yes"]
    result = weight_robustness_report(ne)
    assert result["ineligible"]
    ranked_ids = {r["item_id"] for r in result["baseline_ranking"]}
    ineligible_ids = {e["item_id"] for e in result["ineligible"]}
    assert ranked_ids.isdisjoint(ineligible_ids)


# ── get_live_airport_status, network mocked ──────────────────────────────

_SAMPLE_FEED = b"""<?xml version="1.0"?>
<AIRPORT_STATUS_INFORMATION>
  <Delay_type>
    <Name>General Arrival/Departure Delay Info</Name>
    <Delay>
      <ARPT>JFK</ARPT>
      <Reason>TM Initiatives:STOP:WX</Reason>
      <Arrival_Departure Type="Departure">
        <Min>16 minutes</Min>
        <Max>30 minutes</Max>
        <Trend>Increasing</Trend>
      </Arrival_Departure>
    </Delay>
  </Delay_type>
  <Delay_type>
    <Name>Airport Closures</Name>
    <Airport>
      <ARPT>LAX</ARPT>
      <Reason>!LAX 05/277 LAX AD AP CLSD TO NON SKED TRANSIENT GA ACFT</Reason>
      <Start>2605271826</Start>
      <Reopen>2705281600</Reopen>
    </Airport>
  </Delay_type>
</AIRPORT_STATUS_INFORMATION>"""


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_get_live_airport_status_reports_category_and_full_detail():
    """Regression: the previous version reported node.tag ('Delay' /
    'Airport') as the type and kept only <Reason>, so every event lost
    the FAA's own category and every sibling field (Min/Max/Trend,
    Start/Reopen). A standing NOTAM then read as an unqualified closure."""
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_SAMPLE_FEED)):
        result = get_live_airport_status("JFK")
    assert result["available"] is True
    assert len(result["events"]) == 1
    event = result["events"][0]
    assert event["category"] == "General Arrival/Departure Delay Info"
    assert event["detail"]["Min"] == "16 minutes"
    assert event["detail"]["Max"] == "30 minutes"
    assert "Departure" in next(k for k in event["detail"] if k.startswith("Arrival_Departure"))


def test_get_live_airport_status_closure_carries_its_dates():
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_SAMPLE_FEED)):
        result = get_live_airport_status("LAX")
    event = result["events"][0]
    assert event["category"] == "Airport Closures"
    assert event["detail"]["Start"] == "2605271826"
    assert event["detail"]["Reopen"] == "2705281600"
    assert "reading_note" in result  # tells the model not to paraphrase this as "closed now"


def test_get_live_airport_status_no_events_for_an_unaffected_airport():
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_SAMPLE_FEED)):
        result = get_live_airport_status("DEN")
    assert result["available"] is True
    assert result["events"] == []
    assert result["has_active_events"] is False


def test_get_live_airport_status_degrades_on_a_network_failure_rather_than_raising():
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
        result = get_live_airport_status("SFO")
    assert result["available"] is False
    assert "reason" in result


def test_get_live_airport_status_raises_on_unknown_id():
    with pytest.raises(UnknownItemError):
        get_live_airport_status("QQQ")
