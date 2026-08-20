"""Domain-model tests — the airport criteria, the unmet-demand model,
and the runway geometry behind it. All of this lives in app/tools.py and
app/runway_geometry.py, NOT in app/analytics.py or app/scoring.py.

The split matters: analytics.py holds the generic contract every modelled
metric must satisfy (tested in test_analytics.py) and scoring.py the
generic ranker (test_scoring.py). Both are domain-free and neither
changes when the domain does. The formula, its constants, its causal
reasoning, and the criteria themselves are this assignment's judgement —
so they are tested here, separately, on purpose.
"""
from __future__ import annotations

import pytest

from app import dataset
from app.runway_geometry import (
    DEPENDENT_APPROACH_MIN_SEPARATION_FT,
    INDEPENDENT_APPROACH_MIN_SEPARATION_FT,
    Runway,
    arrival_capacity,
    parallel_pairs,
    perpendicular_separation_ft,
)
from app.tools import (
    DECISIVE_SCORE_GAP,
    DEFAULT_CRITERIA,
    IMC_FRACTION,
    PRACTICAL_CAPACITY_PER_ARRIVAL_STREAM,
    compare_items,
    estimate_unmet_demand,
    find_items,
    list_criteria,
)

# SFO's 28L/28R, from OurAirports' published runway-end coordinates. The
# real separation is ~750 ft; this fixture is the actual data, so the
# geometry test below is a check against reality, not against itself.
SFO_10L = Runway("10L", 37.628742, -122.393410, 118.0, 11870.0)
SFO_10R = Runway("10R", 37.626298, -122.393124, 118.0, 11381.0)


# ─────────────────────────────────────────────────────────────────────────
# Runway geometry — the mechanism behind the "why"
# ─────────────────────────────────────────────────────────────────────────
def test_sfo_parallel_separation_matches_the_published_figure():
    """The whole unmet-demand story rests on SFO's runways being ~750 ft
    apart. If this computation drifts, the explanation becomes fiction —
    so it is pinned against the real-world published number."""
    assert perpendicular_separation_ft(SFO_10L, SFO_10R) == pytest.approx(750, abs=10)


def test_separation_is_perpendicular_not_threshold_to_threshold():
    """Staggered thresholds make the straight-line distance much larger
    than the centerline separation. Using the wrong one moves SFO into
    the wrong FAA band, which is why this is measured perpendicular."""
    import math

    ft_lon = 364_000.0 * math.cos(math.radians(SFO_10L.latitude_deg))
    straight = math.hypot(
        (SFO_10R.longitude_deg - SFO_10L.longitude_deg) * ft_lon,
        (SFO_10R.latitude_deg - SFO_10L.latitude_deg) * 364_000.0,
    )
    assert straight > 850  # the naive number
    assert perpendicular_separation_ft(SFO_10L, SFO_10R) < 800  # the correct one


def test_perpendicular_separation_is_symmetric():
    a = perpendicular_separation_ft(SFO_10L, SFO_10R)
    b = perpendicular_separation_ft(SFO_10R, SFO_10L)
    assert a == pytest.approx(b, abs=1.0)


def test_close_parallels_collapse_to_one_arrival_stream_in_imc():
    cap = arrival_capacity([SFO_10L, SFO_10R])
    assert cap.vmc_streams == 2
    assert cap.imc_streams == 1
    assert cap.weather_degradation == pytest.approx(0.5)


def test_widely_spaced_parallels_stay_independent_in_imc():
    far = Runway("10R", 37.626298 + 0.02, -122.393124, 118.0, 11381.0)
    cap = arrival_capacity([SFO_10L, far])
    assert cap.parallel_pairs[0].separation_ft >= INDEPENDENT_APPROACH_MIN_SEPARATION_FT
    assert cap.imc_streams == 2
    assert cap.weather_degradation == 0.0


def test_three_mutually_close_runways_collapse_to_one_not_two():
    """Union-find, not pairwise counting: three runways each too close to
    the next are ONE arrival stream. Counting pairs would say two."""
    a = Runway("16L", 47.44, -122.31, 180.0, 11900.0)
    b = Runway("16C", 47.44, -122.309, 180.0, 9426.0)
    c = Runway("16R", 47.44, -122.308, 180.0, 8500.0)
    cap = arrival_capacity([a, b, c])
    assert cap.vmc_streams == 3
    assert cap.imc_streams == 1


def test_short_runways_do_not_count_as_arrival_capacity():
    """SNA's short crosswind strip is general-aviation, not an air-carrier
    arrival stream. Counting it would overstate capacity — the exact
    overstatement ASSUMPTIONS.md flags for raw runway_count."""
    short = Runway("06", 33.67, -117.86, 20.0, 2887.0)
    long = Runway("20R", 33.67, -117.87, 200.0, 5701.0)
    cap = arrival_capacity([short, long])
    assert cap.vmc_streams == 1
    assert cap.air_carrier_runways == 1


def test_non_parallel_runways_are_not_paired():
    a = Runway("09", 40.0, -74.0, 90.0, 10000.0)
    b = Runway("18", 40.0, -74.0, 180.0, 10000.0)
    assert parallel_pairs([a, b]) == ()


def test_single_runway_airport_has_no_weather_degradation():
    cap = arrival_capacity([SFO_10L])
    assert cap.vmc_streams == 1
    assert cap.imc_streams == 1
    assert cap.weather_degradation == 0.0
    assert cap.min_parallel_separation_ft is None


def test_faa_separation_bands_are_ordered():
    """Guards the constants themselves: independent must require MORE
    separation than dependent, or the band logic silently inverts."""
    assert INDEPENDENT_APPROACH_MIN_SEPARATION_FT > DEPENDENT_APPROACH_MIN_SEPARATION_FT


# ─────────────────────────────────────────────────────────────────────────
# The unmet-demand model
# ─────────────────────────────────────────────────────────────────────────
def _sfo_like(**overrides):
    kwargs = dict(
        enplanements=26_251_850.0,
        arrival_streams_vmc=4,
        weather_capacity_degradation=0.5,
        traffic_growth=0.0468,
        regional_demand_growth=0.00513,
    )
    kwargs.update(overrides)
    return estimate_unmet_demand(**kwargs)


def test_unmet_demand_factor_magnitudes_sum_to_the_value():
    """The 'why' must reconstruct the 'what' exactly — if factors don't
    sum to the total, the explanation is decorative."""
    result = _sfo_like()
    assert sum(f.magnitude for f in result.factors) == pytest.approx(result.value)


def test_unmet_demand_factor_shares_sum_to_one():
    result = _sfo_like()
    assert sum(f.share_of_total for f in result.factors) == pytest.approx(1.0)


def test_weather_suppression_uses_the_degraded_capacity_and_imc_fraction():
    """Pins the actual arithmetic, not just that a number came out."""
    result = _sfo_like()
    practical = 4 * PRACTICAL_CAPACITY_PER_ARRIVAL_STREAM
    expected = max(0.0, 26_251_850.0 - practical * 0.5) * IMC_FRACTION
    weather = next(f for f in result.factors if f.name == "weather_suppressed_throughput")
    assert weather.magnitude == pytest.approx(expected)


def test_quiet_airport_has_no_unmet_demand_even_when_weather_degraded():
    """The self-gating property, and the reason this is a model rather
    than arithmetic. Anchorage loses a third of its arrival capacity in
    low visibility and still has zero unmet demand, because the remaining
    capacity covers everything that wanted to fly."""
    result = estimate_unmet_demand(
        enplanements=2_729_285.0,
        arrival_streams_vmc=3,
        weather_capacity_degradation=0.33,
        traffic_growth=-0.0139,
        regional_demand_growth=0.00019,
    )
    assert result.value == 0.0
    assert result.confidence == "low"


def test_weather_degradation_increases_unmet_demand():
    low = _sfo_like(weather_capacity_degradation=0.1)
    high = _sfo_like(weather_capacity_degradation=0.6)
    assert high.value > low.value


def test_declining_growth_is_clamped_not_credited_as_headroom():
    """A shrinking airport must never be handed negative unmet demand,
    which would quietly read as spare capacity."""
    shrinking = _sfo_like(traffic_growth=-0.5, regional_demand_growth=-0.5)
    structural = next(
        f for f in shrinking.factors if f.name == "structural_capacity_deficit"
    )
    assert structural.magnitude >= 0.0
    assert shrinking.value >= 0.0


def test_structural_deficit_appears_when_demand_exceeds_good_weather_capacity():
    result = estimate_unmet_demand(
        enplanements=40_000_000.0,
        arrival_streams_vmc=2,
        weather_capacity_degradation=0.0,
        traffic_growth=0.05,
        regional_demand_growth=0.01,
    )
    structural = next(f for f in result.factors if f.name == "structural_capacity_deficit")
    assert structural.magnitude > 0
    assert result.value > 0


def test_low_utilization_drops_confidence_and_says_why():
    result = estimate_unmet_demand(
        enplanements=500_000.0,
        arrival_streams_vmc=4,
        weather_capacity_degradation=0.0,
        traffic_growth=0.01,
        regional_demand_growth=0.001,
    )
    assert result.confidence == "low"
    assert "not this airport's constraint" in result.caveat


def test_high_utilization_is_reported_as_a_lower_bound():
    result = _sfo_like()
    assert result.confidence == "medium"
    assert "LOWER BOUND" in result.caveat


def test_zero_capacity_does_not_crash():
    """No arrival streams means an undefined utilization, which must not
    read as 'constrained' via a NaN comparison."""
    result = estimate_unmet_demand(
        enplanements=1_000.0,
        arrival_streams_vmc=0,
        weather_capacity_degradation=0.0,
        traffic_growth=0.0,
        regional_demand_growth=0.0,
    )
    assert result.confidence == "low"


def test_unmet_demand_always_carries_assumptions_and_a_caveat():
    """The number never travels alone — that is the whole contract."""
    for result in (_sfo_like(), _sfo_like(enplanements=1000.0)):
        assert result.assumptions
        assert result.caveat
        assert result.confidence in {"low", "medium", "high"}


def test_imc_fraction_is_flagged_as_the_weakest_assumption():
    """It is a single national figure applied uniformly. If that stops
    being stated, the model starts overclaiming."""
    assert any("instrument conditions" in a.lower() for a in _sfo_like().assumptions)


# ─────────────────────────────────────────────────────────────────────────
# The criteria themselves
# ─────────────────────────────────────────────────────────────────────────
def test_criteria_weights_match_the_documented_split():
    """25/25/20/15/15. If someone retunes these, DECISIONS.md and
    DESIGN_DOC.md are now wrong and this test says so."""
    weights = {c.name: c.weight for c in DEFAULT_CRITERIA}
    assert weights == {
        "traffic_growth": 25,
        "regional_demand_growth": 25,
        "catchment_monopoly": 20,
        "capacity_pressure": 15,
        "absolute_scale": 15,
    }


def test_size_flavoured_criteria_stay_a_minority_of_the_weight():
    """The brief asks about INCREASED capacity, not current size. The
    two size-correlated criteria must not dominate, or the ranking
    silently answers the wrong question."""
    weights = {c.name: c.weight for c in DEFAULT_CRITERIA}
    size_like = weights["absolute_scale"] + weights["capacity_pressure"]
    assert size_like < sum(weights.values()) / 2
    assert weights["absolute_scale"] <= 0.25 * sum(weights.values())


def test_every_criterion_has_a_real_normalization_window():
    for c in DEFAULT_CRITERIA:
        assert c.lower_bound < c.upper_bound, c.name


def test_criteria_names_match_the_dataset_fields():
    """A criterion no airport supplies data for would silently vanish
    into missing_criteria for every item and never be noticed."""
    supplied = {k for m in dataset.METRICS.values() for k in m}
    assert {c.name for c in DEFAULT_CRITERIA} <= supplied


def test_list_criteria_exposes_every_default_weight_with_no_item_ids():
    """The whole point of the tool: a meta-question about the methodology
    must be answerable without any airport to compare. If this table ever
    drifts from DEFAULT_CRITERIA, the 'proprietary weights' failure mode
    is one incomplete tool result away from recurring."""
    result = list_criteria()
    returned = {c["name"]: c["weight"] for c in result["criteria"]}
    assert returned == {c.name: c.weight for c in DEFAULT_CRITERIA}


def test_list_criteria_percentages_sum_to_100():
    result = list_criteria()
    assert sum(c["weight_pct_of_total"] for c in result["criteria"]) == pytest.approx(100.0)


def test_list_criteria_every_entry_has_a_description():
    result = list_criteria()
    for c in result["criteria"]:
        assert c["description"], c["name"]


def test_list_criteria_via_registry_takes_no_arguments():
    from app.tools import TOOL_REGISTRY

    result = TOOL_REGISTRY["list_criteria"]({})
    assert {c["name"] for c in result["criteria"]} == {c.name for c in DEFAULT_CRITERIA}


def test_lax_does_not_win_the_default_ranking():
    """The single sharpest check that the criteria answer the brief's
    question rather than 'which airport is biggest'."""
    result = compare_items(list(dataset.ELIGIBLE_IDS))
    top_ten = [r["item_id"] for r in result["ranking"][:10]]
    assert "LAX" not in top_ten


def test_ranking_reports_a_statistical_tie_rather_than_a_false_winner():
    """The top two airports are ~0.4% apart. Presenting that as a clear
    winner claims precision the weighting judgement does not have."""
    result = compare_items(list(dataset.ELIGIBLE_IDS))
    ranked = result["ranking"]
    gap = ranked[0]["total_score"] - ranked[1]["total_score"]
    if gap <= DECISIVE_SCORE_GAP:
        assert result["decisive"] is False
        assert len(result["tied_at_top"]) >= 2
    else:
        assert result["decisive"] is True
        assert result["tied_at_top"] == []


def test_airports_missing_a_criterion_still_rank():
    """14 airports have no Census population. They must rank on their
    remaining criteria, not be dropped or scored a spurious zero — the
    missing-data renormalization earning its keep on real ragged data."""
    no_pop = [
        k for k, m in dataset.METRICS.items()
        if "regional_demand_growth" not in m and k in dataset.ELIGIBLE_IDS
    ]
    if not no_pop:
        pytest.skip("no eligible airport is missing population data")
    result = compare_items(no_pop[:3] + ["BOS"])
    ranked_ids = {r["item_id"] for r in result["ranking"]}
    assert no_pop[0] in ranked_ids
    row = next(r for r in result["ranking"] if r["item_id"] == no_pop[0])
    assert row["covered_weight"] < 1.0
    assert "regional_demand_growth" in row["missing_criteria"]


def test_ineligible_airports_cannot_enter_a_ranking_by_the_back_door():
    """The eligibility gate must hold even when the caller passes tiny
    airports directly. Found live: the model called find_items without
    the eligibility filter and New Bedford Regional (3,145 passengers,
    +53% 'growth') came back ranked 4th in New England — the exact
    failure the gate exists to prevent. A filter the caller has to
    remember is a filter that gets forgotten, so it is enforced here."""
    result = compare_items(["BOS", "EWB", "BGR", "PVD"])
    ranked_ids = [r["item_id"] for r in result["ranking"]]
    assert "EWB" not in ranked_ids
    assert "BGR" not in ranked_ids

    # Set aside, never silently dropped — the user asked about them.
    set_aside = {i["item_id"] for i in result["ineligible"]}
    assert set_aside == {"EWB", "BGR"}
    assert all(i["reason"] for i in result["ineligible"])


def test_ineligible_airports_can_be_included_deliberately():
    result = compare_items(["BOS", "EWB"], include_ineligible=True)
    assert {r["item_id"] for r in result["ranking"]} == {"BOS", "EWB"}
    assert result["ineligible"] == []


def test_unknown_category_is_not_reported_as_zero():
    """A category that does not exist and a category with zero rows are
    different answers. Found live: the model asked for 'long haul',
    matched nothing, and said '0% of flights out of Anchorage are long
    haul' — every number correct, the sentence false."""
    from app.tools import aggregate_records

    result = aggregate_records("ANC", "share", "long haul")
    assert result["unknown_category"] is True
    assert result["value"] is None
    assert result["defined"] is False
    assert "domestic" in result["known_categories"]
    assert result["category_semantics"]


def test_known_category_still_aggregates_normally():
    from app.tools import aggregate_records

    result = aggregate_records("ANC", "share", "international")
    assert result["unknown_category"] is False
    assert 0.0 < result["value"] < 1.0


def test_find_items_via_registry_with_no_arguments_does_not_crash():
    """filters={} is a meaningful call ('match everything', per
    find_items' own docstring), not malformed input. Found live in P4
    evals: gpt-4o-mini called find_items with no arguments at all and got
    a raw KeyError('filters') logged as a tool error, wasting a turn on a
    request that was clearly asking for the whole dataset."""
    from app.tools import TOOL_REGISTRY

    result = TOOL_REGISTRY["find_items"]({})
    assert result["match_count"] == len(dataset.AIRPORTS)


def test_find_items_reports_the_real_values_when_a_known_key_matches_nothing():
    """The brief's OWN first question, failing live against gpt-4o-mini:
    "Which airports in New England are strong candidates for terminal
    expansion?" -> find_items({'region': 'New England'}) -> 0 rows ->
    "there are no airports in New England that match."

    `region` is a real key, so unknown_filter_keys was empty and the model
    had no way to tell "you invented a value" from "none exist". It holds
    Census REGIONS; New England is a Census DIVISION. There are 23 such
    airports under {'new_england': 'yes'}.

    Pins that the empty result now carries the real value space, so the
    model can correct itself rather than assert a false absence."""
    result = find_items({"region": "New England"})

    assert result["match_count"] == 0
    # The key IS known — which is precisely why the old signal was silent.
    assert result["unknown_filter_keys"] == []
    assert result["known_values_for_filtered_keys"]["region"]["values"] == [
        "Midwest",
        "Northeast",
        "Other",
        "South",
        "West",
    ]
    assert "New England" in result["guidance"]
    # The route to the right answer must be discoverable from the result.
    assert "new_england" in result["known_attribute_keys"]


def test_find_items_new_england_key_returns_the_airports_that_do_exist():
    """The other half: the correct call must actually work, or the
    guidance above sends the model somewhere equally empty."""
    result = find_items({"new_england": "yes"})
    assert result["match_count"] == 23
    assert "guidance" not in result


def test_find_items_accepts_a_flattened_call_instead_of_matching_everything():
    """Found live on the brief's first question. The model called
    find_items({'new_england': 'yes'}) — filter keys at the top level,
    no 'filters' wrapper. args.get('filters') was None, "no filters"
    means "match everything", so a request for a 23-row SUBSET returned
    all 515 rows and reported success. The model then named New England
    airports from memory, which is the hallucination this tool exists to
    prevent.

    Both shapes must mean the same thing."""
    from app.tools import TOOL_REGISTRY

    flattened = TOOL_REGISTRY["find_items"]({"new_england": "yes"})
    nested = TOOL_REGISTRY["find_items"]({"filters": {"new_england": "yes"}})

    assert flattened["match_count"] == 23
    assert flattened["item_ids"] == nested["item_ids"]
    # The echoed filters are the audit trail: the log shows what was applied.
    assert flattened["filters"] == {"new_england": "yes"}


def test_find_items_no_arguments_still_means_match_everything():
    """The flattening fix must not break the legitimate empty call — that
    behaviour was itself a P4 eval finding and is documented in the
    tool's own docstring."""
    from app.tools import TOOL_REGISTRY

    assert TOOL_REGISTRY["find_items"]({})["match_count"] == len(dataset.AIRPORTS)
    assert TOOL_REGISTRY["find_items"]({"filters": None})["match_count"] == len(
        dataset.AIRPORTS
    )


def test_find_items_success_path_carries_no_diagnostics():
    """The diagnostic fires only when it's needed. A successful filter
    must not pay for it — every extra key here is context the model reads
    on every single call."""
    result = find_items({"hub_class": "L"})
    assert result["match_count"] > 0
    assert "guidance" not in result
    assert "known_values_for_filtered_keys" not in result


# ── focus_criterion: answering a single-dimension question ─────────────
# Origin, from running the brief's own example question against the real
# model: asked to "compare LA and Santa Ana congestion levels" the agent
# read the composite total_score and concluded "LAX has the higher total
# score, therefore LAX is more congested." Both halves true, conclusion
# false — total_score blends five criteria and only one of them is the
# congestion proxy. Two rounds of system-prompt instruction did not stop
# it, so the answer moved into the tool, where the rest of this file's
# numbers already live.


def test_focus_is_absent_unless_asked_for():
    """The default must stay the full weighted ranking. This is the
    regression test for the first attempt at the fix, which made the
    model apply a single-criterion view to 'which airports are strong
    candidates for terminal expansion' and rank New England by traffic
    growth alone — a worse answer than the one being fixed."""
    from app.tools import compare_items

    assert compare_items(["LAX", "SNA"])["focus"] is None


def test_focus_ranks_on_that_criterion_alone_not_on_the_total():
    from app.tools import compare_items

    result = compare_items(["LAX", "SNA"], focus_criterion="traffic_growth")
    focus = result["focus"]
    rows = focus["ranked_by_this_criterion_alone"]
    # SNA has positive traffic growth and LAX negative, while LAX wins the
    # composite. If these two orderings ever agree, this test is no longer
    # checking anything — which is exactly why this pair was chosen.
    assert [r["item_id"] for r in rows] == ["SNA", "LAX"]
    assert result["ranking"][0]["item_id"] == "LAX"
    assert focus["highest"] == "SNA"
    assert focus["lowest"] == "LAX"


def test_focus_numbers_are_the_same_numbers_as_the_main_ranking():
    """Nothing is recomputed — the focus block re-reads components the
    scoring pass already produced. If it ever disagreed with the ranking
    it sits next to, the agent would have two different answers for the
    same question."""
    from app.tools import compare_items

    result = compare_items(["LAX", "SNA", "BOS"], focus_criterion="capacity_pressure")
    from_ranking = {
        entry["item_id"]: next(
            c["raw_value"] for c in entry["components"] if c["criterion"] == "capacity_pressure"
        )
        for entry in result["ranking"]
    }
    from_focus = {r["item_id"]: r["raw_value"] for r in result["focus"]["ranked_by_this_criterion_alone"]}
    assert from_focus == from_ranking


def test_focus_carries_the_warning_not_to_read_it_as_the_total_score():
    from app.tools import compare_items

    note = compare_items(["LAX", "SNA"], focus_criterion="capacity_pressure")["focus"]["note"]
    assert "total_score" in note
    assert "capacity_pressure" in note


def test_focus_reports_an_unknown_criterion_instead_of_ignoring_it():
    """A typo must not degrade quietly into 'no focus block', because the
    model would then answer from the composite — the exact failure this
    parameter exists to prevent."""
    from app.tools import compare_items

    focus = compare_items(["LAX", "SNA"], focus_criterion="congestion")["focus"]
    assert "error" in focus
    assert "capacity_pressure" in focus["error"]  # names the valid options
    assert "ranked_by_this_criterion_alone" not in focus


def test_focus_separates_no_data_from_a_low_value():
    """'No data' and 'the smallest number' are different answers to
    'which is more congested', and collapsing them would invent a fact."""
    ranking = [
        {
            "item_id": "AAA",
            "components": [{"criterion": "capacity_pressure", "raw_value": 100.0, "normalized_score": 0.5}],
        },
        {"item_id": "BBB", "components": [{"criterion": "capacity_pressure", "raw_value": None, "normalized_score": None}]},
    ]
    from app.tools import DEFAULT_CRITERIA, _focus_on_criterion

    focus = _focus_on_criterion("capacity_pressure", ranking, DEFAULT_CRITERIA)
    assert focus["no_data_for"] == ["BBB"]
    assert [r["item_id"] for r in focus["ranked_by_this_criterion_alone"]] == ["AAA"]
    assert focus["lowest"] == "AAA"  # not BBB — BBB has no value at all


def test_focus_is_reachable_through_the_registry():
    """The schema exposes focus_criterion, so the registry dispatch has to
    forward it — a parameter the model can request and the dispatcher
    drops is worse than one that does not exist."""
    from app.tools import TOOL_REGISTRY

    result = TOOL_REGISTRY["compare_items"](
        {"item_ids": ["LAX", "SNA"], "focus_criterion": "capacity_pressure"}
    )
    assert result["focus"]["criterion"] == "capacity_pressure"
    assert TOOL_REGISTRY["compare_items"]({"item_ids": ["LAX", "SNA"]})["focus"] is None


def test_focus_criterion_enum_matches_the_real_criteria():
    """The schema lists the valid criterion names inline for the model.
    If a criterion is renamed and the enum is not, the model requests a
    name the tool rejects."""
    from app.tools import DEFAULT_CRITERIA, TOOL_SCHEMAS

    schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "compare_items")
    enum = schema["function"]["parameters"]["properties"]["focus_criterion"]["enum"]
    assert sorted(enum) == sorted(c.name for c in DEFAULT_CRITERIA)
