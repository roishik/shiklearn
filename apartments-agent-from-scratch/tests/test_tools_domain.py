"""Domain tests for app/tools.py — the ranking KPIs (DEFAULT_CRITERIA),
entity resolution as exposed through the tool surface, the full
compare_items breakdown, and — the single most important thing this file
pins — the eligibility gate holding across EVERY ranking tool, not just
compare_items.

Everything here runs against the REAL, committed dataset
(data/processed_data/{localities,neighborhoods}.json), not a synthetic
fixture. That is deliberate: app/tools.py's docstring and _gate_ids'
docstring both describe a REAL prior-build bug (the gate lived only in
compare_items; three sibling ranking tools reproduced the exact failure
it existed to prevent), and a fixture-based test could not have caught
that — it would have needed the fixture to also include a neighborhood
id, which nobody writing a synthetic fixture is likely to think to add.
Testing against the real 104-locality / 1,394-neighborhood dataset means
the ids used below are real, resolvable things, and the numbers pinned
below were produced by actually running the code on 2026-08-21 against
the dataset committed at that date — see each test's own comment for how
the expected value was obtained.

Companion file: tests/test_tools_uncovered.py covers aggregate_records,
estimate_derived_metric, find_items, rank_by_priorities,
analyze_weight_sensitivity, weight_robustness_report, and
get_current_mortgage_rates in more depth. This file's job is the
criteria themselves, resolve_entity, compare_items' full breakdown, and
the gate.
"""
from __future__ import annotations

import pytest

from app import dataset
from app.tools import (
    CRITERION_DESCRIPTIONS,
    DECISIVE_SCORE_GAP,
    DEFAULT_CRITERIA,
    UnknownItemError,
    analyze_weight_sensitivity,
    compare_items,
    get_item_metrics,
    list_criteria,
    rank_by_priorities,
    resolve_entity,
    weight_robustness_report,
)

# Real ids used throughout this file, named once so a reader doesn't have
# to cross-reference CBS locality codes by hand:
KEFAR_SAVA = "6900"
RAANANA = "8700"
BEER_SHEVA = "9000"
TEL_AVIV = "5000"
JERUSALEM = "3000"
QIRYAT_MOTZKIN = "8200"
JUDEIDE_MAKER = "1292"  # missing rental_yield — see the renormalization test below
# A real neighborhood id (belongs to locality '28', Mazkeret Batya) — the
# flagship case for _gate_ids: a real, resolvable id that is NOT a
# rankable locality. Verified present in data/processed_data/neighborhoods.json.
A_NEIGHBORHOOD_ID = "28:65211075"
AN_UNKNOWN_ID = "not-a-real-id-999999"


# ─────────────────────────────────────────────────────────────────────────
# DEFAULT_CRITERIA / CRITERION_DESCRIPTIONS — the KPIs themselves
# ─────────────────────────────────────────────────────────────────────────
def test_default_criteria_has_the_five_documented_criteria_with_locked_weights():
    """The weight table is a judgement call recorded at length in
    app/tools.py's own header comment (measured correlations moved three
    of the five weights off the original design sketch). Pinning the
    final numbers here means a future edit to that judgement call has to
    touch this test too, rather than silently drifting."""
    weights = {c.name: c.weight for c in DEFAULT_CRITERIA}
    assert weights == {
        "price_level": 30,
        "accessibility": 20,
        "price_momentum_vs_country": 20,
        "socioeconomic_level": 15,
        "rental_yield": 15,
    }
    assert sum(weights.values()) == 100


def test_default_criteria_directions_match_the_documented_intent():
    """price_level and accessibility are lower-is-better (cheaper /
    closer wins); the other three are higher-is-better. Getting even one
    of these backwards would silently invert half a criterion's meaning
    while every other test that only checks ORDER (not direction) would
    stay green."""
    directions = {c.name: c.higher_is_better for c in DEFAULT_CRITERIA}
    assert directions == {
        "price_level": False,
        "accessibility": False,
        "price_momentum_vs_country": True,
        "socioeconomic_level": True,
        "rental_yield": True,
    }


def test_every_criterion_has_a_plain_language_description():
    """CRITERION_DESCRIPTIONS is what list_criteria and _focus_on_criterion
    hand to the model — a criterion with no entry would render as an
    empty string in a user-facing explanation, silently, with no test
    catching a name added to DEFAULT_CRITERIA and forgotten here."""
    for criterion in DEFAULT_CRITERIA:
        assert criterion.name in CRITERION_DESCRIPTIONS
        assert len(CRITERION_DESCRIPTIONS[criterion.name]) > 20


def test_criterion_bounds_are_not_degenerate():
    """A lower_bound == upper_bound would make Criterion.normalize return
    a constant 0.5 for every locality on that criterion (see
    scoring.py's degenerate-bounds branch) — silently flattening it to
    zero information. The frozen percentile bounds should never do this
    for a real, continuously-valued field."""
    for criterion in DEFAULT_CRITERIA:
        assert criterion.lower_bound < criterion.upper_bound


def test_list_criteria_weight_percentages_sum_to_100():
    result = list_criteria()
    total_pct = sum(c["weight_pct_of_total"] for c in result["criteria"])
    assert total_pct == pytest.approx(100.0, abs=0.1)  # rounding to 1 dp each


def test_list_criteria_states_the_score_is_not_investment_advice():
    """list_criteria is the tool a meta-question about methodology routes
    to (see its own docstring) — the 'value for money to live in, not an
    investment score' framing is a load-bearing disclosure locked with
    Roi (_working/PROGRESS.md decision 2), not decoration."""
    result = list_criteria()
    assert "good value to live in" in result["score_meaning"].lower()
    assert "not an investment-return score" in result["score_meaning"].lower()


# ─────────────────────────────────────────────────────────────────────────
# resolve_entity — against the REAL 104-locality catalog
# ─────────────────────────────────────────────────────────────────────────
def test_resolve_entity_modiin_is_genuinely_non_decisive():
    """THE flagship non-decisive case (see _working/PROGRESS.md's own
    'gotchas' section): 'Modiin' is close to BOTH Modi'in-Makkabbim-Re'ut
    (1200) and Modi'in Illit (3797) — two real, different places, not a
    typo of one 'true' answer. A resolver that silently picked the
    higher-confidence candidate here would give a factually wrong answer
    to a real, plausible user query with total confidence. Pinned to the
    hand-verified values in PROGRESS.md so a resolver recalibration that
    accidentally makes this decisive is caught immediately."""
    result = resolve_entity("Modiin")
    assert result["decisive"] is False
    assert result["match_type"] == "locality"
    ids = {c["item_id"] for c in result["candidates"]}
    assert ids == {"1200", "3797"}

    by_id = {c["item_id"]: c for c in result["candidates"]}
    assert by_id["1200"]["confidence"] == pytest.approx(0.8581, abs=1e-4)
    assert by_id["1200"]["name_en"] == "Modi'in-Makkabbim-Re'ut"
    assert by_id["3797"]["confidence"] == pytest.approx(0.8227, abs=1e-4)
    assert by_id["3797"]["name_en"] == "Modi'in Illit"


def test_resolve_entity_hebrew_exact_name_is_decisive():
    """The contrasting case to Modiin: an exact Hebrew name with no real
    competitor resolves cleanly, at full confidence, to Kefar Sava."""
    result = resolve_entity("כפר סבא")
    assert result["decisive"] is True
    assert result["match_type"] == "locality"
    assert result["candidates"][0]["item_id"] == KEFAR_SAVA
    assert result["candidates"][0]["confidence"] == pytest.approx(1.0)
    assert result["candidates"][0]["name_en"] == "Kefar Sava"


def test_resolve_entity_region_query_returns_every_member_not_one_place():
    """'the Krayot' names FOUR real localities, not the best-matching one
    — collapsing a region query to a single locality would silently
    decide something the user never asked. match_type must read
    'region', distinct from an ordinary ambiguous locality match, so the
    model routes to a different explanation ('this names an AREA')
    rather than 'this name is ambiguous between places'."""
    result = resolve_entity("Krayot")
    assert result["match_type"] == "region"
    assert result["decisive"] is False
    ids = {c["item_id"] for c in result["candidates"]}
    # The four real Krayot towns in the eligible set.
    assert ids == {"6800", "9500", "9600", "8200"}
    assert result["clarification_required"]


def test_resolve_entity_no_match_returns_empty_not_a_guess():
    """A nonsense query is a normal fuzzy-search outcome (see
    resolve_entity's own docstring), not an exception and not a
    best-effort substitution of something similar-sounding."""
    result = resolve_entity("zzzznotarealplaceinisrael9999")
    assert result["candidates"] == []
    assert result["decisive"] is False


# ─────────────────────────────────────────────────────────────────────────
# compare_items — full breakdown, pinned against a real pair
# ─────────────────────────────────────────────────────────────────────────
def test_compare_kefar_sava_vs_raanana_full_breakdown():
    """Pinned end-to-end against real nadlan.gov.il/CBS figures, hand
    verified (see _working/agent-logs/tool-tests.md). Kefar Sava wins
    despite Ra'annana being closer to Tel Aviv and having stronger price
    momentum, because Ra'annana is meaningfully MORE expensive
    (3,381,700 vs 2,753,850 ₪) and price_level carries the largest single
    weight (30) — a real, checkable instance of the ranking doing what
    DEFAULT_CRITERIA's own header comment says it should."""
    result = compare_items([KEFAR_SAVA, RAANANA])

    assert result["decisive"] is True
    assert result["tied_at_top"] == []
    assert result["ineligible"] == []
    assert result["excluded"] == []
    assert result["no_items_ranked"] is False

    assert [r["item_id"] for r in result["ranking"]] == [KEFAR_SAVA, RAANANA]
    winner, runner_up = result["ranking"]

    assert winner["name_en"] == "Kefar Sava"
    assert winner["total_score"] == pytest.approx(0.5797, abs=1e-4)
    assert winner["covered_weight"] == pytest.approx(1.0)
    assert winner["missing_criteria"] == []
    assert len(winner["components"]) == 5

    by_criterion = {c["criterion"]: c for c in winner["components"]}
    assert by_criterion["price_level"]["raw_value"] == pytest.approx(2753850.0)
    assert by_criterion["price_level"]["weight"] == pytest.approx(0.30)
    assert by_criterion["accessibility"]["raw_value"] == pytest.approx(16.28)
    assert by_criterion["socioeconomic_level"]["raw_value"] == pytest.approx(8.0)
    assert by_criterion["rental_yield"]["raw_value"] == pytest.approx(3.27)
    # The breakdown must reconstruct the total — the same invariant
    # test_scoring.py's arithmetic tests pin for the generic ranker,
    # checked here again end-to-end through the tool layer's own
    # rounding (compare_items rounds every field independently before
    # returning it, so this is a real check that rounding didn't quietly
    # break the reconstruction property, not a restatement of the
    # scoring.py test).
    assert sum(c["contribution"] for c in winner["components"]) == pytest.approx(
        winner["total_score"], abs=1e-3
    )

    assert runner_up["name_en"] == "Ra'annana"
    assert runner_up["total_score"] == pytest.approx(0.4701, abs=1e-4)
    assert runner_up["total_score"] < winner["total_score"]


def test_compare_items_tie_and_missing_field_renormalization_on_the_real_leaderboard():
    """Two real findings from ranking the FULL eligible set, both pinned
    together because they occur in the SAME real pair and this is the
    exact scenario app/tools.py's own header comment for
    DECISIVE_SCORE_GAP describes hand-checking:

    1. TIE: at the final weights, Qiryat Motzkin (0.7257) and
       Judeide-Maker (0.7245) are the real #1/#2 of the whole eligible
       set and differ by only 0.0012 — inside DECISIVE_SCORE_GAP
       (0.005). This must come back non-decisive with both ids in
       tied_at_top; presenting Qiryat Motzkin as an unqualified winner
       would claim a precision the data does not support.

    2. MISSING-FIELD RENORMALIZATION: Judeide-Maker is one of the real
       6/104 localities missing rental_yield (see dataset.py's
       docstring on join coverage). Its covered_weight is therefore
       0.85 (85/100 = the weight of the other four criteria), and its
       remaining four component weights are renormalized over that 85,
       not the original 100 — e.g. price_level's component weight
       becomes 0.30/0.85 = 0.3529, not 0.30. This is real data
       demonstrating rank_items' renormalization behavior (already unit
       tested in isolation in test_scoring.py) actually firing on the
       committed dataset.
    """
    result = compare_items([QIRYAT_MOTZKIN, JUDEIDE_MAKER])

    assert result["decisive"] is False
    assert set(result["tied_at_top"]) == {QIRYAT_MOTZKIN, JUDEIDE_MAKER}
    assert result["ranking"][0]["item_id"] == QIRYAT_MOTZKIN
    assert result["ranking"][0]["total_score"] == pytest.approx(0.7257, abs=1e-4)

    judeide = next(r for r in result["ranking"] if r["item_id"] == JUDEIDE_MAKER)
    assert judeide["total_score"] == pytest.approx(0.7245, abs=1e-4)
    assert judeide["covered_weight"] == pytest.approx(0.85)
    assert judeide["missing_criteria"] == ["rental_yield"]
    assert len(judeide["components"]) == 4  # rental_yield dropped, not scored as 0

    by_criterion = {c["criterion"]: c for c in judeide["components"]}
    assert "rental_yield" not in by_criterion
    # Renormalized weight = original_weight / covered_weight, e.g.
    # 0.30 / 0.85 — the exact renormalization scoring.py's own tests
    # pin in the abstract, confirmed firing here on a real gap.
    assert by_criterion["price_level"]["weight"] == pytest.approx(0.30 / 0.85, abs=1e-3)
    assert by_criterion["socioeconomic_level"]["weight"] == pytest.approx(0.15 / 0.85, abs=1e-3)
    # Component weights still sum to 1.0 even after renormalization.
    assert sum(c["weight"] for c in judeide["components"]) == pytest.approx(1.0, abs=1e-3)


def test_compare_items_focus_criterion_answers_the_named_dimension_not_total_score():
    """Asked 'which has the better rental yield', a model reading only
    total_score would answer with the wrong place — total_score blends
    all five criteria, and Kefar Sava's total_score win is NOT the same
    fact as Kefar Sava having the higher rental_yield (it does, but for
    a question this specific the tool must say so explicitly rather than
    have the model infer it from a number that doesn't mean that)."""
    result = compare_items([KEFAR_SAVA, RAANANA], focus_criterion="rental_yield")
    focus = result["focus"]
    assert focus is not None
    assert focus["criterion"] == "rental_yield"
    assert focus["best"] == KEFAR_SAVA  # 3.27 vs 2.51 — Kefar Sava has the higher yield
    assert focus["worst"] == RAANANA
    assert focus["higher_is_better"] is True
    ordered_ids = [row["item_id"] for row in focus["ranked_by_this_criterion_alone"]]
    assert ordered_ids == [KEFAR_SAVA, RAANANA]
    # The tool must say plainly this is not the composite score, so a
    # model relaying it can't accidentally conflate the two.
    assert "not" in focus["note"].lower()
    assert "total_score" not in focus["criterion"]


def test_compare_items_focus_criterion_unknown_name_reports_an_error_not_silence():
    """A bad criterion name (a typo the model made up) must fail loudly
    in the response — falling back to describing the composite score
    would be exactly the failure focus_criterion exists to prevent."""
    result = compare_items([KEFAR_SAVA, RAANANA], focus_criterion="walkability")
    assert result["focus"]["error"] is not None
    assert "walkability" in result["focus"]["error"]


def test_compare_items_no_items_ranked_flag_is_true_when_the_list_is_empty():
    """An empty ranking is a real, reachable outcome (every id gated out)
    and must be an AFFIRMATIVE signal, not a payload indistinguishable
    from 'nothing was ranked because the caller passed no ids and this
    silently means agreement'. See compare_items' own comment on this
    field."""
    result = compare_items([A_NEIGHBORHOOD_ID])  # the only id gated out entirely
    assert result["ranking"] == []
    assert result["no_items_ranked"] is True
    assert result["decisive"] is False  # no winner exists when nothing is ranked


# ─────────────────────────────────────────────────────────────────────────
# Errors — unknown ids and neighborhood-not-rankable
# ─────────────────────────────────────────────────────────────────────────
def test_unknown_locality_id_raises_and_points_at_resolve_entity():
    """An id that is neither a known locality nor a known neighborhood is
    a real error (system_prompt.py's NEVER_INVENT_IDS_RULE) — the message
    must name resolve_entity as the correct next step, not just say
    'not found', so a model reading the error has an actionable next
    tool call rather than a dead end."""
    with pytest.raises(UnknownItemError) as excinfo:
        get_item_metrics(AN_UNKNOWN_ID)
    assert "resolve_entity" in str(excinfo.value)
    assert AN_UNKNOWN_ID in str(excinfo.value)


def test_neighborhood_id_passed_to_get_item_metrics_names_it_as_a_neighborhood():
    """get_item_metrics (single-item, non-ranking) hits the SAME
    fetch_item_metrics as every ranking tool, so a neighborhood id here
    must raise the specific 'this is a neighborhood, not a locality'
    error, distinct from an ordinary unknown-id error — the two are
    different facts (a real, un-rankable thing vs. nothing at all) and
    conflating them would misdirect the model's next move."""
    with pytest.raises(UnknownItemError) as excinfo:
        get_item_metrics(A_NEIGHBORHOOD_ID)
    message = str(excinfo.value)
    assert "NEIGHBORHOOD" in message
    assert "28" in message  # names the real parent locality id
    assert "aggregate_records" in message  # points at the tool that DOES answer this


# ─────────────────────────────────────────────────────────────────────────
# THE ELIGIBILITY GATE — parametrized across ALL FOUR ranking tools
# ─────────────────────────────────────────────────────────────────────────
# This is the single most important test in this file. In the prior
# (airport) build, the eligibility gate lived only inside compare_items,
# on the reasoning that "a filter the caller has to remember is a filter
# that gets forgotten" — and then three sibling ranking tools
# (rank_by_priorities, analyze_weight_sensitivity, weight_robustness_report)
# were added that did NOT call it, each one reproducing the exact bug the
# gate was written to prevent: a neighborhood id, which has no
# socioeconomic/accessibility/income data of its own, silently scored on
# a sliver of the criteria and presented as a fairly-ranked peer of a
# real city.
#
# app/tools.py's _gate_ids docstring says this was fixed by calling
# _gate_ids from all four ranking tools, and says so is pinned by "this"
# test. This IS that test — written as a single parametrized case over
# all four callables, specifically so that a FIFTH ranking tool added
# later without wiring in the gate fails this test immediately, rather
# than needing someone to remember to add a new bespoke test for it.
#
# Each wrapper below normalizes the four tools' different signatures
# (they take different required arguments beyond item_ids) down to a
# single (item_ids) -> {"ineligible": [...], <ranked-ids>: [...]} shape,
# so the parametrize body can assert the same two things about all four
# without caring about each tool's own argument list.
def _compare_wrapper(item_ids: list[str]) -> tuple[list[dict], list[str]]:
    result = compare_items(item_ids)
    return result["ineligible"], [r["item_id"] for r in result["ranking"]]


def _rank_by_priorities_wrapper(item_ids: list[str]) -> tuple[list[dict], list[str]]:
    result = rank_by_priorities(item_ids)
    return result["ineligible"], [r["item_id"] for r in result["default_ranking"]]


def _sensitivity_wrapper(item_ids: list[str]) -> tuple[list[dict], list[str]]:
    result = analyze_weight_sensitivity(item_ids, criterion="price_level", factor=1.5)
    return result["ineligible"], [c["item_id"] for c in result["changes"]]


def _robustness_wrapper(item_ids: list[str]) -> tuple[list[dict], list[str]]:
    result = weight_robustness_report(item_ids)
    return result["ineligible"], [r["item_id"] for r in result["baseline_ranking"]]


_RANKING_TOOL_WRAPPERS = [
    pytest.param(_compare_wrapper, id="compare_items"),
    pytest.param(_rank_by_priorities_wrapper, id="rank_by_priorities"),
    pytest.param(_sensitivity_wrapper, id="analyze_weight_sensitivity"),
    pytest.param(_robustness_wrapper, id="weight_robustness_report"),
]


@pytest.mark.parametrize("wrapper", _RANKING_TOOL_WRAPPERS)
def test_gate_enforced_in_all_four_ranking_tools(wrapper):
    """A NEIGHBORHOOD id, mixed in with one real locality, must be set
    aside with a reason in EVERY one of these four tools, and must NOT
    appear in that tool's own ranked-ids output. If a fifth ranking tool
    is ever added to app/tools.py and wired into this parametrize list
    without also calling _gate_ids, this test fails loudly for it,
    rather than silently passing because nobody remembered to write a
    bespoke gate test for the new tool."""
    ineligible, ranked_ids = wrapper([A_NEIGHBORHOOD_ID, KEFAR_SAVA])

    assert len(ineligible) == 1
    assert ineligible[0]["item_id"] == A_NEIGHBORHOOD_ID
    assert "NEIGHBORHOOD" in ineligible[0]["reason"]
    assert ineligible[0]["parent_locality_id"] == "28"

    assert A_NEIGHBORHOOD_ID not in ranked_ids
    assert KEFAR_SAVA in ranked_ids


@pytest.mark.parametrize("wrapper", _RANKING_TOOL_WRAPPERS)
def test_gate_with_only_a_neighborhood_id_ranks_nothing_but_still_explains_why(wrapper):
    """The degenerate case of the same gate: EVERY requested id is a
    neighborhood, so nothing is rankable at all. This must not raise or
    silently return an empty result with no explanation — 'ineligible'
    must still carry the reason, for all four tools."""
    ineligible, ranked_ids = wrapper([A_NEIGHBORHOOD_ID])
    assert ranked_ids == []
    assert len(ineligible) == 1
    assert ineligible[0]["item_id"] == A_NEIGHBORHOOD_ID


def test_gate_an_unknown_id_mixed_with_a_neighborhood_id_is_not_silently_absorbed():
    """_gate_ids' own docstring distinguishes 'a real thing that isn't
    rankable' (a neighborhood — comes back in `ineligible`) from 'not in
    the dataset at all' (an unknown id — must raise, via
    fetch_item_metrics, once the gate passes it through unchanged).
    Checked once against compare_items; the other three tools share the
    exact same _gate_ids + fetch_item_metrics call sequence, so this is
    not repeated per-tool the way the gate-itself test above is."""
    with pytest.raises(UnknownItemError):
        compare_items([A_NEIGHBORHOOD_ID, AN_UNKNOWN_ID])


def test_gate_does_not_touch_a_request_containing_only_real_localities():
    """The gate must be a no-op when nothing needs gating — every id
    passed through untouched, ineligible empty. Guards against an
    over-eager implementation that filters or reorders ids it had no
    reason to touch."""
    result = compare_items([KEFAR_SAVA, RAANANA, BEER_SHEVA])
    assert result["ineligible"] == []
    assert {r["item_id"] for r in result["ranking"]} == {KEFAR_SAVA, RAANANA, BEER_SHEVA}


# ─────────────────────────────────────────────────────────────────────────
# Every locality in the eligible set has a real, resolvable name — a
# cheap but real sanity check that dataset.py and tools.py agree on the
# item universe (see dataset.py's own module docstring on the
# LOCALITIES/NEIGHBORHOODS split).
# ─────────────────────────────────────────────────────────────────────────
def test_eligible_ids_are_all_real_localities_not_neighborhoods():
    for item_id in dataset.ELIGIBLE_IDS:
        assert item_id in dataset.LOCALITIES
        assert item_id not in dataset.NEIGHBORHOODS
