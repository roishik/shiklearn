"""
tools.py — the tools the agent can call, and this domain's actual KPIs.

The point of this file: the LLM never sees a KPI it has to compute. It
only ever sees numbers app/scoring.py already computed, PLUS the raw
inputs, so it can talk about them without ever being trusted to do the
arithmetic. Every tool below returns a per-component breakdown (raw
value, normalized score, weight, contribution) — never a bare number.

This is also the ONLY module that holds domain judgement. scoring.py has
the generic weighted ranker, analytics.py the generic filter/aggregate/
derived-metric contracts, runway_geometry.py the generic airfield
geometry — none of them know what an airport is worth. The criteria,
their weights and bounds, and the unmet-demand model all live here, so
"what did you decide, and why" has one address.

Data comes from app/dataset.py (static JSON rebuilt by
data/refresh_data.py from FAA, OurAirports, US Census and BTS — all
public, all keyless). The one genuinely live call is FAA NAS Status; it
is deliberately kept OUT of the scored path, because a ground stop today
says nothing about whether an airport is worth expanding over a decade.
"""
from __future__ import annotations

import urllib.error
import urllib.request
from typing import Any, Callable
from xml.etree import ElementTree

from app.analytics import (
    SUPPORTED_OPERATIONS,
    DerivedMetricResult,
    FactorInput,
    aggregate,
    build_derived_metric,
    filter_items,
    known_attribute_keys,
    known_attribute_values,
)
from app import dataset
from app.entity_resolution import resolve
from app.scoring import (
    PRIORITY_EMPHASIS_FACTOR,
    Criterion,
    apply_priority_emphasis,
    find_weight_flip_point,
    rank_items,
    scale_criterion_weight,
    sensitivity_analysis,
)

# ── This assignment's criteria ───────────────────────────────────────────
# The brief asks where renovation is "most profitable based on INCREASED
# flight and passenger capacity" — a headroom question, not a size
# question. Ranking on present-day size answers a different question and
# puts LAX first regardless of whether expanding LAX returns anything.
#
# So the two genuinely forward-looking signals carry 50% between them,
# and the two size-flavoured ones 30%, with absolute_scale alone held to
# 15%. Sanity check that this works: LAX ranks 67th of 144, not 1st.
#
# Bounds are the 5th/95th percentile of the ELIGIBLE set, measured
# 2026-08-18 (capacity_pressure's floor recomputed 2026-08-19 — it had
# drifted; see DECISIONS.md's final-review entry) and frozen as constants
# rather than recomputed per query —
# a ranking has to be reproducible, and bounds that shift with whatever
# subset was passed in would make two runs silently incomparable.
# Criterion.normalize clamps anything outside them, so an out-of-range
# value degrades gracefully instead of escaping [0,1].
#
# Every one of these is higher_is_better: more growth, more regional
# demand, more isolation from competitors, more pressure, more scale.
DEFAULT_CRITERIA: list[Criterion] = [
    # Is traffic already rising? FAA's own CY2024->CY2025 change.
    Criterion(name="traffic_growth", weight=25, lower_bound=-0.07884, upper_bound=0.13817),
    # Is the region it serves growing? Census county population CAGR,
    # 2022->2025. The only demand-side signal in the set, and the one
    # that genuinely decorrelates the ranking from airport size.
    Criterion(name="regional_demand_growth", weight=25, lower_bound=-0.00250, upper_bound=0.02411),
    # Can demand escape to another airport? Distance to the nearest
    # commercial-service competitor, in miles.
    Criterion(name="catchment_monopoly", weight=20, lower_bound=10.7, upper_bound=100.61),
    # Passengers per air-carrier runway. Correlates r=0.89 with
    # absolute_scale — kept because it is the only congestion proxy
    # available and the LA/Santa-Ana comparison needs one, but held to
    # 15% and disclosed rather than hidden. See DECISIONS.md [18:20].
    Criterion(name="capacity_pressure", weight=15, lower_bound=287264.425, upper_bound=7823094.2),
    # Size of the prize. Deliberately the joint-smallest weight.
    Criterion(name="absolute_scale", weight=15, lower_bound=564368.0, upper_bound=26519646.05),
]

# Plain-language gloss for each criterion, condensed from the comments
# above. Kept as a separate dict rather than a field on Criterion because
# scoring.py's docstring requires that dataclass shape to stay identical
# stable — this is domain text, not scoring logic.
CRITERION_DESCRIPTIONS: dict[str, str] = {
    "traffic_growth": "Year-over-year passenger traffic growth (FAA CY2024->CY2025 change).",
    "regional_demand_growth": (
        "Growth of the population and economic activity in the airport's home region "
        "(Census county population CAGR, 2022->2025) — the only demand-side signal, and "
        "the one that decorrelates the ranking from raw airport size."
    ),
    "catchment_monopoly": (
        "Distance to the nearest commercial-service competitor airport, in miles — how "
        "much local demand could otherwise escape to a rival airport."
    ),
    "capacity_pressure": (
        "Passengers per air-carrier runway, a congestion proxy. Correlates with "
        "absolute_scale (r=0.89), so held to a smaller weight on purpose."
    ),
    "absolute_scale": (
        "Current passenger volume — the size of the prize. Deliberately the "
        "joint-smallest weight: being large today is not evidence that expanding "
        "returns anything."
    ),
}


class UnknownItemError(KeyError):
    pass


def _unknown_airport(item_id: str) -> UnknownItemError:
    """One error message shape for every 'no such airport' case. Names a
    few real ids rather than dumping 515, and points at resolve_entity —
    the model's next move should be to resolve the name, not to guess
    another code."""
    return UnknownItemError(
        f"unknown airport id={item_id!r}. Ids are FAA LocIDs (usually the IATA code): "
        f"e.g. {', '.join(list(dataset.AIRPORTS)[:5])}. "
        "Call resolve_entity first to turn a name or city into an id."
    )


def fetch_item_metrics(item_id: str) -> dict[str, float]:
    """Raw criterion inputs for one airport, straight from the built
    dataset. Any criterion the airport has no data for is simply absent —
    scoring.py renormalizes each item's weights over what IS present, so
    a gap costs that airport the criterion, not the whole ranking."""
    if item_id not in dataset.METRICS:
        raise _unknown_airport(item_id)
    return dict(dataset.METRICS[item_id])


# ── OpenAI-style tool schemas (Anthropic provider converts these; see
#    app/providers/llm/anthropic_llm.py) ────────────────────────────────────
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "compare_items",
            "description": (
                "Fetch metrics for the given item ids and rank them using the "
                "deterministic scoring function. Returns, for every ranked "
                "item, its total score and a per-criterion breakdown (raw "
                "value, normalized score, weight, contribution), plus a "
                "separate 'excluded' list of items that had too little data "
                "to score fairly (see covered_weight/missing_criteria on "
                "each) — mention exclusions to the user rather than ignoring "
                "them. Always call this tool for any ranking or comparison "
                "question — never estimate or guess a score or ranking "
                "yourself. Airports below FAA primary-airport hub class "
                "(L/M/S) are automatically set aside and returned in "
                "'ineligible' — percentage growth on a tiny base is not an "
                "investment signal. When 'ineligible' is non-empty, tell the "
                "user which airports were set aside and why; do not present a "
                "silently shortened list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Airport ids to compare — FAA LocIDs, usually the IATA "
                            "code, e.g. ['LAX', 'SNA']. Use resolve_entity or "
                            "find_items to obtain these; do not type them from memory."
                        ),
                    },
                    "focus_criterion": {
                        "type": "string",
                        "enum": [
                            "traffic_growth",
                            "regional_demand_growth",
                            "catchment_monopoly",
                            "capacity_pressure",
                            "absolute_scale",
                        ],
                        "description": (
                            "OMIT THIS BY DEFAULT. Any question about which airports are "
                            "strong/best/promising candidates, about expansion, about "
                            "investment, or any general ranking must leave it unset and "
                            "use the full weighted score. Set it ONLY when the user "
                            "explicitly names a single measurable dimension and asks to "
                            "compare on that dimension. 'Compare their congestion levels' -> "
                            "capacity_pressure. 'Which is growing faster' -> "
                            "traffic_growth. 'Which is bigger' -> absolute_scale. "
                            "'Which region is growing' -> regional_demand_growth. "
                            "'Which has less competition nearby' -> catchment_monopoly. "
                            "The response then carries a 'focus' block ranking the "
                            "airports on that criterion alone, with the leader already "
                            "identified — report that block. Do NOT answer a "
                            "single-dimension question from total_score: it blends all "
                            "five criteria and means something different."
                        ),
                    },
                },
                "required": ["item_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_metrics",
            "description": (
                "Fetch raw metrics for ONE airport, with no scoring applied. "
                "Use this only for a factual question about a single airport. "
                "Do NOT call it twice to compare two airports — raw metrics "
                "are on wildly different scales (passengers vs. miles vs. "
                "percentages) and are not comparable as-is. Any question that "
                "compares, ranks, or asks which is more congested must go "
                "through compare_items, which normalizes them and returns "
                "weighted contributions."
            ),
            "parameters": {
                "type": "object",
                "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_criteria",
            "description": (
                "Return the scoring methodology itself: every criterion this agent "
                "ranks airports on, its weight, and what it measures. Call this for "
                "any meta-question about HOW the ranking works — 'what criteria/weights "
                "do you use', 'how is the score calculated', 'what goes into the "
                "ranking' — never answer that from memory or call it proprietary. "
                "This is distinct from compare_items, which needs specific airports to "
                "rank; list_criteria takes none because it describes the method, not a "
                "result. Takes no arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_entity",
            "description": (
                "Turn a user's free-text reference to an airport ('LA', "
                "'Santa Ana', 'Logan', a partial or misspelled name) into "
                "concrete airport ids. "
                "ALWAYS call this before compare_items or get_item_metrics "
                "when the user named something in words rather than giving an "
                "exact id — never guess or invent an id yourself. Returns "
                "candidates with a confidence and a per-signal breakdown, plus "
                "a 'decisive' flag. If decisive is false, you MUST NOT silently "
                "pick the top candidate: either ask the user which one they "
                "meant, or state plainly which one you assumed and why before "
                "continuing. If candidates is empty, say nothing matched — do "
                "not substitute a similar-sounding item."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's own words, e.g. 'Santa Ana' or 'the LA airport'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_items",
            "description": (
                "Find every item matching a set of attribute filters (all "
                "filters must match — AND, not OR). Use this whenever the user "
                "describes a GROUP rather than naming airports ('airports in "
                "New England', 'large hubs in the West'). NEVER list ids from "
                "memory to build such a group — call this. Returns the "
                "matching ids plus the attribute keys that actually exist, so "
                "you can tell the user when they asked about a field the "
                "dataset doesn't have."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "description": (
                            "Attribute name -> required value, e.g. "
                            "{'new_england': 'yes'} or {'region': 'West', "
                            "'hub_class': 'L'}. Available keys include: state "
                            "(2-letter), region (Census region — the ONLY "
                            "values are Northeast/Midwest/South/West/Other; "
                            "New England is a Census DIVISION, not a region, "
                            "so match it with the new_england key below, "
                            "never with region), new_england "
                            "(yes/no), hub_class (L/M/S/N/None), municipality, "
                            "weather_constrained (yes/no/unknown -- 'unknown' means the "
                            "runway geometry could not be measured, NOT that the "
                            "airport is unconstrained), ranking_eligible "
                            "(yes/no). Matching is case-insensitive."
                        ),
                        "additionalProperties": {"type": "string"},
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_records",
            "description": (
                "Compute a deterministic aggregate (share, mean, count, sum) "
                "over ONE item's sub-records, optionally restricted to a "
                "category. This is the tool for single-entity statistics like "
                "'what percentage of flights out of Anchorage are long haul?' — "
                "that is NOT a ranking question, so do not use compare_items "
                "for it. Departure-mix records exist for Anchorage (ANC) only; "
                "for any other airport this tool reports that no record-level "
                "data exists, which you must relay rather than estimating. "
                "The 'category' argument must be one of the dataset's OWN "
                "category values, not the user's phrasing: call with no "
                "category first to see 'known_categories' and "
                "'category_semantics', then call again with a real one. If "
                "'unknown_category' comes back true you asked for something "
                "that does not exist — that is NOT a zero result and you must "
                "never report it as 0%. Always relay 'category_semantics' when "
                "it says the figure is a proxy. "
                "'share' is computed on units/magnitude; the record counts are "
                "returned too, so if the user meant share-by-count you can give "
                "that instead. Never compute a percentage yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "operation": {
                        "type": "string",
                        "enum": list(SUPPORTED_OPERATIONS),
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional record category, e.g. 'international' or 'domestic'.",
                    },
                },
                "required": ["item_id", "operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_derived_metric",
            "description": (
                "Estimate a MODELLED quantity that exists in no dataset — "
                "currently 'unmet_demand' — and return it with the contributing "
                "factors, their magnitudes, the model's assumptions, a "
                "confidence level, and a caveat. Use this for 'what is the "
                "unmet demand at X, and why?'-shaped questions. The two factors "
                "are weather-suppressed throughput (parallel runways too close "
                "to fly independently in low visibility, so arrival capacity "
                "collapses) and structural capacity deficit (demand projected "
                "forward exceeds good-weather runway capacity). When explaining "
                "the 'why', use ONLY the returned factors and their magnitudes; "
                "never invent a cause. Always report the confidence and caveat "
                "— this is a model output, not an observation, and presenting "
                "it as a measured fact is wrong."
            ),
            "parameters": {
                "type": "object",
                "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_by_priorities",
            "description": (
                "Rank items with the weighting adjusted to priorities the user "
                "stated in their own words ('I care about being fast and "
                "cheap', 'quality matters most, budget is flexible'). Map their "
                "words onto criterion NAMES and pass those; you do not choose "
                "how much to reweight — the tool applies a fixed factor. "
                "Returns BOTH the default ranking and the adjusted one, plus an "
                "'assumption_to_state' string. You MUST tell the user which "
                "criteria you emphasized and that it reflects your reading of "
                "their words — never present a reweighted ranking as if it were "
                "the neutral one. If the user states priorities but names no "
                "items, call find_items or ask which items they mean first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_ids": {"type": "array", "items": {"type": "string"}},
                    "emphasize": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Criterion names the user cares MORE about, e.g. "
                            "['traffic_growth', 'regional_demand_growth']. Valid names: "
                            "traffic_growth, regional_demand_growth, catchment_monopoly, "
                            "capacity_pressure, absolute_scale."
                        ),
                    },
                    "deemphasize": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Criterion names the user cares LESS about.",
                    },
                },
                "required": ["item_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_weight_sensitivity",
            "description": (
                "Re-rank the given items with ONE criterion's weight scaled by "
                "a factor, and report what moved: whether the winner changed, "
                "how far each item shifted, and a Kendall tau rank-correlation "
                "(1.0 = order unchanged). Call this when the user asks why a "
                "weight was chosen, what happens if it's wrong, or how "
                "sensitive the ranking is. Report the result honestly even "
                "when it shows the ranking is fragile."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_ids": {"type": "array", "items": {"type": "string"}},
                    "criterion": {"type": "string", "description": "Which criterion's weight to perturb."},
                    "factor": {
                        "type": "number",
                        "description": "Multiplier on that weight; 0.5 halves it, 2.0 doubles it.",
                    },
                },
                "required": ["item_ids", "criterion", "factor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weight_robustness_report",
            "description": (
                "For every criterion, find the smallest weight multiplier that "
                "would change the top-ranked item. Use this to answer 'how "
                "confident are you in this ranking?' or 'which weight matters "
                "most?'. A flip factor near 1.0 means the result hangs on that "
                "weight and the top items should be described as close rather "
                "than as a clear winner — say so plainly when that's the case."
            ),
            "parameters": {
                "type": "object",
                "properties": {"item_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["item_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_airport_status",
            "description": (
                "Fetch CURRENT operational status for one airport from FAA's "
                "live NAS Status feed — ground stops, ground delay programs, "
                "closures, arrival/departure delays. Use this only when the "
                "user asks what is happening NOW. This is live operational "
                "colour and is NOT part of the investment score: never present "
                "a ground delay as evidence for or against expanding an "
                "airport, and say so if the user conflates them. If "
                "'available' is false the feed was unreachable — report the "
                "rest of the answer without it rather than guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
            },
        },
    },
]


# Two scores this close are the same score. Chosen from the real data
# rather than picked round: at the final weights the top two airports
# (Nashville 0.6333, Denver 0.6314) differ by 0.0019, and the winner
# flips on a 5% change to a single weight — so anything inside this band
# is a tie the tool has no business breaking silently.
#
# Same shape as entity_resolution's `decisive` flag, deliberately: a
# confidence floor plus a required GAP. One convention for "the data does
# not actually separate these," used in both places it comes up.
DECISIVE_SCORE_GAP = 0.005


def _tie_group(ranked: list[dict[str, Any]]) -> list[str]:
    """Ids at the top that are within DECISIVE_SCORE_GAP of first place.

    Returns [] when the leader is clear. Length >= 2 means "these are
    tied" — never "here is the winner" — and the system prompt requires
    the model to say so.
    """
    if len(ranked) < 2:
        return []
    top = ranked[0]["total_score"]
    tied = [r["item_id"] for r in ranked if top - r["total_score"] <= DECISIVE_SCORE_GAP]
    return tied if len(tied) > 1 else []


def _gate_ids(
    item_ids: list[str], *, include_ineligible: bool = False
) -> tuple[list[str], list[dict[str, Any]]]:
    """Split requested ids into (rankable, set-aside-with-a-reason).

    The FAA hub-class gate exists because percentage growth on a
    near-zero base is not an investment signal. The concrete failure:
    asked for New England candidates, New Bedford Regional — 3,145
    passengers a year, +52.7% — came back ranked 4th, ahead of airports
    a thousand times its size.

    This lives in one function called by EVERY ranking tool, and that is
    the whole point. It was originally written inline inside
    `compare_items`, on the argument that "a filter the caller has to
    remember is a filter that gets forgotten" — and then three sibling
    ranking tools were added that did not call it, and each one
    reproduced the exact failure the gate was written to prevent. The
    argument was right; it just had not been applied to itself.

    Set-aside airports are RETURNED with a reason, never silently
    dropped: the user asked about them and is owed an explanation for
    their absence.
    """
    if include_ineligible:
        return list(item_ids), []

    eligible_ids = set(dataset.ELIGIBLE_IDS)
    kept: list[str] = []
    ineligible: list[dict[str, Any]] = []
    for item_id in item_ids:
        # An unknown id passes through rather than being reported as
        # ineligible — "not in the dataset" is a different answer from
        # "too small to rank", and the downstream lookup raises a proper
        # unknown-airport error for it.
        if item_id in eligible_ids or item_id not in dataset.AIRPORTS:
            kept.append(item_id)
            continue
        airport = dataset.AIRPORTS[item_id]
        ineligible.append(
            {
                "item_id": item_id,
                "name": airport.get("name"),
                "faa_hub_class": airport.get("faa_hub_class"),
                "enplanements": airport.get("enplanements_cy2025_prelim"),
                "reason": (
                    "Below FAA primary-airport hub class (L/M/S). Percentage growth on a "
                    "base this small is not a meaningful investment signal, so it is "
                    "excluded from ranking rather than allowed to top it."
                ),
            }
        )
    return kept, ineligible


def _focus_on_criterion(
    criterion_name: str, ranking: list[dict[str, Any]], criteria: list[Criterion]
) -> dict[str, Any]:
    """Re-rank an already-computed ranking on ONE criterion.

    Pure presentation of numbers already computed — it re-reads the
    per-component breakdown `rank_items` produced and orders by that one
    component. No new arithmetic, nothing recomputed, so it cannot
    disagree with the main ranking.

    Ordering is by raw_value descending, not by normalized_score, and the
    distinction matters for a criterion where lower is better: "which is
    MORE congested" is a question about the raw quantity, not about which
    airport scores better on it. `higher_is_better` is returned alongside
    so the reader knows which direction is favourable for investment,
    which is a separate question from which value is larger.
    """
    match = next((c for c in criteria if c.name == criterion_name), None)
    if match is None:
        # A bad criterion name is the caller's error and must not be
        # silently ignored — an unnoticed typo here would mean the model
        # quietly falls back to describing the composite, which is the
        # exact failure this parameter exists to prevent.
        return {
            "criterion": criterion_name,
            "error": (
                f"Unknown criterion {criterion_name!r}. Valid criteria: "
                f"{', '.join(c.name for c in criteria)}. Call list_criteria for what each means."
            ),
        }

    rows = []
    for entry in ranking:
        component = next((c for c in entry["components"] if c["criterion"] == criterion_name), None)
        if component is None or component["raw_value"] is None:
            # Missing rather than zero. A dropped criterion is renormalized
            # away in the composite; here it has to be said out loud,
            # because "no data" and "the lowest value" are different
            # answers to "which is more congested".
            rows.append({"item_id": entry["item_id"], "raw_value": None, "normalized_score": None})
            continue
        rows.append(
            {
                "item_id": entry["item_id"],
                "raw_value": component["raw_value"],
                "normalized_score": component["normalized_score"],
            }
        )

    with_values = sorted(
        [r for r in rows if r["raw_value"] is not None], key=lambda r: r["raw_value"], reverse=True
    )
    for position, row in enumerate(with_values, start=1):
        row["rank_on_criterion"] = position
    missing = [r["item_id"] for r in rows if r["raw_value"] is None]

    return {
        "criterion": criterion_name,
        "description": CRITERION_DESCRIPTIONS.get(criterion_name, ""),
        "weight_in_total_score": match.weight,
        "higher_is_better_for_investment": match.higher_is_better,
        "ranked_by_this_criterion_alone": with_values,
        "highest": with_values[0]["item_id"] if with_values else None,
        "lowest": with_values[-1]["item_id"] if with_values else None,
        "no_data_for": missing,
        "note": (
            f"This ordering is on {criterion_name} alone. It is NOT the total expansion score, "
            f"which blends all {len(criteria)} criteria and answers a different question. Report "
            f"this ordering, and do not describe total_score as a measure of {criterion_name}."
        ),
    }


def compare_items(
    item_ids: list[str],
    criteria: list[Criterion] | None = None,
    coverage_threshold: float = 0.5,
    include_ineligible: bool = False,
    focus_criterion: str | None = None,
) -> dict[str, Any]:
    """Rank the given airports on the deterministic criteria.

    ELIGIBILITY IS ENFORCED HERE, not upstream, and that placement is the
    point. The gate (FAA hub class L/M/S) exists because percentage growth
    on a near-zero base is meaningless — but a filter the caller has to
    remember is a filter that gets forgotten. It was: asked for New
    England candidates, the model called find_items without the
    eligibility filter and New Bedford Regional (3,145 passengers, +53%
    "growth") came back ranked 4th, which is the precise failure the gate
    was introduced to prevent.

    So the ranking tool refuses to rank ineligible airports by default,
    regardless of how the ids were obtained. They are RETURNED in
    `ineligible` with a reason rather than silently dropped — the user
    asked about them and deserves to be told why they are not in the list.
    `include_ineligible=True` overrides it for a caller who genuinely
    wants the long tail.

    `focus_criterion` exists because of one specific, observed failure.
    Asked "compare LA and Santa Ana congestion levels" — a question about
    ONE dimension, not about which airport is the better investment — the
    model read the composite `total_score` and reported "LAX has the
    higher total score, therefore LAX is more congested." Both halves of
    that sentence are true and the conclusion is false: `total_score` is a
    weighted blend of five criteria, only one of which is the congestion
    proxy. Two rounds of system-prompt instruction did not stop it.

    The fix is the same principle the rest of this file runs on: if the
    model keeps deriving a wrong answer, stop asking it to derive one.
    With `focus_criterion` set, the tool returns a `focus` block that
    ranks the items on that criterion ALONE, already ordered, with the
    leader named. The model reports it instead of inferring it — the same
    reason every other tool here returns a per-component breakdown rather
    than a bare number.
    """
    criteria = criteria or DEFAULT_CRITERIA

    item_ids, ineligible = _gate_ids(item_ids, include_ineligible=include_ineligible)

    items = {item_id: fetch_item_metrics(item_id) for item_id in item_ids}
    result = rank_items(items, criteria, coverage_threshold=coverage_threshold)
    ranking = [
        {
            "rank": r.rank,
            "item_id": r.item_id,
            "total_score": round(r.total_score, 4),
            "covered_weight": round(r.covered_weight, 4),
            "missing_criteria": list(r.missing_criteria),
            "components": [
                {
                    "criterion": comp.criterion,
                    "raw_value": comp.raw_value,
                    "normalized_score": round(comp.normalized_score, 4),
                    "weight": round(comp.weight, 4),
                    "contribution": round(comp.contribution, 4),
                }
                for comp in r.components
            ],
        }
        for r in result.ranked
    ]
    tied = _tie_group(ranking)
    # `decisive` answers "is there a clear winner". With nothing ranked
    # there is no winner at all, and `not tied` would report true —
    # reachable on a real question ("compare these three small regionals"),
    # where every airport is correctly gated out and the payload then
    # asserts confidence in a list that does not exist.
    decisive = bool(ranking) and not tied
    focus = _focus_on_criterion(focus_criterion, ranking, criteria) if focus_criterion else None
    return {
        "criteria": [
            {"name": c.name, "weight": c.weight, "higher_is_better": c.higher_is_better}
            for c in criteria
        ],
        "coverage_threshold": coverage_threshold,
        # Non-empty means the top of this ranking is a statistical tie and
        # must be reported as one. The scores are real and the ordering is
        # deterministic, but the separation is smaller than the weighting
        # judgement that produced it, so calling #1 a winner would claim a
        # precision the method does not have.
        "tied_at_top": tied,
        "decisive": decisive,
        # An affirmative signal rather than an absent one: the model
        # should say "none of these could be ranked, here is why" instead
        # of reading an empty list as agreement.
        "no_items_ranked": not ranking,
        "tie_threshold": DECISIVE_SCORE_GAP,
        # Airports the caller asked about that are below FAA primary-airport
        # hub class. Returned, not silently dropped — tell the user these
        # were set aside and why, rather than showing a shorter list with no
        # explanation.
        "ineligible": ineligible,
        "ranking": ranking,
        # Items with SOME data but too little of it to score fairly, per
        # coverage_threshold — surfaced explicitly rather than silently dropped,
        # so the LLM can tell the user "N excluded, here's why" instead of
        # producing a ranking that quietly omits them with no explanation.
        "excluded": [
            {
                "item_id": e.item_id,
                "covered_weight": round(e.covered_weight, 4),
                "missing_criteria": list(e.missing_criteria),
                "reason": e.reason,
            }
            for e in result.excluded
        ],
        # Present only when the caller asked about one dimension. See the
        # docstring: this is the answer to "who is more congested",
        # computed here so the model never has to derive it from the
        # composite score, which does not mean that.
        "focus": focus,
    }


def get_item_metrics(item_id: str) -> dict[str, Any]:
    return {"item_id": item_id, "metrics": fetch_item_metrics(item_id)}


def list_criteria() -> dict[str, Any]:
    """The scoring methodology itself: every criterion this agent ranks
    on, its weight, and what it measures. Takes no item_ids because it
    isn't a ranking — it exists so a meta-question ('what are your
    criteria/weights', 'how is the score calculated') has a tool to route
    to, instead of the model either reaching for compare_items (which
    needs items to compare and won't answer a question with none) or
    answering from its own training-data reflex about scoring algorithms
    being proprietary. These weights are static config, not a computed
    score, so returning them is not a NEVER_COMPUTE_RULE violation."""
    total = sum(c.weight for c in DEFAULT_CRITERIA)
    return {
        "criteria": [
            {
                "name": c.name,
                "weight": c.weight,
                "weight_pct_of_total": round(100 * c.weight / total, 1),
                "higher_is_better": c.higher_is_better,
                "description": CRITERION_DESCRIPTIONS.get(c.name, ""),
            }
            for c in DEFAULT_CRITERIA
        ],
        "note": (
            "These are the default weights, applied unless the user's own stated "
            "priorities lead you to call rank_by_priorities instead — that "
            "reweights per-query, it does not change these defaults."
        ),
    }


def resolve_entity(query: str) -> dict[str, Any]:
    """Free-text name -> candidate item ids, with confidence and a
    decisive/ambiguous verdict.

    Zero matches is a normal outcome of a fuzzy search, not a caller
    mistake, so it comes back as an empty candidate list with
    decisive=false — NOT as an exception. That's the deliberate contrast
    with fetch_item_metrics's UnknownItemError, where being handed an id
    that doesn't exist really is a bug worth naming loudly.
    """
    # A metro name is not a failed airport match — it is a DIFFERENT kind
    # of answer, and collapsing it to the busiest airport would silently
    # decide something the user didn't. Checked first, and returned as an
    # explicitly non-decisive result so the model has to say which
    # reading it took.
    metro = dataset.resolve_metro(query)
    if metro is not None:
        name, ids = metro
        return {
            "query": query,
            "decisive": False,
            "match_type": "metro_area",
            "metro_name": name,
            "candidates": [
                {
                    "item_id": i,
                    "matched_text": name,
                    "confidence": 1.0,
                    "signals": {"metro_area_member": 1.0},
                }
                for i in ids
            ],
            "clarification_required": (
                f"{query!r} names a metropolitan area with {len(ids)} commercial airports "
                f"({', '.join(ids)}), not a single airport. State which reading you are using — "
                f"the primary airport ({ids[0]}) or the whole metro — before answering, or ask "
                "the user which they meant. Do not silently pick one."
            ),
        }

    result = resolve(query, dataset.ENTITY_CATALOG)
    return {
        "query": result.query,
        "decisive": result.decisive,
        "match_type": "airport",
        "candidates": [
            {
                "item_id": c.item_id,
                "matched_text": c.matched_text,
                "confidence": c.confidence,
                "signals": dict(c.signals),
            }
            for c in result.candidates
        ],
    }


def find_items(filters: dict[str, str]) -> dict[str, Any]:
    """Attribute filter -> matching ids. Also returns the attribute keys
    that actually exist, so an unknown field reads as "there is no such
    field" rather than as "nothing matched" — those are different
    answers and conflating them misleads the user.

    On an empty result it additionally returns the values each filtered
    key actually takes, because a known key with an unknown VALUE is a
    third distinct answer that used to be indistinguishable from the
    second. See the comment on the empty-result branch below."""
    matched = filter_items(dataset.ATTRIBUTES, filters)
    known_keys = known_attribute_keys(dataset.ATTRIBUTES)
    unknown_keys = [k for k in filters if k.casefold() not in known_keys]
    result = {
        "filters": dict(filters),
        "item_ids": list(matched),
        "match_count": len(matched),
        "known_attribute_keys": list(known_keys),
        # Non-empty means the caller filtered on a field that doesn't
        # exist, so item_ids is empty for a REASON — not because nothing
        # in the dataset qualifies.
        "unknown_filter_keys": unknown_keys,
    }

    # A known KEY carrying a value the dataset never uses matched nothing,
    # and "nothing matched" reads as "none exist". Found live on the
    # brief's own first question: the model called
    # {'region': 'New England'} — `region` is a real key, but it holds
    # Census REGIONS (Northeast/Midwest/South/West) and New England is a
    # Census DIVISION — got zero rows, and answered "there are no airports
    # in New England that match." There are 23, under {'new_england':
    # 'yes'}. Every number in that reply was correct and the reply was
    # false.
    #
    # Exactly the failure aggregate_records already guards for an unknown
    # category VALUE; this is that guard applied to a filter value. The
    # cure is the same: hand back the real value space and forbid the
    # confident negative, so the model can correct itself in one more turn
    # instead of confidently reporting an absence.
    if filters and not matched:
        result["known_values_for_filtered_keys"] = known_attribute_values(
            dataset.ATTRIBUTES,
            [k for k in filters if k.casefold() in known_keys],
        )
        result["guidance"] = (
            "Nothing matched every filter. This is NOT evidence that no such item "
            "exists — check your VALUES before answering. "
            "'known_values_for_filtered_keys' lists the values each key you "
            "filtered on actually takes. If a value you passed is not in that "
            "list, you filtered on something this dataset never stores (a common "
            "case: a place name that is a Census DIVISION, like 'New England', "
            "passed as 'region', which only holds Northeast/Midwest/South/West) — "
            "call this tool again with a real value, or with the key that does "
            "express what was asked; 'known_attribute_keys' has the full list. "
            "Only report that nothing exists after every value you used appears "
            "in the lists above."
        )
    return result


def aggregate_records(item_id: str, operation: str, category: str | None = None) -> dict[str, Any]:
    """Single-entity statistic over sub-records. Not a ranking, and
    deliberately a separate code path from scoring.py — a share is a
    count over one entity's own rows, with no weights and nothing to
    normalize."""
    if item_id not in dataset.RECORDS:
        if item_id in dataset.AIRPORTS:
            raise UnknownItemError(
                f"no record-level departure data for {item_id!r}. BTS publishes per-origin "
                f"segment summaries for {', '.join(dataset.RECORDS)} only in this dataset; the "
                "per-route microdata that would cover every airport is TranStats-only and "
                "bot-blocked. Say so rather than estimating a share."
            )
        raise _unknown_airport(item_id)

    records = dataset.RECORDS[item_id]
    known_categories = sorted({str(r["category"]) for r in records if "category" in r})

    # "No such category" and "a real category with zero rows" are DIFFERENT
    # answers, and conflating them is how a tool produces a confident
    # falsehood. Found by running the brief's own question: the model asked
    # for category "long haul" (which does not exist here — the categories
    # are domestic/international), matched nothing, and reported "0% of
    # flights out of Anchorage are long haul." Every number in that
    # sentence was correct and the sentence was false.
    #
    # Same guard find_items already had via unknown_filter_keys; this is
    # that idea applied to a category VALUE rather than a filter KEY.
    unknown_category = category is not None and category.casefold() not in {
        c.casefold() for c in known_categories
    }
    if unknown_category:
        return {
            "item_id": item_id,
            "operation": operation,
            "category": category,
            "value": None,
            "defined": False,
            "known_categories": known_categories,
            "unknown_category": True,
            "category_semantics": dataset.CATEGORY_SEMANTICS.get(item_id),
            "guidance": (
                f"{category!r} is not a category in this dataset — the available categories are "
                f"{known_categories}. This is NOT the same as a zero result: do not report "
                f"'0%' or 'none'. Read 'category_semantics' above: it says which available "
                "category is the right proxy for what was asked. Call this tool again with that "
                "category and answer the question — do not stop to ask the user which category "
                "they want when the semantics already say which one applies. Then state plainly "
                "that the figure is a proxy and what its limitation is."
            ),
        }

    # A share needs something to take a share OF. With no category the
    # arithmetic is matching_units / total_units over the SAME set, which
    # is 1.0 — a confident, fully "defined" 100%.
    #
    # This is not hypothetical: the schema tells the model to call this
    # tool with no category first, to discover what the categories are.
    # Asked "what percentage of flights out of Anchorage are long haul",
    # the model follows that instruction, receives value 1.0 with
    # defined:true and unknown_category:false, and has everything it needs
    # to answer "100%". Same failure class as the unknown-category guard
    # above, on the path the tool's own documentation routes through.
    if operation == "share" and category is None:
        return {
            "item_id": item_id,
            "operation": operation,
            "category": None,
            "value": None,
            "defined": False,
            "known_categories": known_categories,
            "unknown_category": False,
            "category_semantics": dataset.CATEGORY_SEMANTICS.get(item_id),
            "guidance": (
                "A share of everything is 100% by definition, so no value is returned here. "
                f"This call is for discovery: the categories are {known_categories}. Read "
                "'category_semantics' to see which one answers the question — including where "
                "it is a proxy rather than the exact measurement asked for — then call this "
                "tool again with that category."
            ),
        }

    result = aggregate(
        records,
        operation=operation,
        value_field="units",
        group_by_field="category" if category is not None else None,
        group_value=category,
    )
    value = result.value
    # NaN is not JSON-serializable in a way any consumer handles well, and
    # "undefined" is the honest word for mean-of-nothing or share-of-zero.
    is_defined = value == value
    return {
        "item_id": item_id,
        "operation": result.operation,
        "category": result.group_value,
        "value": round(value, 6) if is_defined else None,
        "defined": is_defined,
        "known_categories": known_categories,
        "unknown_category": False,
        # How to read these categories in the user's vocabulary, including
        # where they are a proxy rather than the requested measurement.
        "category_semantics": dataset.CATEGORY_SEMANTICS.get(item_id),
        # The arithmetic, exposed — so the model explains a number it
        # never computed, same contract as compare_items' components.
        "matching_records": result.matching_records,
        "total_records": result.total_records,
        "matching_units": result.matching_units,
        "total_units": result.total_units,
        # The other reading of "what percentage", so the model can offer
        # it if the user meant share-by-count rather than share-by-volume.
        "share_by_record_count": (
            round(result.matching_records / result.total_records, 6) if result.total_records else None
        ),
    }


# ── The DOMAIN model behind estimate_derived_metric ──────────────────────
# "Unmet demand at SFO, and why" is the question this exists for, and the
# "why" is the hard half. A correlation would not survive being asked
# twice; a mechanism does. So the model is built on a physical, published
# constraint rather than a fitted relationship.
#
# THE MECHANISM. SFO's two parallel arrival runways (28L/28R) sit about
# 750 ft apart — computed, not looked up: app/runway_geometry.py measures
# 746.8 ft from OurAirports' published runway-end coordinates, against a
# published figure of 750. FAA requires 2,500 ft for even dependent
# simultaneous approaches and 4,300 ft for independent ones, so when the
# marine layer drops the ceiling, SFO's two arrival streams collapse into
# ONE and its arrival rate roughly halves. The published schedule does
# not halve. That gap — demand that exists, was scheduled, and physically
# could not be flown — is the unmet demand, and no amount of terminal
# construction fixes it.
#
# The same computation runs on all 515 airports, which is what makes this
# a model rather than a fact about SFO with arithmetic wrapped round it.
# It independently recovers things nobody told it: Denver, deliberately
# built with runways 2,510+ ft apart, shows ZERO weather degradation;
# Seattle, whose 16C/16L are 738 ft apart, shows 0.67.

# Practical annual enplanements one arrival stream can sustain. NOT
# invented — it is the 95th percentile of enplanements-per-arrival-stream
# actually achieved across the 144 eligible airports (measured
# 2026-08-18), i.e. "what a very well-utilized comparable airport really
# does," not a theoretical throughput.
#
# The honest caveat: the true maximum observed is San Diego at 12.7M on a
# single runway — the busiest single-runway commercial airport in the US.
# Using p95 rather than that maximum is a deliberate conservatism, and it
# means this model UNDERSTATES how much traffic a constrained airport
# could theoretically absorb. Stated because the estimate moves roughly
# linearly with this constant, so it is the first number to push on.
PRACTICAL_CAPACITY_PER_ARRIVAL_STREAM = 7_800_000.0

# Fraction of the year an airport is in instrument meteorological
# conditions, i.e. ceiling/visibility low enough that the parallel-runway
# separation rules above start binding.
#
# THIS IS THE MODEL'S WEAKEST INPUT and it is stated first for that
# reason. It is a single national figure applied uniformly, when the real
# rate is intensely local — SFO's summer marine layer is a daily
# morning event, Phoenix is near-permanently clear. Doing this properly
# means historical METAR ceiling/visibility per airport (aviationweather.gov
# publishes it, keylessly) and is the single highest-value upgrade to
# this model. Not built here for time; see ASSUMPTIONS.md.
IMC_FRACTION = 0.12

# Utilization at or above which an airport is treated as
# capacity-constrained, meaning observed traffic is being suppressed BY
# the constraint rather than merely sitting below it.
CAPACITY_CONSTRAINED_UTILIZATION = 0.85


def estimate_unmet_demand(
    enplanements: float,
    arrival_streams_vmc: int,
    weather_capacity_degradation: float,
    traffic_growth: float,
    regional_demand_growth: float,
) -> DerivedMetricResult:
    """Model passenger demand that exists but cannot be flown.

    Unmet demand appears in no dataset by construction: nobody records the
    passenger who was never scheduled, or the flight that was never filed
    because the slot does not exist. It has to be modelled from observable
    proxies, with the assumptions travelling attached to the number.

    Two additive terms, each with its own mechanism:

      1. WEATHER-SUPPRESSED THROUGHPUT. When the ceiling drops, runways
         too close together to fly independently collapse into a single
         arrival stream (see runway_geometry.py for the FAA thresholds).
         Capacity falls; the schedule does not. During that fraction of
         the year, demand above the degraded capacity cannot be flown.
         This is SFO's entire story and it is why the answer to "why?"
         is geometric rather than commercial.

      2. STRUCTURAL CAPACITY DEFICIT. Demand projected one year forward
         at the airport's own traffic and regional-population growth,
         minus what its runways can sustain even in perfect weather.
         Zero for most airports; non-zero only where the airfield is
         genuinely undersized for its market.

    Self-gating, which is what stops it being arithmetic-for-its-own-sake:
    an airport with capacity to spare produces max(0, negative) = 0 on
    BOTH terms. A quiet airport losing half its arrival rate in fog has
    no unmet demand, correctly, because the remaining half still covers
    everything that wanted to fly.

    THE OBJECTION TO HAVE AN ANSWER FOR — "isn't that just a capacity
    ceiling, not demand?" They are the same phenomenon from two sides: the
    ceiling is the CAUSE, unmet demand the EFFECT measured through it.
    That is exactly why utilization gates confidence rather than the
    number: below the constrained threshold the same arithmetic yields a
    far weaker claim, and the caveat says so.
    """
    practical_capacity = arrival_streams_vmc * PRACTICAL_CAPACITY_PER_ARRIVAL_STREAM
    utilization = enplanements / practical_capacity if practical_capacity else float("nan")

    # Term 1. Capacity while degraded, and the demand that exceeds it,
    # annualized over the fraction of the year spent in those conditions.
    degraded_capacity = practical_capacity * (1.0 - weather_capacity_degradation)
    weather_suppressed = max(0.0, enplanements - degraded_capacity) * IMC_FRACTION

    # Term 2. Growth clamped at zero in both components: a shrinking
    # airport or a shrinking county is evidence of LESS pressure, and
    # letting it go negative would quietly credit decline as headroom.
    projected_demand = (
        enplanements * (1.0 + max(0.0, traffic_growth)) * (1.0 + max(0.0, regional_demand_growth))
    )
    structural_deficit = max(0.0, projected_demand - practical_capacity)

    contributions = [
        FactorInput(
            name="weather_suppressed_throughput",
            magnitude=weather_suppressed,
            source_field="min_parallel_separation_ft",
            explanation=(
                f"Arrival capacity falls {weather_capacity_degradation:.0%} in low visibility "
                f"because of parallel-runway separation, applied over the {IMC_FRACTION:.0%} of "
                "the year assumed to be instrument conditions. Demand above the degraded rate "
                "in those periods cannot be flown. A physical airfield constraint — terminal "
                "construction does not change it."
            ),
        ),
        FactorInput(
            name="structural_capacity_deficit",
            magnitude=structural_deficit,
            source_field="enplanements_cy2025_prelim",
            explanation=(
                f"Demand projected one year forward at this airport's own traffic growth "
                f"({traffic_growth:+.1%}) and county population growth "
                f"({regional_demand_growth:+.2%}), minus what {arrival_streams_vmc} arrival "
                "stream(s) can sustain in good weather. Zero unless the airfield is undersized "
                "for its market even before weather is considered."
            ),
        ),
    ]

    # NaN-safe: an unknown utilization must not read as "constrained".
    capacity_constrained = (
        utilization == utilization and utilization >= CAPACITY_CONSTRAINED_UTILIZATION
    )
    weather_exposed = weather_capacity_degradation > 0 and weather_suppressed > 0

    if capacity_constrained or weather_exposed:
        confidence = "medium"
        caveat = (
            f"Utilization is {utilization:.0%} of modelled good-weather capacity"
            + (
                f", and arrival capacity drops {weather_capacity_degradation:.0%} in low "
                "visibility because the parallel runways are too close together to fly "
                "independently. The runway geometry is the mechanism producing this number, "
                "not a competing explanation for it. "
                if weather_exposed
                else ". "
            )
            + "This is a LOWER BOUND: demand that was never scheduled because the constraint "
            "is well known to airlines is invisible here, and so is every passenger who drove "
            "to a competing airport instead."
        )
    else:
        confidence = "low"
        caveat = (
            f"Utilization is only {utilization:.0%} of modelled good-weather capacity, below "
            f"the {CAPACITY_CONSTRAINED_UTILIZATION:.0%} threshold where the airfield plausibly "
            "binds. Capacity is not this airport's constraint, so this figure should not be "
            "read as 'traffic we could win by expanding' — route economics, airline network "
            "decisions or catchment size are the more likely limits. Same arithmetic, much "
            "weaker claim."
        )

    return build_derived_metric(
        metric="unmet_demand",
        unit="annual enplanements",
        contributions=contributions,
        assumptions=(
            f"One arrival stream sustains {PRACTICAL_CAPACITY_PER_ARRIVAL_STREAM:,.0f} annual "
            "enplanements — the 95th percentile actually achieved across the 144 eligible "
            "airports, not a theoretical rate. The estimate moves roughly linearly with it.",
            f"Instrument conditions are assumed to occur {IMC_FRACTION:.0%} of the year, "
            "uniformly at every airport. This is the weakest input in the model: the real rate "
            "is intensely local (SFO's summer marine layer vs. Phoenix). Per-airport METAR "
            "history would fix it.",
            "Arrival streams are counted from runway geometry alone. Crossing-runway conflicts, "
            "wake-turbulence spacing, noise curfews and terrain-driven approach restrictions all "
            "reduce real capacity further, so good-weather capacity here is an UPPER bound — "
            "which makes the unmet-demand figure a lower bound.",
            "Demand is projected one year forward. Growth is clamped at zero, so a declining "
            "airport is never credited with negative unmet demand.",
        ),
        confidence=confidence,
        caveat=caveat,
    )


def estimate_derived_metric(item_id: str) -> dict[str, Any]:
    """A modelled quantity plus the factors that produced it. The number
    never travels without its assumptions, confidence, and caveat."""
    if item_id not in dataset.AIRPORTS:
        raise _unknown_airport(item_id)

    airport = dataset.AIRPORTS[item_id]

    # `or 0.0` cannot tell "absent" from "zero", and this model runs on
    # ragged public data where both occur. 14 of 515 airports have no
    # county population figure at all (Puerto Rico and the island
    # territories, where the Census county join has nothing to join to).
    # For those the tool used to state "county population growth
    # (+0.00%)" inside the factor explanation — the exact text the system
    # prompt tells the model to quote when explaining the "why" — with
    # nothing anywhere in the payload marking it as absent rather than
    # measured.
    #
    # scoring.py already solves this properly for the ranking path, with
    # missing_criteria, covered_weight, and renormalization over the
    # weight that is actually present. That discipline exists; this code
    # path simply did not participate in it. Tracking the gaps here is
    # that same rule applied where it was missing.
    missing_inputs: list[str] = []

    def numeric(field: str, label: str, default: float = 0.0) -> float:
        raw = airport.get(field)
        if raw is None or raw == "":
            missing_inputs.append(label)
            return default
        return float(raw)

    inputs = {
        "enplanements": numeric("enplanements_cy2025_prelim", "enplanements"),
        "arrival_streams_vmc": int(airport.get("arrival_streams_vmc") or 0),
        "arrival_streams_imc": int(airport.get("arrival_streams_imc") or 0),
        "weather_capacity_degradation": numeric(
            "weather_capacity_degradation", "weather_capacity_degradation"
        ),
        "min_parallel_separation_ft": airport.get("min_parallel_separation_ft"),
        "traffic_growth": numeric("yoy_pct_change_2025", "traffic_growth"),
        "regional_demand_growth": numeric(
            "county_population_cagr_recent", "regional_demand_growth"
        ),
    }
    if not inputs["arrival_streams_vmc"]:
        raise UnknownItemError(
            f"no usable runway geometry for {item_id!r} (no runway long enough for air-carrier "
            "arrivals, or missing coordinates), so arrival capacity cannot be modelled. "
            "The unmet-demand estimate is unavailable for this airport."
        )

    result = estimate_unmet_demand(
        enplanements=inputs["enplanements"],
        arrival_streams_vmc=inputs["arrival_streams_vmc"],
        weather_capacity_degradation=inputs["weather_capacity_degradation"],
        traffic_growth=inputs["traffic_growth"],
        regional_demand_growth=inputs["regional_demand_growth"],
    )
    # A modelled number built partly on absent inputs is not the same
    # number, and must not be presented as one. Confidence is forced
    # down, the caveat says which inputs were missing, and the payload
    # carries the list so the model cannot state the figure without also
    # having the reason to qualify it.
    confidence = result.confidence
    caveat = result.caveat
    if missing_inputs:
        confidence = "low"
        caveat = (
            f"No data for {', '.join(missing_inputs)} at this airport — treated as zero in the "
            "model, which understates projected demand. Report this estimate as a lower bound "
            "built on incomplete inputs, and name the missing ones. "
        ) + (caveat or "")

    return {
        "item_id": item_id,
        "metric": result.metric,
        "value": round(result.value, 4),
        "unit": result.unit,
        "confidence": confidence,
        "caveat": caveat,
        # Empty for a fully-measured airport. Non-empty means at least one
        # number in `observed_inputs` is a stand-in, not a measurement.
        "missing_inputs": missing_inputs,
        "assumptions": list(result.assumptions),
        "observed_inputs": dict(inputs),
        "factors": [
            {
                "name": f.name,
                "magnitude": round(f.magnitude, 4),
                "share_of_total": round(f.share_of_total, 4),
                "source_field": f.source_field,
                "explanation": f.explanation,
            }
            for f in result.factors
        ],
    }


def rank_by_priorities(
    item_ids: list[str], emphasize: list[str] | None = None, deemphasize: list[str] | None = None
) -> dict[str, Any]:
    """Rank items with the weights adjusted to the user's stated
    priorities ("I care about being fast and cheap").

    Returns BOTH the default ranking and the adjusted one, deliberately.
    Handing back only the adjusted list would let a reweighting change
    the answer invisibly — the user asked for their priorities to be
    honored, not for the default result to be quietly replaced. Showing
    both makes the effect of their own stated preference legible, and
    makes it obvious when the preference changed nothing.
    """
    emphasize = emphasize or []
    deemphasize = deemphasize or []
    item_ids, ineligible = _gate_ids(item_ids)
    items = {item_id: fetch_item_metrics(item_id) for item_id in item_ids}

    adjusted_criteria = apply_priority_emphasis(
        DEFAULT_CRITERIA, emphasize=emphasize, deemphasize=deemphasize
    )
    default_result = rank_items(items, DEFAULT_CRITERIA)
    adjusted_result = rank_items(items, adjusted_criteria)

    def as_rows(result: Any) -> list[dict[str, Any]]:
        return [
            {
                "rank": r.rank,
                "item_id": r.item_id,
                "total_score": round(r.total_score, 4),
                "components": [
                    {
                        "criterion": c.criterion,
                        "raw_value": c.raw_value,
                        "normalized_score": round(c.normalized_score, 4),
                        "weight": round(c.weight, 4),
                        "contribution": round(c.contribution, 4),
                    }
                    for c in r.components
                ],
            }
            for r in result.ranked
        ]

    default_rows = as_rows(default_result)
    adjusted_rows = as_rows(adjusted_result)
    default_top = default_result.ranked[0].item_id if default_result.ranked else None
    adjusted_top = adjusted_result.ranked[0].item_id if adjusted_result.ranked else None

    # Tie detection, for the same reason compare_items has it: two scores
    # inside DECISIVE_SCORE_GAP are the same score. Without this the tool
    # reported "your priorities changed the winner" off a 0.0020 gap —
    # announcing an effect on a difference this file elsewhere calls
    # noise. A reweighting only changed the winner if the new leader was
    # outside the old leader's tie band.
    default_tied = _tie_group(default_rows)
    adjusted_tied = _tie_group(adjusted_rows)
    changed_the_winner = bool(
        default_top and adjusted_top and adjusted_top != default_top
        and adjusted_top not in default_tied
    )

    return {
        "emphasized": emphasize,
        "deemphasized": deemphasize,
        "emphasis_factor": PRIORITY_EMPHASIS_FACTOR,
        "default_weights": {c.name: c.weight for c in DEFAULT_CRITERIA},
        "adjusted_weights": {c.name: c.weight for c in adjusted_criteria},
        "default_ranking": default_rows,
        "adjusted_ranking": adjusted_rows,
        # Same eligibility gate every other ranking tool applies, and
        # returned for the same reason — the user asked about these.
        "ineligible": ineligible,
        "default_tied_at_top": default_tied,
        "adjusted_tied_at_top": adjusted_tied,
        "priorities_changed_the_winner": changed_the_winner,
        "assumption_to_state": (
            f"Ranked with {', '.join(emphasize) or 'no criteria'} weighted "
            f"{PRIORITY_EMPHASIS_FACTOR}x higher"
            + (f" and {', '.join(deemphasize)} weighted lower" if deemphasize else "")
            + ", based on the priorities you described. Tell the user this — the "
            "weighting reflects an interpretation of their words, not a fact about the data."
        ),
    }


def analyze_weight_sensitivity(item_ids: list[str], criterion: str, factor: float) -> dict[str, Any]:
    """Re-rank with one criterion's weight scaled by `factor`, and report
    what actually moved. Answers "what happens if this weight is wrong?"
    with evidence instead of reassurance."""
    # A negative factor produces a NEGATIVE weight, which silently breaks
    # two invariants scoring.py annotates as guaranteed: total_score in
    # 0..1, and component weights summing to 1. factor=-5 yields
    # total_score=-2.68 with covered_weight=1.0 — a payload that reads as
    # fully valid. apply_priority_emphasis already guards this; the guard
    # simply was never applied to the other entry point.
    if factor <= 0:
        raise ValueError(
            f"factor must be greater than 0 (got {factor}). A zero or negative weight "
            "multiplier produces scores outside the 0..1 range the scoring contract "
            "guarantees. Use a value below 1 to reduce a weight and above 1 to raise it."
        )
    item_ids, ineligible = _gate_ids(item_ids)
    items = {item_id: fetch_item_metrics(item_id) for item_id in item_ids}
    scaled = scale_criterion_weight(DEFAULT_CRITERIA, criterion, factor)
    overrides = {c.name: c.weight for c in scaled if c.name == criterion}
    result = sensitivity_analysis(items, DEFAULT_CRITERIA, overrides)

    return {
        "criterion": criterion,
        "factor": factor,
        "ineligible": ineligible,
        "baseline_weights": dict(result.baseline_weights),
        "perturbed_weights": dict(result.perturbed_weights),
        "top_item": {
            "before": result.baseline_top,
            "after": result.perturbed_top,
            "changed": result.top_changed,
        },
        # Kendall tau: +1 means the order is untouched, lower means churn.
        # One number for "did this weight actually matter", which is more
        # honest than eyeballing two lists that look similar.
        "kendall_tau": round(result.kendall_tau, 4),
        "items_moved": result.items_moved,
        "max_rank_movement": result.max_rank_movement,
        # A weight change alters covered_weight too, so items can enter or
        # leave the ranking entirely — not just move within it.
        "items_entered_ranking": list(result.items_entered),
        "items_left_ranking": list(result.items_left),
        "changes": [
            {
                "item_id": c.item_id,
                "rank_before": c.baseline_rank,
                "rank_after": c.perturbed_rank,
                "rank_delta": c.rank_delta,
                "score_before": round(c.baseline_score, 4) if c.baseline_score is not None else None,
                "score_after": round(c.perturbed_score, 4) if c.perturbed_score is not None else None,
            }
            for c in result.changes
        ],
    }


def weight_robustness_report(item_ids: list[str]) -> dict[str, Any]:
    """For every criterion, the smallest weight multiplier that changes
    the winner. A criterion with a flip point close to 1.0 is one the
    result HANGS ON; one that never flips within the search range is not
    load-bearing and its exact weight barely matters.

    This is the tool for "how confident are you in this ranking?" — and
    it can legitimately return "not very", which is the point.
    """
    item_ids, ineligible = _gate_ids(item_ids)
    items = {item_id: fetch_item_metrics(item_id) for item_id in item_ids}
    baseline = rank_items(items, DEFAULT_CRITERIA)

    findings = []
    for criterion in DEFAULT_CRITERIA:
        flip = find_weight_flip_point(items, DEFAULT_CRITERIA, criterion.name)
        findings.append(
            {
                "criterion": criterion.name,
                "current_weight": criterion.weight,
                "flip_factor": flip,
                "interpretation": (
                    "No weight multiplier up to 10x changes the winner — this criterion's "
                    "exact weight is not load-bearing."
                    if flip is None
                    else f"Scaling this weight by {flip}x changes the top-ranked item. "
                    + (
                        "That is a small change, so the ranking is SENSITIVE to this weight "
                        "and the top result should be presented as close, not decisive."
                        if abs(flip - 1.0) <= 0.5
                        else "That is a large change, so the ranking is robust to this weight."
                    )
                ),
            }
        )

    tightest = min(
        (f for f in findings if f["flip_factor"] is not None),
        key=lambda f: abs(float(f["flip_factor"]) - 1.0),  # type: ignore[arg-type]
        default=None,
    )

    return {
        "baseline_top": baseline.ranked[0].item_id if baseline.ranked else None,
        # Same gate as every other ranking tool. Without it this report
        # computed flip factors over a population compare_items would
        # never rank, so the confidence evidence described a ranking the
        # user was never shown.
        "ineligible": ineligible,
        "baseline_ranking": [
            {"rank": r.rank, "item_id": r.item_id, "total_score": round(r.total_score, 4)}
            for r in baseline.ranked
        ],
        "criteria": findings,
        "most_sensitive_criterion": tightest["criterion"] if tightest else None,
        "summary": (
            "No single weight change up to 10x alters the winner; the ranking is robust."
            if tightest is None
            else f"The result is most sensitive to {tightest['criterion']!r}, which flips the "
            f"winner at {tightest['flip_factor']}x its current weight."
        ),
    }


# ── The one genuinely live call ──────────────────────────────────────────
# FAA NAS Status: free, no key, no signup, real-time. It satisfies the
# brief's "use public APIs" requirement with an actual live request
# rather than a static file that was fetched once.
#
# DELIBERATELY OUTSIDE THE SCORED PATH, and that is the interesting
# decision. A ground stop at SNA this afternoon says exactly nothing
# about whether SNA is worth a terminal investment over the next decade —
# it is weather, or a runway closure, or an equipment outage. Feeding
# transient operational status into a capital-planning score would be
# indefensible in about ten seconds of questioning. So it is presented as
# live operational colour alongside the ranking, never as an input to it.
#
# Note it returns XML, not JSON, which is why this parses rather than
# json.loads. Kept in the standard library — adding a dependency to read
# one small document is not worth it.
NAS_STATUS_URL = "https://nasstatus.faa.gov/api/airport-status-information"
NAS_STATUS_TIMEOUT_SECONDS = 6.0


def get_live_airport_status(item_id: str) -> dict[str, Any]:
    """Current FAA operational status: ground stops, ground delay
    programs, closures, arrival/departure delays.

    Every failure mode here returns `{"available": false, "reason": ...}`
    rather than raising, because this tool is decoration on an answer that
    must still work without it. A timeout on a live feed should degrade
    the response, not fail the question — the agent loop would otherwise
    surface a tool error for something genuinely optional.
    """
    if item_id not in dataset.AIRPORTS:
        raise _unknown_airport(item_id)

    try:
        with urllib.request.urlopen(NAS_STATUS_URL, timeout=NAS_STATUS_TIMEOUT_SECONDS) as resp:
            root = ElementTree.fromstring(resp.read())
    except (urllib.error.URLError, ElementTree.ParseError, OSError) as exc:
        return {
            "item_id": item_id,
            "available": False,
            "reason": f"FAA NAS Status unreachable ({type(exc).__name__}). "
                      "Report the ranking without live status rather than guessing it.",
            "source": NAS_STATUS_URL,
        }

    # ElementTree has no parent pointers, so build the map once. It is
    # needed because the CATEGORY of an event lives on an ancestor
    # (<Delay_type><Name>Airport Closures</Name>), not on the node
    # carrying the airport code.
    parents = {child: parent for parent in root.iter() for child in parent}

    def category_of(node: ElementTree.Element) -> str:
        """Nearest labelled ancestor — the FAA's own name for this kind of
        event."""
        current = parents.get(node)
        while current is not None:
            name = (current.findtext("Name") or "").strip()
            if name:
                return name
            current = parents.get(current)
        return "Unclassified"

    events: list[dict[str, Any]] = []
    # The previous version flattened the tree with root.iter() and reported
    # node.tag as the type, so every event came back as "Delay" or
    # "Airport" and the feed's own labels were unreachable. It also kept
    # only <Reason>, discarding <Min>/<Max>/<Trend> on delays and
    # <Start>/<Reopen> on closures.
    #
    # That combination produced confidently wrong statements. A real
    # example: LAX carries a standing NOTAM reading "LAX AD AP CLSD TO NON
    # SKED TRANSIENT GA ACFT EXC 24HR PPR", valid for a year. Stripped of
    # its category and its dates it reads as "LAX AD AP CLSD", and the
    # agent reported LAX as closed. It is open; it is restricted to
    # scheduled traffic without prior permission. Meanwhile JFK, with a
    # genuine 16-30 minute increasing departure delay, returned the least
    # informative payload of the three, because all of its numbers were in
    # exactly the sibling elements being dropped.
    for node in root.iter():
        code = ""
        for tag in ("ARPT", "IATA", "Airport"):
            child = node.find(tag)
            if child is not None and child.text and child.text.strip():
                code = child.text.strip()
                break
        if code != item_id:
            continue
        # Keep the whole matched subtree. Which fields are present depends
        # on the event kind, and guessing wrong is how the delay
        # magnitudes were lost the first time.
        detail: dict[str, Any] = {}
        for child in node.iter():
            if child is node:
                continue
            has_text = bool((child.text or "").strip())
            if not has_text and not child.attrib:
                # A pure grouping element (no text of its own, no
                # attributes) — e.g. the wrapper around Min/Max/Trend.
                # Its children are still walked, since node.iter() does
                # not stop at it; there is just nothing to record here.
                continue
            key = child.tag
            if child.attrib:
                # <Arrival_Departure Type="Departure"> carries no text of
                # its own — its meaning is entirely in the attribute — and
                # is exactly the case a text-only check drops silently.
                # It's the qualifier that says whether "16 minutes" is
                # arrivals or departures, which is most of the meaning.
                key = f"{key}[{','.join(child.attrib.values())}]"
            detail[key] = child.text.strip() if has_text else True
        events.append({"category": category_of(node), "element": node.tag, "detail": detail})

    return {
        "item_id": item_id,
        "available": True,
        "has_active_events": bool(events),
        "events": events,
        "source": NAS_STATUS_URL,
        # Stated in the payload, not just in a comment, because the model
        # is what has to relay it to the user.
        "scope_note": (
            "Live operational status only. This is NOT part of the investment score and must "
            "not be described as if it were — a ground delay today is weather or an equipment "
            "outage, not evidence about whether this airport is worth expanding."
        ),
        # The feed mixes live delay programs with standing NOTAMs that can
        # run for a year, and a NOTAM's text often begins with something
        # that reads like "AP CLSD". Report what the fields say, including
        # the dates, rather than summarizing an event as a closure.
        "reading_note": (
            "Each event carries its FAA category and its own fields. Do not paraphrase an "
            "event as 'the airport is closed' unless a field says so: many entries are "
            "standing NOTAMs restricting a class of traffic, with validity dates in the "
            "text, not closures happening now. Quote the category and the relevant fields."
        ),
    }


# Dispatch table used by agent_loop.py: tool name -> callable(args_dict).
# Kept as a plain dict, not a decorator/registry framework — this is the
# entire "tool registry" a hand-rolled loop needs.
def _find_items_filters(args: dict[str, Any]) -> dict[str, str]:
    """Normalize find_items' arguments. Three shapes arrive in practice and
    they do NOT mean the same thing:

        {}                       -> match everything. Legitimate and
                                    documented; a model asking for the whole
                                    dataset. args.get (not args[...]) because
                                    a raw KeyError('filters') here was logged
                                    as a tool error in P4 evals instead of the
                                    empty-filter result clearly intended.
        {"filters": {...}}       -> the schema's shape.
        {"new_england": "yes"}   -> the model dropped the wrapper.

    The third used to collapse into the first: args.get("filters") returned
    None, "no filters" means "match everything", and a request for a SUBSET
    silently returned all 515 rows. The model then listed New England
    airports from its own memory — the exact hallucination find_items exists
    to prevent — wearing a successful tool call as cover. Found live on the
    brief's own first question.

    So a flattened call is read as filters rather than as "no filters".
    Nothing is guessed: an unrecognized key still lands in
    `unknown_filter_keys` downstream, which already tells the model it
    filtered on a field that does not exist. Only scalars are taken; a
    stray list or object is another tool's argument, not a filter value.
    """
    if "filters" in args:
        return args.get("filters") or {}
    return {k: v for k, v in args.items() if isinstance(v, (str, int, float, bool))}


TOOL_REGISTRY: dict[str, Callable[[dict[str, Any]], Any]] = {
    "get_live_airport_status": lambda args: get_live_airport_status(item_id=args["item_id"]),
    "compare_items": lambda args: compare_items(
        item_ids=args["item_ids"], focus_criterion=args.get("focus_criterion")
    ),
    "get_item_metrics": lambda args: get_item_metrics(item_id=args["item_id"]),
    "list_criteria": lambda _: list_criteria(),
    "resolve_entity": lambda args: resolve_entity(query=args["query"]),
    "find_items": lambda args: find_items(filters=_find_items_filters(args)),
    "aggregate_records": lambda args: aggregate_records(
        item_id=args["item_id"], operation=args["operation"], category=args.get("category")
    ),
    "estimate_derived_metric": lambda args: estimate_derived_metric(item_id=args["item_id"]),
    "analyze_weight_sensitivity": lambda args: analyze_weight_sensitivity(
        item_ids=args["item_ids"], criterion=args["criterion"], factor=args["factor"]
    ),
    "weight_robustness_report": lambda args: weight_robustness_report(item_ids=args["item_ids"]),
    "rank_by_priorities": lambda args: rank_by_priorities(
        item_ids=args["item_ids"],
        emphasize=args.get("emphasize"),
        deemphasize=args.get("deemphasize"),
    ),
}
