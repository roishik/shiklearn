"""
tools.py — the tools the agent can call, and this domain's actual KPIs.

The point of this file: the LLM never sees a KPI it has to compute. It
only ever sees numbers app/scoring.py already computed, PLUS the raw
inputs, so it can talk about them without ever being trusted to do the
arithmetic. Every tool below returns a per-component breakdown (raw
value, normalized score, weight, contribution) — never a bare number.

This is also the ONLY module that holds domain judgement. scoring.py has
the generic weighted ranker, analytics.py the generic filter/aggregate/
derived-metric contracts, affordability.py the generic mortgage math —
none of them know what a "good value" Israeli locality is. The criteria,
their weights and bounds, the eligibility gate, and the required-income
model all live here, so "what did you decide, and why" has one address.

Data comes from app/dataset.py (static JSON rebuilt by
data/refresh_data.py from nadlan.gov.il and CBS — all public, all
keyless). The one genuinely live call is the Bank of Israel's known
interest rate; it is deliberately kept OUT of the scored path (see
system_prompt.py rule 7 and get_current_mortgage_rates' own docstring) —
a rate change moves what every buyer can afford everywhere at once, so
it says nothing about whether one locality is better VALUE than another.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from app import dataset
from app.affordability import (
    DEFAULT_MORTGAGE_TERM_YEARS,
    DEFAULT_REPAYMENT_CAP,
    FIRST_HOME_MAX_LTV,
    monthly_payment,
    years_of_income_to_buy,
)
from app.analytics import (
    DerivedMetricResult,
    FactorInput,
    aggregate,
    build_derived_metric,
    filter_items,
    known_attribute_keys,
    known_attribute_values,
)
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

# ─────────────────────────────────────────────────────────────────────────
# This assignment's criteria: DEFAULT_CRITERIA
# ─────────────────────────────────────────────────────────────────────────
# Default score meaning (locked with Roi, see _working/PROGRESS.md): "value
# for money to live in" — affordability weighed against quality of place.
# It is NOT an investment-return score and NOT a price forecast.
#
# BOUNDS. Each lower_bound/upper_bound is the 5th/95th percentile of the
# raw field, measured on the 104 ELIGIBLE localities (linear interpolation,
# i.e. the "R type 7" method numpy.percentile/statistics use by default),
# on 2026-08-20. Frozen here as literal constants rather than recomputed
# per query — a ranking has to be reproducible, and bounds that shift with
# whatever subset of ids happened to be passed to compare_items would make
# two runs of the same question silently incomparable (today's top-5 could
# stop meaning the same thing as last week's top-5 for no reason connected
# to the places themselves). Criterion.normalize clamps anything outside
# [lower_bound, upper_bound], so a value outside the measured range (a new
# data refresh, or a locality this specific query didn't include in the
# percentile calculation) degrades gracefully to 0 or 1 instead of
# escaping [0, 1] or crashing.
#
# WEIGHTS. The design intent (see _working/PROGRESS.md's original table)
# was price_level=30, socioeconomic_level=20, accessibility=20,
# price_momentum_vs_country=15, rental_yield=15. The FINAL weights below
# differ from that intent, and they differ because the measured pairwise
# Pearson correlation matrix on the eligible 104 said to change them, not
# because of a change of mind:
#
#                     price_level  socioecon  accessibility  momentum  rental_yield
#     price_level         1.000      0.666        -0.371     -0.003       -0.391
#     socioecon            0.666      1.000       -0.168     -0.097       -0.158
#     accessibility       -0.371     -0.168        1.000      0.020        0.405
#     momentum            -0.003     -0.097        0.020      1.000       -0.195
#     rental_yield        -0.391     -0.158         0.405     -0.195        1.000
#
#   (pairwise Pearson r, eligible 104, pairwise-complete n where a field
#   has missing values — see FROZEN BOUNDS table below for each n. Full
#   measurement recorded in _working/agent-logs/tools-surface.md.)
#
#   - socioeconomic_level: 20 -> 15. It correlates r=0.666 with
#     price_level — by far the strongest pair in the matrix. Expensive
#     places have richer residents; at the original 20+30=50, HALF the
#     total weight leaned on two criteria that substantially overlap, so
#     the score would have partly measured "expensive" twice under two
#     different names. Held to 15 and DISCLOSED here, not dropped to
#     zero — the residual (uncorrelated) variance is exactly where the
#     interesting answers live: somewhere cheap that is ALSO a good
#     place to live, which is a real and findable thing in this data
#     (see the actual top-10 this produces — none of them are Tel Aviv).
#   - price_momentum_vs_country: 15 -> 20. It is the ONLY criterion in
#     the set with |r| <= 0.10 against every other criterion — genuinely
#     decorrelated, not just "less correlated". The weight moved from the
#     criterion that duplicates price_level (socioeconomic_level) to the
#     one that adds independent information the other four don't carry.
#   - rental_yield: stays at 15. It DOES partly re-express price_level
#     (r=-0.391, moderate) — cheaper places tend to have higher yields,
#     mechanically, since yield is rent/price. It earns its place as an
#     opportunity-cost check (is the price supported by what the home is
#     actually worth to live in or rent out, not just speculation), but
#     it is not raised above 15, because doing so would push in the same
#     direction price_level already pushes, for the same underlying
#     reason socioeconomic_level was cut.
#   - accessibility: unchanged at 20. Its strongest correlation is with
#     rental_yield (r=0.405, moderate — closer-to-core places tend to
#     have lower yields, plausibly because their price already prices in
#     the location premium) and a moderate r=-0.371 with price_level. Not
#     as extreme as the price_level/socioeconomic_level pair, so left as
#     originally weighted.
#   - price_level: unchanged at 30. It stays the largest single weight on
#     purpose — for most buyers "can I actually afford it" dominates
#     "how much better is it than the alternative" — and the correlation
#     evidence argues for CUTTING the criteria that duplicate it, not for
#     cutting price_level itself.
#
# Every criterion is listed with its true statistical direction
# (higher_is_better), not forced positive the way a "bigger number always
# wins" convention would — Criterion.normalize already handles a
# lower-is-better criterion correctly (see scoring.py), so there is no
# reason to invert the sign here and lose the literal meaning of the raw
# value.
DEFAULT_CRITERIA: list[Criterion] = [
    # Median 4-room price vs. the market: nadlan.gov.il settlement-level
    # median. LOWER is better — this is the affordability half of "value".
    Criterion(
        name="price_level",
        weight=30,
        lower_bound=757500.0,
        upper_bound=3467262.5,
        higher_is_better=False,
    ),
    # Straight-line (haversine) distance to whichever of Tel Aviv /
    # Jerusalem / Haifa is nearest — a proxy for access to Israel's three
    # biggest employment cores, computed in data/refresh_data.py, not a
    # measured commute time. LOWER is better.
    Criterion(
        name="accessibility",
        weight=20,
        lower_bound=4.617,
        upper_bound=69.7465,
        higher_is_better=False,
    ),
    # Local 5-year price CAGR minus the NATIONAL 5-year price CAGR — is
    # this place gaining or losing ground on the market as a whole, not
    # just "are prices rising" (almost everywhere's are). HIGHER is
    # better, and this is the one criterion that is genuinely
    # decorrelated from all four others (see the matrix above).
    Criterion(
        name="price_momentum_vs_country",
        weight=20,
        lower_bound=-0.038353,
        upper_bound=0.053857,
        higher_is_better=True,
    ),
    # CBS Socio-Economic Index cluster, 1 (lowest) to 10 (highest) — a
    # broad index blending income, education and employment. HIGHER is
    # better. Cut from the original 20 to 15 (see the correlation
    # discussion above) because of its r=0.666 overlap with price_level.
    Criterion(
        name="socioeconomic_level",
        weight=15,
        lower_bound=2.0,
        upper_bound=9.0,
        higher_is_better=True,
    ),
    # Annual rent as a percentage of purchase price (nadlan.gov.il's own
    # yield index). HIGHER is better — the opportunity-cost check: is the
    # price supported by what the home is worth to actually live in or
    # rent out, not just appreciation speculation.
    Criterion(
        name="rental_yield",
        weight=15,
        lower_bound=2.1325,
        upper_bound=3.776,
        higher_is_better=True,
    ),
]

# Plain-language gloss for each criterion, in words a first-time buyer
# could follow with no statistics background — condensed from the design
# comments above, but written for the tool's CALLER (the LLM, and through
# it the user), not for a future maintainer reading the source. Kept as a
# separate dict rather than a field on Criterion, same reasoning as the
# airport-domain build this replaces: scoring.py's dataclass shape must
# stay domain-free, and this is domain text, not scoring logic.
CRITERION_DESCRIPTIONS: dict[str, str] = {
    "price_level": (
        "How expensive a typical 4-room home is here, compared with every other eligible "
        "place in Israel. Cheaper scores better — this is the single biggest factor in "
        "'value for money' here, and it carries the largest weight (30) for that reason."
    ),
    "accessibility": (
        "Straight-line distance to whichever of Tel Aviv, Jerusalem, or Haifa is closer — "
        "a stand-in for how far this place is from Israel's three biggest job markets. "
        "Closer scores better. This is a straight-line distance, not a measured commute "
        "time, so it can understate the real commute where roads or rail don't run direct."
    ),
    "price_momentum_vs_country": (
        "Whether prices here have grown faster or slower than the NATIONAL average over "
        "the last 5 years, not just whether they went up (almost everywhere's did). "
        "Growing faster than the country scores better. This is the one factor in the "
        "ranking that has almost nothing to do with the other four, so it's weighted at "
        "20 to make sure it actually moves the answer rather than being drowned out."
    ),
    "socioeconomic_level": (
        "The CBS government's own socioeconomic ranking of the area, from 1 (lowest) to "
        "10 (highest) — a broad measure of local income, education, and employment. "
        "Higher scores better. It's held to a lower weight (15, not 20) than you might "
        "expect, because expensive places and high-ranked places are largely the SAME "
        "places here — weighting both heavily would count that fact twice."
    ),
    "rental_yield": (
        "Annual rent as a percentage of the purchase price. Higher scores better — a "
        "high yield means the price is backed by what the home is actually worth to live "
        "in or rent out today, rather than a bet that it will simply keep getting more "
        "expensive."
    ),
}


class UnknownItemError(KeyError):
    pass


def _unknown_item(item_id: str) -> UnknownItemError:
    """One error message shape for every 'no such locality' case. Names a
    few real ids rather than dumping all 104, and points at
    resolve_entity — the model's next move should be to resolve the name,
    never to guess another id or reconstruct one from memory (see
    system_prompt.py's NEVER_INVENT_IDS_RULE)."""
    sample = ", ".join(list(dataset.LOCALITIES)[:5])
    return UnknownItemError(
        f"unknown locality id={item_id!r}. Ids are CBS locality codes, e.g. {sample}. "
        "Call resolve_entity first to turn a name (Hebrew, English, or a transliteration) "
        "into an id — never guess or invent one."
    )


def fetch_item_metrics(item_id: str) -> dict[str, float]:
    """Raw criterion inputs for one locality, straight from the built
    dataset. Any criterion the locality has no data for is simply absent
    — scoring.py renormalizes each item's weights over what IS present,
    so a gap (e.g. socioeconomic_cluster: 102/104, rental_yield: 98/104 —
    see LOCALITIES_META['join_coverage']) costs that locality the one
    criterion, not the whole ranking."""
    if item_id not in dataset.METRICS:
        if item_id in dataset.NEIGHBORHOODS:
            raise _neighborhood_not_rankable(item_id)
        raise _unknown_item(item_id)
    return dict(dataset.METRICS[item_id])


def _neighborhood_not_rankable(item_id: str) -> UnknownItemError:
    """A NEIGHBORHOOD id was handed to something that expects a
    LOCALITY. See _gate_ids' docstring for why this is a distinct error
    from 'unknown id' — a neighborhood is a real, known thing, just not
    the kind of thing this tool ranks."""
    # Every row in dataset.NEIGHBORHOODS carries parent_id by construction
    # of data/processed_data/neighborhoods.json (verified: 0 of 1,394 rows
    # missing it). Asserting rather than silently formatting a "parent
    # locality is None" message into a user-facing error keeps that
    # invariant honest instead of letting a data-pipeline regression show
    # up as a confusing string three calls away from its actual cause.
    parent_id = dataset.NEIGHBORHOODS[item_id].get("parent_id")
    assert parent_id is not None, f"data invariant violated: neighborhood {item_id!r} has no parent_id"
    parent_name = dataset.LOCALITIES.get(parent_id, {}).get("name_en", parent_id)
    return UnknownItemError(
        f"{item_id!r} is a NEIGHBORHOOD, not a locality — nadlan.gov.il and CBS both publish "
        "socioeconomic/accessibility/income figures at the LOCALITY level only, so a "
        "neighborhood has no data of its own to rank or fetch metrics for. Its parent "
        f"locality is {parent_id!r} ({parent_name}). To ask about this neighborhood, use "
        f"aggregate_records({parent_id!r}, ...) — it answers questions about how a locality's "
        "neighborhoods compare to each other, which is the shape of question a neighborhood "
        "id actually supports here."
    )


# ── OpenAI-style tool schemas (Anthropic provider converts these; see
#    app/providers/llm/anthropic_llm.py) ────────────────────────────────────
_CRITERION_NAMES: list[str] = [c.name for c in DEFAULT_CRITERIA]

# app/analytics.py's aggregate() supports share/mean/count/sum, because it is a
# general-purpose module and those are the four aggregates any record set might
# want. This tool deliberately offers only TWO of them.
#
# The reason is the shape of THIS domain's records: one record is one
# neighborhood, carrying a constant units=1.0 (see app/dataset.py) so that
# "share" comes out as a share BY COUNT of neighborhoods -- which is what "what
# share of Tel Aviv's neighborhoods are above the city median" actually asks.
# That constant makes the other two degenerate: `mean` can only ever return
# 1.0, and `sum` is always exactly `count`.
#
# Both would still "work" -- nothing crashes, nothing lies. But a tool schema is
# a promise to the model about what it can usefully ask for, and advertising an
# operation whose answer is always 1.0 invites the model to call it and then
# explain a meaningless number to a user. Narrowing the enum is cheaper and
# more honest than documenting a trap.
#
# Summing or averaging area medians would be the wrong statistic anyway even if
# units carried price: the mean of 41 neighborhood medians is not the median of
# the city, and their sum means nothing at all.
AGGREGATE_OPERATIONS_OFFERED: tuple[str, ...] = ("share", "count")


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "compare_items",
            "description": (
                "Fetch metrics for the given locality ids and rank them using the "
                "deterministic scoring function. Returns, for every ranked locality, its "
                "total score and a per-criterion breakdown (raw value, normalized score, "
                "weight, contribution), plus a separate 'excluded' list of localities that "
                "had too little data to score fairly (see covered_weight/missing_criteria "
                "on each) — mention exclusions rather than ignoring them. Always call this "
                "tool for any ranking or comparison question — never estimate or guess a "
                "score yourself. A NEIGHBORHOOD id (rather than a locality id) is "
                "automatically set aside and returned in 'ineligible' with a reason — "
                "neighborhoods have no socioeconomic/accessibility/income data of their "
                "own to rank on. When 'ineligible' is non-empty, tell the user which ids "
                "were set aside and why; do not present a silently shortened list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Locality ids to compare — CBS locality codes, e.g. "
                            "['5000', '3000']. Use resolve_entity or find_items to obtain "
                            "these; do not type them from memory."
                        ),
                    },
                    "focus_criterion": {
                        "type": "string",
                        "enum": _CRITERION_NAMES,
                        "description": (
                            "OMIT THIS BY DEFAULT. Any question about which places are the "
                            "best VALUE, good places to buy, or any general ranking must "
                            "leave it unset and use the full weighted score. Set it ONLY "
                            "when the user explicitly names a single measurable dimension. "
                            "'Which is cheaper' -> price_level. 'Which has the better rental "
                            "yield' -> rental_yield. 'Which is closer to Tel Aviv' -> "
                            "accessibility. 'Which is growing faster' -> "
                            "price_momentum_vs_country. 'Which has the higher socioeconomic "
                            "level' -> socioeconomic_level. The response then carries a "
                            "'focus' block ranking the localities on that criterion alone, "
                            "with the leader already identified — report that block. Do NOT "
                            "answer a single-dimension question from total_score: it blends "
                            "all five criteria and measures value-for-money, not any one "
                            "of them."
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
                "Fetch raw metrics for ONE locality, with no scoring applied. Use this "
                "only for a factual question about a single locality. Do NOT call it "
                "twice to compare two localities — raw metrics are on wildly different "
                "scales (₪ vs. km vs. a 1-10 cluster) and are not comparable as-is. Any "
                "question that compares, ranks, or asks which is 'better' must go through "
                "compare_items, which normalizes them and returns weighted contributions."
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
                "Return the scoring methodology itself: every criterion this agent ranks "
                "localities on, its weight, and what it measures. Call this for any "
                "meta-question about HOW the ranking works — 'what criteria/weights do you "
                "use', 'how is the score calculated', 'what goes into the ranking' — never "
                "answer that from memory or call it proprietary; these weights are "
                "disclosed by design. Distinct from compare_items, which needs specific "
                "localities to rank; list_criteria takes none because it describes the "
                "method, not a result. Takes no arguments."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_entity",
            "description": (
                "Turn a user's free-text reference to a place ('Tel Aviv', 'kiryat ata', "
                "a Hebrew name, a partial or misspelled name, or a REGION like 'the "
                "Krayot' or 'Gush Dan') into concrete locality ids. ALWAYS call this "
                "before compare_items, get_item_metrics, aggregate_records, or "
                "estimate_derived_metric when the user named something in words rather "
                "than giving an exact id — never guess or invent an id yourself. Returns "
                "candidates with a confidence and a per-signal breakdown, plus a "
                "'decisive' flag. If decisive is false, you MUST NOT silently pick the "
                "top candidate: either ask the user which one they meant, or state "
                "plainly which one you assumed and why before continuing. If "
                "match_type is 'region', the query named an AREA containing several "
                "localities, not one place — see 'candidates' for all of them and "
                "'members_not_in_eligible_set' for any real places in that area too small "
                "to be ranked here. If candidates is empty, say nothing matched — do not "
                "substitute a similar-sounding place."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's own words, e.g. 'Kiryat Ata' or 'העיר חיפה'.",
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
                "Find every locality matching a set of attribute filters (all filters "
                "must match — AND, not OR). Use this whenever the user describes a GROUP "
                "rather than naming localities ('places up north', 'urban localities in "
                "the Center district'). NEVER list ids from memory to build such a group "
                "— call this. Returns the matching ids plus the attribute keys that "
                "actually exist, so you can tell the user when they asked about a field "
                "the dataset doesn't have."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "description": (
                            "Attribute name -> required value, e.g. {'district': 'North'} "
                            "or {'locality_type': 'urban'}. Available keys include: "
                            "district (Jerusalem/North/Haifa/Center/Tel Aviv/South/Judea "
                            "and Samaria/Unknown — the six official CBS districts plus "
                            "Judea and Samaria and an Unknown fallback), subdistrict_name "
                            "(a finer CBS נפה grouping, e.g. 'Sharon', 'Haifa' — NOT the "
                            "same list as district), subdistrict_code (the raw CBS נפה "
                            "numeric code as a string), global_type (the Hebrew "
                            "GLOBAL_TYPE value verbatim), locality_type (urban/Arab/"
                            "community — the English gloss of global_type). Matching is "
                            "case-insensitive. If a filter value doesn't match anything, "
                            "the result tells you the values that actually occur for that "
                            "key — check that before concluding nothing exists."
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
                "Compute a deterministic aggregate (share, count) over ONE locality's "
                "own NEIGHBORHOODS, optionally restricted to a category. This is the "
                "tool for single-locality statistics like 'what share of Tel Aviv's "
                "neighborhoods are above the city median price?' — that is NOT a "
                "ranking question, so do not use compare_items for it. 'mean' and "
                "'sum' are NOT offered here: every neighborhood record is a single "
                "unit, so a mean would always be 1.0 and a sum would just restate the "
                "count — neither means anything. Only priced neighborhoods are counted "
                "in the numerator AND denominator here — a locality can have "
                "neighborhoods nadlan.gov.il tracks but never priced, and this tool "
                "reports that count honestly (see 'neighborhoods_without_price_data') "
                "rather than pretending the priced subset is the whole picture. The "
                "'category' argument must be one of the dataset's OWN category values, "
                "not the user's phrasing: call with no category first to see "
                "'known_categories' and 'category_semantics', then call again with a "
                "real one. If 'unknown_category' comes back true you asked for "
                "something that does not exist — that is NOT a zero result and you "
                "must never report it as 0%. Always relay 'category_semantics'. "
                "'share' is a share BY COUNT of neighborhoods; the raw counts are "
                "returned too. Never compute a percentage yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "operation": {"type": "string", "enum": list(AGGREGATE_OPERATIONS_OFFERED)},
                    "category": {
                        "type": "string",
                        "description": (
                            f"Optional record category — one of {list(dataset.KNOWN_NEIGHBORHOOD_CATEGORIES)}."
                        ),
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
                "Estimate the MODELLED monthly household income needed to buy this "
                "locality's median 4-room home under standard Israeli mortgage terms, "
                "with the contributing factors, assumptions, a confidence level, and a "
                "caveat. This quantity exists in no dataset — it is built from "
                "app/affordability.py's mortgage arithmetic, not looked up. Use this for "
                "'what income do you need to buy here, and why?'-shaped questions. When "
                "explaining the 'why', use ONLY the returned factors and their "
                "magnitudes; never invent a cause. Always report the confidence and "
                "caveat — this is a model output, not a measurement. IMPORTANT: the "
                "result also compares the required income against this locality's ACTUAL "
                "median household income, but that income figure is from a 2021 CBS "
                "survey while prices are current — always relay the vintage-mismatch "
                "caveat when you state that comparison; it means today's true burden is "
                "understated, never overstated, by this figure."
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
                "Rank localities with the weighting adjusted to priorities the user "
                "stated in their own words ('I care about a short commute and don't mind "
                "paying more', 'cheapest place that isn't declining'). Map their words "
                "onto criterion NAMES and pass those; you do not choose how much to "
                "reweight — the tool applies a fixed factor. Returns BOTH the default "
                "ranking and the adjusted one, plus an 'assumption_to_state' string. You "
                "MUST tell the user which criteria you emphasized and that it reflects "
                "your reading of their words — never present a reweighted ranking as if "
                "it were the neutral one. If the user states priorities but names no "
                "localities, call find_items or ask which places they mean first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_ids": {"type": "array", "items": {"type": "string"}},
                    "emphasize": {
                        "type": "array",
                        "items": {"type": "string", "enum": _CRITERION_NAMES},
                        "description": "Criterion names the user cares MORE about.",
                    },
                    "deemphasize": {
                        "type": "array",
                        "items": {"type": "string", "enum": _CRITERION_NAMES},
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
                "Re-rank the given localities with ONE criterion's weight scaled by a "
                "factor, and report what moved: whether the winner changed, how far each "
                "locality shifted, and a Kendall tau rank-correlation (1.0 = order "
                "unchanged). Call this when the user asks why a weight was chosen, what "
                "happens if it's wrong, or how sensitive the ranking is. Report the "
                "result honestly even when it shows the ranking is fragile."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_ids": {"type": "array", "items": {"type": "string"}},
                    "criterion": {
                        "type": "string",
                        "enum": _CRITERION_NAMES,
                        "description": "Which criterion's weight to perturb.",
                    },
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
                "For every criterion, find the smallest weight multiplier that would "
                "change the top-ranked locality. Use this to answer 'how confident are "
                "you in this ranking?' or 'which weight matters most?'. A flip factor "
                "near 1.0 means the result hangs on that weight and the top places should "
                "be described as close rather than as a clear winner — say so plainly "
                "when that's the case."
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
            "name": "get_current_mortgage_rates",
            "description": (
                "Fetch the CURRENT Bank of Israel known interest rate, live, plus an "
                "estimated Prime-tracked mortgage rate built from it. Use this ONLY to "
                "answer 'what can I afford right now' or 'what would the payment be at "
                "today's rate' — it is NOT part of the value-for-money score and must "
                "never be presented as evidence for or against a particular locality: a "
                "rate change moves what every buyer can afford everywhere at once, so it "
                "says nothing about relative value. If 'available' is false the live feed "
                "was unreachable and a clearly labelled stated-assumption default is "
                "returned instead — say so plainly if you relay it; never present it as a "
                "live measurement."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# Two scores this close are the same score. Chosen from the real data, not
# picked round: at the final weights the top two localities (Qiryat
# Motzkin 0.7257, Judeide-Maker 0.7245) differ by only 0.00124, and
# weight_robustness_report on the full eligible set shows the winner
# flips at just a 5% change to ANY of price_level, accessibility,
# socioeconomic_level, or rental_yield (flip factors measured: 1.05,
# 0.95, 0.95, 1.05) and at a 15% change to price_momentum_vs_country
# (0.85) — measured 2026-08-21. A difference this small, moved by a
# weighting change this small, is not a real separation between two
# places; it is noise relative to the judgement calls that produced it.
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


def _gate_ids(item_ids: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    """Split requested ids into (rankable localities, set-aside-with-a-reason).

    THE ELIGIBILITY GATE THIS DOMAIN ACTUALLY NEEDS. In the prior
    (airport) build, this gate excluded small airports by a domain rule
    (FAA hub class) computed at RANKING TIME. Here that domain rule
    (locality type + population floor + real deal data — see
    dataset.LOCALITIES_META['eligibility_rule']) already ran UPSTREAM, in
    data/refresh_data.py: every id in dataset.LOCALITIES has already
    passed it, so there is no further population/type cut to apply here.

    What IS still this layer's job, and the reason this function exists
    at all: this dataset carries a SECOND kind of id — 1,394 neighborhood
    ids (see dataset.py's module docstring on the LOCALITIES/NEIGHBORHOODS
    split) — that are real, resolvable things but are NOT independently
    rankable, because nadlan.gov.il and CBS publish
    socioeconomic/accessibility/income data at the locality level only.
    A neighborhood id handed to a ranking tool must be rejected with a
    reason, not silently scored on a sliver of the criteria it has no
    data for and left to masquerade as a fairly-ranked peer of a city.

    THIS LIVES IN ONE FUNCTION CALLED BY EVERY RANKING TOOL, and that is
    the whole point — carried over verbatim from the prior build's own
    postmortem: the gate was originally written inline inside
    compare_items, on the argument that "a filter the caller has to
    remember is a filter that gets forgotten," and three sibling ranking
    tools (rank_by_priorities, analyze_weight_sensitivity,
    weight_robustness_report) were then added that did not call it, each
    one reproducing the exact failure the gate was written to prevent.
    The argument was right; it had just never been applied to itself.
    Enforced here, in all four, and pinned by
    tests/test_tools_domain.py's test_gate_enforced_in_all_four_ranking_tools.

    Set-aside neighborhoods are RETURNED with a reason, never silently
    dropped: the user asked about that id and is owed an explanation for
    its absence. An id that is neither a known locality nor a known
    neighborhood passes through unchanged — "not in the dataset at all"
    is a different answer from "a real thing that isn't rankable", and
    the downstream lookup (fetch_item_metrics) raises the proper
    unknown-locality error for it instead.
    """
    eligible_ids = set(dataset.ELIGIBLE_IDS)
    kept: list[str] = []
    ineligible: list[dict[str, Any]] = []
    for item_id in item_ids:
        if item_id in eligible_ids or item_id not in dataset.NEIGHBORHOODS:
            kept.append(item_id)
            continue
        row = dataset.NEIGHBORHOODS[item_id]
        parent_id = row.get("parent_id")
        assert parent_id is not None, f"data invariant violated: neighborhood {item_id!r} has no parent_id"
        parent_name = dataset.LOCALITIES.get(parent_id, {}).get("name_en", parent_id)
        ineligible.append(
            {
                "item_id": item_id,
                "name_he": row.get("name_he"),
                "parent_locality_id": parent_id,
                "parent_locality_name": parent_name,
                "reason": (
                    "This is a NEIGHBORHOOD id, not a locality id. Neighborhoods have no "
                    "socioeconomic/accessibility/income data of their own — nadlan.gov.il "
                    "and CBS both publish those at the locality level only — so ranking "
                    "one against a city would either fabricate city-level data as if it "
                    f"were the neighborhood's own, or score it unfairly on a sliver of the "
                    f"criteria. Use aggregate_records({parent_id!r}, ...) instead to compare "
                    f"{parent_name}'s neighborhoods to each other."
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

    Ordering is by raw_value, in the criterion's own favourable
    direction (cheapest first for price_level, closest first for
    accessibility, highest first for the rest) — not by normalized_score
    — because "which is CHEAPER" is a question about the raw quantity in
    its natural direction, not about which locality scores better on a
    0..1 scale. `higher_is_better` is returned alongside so the reader
    knows which direction is favourable, a separate question from which
    raw value is larger.
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
            # because "no data" and "the worst value" are different
            # answers to "which is cheaper".
            rows.append({"item_id": entry["item_id"], "raw_value": None, "normalized_score": None})
            continue
        rows.append(
            {
                "item_id": entry["item_id"],
                "raw_value": component["raw_value"],
                "normalized_score": component["normalized_score"],
            }
        )

    # Favourable direction: ascending (smallest first) when lower is
    # better, descending otherwise. reverse=match.higher_is_better puts
    # the BEST value first either way.
    with_values = sorted(
        [r for r in rows if r["raw_value"] is not None],
        key=lambda r: r["raw_value"],
        reverse=match.higher_is_better,
    )
    for position, row in enumerate(with_values, start=1):
        row["rank_on_criterion"] = position
    missing = [r["item_id"] for r in rows if r["raw_value"] is None]

    return {
        "criterion": criterion_name,
        "description": CRITERION_DESCRIPTIONS.get(criterion_name, ""),
        "weight_in_total_score": match.weight,
        "higher_is_better": match.higher_is_better,
        "ranked_by_this_criterion_alone": with_values,
        "best": with_values[0]["item_id"] if with_values else None,
        "worst": with_values[-1]["item_id"] if with_values else None,
        "no_data_for": missing,
        "note": (
            f"This ordering is on {criterion_name} alone. It is NOT the total value-for-money "
            f"score, which blends all {len(criteria)} criteria and answers a different question. "
            f"Report this ordering, and do not describe total_score as a measure of {criterion_name}."
        ),
    }


def compare_items(
    item_ids: list[str],
    criteria: list[Criterion] | None = None,
    coverage_threshold: float = 0.5,
    focus_criterion: str | None = None,
) -> dict[str, Any]:
    """Rank the given localities on the deterministic criteria.

    ELIGIBILITY IS ENFORCED HERE (via `_gate_ids`), and every sibling
    ranking tool enforces it too — see `_gate_ids`' docstring for why
    that duplication is deliberate rather than a shared "call the gate
    once upstream" design: the prior (airport) build tried the latter
    and three ranking tools silently skipped it.

    `focus_criterion` exists for the same reason it did in the prior
    build: asked a single-dimension question ("which is cheaper"), a
    model reading only `total_score` will report the wrong answer with
    perfect internal consistency, because total_score is a five-way
    blend and only one of the blended criteria is the one dimension that
    was actually asked about. With `focus_criterion` set, the tool
    returns a `focus` block that ranks the localities on that criterion
    ALONE, already ordered, with the leader named — the model reports it
    instead of inferring it, same reason every other tool here returns a
    per-component breakdown rather than a bare number.
    """
    criteria = criteria or DEFAULT_CRITERIA

    item_ids, ineligible = _gate_ids(item_ids)

    items = {item_id: fetch_item_metrics(item_id) for item_id in item_ids}
    result = rank_items(items, criteria, coverage_threshold=coverage_threshold)
    ranking = [
        {
            "rank": r.rank,
            "item_id": r.item_id,
            "name_en": dataset.LOCALITIES.get(r.item_id, {}).get("name_en"),
            "name_he": dataset.LOCALITIES.get(r.item_id, {}).get("name_he"),
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
    # reachable on a real question ("compare these two neighborhoods"),
    # where both ids are correctly gated out and the payload then asserts
    # confidence in a list that does not exist.
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
        # Requested ids that were NEIGHBORHOODS, not localities. Returned,
        # not silently dropped — tell the user these were set aside and
        # why, rather than showing a shorter list with no explanation.
        "ineligible": ineligible,
        "ranking": ranking,
        # Localities with SOME data but too little of it to score fairly,
        # per coverage_threshold — surfaced explicitly rather than
        # silently dropped, so the model can tell the user "N excluded,
        # here's why" instead of producing a ranking that quietly omits
        # them with no explanation.
        "excluded": [
            {
                "item_id": e.item_id,
                "name_en": dataset.LOCALITIES.get(e.item_id, {}).get("name_en"),
                "covered_weight": round(e.covered_weight, 4),
                "missing_criteria": list(e.missing_criteria),
                "reason": e.reason,
            }
            for e in result.excluded
        ],
        # Present only when the caller asked about one dimension. See the
        # docstring: this is the answer to "which is cheaper", computed
        # here so the model never has to derive it from the composite
        # score, which does not mean that.
        "focus": focus,
    }


def get_item_metrics(item_id: str) -> dict[str, Any]:
    metrics = fetch_item_metrics(item_id)  # raises _unknown_item / _neighborhood_not_rankable
    row = dataset.LOCALITIES[item_id]
    return {
        "item_id": item_id,
        "name_en": row.get("name_en"),
        "name_he": row.get("name_he"),
        "metrics": metrics,
    }


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
        "score_meaning": (
            "The default score answers 'where is a home good value to live in' — it weighs "
            "what a place costs against what you get for it, rather than rewarding expensive "
            "places for being expensive. It is NOT an investment-return score and it is not a "
            "prediction of future prices."
        ),
        "note": (
            "These are the default weights, applied unless the user's own stated priorities "
            "lead you to call rank_by_priorities instead — that reweights per-query, it does "
            "not change these defaults."
        ),
    }


def resolve_entity(query: str) -> dict[str, Any]:
    """Free-text name -> candidate locality ids, with confidence and a
    decisive/ambiguous verdict.

    A REGION is not a failed locality match — it is a DIFFERENT kind of
    answer ("the Krayot" names four localities, not one), and collapsing
    it to a single "best-matching" locality would silently decide
    something the user didn't ask. Checked first, via
    dataset.resolve_region, and returned as an explicitly non-decisive
    result (match_type='region') so the model has to say which reading
    it took — see system_prompt.py rule 8.

    Zero matches is a normal outcome of a fuzzy search, not a caller
    mistake, so it comes back as an empty candidate list with
    decisive=false — NOT as an exception. That's the deliberate contrast
    with fetch_item_metrics's UnknownItemError, where being handed an id
    that doesn't exist really is a bug worth naming loudly.
    """
    region = dataset.resolve_region(query)
    if region is not None:
        return {
            "query": query,
            "decisive": False,
            "match_type": "region",
            "region_name": region["region_name"],
            "candidates": [
                {
                    "item_id": item_id,
                    "matched_text": region["region_name"],
                    "confidence": 1.0,
                    "signals": {"region_member": 1.0},
                    "name_en": dataset.LOCALITIES.get(item_id, {}).get("name_en"),
                }
                for item_id in region["item_ids"]
            ],
            # Real places that belong to this region but did not survive
            # the dataset's eligibility gate (too small, or a co-op
            # housing type) — e.g. Efrat in Gush Etzion. Reported, not
            # silently dropped, same principle as `ineligible` in _gate_ids.
            "members_not_in_eligible_set": region["members_not_in_eligible_set"],
            "clarification_required": (
                f"{query!r} names the {region['region_name']} area, containing "
                f"{len(region['item_ids'])} eligible localities, not a single place. State "
                "plainly which reading you are using (all of them, or one in particular) "
                "before answering — do not silently pick one. Prefer answering with a "
                "stated assumption over stopping to ask, unless the choice would "
                "materially change the answer (see system_prompt.py rule 8)."
            ),
        }

    result = resolve(query, dataset.ENTITY_CATALOG)
    return {
        "query": result.query,
        "decisive": result.decisive,
        "match_type": "locality",
        "candidates": [
            {
                "item_id": c.item_id,
                "matched_text": c.matched_text,
                "confidence": c.confidence,
                "signals": dict(c.signals),
                "name_en": dataset.LOCALITIES.get(c.item_id, {}).get("name_en"),
            }
            for c in result.candidates
        ],
    }


def find_items(filters: dict[str, str | int | float | bool]) -> dict[str, Any]:
    """Attribute filter -> matching locality ids. Also returns the
    attribute keys that actually exist, so an unknown field reads as
    "there is no such field" rather than as "nothing matched" — those are
    different answers and conflating them misleads the user.

    On an empty result it additionally returns the values each filtered
    key actually takes, because a known key with an unknown VALUE is a
    third distinct answer that would otherwise be indistinguishable from
    the second. See the comment on the empty-result branch below."""
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
    # and "nothing matched" reads as "none exist". `district` holds the
    # six official CBS districts (plus Judea and Samaria and Unknown) —
    # a caller filtering {'district': 'Sharon'} (a SUBDISTRICT, not a
    # district) gets zero rows, and reporting that as "there are no
    # localities in the Sharon" would be false; there are several, under
    # {'subdistrict_name': 'Sharon'}.
    #
    # Exactly the failure aggregate_records already guards for an unknown
    # category VALUE (see its own comment on the same pattern); this is
    # that guard applied to a filter value.
    if filters and not matched:
        result["known_values_for_filtered_keys"] = known_attribute_values(
            dataset.ATTRIBUTES,
            [k for k in filters if k.casefold() in known_keys],
        )
        result["guidance"] = (
            "Nothing matched every filter. This is NOT evidence that no such locality "
            "exists — check your VALUES before answering. "
            "'known_values_for_filtered_keys' lists the values each key you filtered on "
            "actually takes. If a value you passed is not in that list, you filtered on "
            "something this dataset never stores (a common case: a SUBDISTRICT name, like "
            "'Sharon', passed as 'district', which only holds the six official CBS "
            "districts) — call this tool again with a real value, or with the key that "
            "does express what was asked; 'known_attribute_keys' has the full list. Only "
            "report that nothing exists after every value you used appears in the lists "
            "above."
        )
    return result


def aggregate_records(item_id: str, operation: str, category: str | None = None) -> dict[str, Any]:
    """Single-locality statistic over its own neighborhoods. Not a
    ranking, and deliberately a separate code path from scoring.py — a
    share is a count over one locality's own sub-records, with no
    weights and nothing to normalize.

    DENOMINATOR HONESTY. dataset.RECORDS only contains neighborhoods that
    HAVE a median_price_4room — 621 of 1,394 nationwide (see
    dataset.NEIGHBORHOODS_META['counts']). That is the correct set to
    take a price-comparison share OVER (a neighborhood with no price
    cannot be compared to the city median at all), but it means
    `total_records` below is the PRICED count for this locality, not
    its full neighborhood count. Reporting only that number, with no
    context, would let "70% of X's neighborhoods are above the median"
    quietly mean "70% of the 12 we have a price for" when X actually has
    40 neighborhoods and 28 of them are silently invisible to the
    question. So every response here also carries
    total_neighborhoods_tracked and neighborhoods_without_price_data —
    the real, full denominator situation — never just the priced count
    dressed up as the total.
    """
    if item_id not in dataset.LOCALITIES:
        if item_id in dataset.NEIGHBORHOODS:
            raise _neighborhood_not_rankable(item_id)
        raise _unknown_item(item_id)

    # Belt-and-braces behind the schema enum. The enum already narrows the
    # model's choices to AGGREGATE_OPERATIONS_OFFERED, but a schema is a
    # request, not a guarantee: providers do hallucinate enum values, and
    # analytics.aggregate() would happily accept "mean" or "sum" and return
    # a technically-correct, semantically-empty number (see the comment on
    # AGGREGATE_OPERATIONS_OFFERED). Failing loudly with the reason is far
    # better than handing the model 1.0 and letting it narrate that to a
    # user as if it meant something.
    if operation not in AGGREGATE_OPERATIONS_OFFERED:
        raise ValueError(
            f"operation={operation!r} is not available for neighborhood records. "
            f"Supported here: {list(AGGREGATE_OPERATIONS_OFFERED)}. "
            "'mean' and 'sum' are deliberately not offered: each record is one "
            "neighborhood carrying a constant unit, so their mean is always 1.0 "
            "and their sum is always the count. To compare neighborhood price "
            "levels, use 'share' against a category instead."
        )

    records = dataset.RECORDS[item_id]
    counts = dataset.NEIGHBORHOOD_COUNTS[item_id]
    total_neighborhoods = counts["total_neighborhoods"]
    priced_neighborhoods = counts["with_price_data"]
    name = dataset.LOCALITIES[item_id].get("name_en", item_id)

    # Four distinct outcomes, not two — see
    # _working/agent-logs/tools-surface.md's gotcha: a locality can (a)
    # have zero neighborhoods tracked by nadlan.gov.il at all, (b) have
    # neighborhoods tracked but none of them priced, (c) have some priced
    # and some not, or (d) not be a locality at all (handled above). (a)
    # and (b) both mean "no aggregate can be computed here", but for
    # DIFFERENT reasons, and conflating them with "zero" or with a normal
    # empty-category result would misrepresent which one is true.
    if total_neighborhoods == 0:
        return {
            "item_id": item_id,
            "name_en": name,
            "operation": operation,
            "category": category,
            "value": None,
            "defined": False,
            "total_neighborhoods_tracked": 0,
            "neighborhoods_with_price_data": 0,
            "neighborhoods_without_price_data": 0,
            "known_categories": [],
            "unknown_category": False,
            "guidance": (
                f"nadlan.gov.il tracks ZERO neighborhoods under {name} at all — this is not a "
                "priced-vs-unpriced gap, there is no neighborhood-level breakdown for this "
                "locality in the source data. Say so rather than reporting 0% or 'none are "
                "above the median', which would misstate a missing breakdown as a real "
                "measured answer."
            ),
        }
    if priced_neighborhoods == 0:
        return {
            "item_id": item_id,
            "name_en": name,
            "operation": operation,
            "category": category,
            "value": None,
            "defined": False,
            "total_neighborhoods_tracked": total_neighborhoods,
            "neighborhoods_with_price_data": 0,
            "neighborhoods_without_price_data": total_neighborhoods,
            "known_categories": [],
            "unknown_category": False,
            "guidance": (
                f"nadlan.gov.il tracks {total_neighborhoods} neighborhood(s) under {name}, but "
                "NONE of them have their own price data — every summary/price field nadlan "
                "publishes for them is null. No price-comparison aggregate can be computed. "
                "Say so rather than reporting 0%, which would read as a measured answer "
                "rather than an absence of data."
            ),
        }

    known_categories = sorted({str(r["category"]) for r in records if "category" in r})

    # "No such category" and "a real category with zero rows" are DIFFERENT
    # answers, and conflating them is how a tool produces a confident
    # falsehood. Same guard find_items already applies to a filter KEY,
    # applied here to a category VALUE — see that function's comment for
    # the original instance of this failure mode.
    unknown_category = category is not None and category.casefold() not in {
        c.casefold() for c in known_categories
    }
    if unknown_category:
        return {
            "item_id": item_id,
            "name_en": name,
            "operation": operation,
            "category": category,
            "value": None,
            "defined": False,
            "total_neighborhoods_tracked": total_neighborhoods,
            "neighborhoods_with_price_data": priced_neighborhoods,
            "neighborhoods_without_price_data": total_neighborhoods - priced_neighborhoods,
            "known_categories": known_categories,
            "unknown_category": True,
            "category_semantics": dataset.CATEGORY_SEMANTICS.get(item_id),
            "guidance": (
                f"{category!r} is not a category in this dataset — the available categories "
                f"are {known_categories}. This is NOT the same as a zero result: do not "
                "report '0%' or 'none'. Read 'category_semantics' above: it says exactly "
                "what these categories compare. Call this tool again with a real category "
                "and answer the question."
            ),
        }

    # A share needs something to take a share OF. With no category the
    # arithmetic is matching_units / total_units over the SAME set, which
    # is a confident, fully "defined" 100% — not useful, and not what a
    # caller asking for a bare share meant. This branch exists so the
    # documented discovery flow (call once with no category to see
    # known_categories, then call again with a real one) never itself
    # produces a misleadingly confident number.
    if operation == "share" and category is None:
        return {
            "item_id": item_id,
            "name_en": name,
            "operation": operation,
            "category": None,
            "value": None,
            "defined": False,
            "total_neighborhoods_tracked": total_neighborhoods,
            "neighborhoods_with_price_data": priced_neighborhoods,
            "neighborhoods_without_price_data": total_neighborhoods - priced_neighborhoods,
            "known_categories": known_categories,
            "unknown_category": False,
            "category_semantics": dataset.CATEGORY_SEMANTICS.get(item_id),
            "guidance": (
                "A share of everything is 100% by definition, so no value is returned here. "
                f"This call is for discovery: the categories are {known_categories}. Read "
                "'category_semantics' to see exactly what they compare, then call this tool "
                "again with a real category."
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
        "name_en": name,
        "operation": result.operation,
        "category": result.group_value,
        "value": round(value, 6) if is_defined else None,
        "defined": is_defined,
        # The REAL denominator situation — see this function's docstring.
        "total_neighborhoods_tracked": total_neighborhoods,
        "neighborhoods_with_price_data": priced_neighborhoods,
        "neighborhoods_without_price_data": total_neighborhoods - priced_neighborhoods,
        "known_categories": known_categories,
        "unknown_category": False,
        # What these categories mean, and where the comparison is a proxy
        # (e.g. a neighborhood priced only on an all-rooms fallback rather
        # than a true 4-room series) rather than an exact like-for-like.
        "category_semantics": dataset.CATEGORY_SEMANTICS.get(item_id),
        # The arithmetic, exposed — so the model explains a number it
        # never computed, same contract as compare_items' components.
        # matching_records/total_records here are counts over the PRICED
        # subset only (dataset.RECORDS), i.e. total_records ==
        # neighborhoods_with_price_data, never the full neighborhood count.
        "matching_records": result.matching_records,
        "total_records": result.total_records,
        "matching_units": result.matching_units,
        "total_units": result.total_units,
        # The other reading of "what share", so the model can offer it if
        # the user meant something other than a share by neighborhood count
        # (both readings are the same here, since each record's `units` is
        # a constant 1.0 — a share BY count of priced neighborhoods).
        "share_by_record_count": (
            round(result.matching_records / result.total_records, 6) if result.total_records else None
        ),
    }


# ─────────────────────────────────────────────────────────────────────────
# The domain model behind estimate_derived_metric: required household
# income
# ─────────────────────────────────────────────────────────────────────────
# "What income do you need to buy the median home here, and why?" is the
# question this exists for, and it is answered with a MECHANISM (the same
# amortization arithmetic every bank underwriter actually uses — see
# app/affordability.py), not a correlation. The required income is not
# looked up anywhere; it is the income at which the standard
# repayment-to-income cap is exactly met on a mortgage sized to this
# locality's own median 4-room price.
#
# Because this is a MODELLED quantity, every rate/term/ltv/cap input is a
# documented, overridable assumption (same rule affordability.py's own
# header states), not a value read live off a feed — see the constant
# below for why the live Bank of Israel rate is deliberately NOT called
# from inside this function.
REQUIRED_INCOME_METRIC = "required_monthly_household_income_for_median_4room_home"

# NOT fetched live from get_current_mortgage_rates, and that is a
# deliberate design choice, not an oversight: the live call is kept
# OUTSIDE every scored/modelled path (system_prompt.py rule 7) because a
# rate change moves what EVERY buyer can afford, everywhere, at once. If
# this function called it, running estimate_derived_metric on the SAME
# locality on two different days would return two different required-
# income figures with nothing about the locality having changed — which
# would look like new information about the PLACE when it is really just
# the news cycle. A fixed, dated, stated assumption keeps this tool
# reproducible, exactly like affordability.py's own DEFAULT_MORTGAGE_TERM_
# YEARS / DEFAULT_REPAYMENT_CAP conventions.
#
# Value: the Bank of Israel known rate observed 2026-08-21 via
# get_current_mortgage_rates (3.50%), plus the standard ~1.5
# percentage-point spread Israeli banks price into a Prime-tracked
# mortgage tranche (מסלול פריים) — the most commonly used variable-rate
# mortgage track in Israel. This is a citable market convention, not a
# fitted number, and a real quote from a specific bank on a specific day
# will differ from it.
ASSUMED_MORTGAGE_ANNUAL_RATE = 0.05  # 3.50% BOI known rate + 1.5pp Prime spread, stated 2026-08-21


def estimate_derived_metric(item_id: str) -> dict[str, Any]:
    """The monthly household income needed to buy this locality's median
    4-room home, plus the factors that produced it. The number never
    travels without its assumptions, confidence, and caveat."""
    if item_id not in dataset.LOCALITIES:
        if item_id in dataset.NEIGHBORHOODS:
            raise _neighborhood_not_rankable(item_id)
        raise _unknown_item(item_id)

    row = dataset.LOCALITIES[item_id]
    median_price = row.get("median_price_4room")
    if median_price is None:
        # Never observed in the real data (median_price_4room is
        # 104/104 — it is a precondition of the eligibility gate itself,
        # see LOCALITIES_META['eligibility_rule']) but guarded rather
        # than assumed, since a bare KeyError here would be an opaque
        # failure for something that should be structurally impossible.
        raise UnknownItemError(
            f"{item_id!r} has no median_price_4room in this dataset, so a required-income "
            "figure cannot be modelled for it. This should not happen for an eligible "
            "locality — report it as a data problem rather than guessing a price."
        )

    ltv = FIRST_HOME_MAX_LTV
    years = float(DEFAULT_MORTGAGE_TERM_YEARS)
    rate = ASSUMED_MORTGAGE_ANNUAL_RATE
    cap = DEFAULT_REPAYMENT_CAP

    principal = median_price * ltv
    down_payment = median_price - principal
    payment = monthly_payment(principal, rate, years)
    required_monthly_income = payment / cap
    # The two terms below sum to required_monthly_income EXACTLY, by
    # construction (buffer is defined as the remainder) — this is the
    # invariant build_derived_metric's own docstring requires: the
    # factors reconstruct the number, they don't just gesture at it.
    repayment_cap_buffer = required_monthly_income - payment

    contributions = [
        FactorInput(
            name="principal_debt_service",
            magnitude=payment,
            source_field="median_price_4room",
            explanation=(
                f"The mortgage payment itself: {payment:,.0f} ₪/month on a {principal:,.0f} ₪ "
                f"loan ({ltv:.0%} of the {median_price:,.0f} ₪ median 4-room price, per Bank of "
                f"Israel Directive 329's first-home LTV ceiling) at an assumed {rate:.1%} annual "
                f"rate over {years:.0f} years."
            ),
        ),
        FactorInput(
            name="repayment_cap_buffer",
            magnitude=repayment_cap_buffer,
            source_field="DEFAULT_REPAYMENT_CAP",
            explanation=(
                f"On top of the payment itself: a bank underwriting to the standard "
                f"{cap:.0%}-of-income repayment cap requires the borrower's income to be "
                f"{1 / cap:.2f}x the payment, not merely cover it — this is the extra income "
                "cushion that underwriting policy demands, not a cost of the loan itself."
            ),
        ),
    ]

    # Compare the modelled requirement against this locality's ACTUAL
    # income, where it exists. Absent for exactly the 2 of 104 localities
    # missing a CBS socioeconomic-index row (see
    # LOCALITIES_META['join_coverage']) — `or 0.0` would silently turn
    # "no data" into "zero income", so it is tracked explicitly instead,
    # same discipline scoring.py applies to a missing criterion.
    actual_monthly_income = row.get("income_per_household_monthly")
    missing_inputs: list[str] = []
    income_gap_monthly: float | None = None
    years_of_income: float | None = None
    if actual_monthly_income is None:
        missing_inputs.append("income_per_household_monthly")
    else:
        income_gap_monthly = required_monthly_income - actual_monthly_income
        years_of_income = years_of_income_to_buy(median_price, actual_monthly_income * 12.0)

    confidence = "medium"
    caveat = (
        f"MODELLED, not measured: assumes a {rate:.1%} annual mortgage rate (Bank of Israel "
        f"known rate plus the standard Prime-track spread, stated as of 2026-08-21 — a real "
        f"bank quote will differ), a {years:.0f}-year term, and {ltv:.0%} loan-to-value (BoI "
        "Directive 329's ceiling for a FIRST home only — a buyer upgrading or investing faces "
        "a lower LTV ceiling and therefore needs MORE income than this figure, not less)."
    )
    if actual_monthly_income is not None:
        caveat += (
            " The comparison against this locality's actual median household income is ITSELF "
            "stale in a specific, one-directional way: that income figure is from the CBS 2021 "
            "socioeconomic survey while median_price_4room is a live 2026 nadlan.gov.il figure, "
            "so five years of nominal wage growth are missing from the income side. That means "
            "this comparison UNDERSTATES how affordable the locality is relative to current "
            "incomes, i.e. it OVERSTATES the true income gap — never the reverse. State this "
            "when relaying the comparison."
        )
    else:
        confidence = "low"
        caveat += (
            f" No income_per_household_monthly figure exists for {item_id!r} — it is one of the "
            "2 of 104 eligible localities the CBS socioeconomic survey does not cover (it is not "
            "itself a local authority). The required-income figure below cannot be compared "
            "against this locality's own actual income at all."
        )

    result: DerivedMetricResult = build_derived_metric(
        metric=REQUIRED_INCOME_METRIC,
        unit="₪/month (household)",
        contributions=contributions,
        assumptions=(
            f"{years:.0f}-year mortgage term — the standard Israeli maximum "
            "(app.affordability.DEFAULT_MORTGAGE_TERM_YEARS).",
            f"{ltv:.0%} loan-to-value — Bank of Israel Directive 329's ceiling for a FIRST "
            "home (app.affordability.FIRST_HOME_MAX_LTV); an upgrader or investment buyer "
            "faces a lower ceiling.",
            f"{rate:.1%} assumed annual interest rate — Bank of Israel known rate (3.50%, "
            "observed 2026-08-21) plus the standard ~1.5 percentage-point Prime-track spread "
            "Israeli banks price in; call get_current_mortgage_rates for today's live rate, "
            "which this figure does NOT auto-update with (see ASSUMED_MORTGAGE_ANNUAL_RATE's "
            "own comment for why it is fixed rather than live).",
            f"{cap:.0%} repayment-to-income cap — standard Israeli bank underwriting "
            "convention (app.affordability.DEFAULT_REPAYMENT_CAP), not a single published "
            "regulatory ceiling the way LTV is.",
        ),
        confidence=confidence,
        caveat=caveat,
    )

    return {
        "item_id": item_id,
        "name_en": row.get("name_en"),
        "metric": result.metric,
        "value": round(result.value, 2),
        "unit": result.unit,
        "confidence": result.confidence,
        "caveat": result.caveat,
        # Empty for a fully-measured locality. Non-empty means at least
        # one number in the comparison below is a stand-in, not a
        # measurement.
        "missing_inputs": missing_inputs,
        "assumptions": list(result.assumptions),
        "inputs": {
            "median_price_4room": median_price,
            "down_payment": round(down_payment, 2),
            "loan_principal": round(principal, 2),
            "ltv": ltv,
            "annual_rate": rate,
            "years": years,
            "repayment_cap": cap,
        },
        "factors": [
            {
                "name": f.name,
                "magnitude": round(f.magnitude, 2),
                "share_of_total": round(f.share_of_total, 4),
                "source_field": f.source_field,
                "explanation": f.explanation,
            }
            for f in result.factors
        ],
        # The comparison to reality — 2021-vintage income, see the caveat
        # for the direction of the resulting bias.
        "actual_monthly_household_income": actual_monthly_income,
        "actual_household_income_vintage": "CBS 2021 socioeconomic survey" if actual_monthly_income is not None else None,
        "income_gap_monthly": round(income_gap_monthly, 2) if income_gap_monthly is not None else None,
        "years_of_income_to_buy_at_actual_income": (
            round(years_of_income, 2) if years_of_income is not None else None
        ),
    }


def rank_by_priorities(
    item_ids: list[str], emphasize: list[str] | None = None, deemphasize: list[str] | None = None
) -> dict[str, Any]:
    """Rank localities with the weights adjusted to the user's stated
    priorities ("I care about a short commute and don't mind paying
    more").

    Returns BOTH the default ranking and the adjusted one, deliberately.
    Handing back only the adjusted list would let a reweighting change
    the answer invisibly — the user asked for their priorities to be
    honored, not for the default result to be quietly replaced. Showing
    both makes the effect of their own stated preference legible, and
    makes it obvious when the preference changed nothing.

    Same eligibility gate every other ranking tool applies (see
    `_gate_ids`), applied here directly rather than inherited — this is
    exactly the sibling tool whose missing gate was the prior build's
    real, observed bug.
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
                "name_en": dataset.LOCALITIES.get(r.item_id, {}).get("name_en"),
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
    # could report "your priorities changed the winner" off a gap this
    # file elsewhere calls noise. A reweighting only changed the winner if
    # the new leader was outside the old leader's tie band.
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
        # returned for the same reason — the user asked about these ids.
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
    with evidence instead of reassurance.

    Same eligibility gate every other ranking tool applies (see
    `_gate_ids`) — without it, this report would compute flip factors
    over a population compare_items would never actually rank, so the
    "confidence" evidence would describe a ranking the user was never
    shown.
    """
    # A negative factor produces a NEGATIVE weight, which silently breaks
    # two invariants scoring.py guarantees: total_score in 0..1, and
    # component weights summing to 1. apply_priority_emphasis already
    # guards this (factor must be > 0); the guard is repeated here
    # because this is a different entry point into the same underlying
    # risk — scale_criterion_weight has no floor of its own.
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
        # One number for "did this weight actually matter", more honest
        # than eyeballing two lists that look similar.
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
                "name_en": dataset.LOCALITIES.get(c.item_id, {}).get("name_en"),
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
    it can legitimately return "not very", which is the point. Same
    eligibility gate every other ranking tool applies (see `_gate_ids`).
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
                    else f"Scaling this weight by {flip}x changes the top-ranked locality. "
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
            {
                "rank": r.rank,
                "item_id": r.item_id,
                "name_en": dataset.LOCALITIES.get(r.item_id, {}).get("name_en"),
                "total_score": round(r.total_score, 4),
            }
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
# The Bank of Israel publishes its own known/monetary interest rate
# (הריבית הידועה) keylessly, in real time, at the URL below — verified by
# hand (curl, and a plain urllib.request.urlopen with no spoofed headers)
# on 2026-08-20/21; it returns clean JSON with no CAPTCHA or bot wall.
#
# DELIBERATELY OUTSIDE THE SCORED PATH, and that is the interesting
# decision, same shape as the prior (airport) build's FAA NAS Status
# call. A mortgage-rate move changes what EVERY buyer can afford,
# everywhere, on the same day — it says nothing about whether one
# LOCALITY is better value than another, so feeding it into the ranking
# would conflate a national macro fact with a place-specific one. See
# system_prompt.py rule 7.
#
# This is NOT itself a mortgage rate: it is the central bank's own policy
# rate, which most Israeli banks then use as the base for a Prime-tracked
# lending tranche (מסלול פריים) by adding a standard spread. Both numbers
# are returned, clearly labelled, rather than presenting the base rate as
# if it were what a borrower would actually be quoted.
BOI_INTEREST_RATE_URL = "https://boi.org.il/PublicApi/GetInterest"
BOI_TIMEOUT_SECONDS = 6.0

# Standard spread Israeli banks price into a Prime-tracked mortgage
# tranche on top of the Bank of Israel's own known rate — a widely cited
# market convention (not a regulatory figure the way LTV is, so treated,
# like affordability.py's DEFAULT_REPAYMENT_CAP, as a documented,
# overridable assumption rather than an authoritative constant).
PRIME_MORTGAGE_SPREAD = 0.015  # 1.5 percentage points

# Used ONLY when the live feed is unreachable — see the except branch
# below. Bank of Israel known rate as last observed by this project
# (2026-08-21: 3.50%), plus PRIME_MORTGAGE_SPREAD. Matches
# ASSUMED_MORTGAGE_ANNUAL_RATE exactly on purpose: it is the same
# real-world number, stated once as a constant here and reused there
# rather than risking the two drifting apart.
FALLBACK_BOI_KNOWN_RATE = 0.035  # 3.50%, observed 2026-08-21
FALLBACK_ASSUMED_PRIME_MORTGAGE_RATE = FALLBACK_BOI_KNOWN_RATE + PRIME_MORTGAGE_SPREAD
FALLBACK_STATED_DATE = "2026-08-21"


def get_current_mortgage_rates() -> dict[str, Any]:
    """Live Bank of Israel known rate, plus an estimated Prime-track
    mortgage rate built from it.

    Every failure mode here returns a payload with `available: False`
    and a clearly-labelled STATED ASSUMPTION default, rather than
    raising or silently substituting a scrape of some other source —
    per the standing rule that a blocked data source gets reported, not
    worked around (see _working/PROGRESS.md). The failure is visible IN
    the payload itself (`available`, `is_stated_assumption`, `reason`),
    not just in a log line the model never sees, because this tool is
    decoration on an answer that must still work without it — a timeout
    on a live feed should degrade the response, not fail the question.
    """
    try:
        request = urllib.request.Request(
            BOI_INTEREST_RATE_URL, headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=BOI_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read())
        # The API returns e.g. 3.5 meaning 3.5%, not a 0..1 fraction —
        # verified by hand against the published rate (3.50% as of the
        # last Monetary Committee decision before 2026-08-21).
        boi_known_rate = float(payload["currentInterest"]) / 100.0
        estimated_prime_rate = boi_known_rate + PRIME_MORTGAGE_SPREAD
        return {
            "available": True,
            "is_stated_assumption": False,
            "source": BOI_INTEREST_RATE_URL,
            "boi_known_rate": boi_known_rate,
            "boi_known_rate_pct": f"{boi_known_rate:.2%}",
            "prime_mortgage_spread": PRIME_MORTGAGE_SPREAD,
            "estimated_prime_mortgage_rate": estimated_prime_rate,
            "estimated_prime_mortgage_rate_pct": f"{estimated_prime_rate:.2%}",
            "last_published_date": payload.get("lastPublishedDate"),
            "next_interest_decision_date": payload.get("nextInterestDate"),
            "note": (
                "boi_known_rate is the Bank of Israel's own known/monetary interest rate "
                "(הריבית הידועה), fetched live just now. estimated_prime_mortgage_rate adds "
                f"the standard {PRIME_MORTGAGE_SPREAD:.1%} spread Israeli banks price into a "
                "Prime-tracked mortgage tranche (מסלול פריים) — a citable convention, not "
                "this specific borrower's actual quote, which depends on the bank and the "
                "borrower's own profile. This is NOT part of the value-for-money score and "
                "must never be used as evidence for or against a locality."
            ),
        }
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, OSError) as exc:
        return {
            "available": False,
            "is_stated_assumption": True,
            "source": BOI_INTEREST_RATE_URL,
            "reason": (
                f"Bank of Israel public rate API unreachable or returned an unexpected shape "
                f"({type(exc).__name__}: {exc})."
            ),
            "boi_known_rate": FALLBACK_BOI_KNOWN_RATE,
            "boi_known_rate_pct": f"{FALLBACK_BOI_KNOWN_RATE:.2%}",
            "prime_mortgage_spread": PRIME_MORTGAGE_SPREAD,
            "estimated_prime_mortgage_rate": FALLBACK_ASSUMED_PRIME_MORTGAGE_RATE,
            "estimated_prime_mortgage_rate_pct": f"{FALLBACK_ASSUMED_PRIME_MORTGAGE_RATE:.2%}",
            "stated_assumption_source": (
                f"Bank of Israel known rate as last observed by this project ({FALLBACK_STATED_DATE}: "
                f"{FALLBACK_BOI_KNOWN_RATE:.2%}) plus the standard {PRIME_MORTGAGE_SPREAD:.1%} "
                "Prime-track spread."
            ),
            "stated_assumption_date": FALLBACK_STATED_DATE,
            "note": (
                "The live Bank of Israel rate feed was unreachable just now, so this is a "
                "STATED, DATED ASSUMPTION, not a live measurement — say so if you relay it. "
                "Never present it as current."
            ),
        }


# Dispatch table used by agent_loop.py: tool name -> callable(args_dict).
# Kept as a plain dict, not a decorator/registry framework — this is the
# entire "tool registry" a hand-rolled loop needs.
def _find_items_filters(args: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Normalize find_items' arguments. Three shapes arrive in practice
    and they do NOT mean the same thing — carried over verbatim from the
    prior build, which found this live on its own first eval question:

        {}                       -> match everything. Legitimate and
                                    documented; a model asking for the
                                    whole dataset. args.get (not
                                    args[...]) because a raw
                                    KeyError('filters') here would read
                                    as a tool error instead of the
                                    empty-filter result clearly intended.
        {"filters": {...}}       -> the schema's shape.
        {"district": "North"}    -> the model dropped the wrapper.

    The third used to collapse into the first: args.get("filters")
    returned None, "no filters" means "match everything", and a request
    for a SUBSET silently returned every locality. So a flattened call is
    read as filters rather than as "no filters". Nothing is guessed: an
    unrecognized key still lands in `unknown_filter_keys` downstream,
    which already tells the model it filtered on a field that does not
    exist. Only scalars are taken; a stray list or object is another
    tool's argument, not a filter value.
    """
    if "filters" in args:
        return args.get("filters") or {}
    return {k: v for k, v in args.items() if isinstance(v, (str, int, float, bool))}


TOOL_REGISTRY: dict[str, Callable[[dict[str, Any]], Any]] = {
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
    "get_current_mortgage_rates": lambda _: get_current_mortgage_rates(),
}
