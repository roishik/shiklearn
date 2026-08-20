"""
dataset.py -- the one place that reads data/ off disk.

Everything else in app/ is handed data rather than fetching it, which is
what keeps scoring.py, analytics.py, entity_resolution.py and
runway_geometry.py pure and testable with no fixtures. This module is the
seam: it does the I/O, shapes the result, and hands plain dicts onward.

Loaded once at import. The dataset is a few hundred KB of static JSON
rebuilt by data/refresh_data.py, not a live query, so re-reading it per
tool call would buy nothing.

ELIGIBILITY. `ELIGIBLE_IDS` is FAA hub class L/M/S -- 144 of the 515
airports. Percentage growth on a near-zero base is meaningless, and
scoring all 515 put airports like Jack Edwards National (+126,403% YoY
on 37,951 passengers) into the top 50 and made New Bedford Regional read
as New England's second-best expansion candidate. The threshold is FAA's
own primary-airport classification rather than a round number we picked.
Note it is a RANKING filter, not a data cut: every one of the 515 stays
addressable by id, by find_items, and by entity resolution. See
DECISIONS.md [18:20] and ASSUMPTIONS.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed_data"

# FAA hub classes that count as primary airports -- see module docstring.
ELIGIBLE_HUB_CLASSES = frozenset({"L", "M", "S"})

# US Census regions, used so "airports in the West" works as a filter
# rather than only the four regions the example questions happen to name.
_CENSUS_REGION = {
    "Northeast": {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"},
    "Midwest": {"IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"},
    "South": {"DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY",
              "MS", "TN", "AR", "LA", "OK", "TX"},
    "West": {"AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"},
}
_STATE_TO_REGION = {st: region for region, states in _CENSUS_REGION.items() for st in states}


def _load(name: str) -> dict[str, Any]:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


_candidates = _load("candidates.json")
_anc_traffic = _load("anc_traffic_mix.json")

AIRPORTS: dict[str, dict[str, Any]] = _candidates["airports"]
DATA_META: dict[str, Any] = _candidates["_meta"]

ELIGIBLE_IDS: tuple[str, ...] = tuple(
    sorted(k for k, v in AIRPORTS.items() if v.get("faa_hub_class") in ELIGIBLE_HUB_CLASSES)
)


def state_of(airport: dict[str, Any]) -> str:
    """'US-CA' -> 'CA'. OurAirports' iso_region is the only state field
    in the set, so it is parsed rather than stored twice."""
    region = airport.get("iso_region") or ""
    return region.split("-", 1)[1] if "-" in region else region


# ── Numeric metrics: the raw inputs the scorer normalizes ────────────────
# Keys here MUST match DEFAULT_CRITERIA's names in tools.py. Any airport
# missing one of these simply omits the key -- scoring.py renormalizes
# each item's weights across the criteria actually present, which is what
# lets the 14 airports with no Census population still rank.
def _metrics(airport: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    enplanements = airport.get("enplanements_cy2025_prelim")
    if enplanements is not None:
        out["absolute_scale"] = float(enplanements)
        runways = airport.get("air_carrier_runway_count") or airport.get("runway_count")
        if runways:
            # Passengers per air-carrier runway. Correlates r=0.89 with
            # absolute_scale -- kept deliberately, weighted low, and
            # disclosed rather than hidden. See DECISIONS.md [18:20].
            out["capacity_pressure"] = float(enplanements) / float(runways)
    if airport.get("yoy_pct_change_2025") is not None:
        out["traffic_growth"] = float(airport["yoy_pct_change_2025"])
    if airport.get("nearest_competitor_mi") is not None:
        out["catchment_monopoly"] = float(airport["nearest_competitor_mi"])
    if airport.get("county_population_cagr_recent") is not None:
        out["regional_demand_growth"] = float(airport["county_population_cagr_recent"])
    return out


# ── Categorical attributes: what you FILTER on, never what you score on ──
# Kept in a separate dict from the metrics above on purpose. Mixing them
# is how a categorical value ends up normalized as if it were a KPI.
def _weather_constrained(airport: dict) -> str:
    """"yes" / "no" / "unknown" -- see the call site for why the third
    value exists. Unmeasurable is not the same as unconstrained."""
    if not airport.get("runway_geometry_complete"):
        return "unknown"
    return "yes" if (airport.get("weather_capacity_degradation") or 0) > 0 else "no"


def _attributes(airport: dict[str, Any]) -> dict[str, str]:
    state = state_of(airport)
    return {
        "state": state,
        "region": _STATE_TO_REGION.get(state, "Other"),
        "new_england": "yes" if airport.get("is_new_england") else "no",
        "hub_class": str(airport.get("faa_hub_class") or "None"),
        "municipality": str(airport.get("municipality") or ""),
        # Whether the airport loses arrival capacity in low visibility --
        # a categorical read on the runway geometry, so "which congested
        # airports are weather-constrained" is a filter, not an essay.
        #
        # THREE values, not two, and that is the point. 23 airports have
        # incomplete runway geometry (missing coordinates, or no runway
        # long enough for air-carrier arrivals), so nothing was measured
        # for them. Collapsing that into "no" reports an absence of
        # measurement as a measurement of absence — and three of the 23
        # are ranking-eligible, one of them currently 3rd nationally. A
        # user asking "which top-ranked airports are NOT weather
        # constrained" would get it back as an affirmative finding.
        # `runway_geometry_complete` is already in the data; it just was
        # not being read.
        "weather_constrained": _weather_constrained(airport),
        "ranking_eligible": "yes" if airport.get("faa_hub_class") in ELIGIBLE_HUB_CLASSES else "no",
    }


METRICS: dict[str, dict[str, float]] = {k: _metrics(v) for k, v in AIRPORTS.items()}
ATTRIBUTES: dict[str, dict[str, str]] = {k: _attributes(v) for k, v in AIRPORTS.items()}


# ── Entity resolution catalog ────────────────────────────────────────────
# Built from the dataset's own name/code/city columns, never hand-written.
# Includes the bare id so an exact-id query resolves through the same path
# with no special case, and both the FAA LocID and the IATA code because
# for ~30 airports they differ (FAA 'SAW' is Marquette, IATA MQT).
def _aliases(locid: str, airport: dict[str, Any]) -> list[str]:
    name = str(airport.get("name") or "")
    municipality = str(airport.get("municipality") or "")
    aliases = {locid, name, municipality}
    iata = airport.get("iata_code")
    if iata:
        aliases.add(str(iata))
    # "Boston Logan" as well as "Boston Logan International Airport" --
    # users rarely type the suffix, and dropping it costs nothing since
    # resolution scores against every alias and keeps the best.
    for suffix in (" International Airport", " Regional Airport", " Airport", " International"):
        if name.endswith(suffix):
            aliases.add(name[: -len(suffix)])
            break
    if municipality and name and municipality not in name:
        aliases.add(f"{municipality} {name}")
    return sorted(a for a in aliases if a)


ENTITY_CATALOG: dict[str, list[str]] = {k: _aliases(k, v) for k, v in AIRPORTS.items()}


# ── Multi-airport metros ─────────────────────────────────────────────────
# The one HAND-WRITTEN structure in this module, and it earns the
# exception. "Compare LA and Santa Ana" is a real question from the brief,
# and "LA" does not name an airport — it names a region containing five
# of them. No column in FAA's, OurAirports' or Census's data says which
# airports share a metro market: county is too small (LAX and BUR are in
# the same county as each other but ONT is not), and CBSA would need a
# metro-boundary file we do not carry.
#
# Deliberately NOT resolved silently to the busiest airport. Answering
# "compare LA and Santa Ana" with LAX-vs-SNA hides a choice the user did
# not make — the LA basin's traffic is genuinely split, and SNA competes
# with BUR/LGB/ONT as much as with LAX. So this returns the whole set and
# the agent is required to state which reading it used.
#
# Scope: metros where a colloquial name maps to several airports. A
# single-airport city needs no entry — _aliases() already covers it from
# the municipality column.
METRO_AIRPORTS: dict[str, list[str]] = {
    "Los Angeles": ["LAX", "BUR", "LGB", "ONT", "SNA"],
    "LA": ["LAX", "BUR", "LGB", "ONT", "SNA"],
    "San Francisco Bay Area": ["SFO", "OAK", "SJC"],
    "Bay Area": ["SFO", "OAK", "SJC"],
    "New York": ["JFK", "LGA", "EWR"],
    "NYC": ["JFK", "LGA", "EWR"],
    "Washington DC": ["DCA", "IAD", "BWI"],
    "DC": ["DCA", "IAD", "BWI"],
    "Chicago": ["ORD", "MDW"],
    "Houston": ["IAH", "HOU"],
    "Dallas": ["DFW", "DAL"],
    "Miami": ["MIA", "FLL", "PBI"],
}


def resolve_metro(query: str) -> tuple[str, list[str]] | None:
    """Exact (case-insensitive) metro-name match -> its airports, keeping
    only ids actually present in the dataset. Exact-match only: fuzzy
    metro matching would reintroduce exactly the short-query noise
    MIN_FUZZY_QUERY_LENGTH exists to stop."""
    folded = query.strip().casefold()
    for name, ids in METRO_AIRPORTS.items():
        if name.casefold() == folded:
            present = [i for i in ids if i in AIRPORTS]
            if present:
                return name, present
    return None


# ── Sub-records for single-entity aggregates ─────────────────────────────
# Only Anchorage has a departure-mix breakdown, because BTS publishes
# per-origin segment summaries and the true per-route microdata is
# TranStats-only (bot-blocked). Every other airport raises a clear "no
# record-level data for this airport" rather than getting a fabricated
# one -- see ASSUMPTIONS.md.
def _anc_records() -> list[dict[str, Any]]:
    year = _anc_traffic["_meta"]["most_recent_full_calendar_year"]
    a = _anc_traffic["annual"][year]
    return [
        {
            "category": "international",
            "units": a["outbound_international_departures"],
            "avg_distance_mi": a["avg_outbound_international_distance_mi"],
            "year": year,
        },
        {
            "category": "domestic",
            "units": a["domestic_departures"],
            "avg_distance_mi": a["avg_domestic_distance_mi"],
            "year": year,
        },
    ]


RECORDS: dict[str, list[dict[str, Any]]] = {"ANC": _anc_records()}

# What the categories MEAN, in the user's vocabulary rather than the
# dataset's. Surfaced with every aggregate so the model can map "long
# haul" onto what the data actually distinguishes, instead of asking for
# a category called "long haul", matching nothing, and reporting 0%.
#
# The honest framing matters here: this is a PROXY, not the requested
# measurement. No publicly downloadable BTS table carries per-route
# distances, so a true "% of flights over N miles" is not computable —
# see ASSUMPTIONS.md and DECISIONS.md [15:38].
CATEGORY_SEMANTICS: dict[str, str] = {
    "ANC": (
        "Categories are 'domestic' and 'international' departures. For a LONG-HAUL question, "
        "international share is the available proxy: Anchorage's outbound international flights "
        "averaged 4,322 mi in 2025 (trans-Pacific cargo and passenger routes) against 1,458 mi "
        "for domestic. State that this is a domestic/international split used as a long-haul "
        "proxy, NOT a true distance-threshold percentage — per-route distances are not in any "
        "public BTS table. Note too that the domestic average is itself long-haul-heavy "
        "(Anchorage-Seattle alone is ~1,448 mi), so the international share UNDERSTATES how "
        "many long flights leave Anchorage."
    )
}
ANC_ANNUAL: dict[str, Any] = _anc_traffic["annual"]
ANC_META: dict[str, Any] = _anc_traffic["_meta"]
