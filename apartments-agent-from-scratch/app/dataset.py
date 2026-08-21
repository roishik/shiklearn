"""
dataset.py -- the one place that reads data/ off disk.

Everything else in app/ is handed data rather than fetching it, which is
what keeps scoring.py, analytics.py, entity_resolution.py and
affordability.py pure and testable with no fixtures. This module is the
seam: it does the I/O, shapes the result, and hands plain dicts onward.

Loaded once at import. The dataset is a few hundred KB of static JSON
rebuilt by data/refresh_data.py, not a live query, so re-reading it per
tool call would buy nothing.

THE ITEM UNIVERSE, AND WHY IT'S SPLIT THIS WAY. This dataset carries two
tables at genuinely different granularity: 104 LOCALITIES (cities/towns,
each with its own socioeconomic cluster, income, peripherality and
distance-to-core figures -- the row shape DEFAULT_CRITERIA in tools.py is
built on) and 1,394 NEIGHBORHOODS (sub-areas of those localities, carrying
little beyond a price series, and that series is present for only 621 of
them). A neighborhood has no socioeconomic/accessibility/income figure of
its own -- nadlan.gov.il and CBS both publish those at the LOCALITY level
only -- so scoring a neighborhood against DEFAULT_CRITERIA the same way a
locality is scored would either silently fabricate city-level data as if
it were neighborhood-level (misattribution) or score it on a sliver of the
criteria and let it masquerade as a fairly-ranked peer of a city.

So: LOCALITIES are the sole entity this module resolves by name
(ENTITY_CATALOG), exposes metrics for (METRICS), and offers for filtering
(ATTRIBUTES). NEIGHBORHOODS are reachable only as SUB-RECORDS of their
parent locality (RECORDS / NEIGHBORHOOD_COUNTS), consumed by
aggregate_records -- the same shape the old airport-domain build used for
Anchorage's departure-mix records under one airport. app/tools.py's
eligibility gate (`_gate_ids`) enforces this split at the tool boundary: a
neighborhood id handed to a ranking tool comes back in `ineligible` with a
reason, not silently scored on partial data.

ELIGIBILITY. `ELIGIBLE_IDS` is every locality in localities.json, which is
ALL of them -- data/refresh_data.py already applied the domain eligibility
rule (GLOBAL_TYPE in {urban, Arab, community} AND population >= 5,000 AND
real nadlan deal data) when it built the file; see `LOCALITIES_META
["eligibility_rule"]` for the rule's own text and
`LOCALITIES_META["counts"]` for how many of the original 1,509 candidate
localities were dropped and why. There is nothing left to filter at this
layer for LOCALITIES. What IS still this layer's job: making sure a
neighborhood id (a different kind of thing that happens to also have an
id) never gets treated as an eligible locality. See app/tools.py's
`_gate_ids` for where that is enforced, and its docstring for the concrete
failure this mirrors from the prior (airport) build.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.entity_resolution import resolve as _resolve_entity
from app.hebrew import REGION_GROUPS, TRANSLITERATION_VARIANTS, match_region

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed_data"


def _load(name: str) -> dict[str, Any]:
    """Read one processed-data JSON file, failing LOUDLY and specifically
    if it is missing -- this module is the only place a missing/renamed
    data file should ever surface, and it should surface as "run
    data/refresh_data.py", not as an opaque FileNotFoundError three frames
    into some unrelated tool call."""
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"missing dataset file {path} -- run `python -m data.refresh_data` (or "
            "`python data/refresh_data.py`) to (re)build data/processed_data/*.json "
            "before importing app.dataset. This module refuses to start with stale "
            "or absent data rather than silently running on nothing."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_localities_doc = _load("localities.json")
_neighborhoods_doc = _load("neighborhoods.json")

LOCALITIES: dict[str, dict[str, Any]] = _localities_doc["localities"]
LOCALITIES_META: dict[str, Any] = _localities_doc["_meta"]
NEIGHBORHOODS: dict[str, dict[str, Any]] = _neighborhoods_doc["neighborhoods"]
NEIGHBORHOODS_META: dict[str, Any] = _neighborhoods_doc["_meta"]

# See the module docstring's ELIGIBILITY section: the domain rule already
# ran upstream in data/refresh_data.py, so this is every locality in the
# file, not a further cut. Kept as a named, exported tuple (rather than
# inlining `LOCALITIES.keys()` at every call site) so app/tools.py has one
# stable thing to gate against, and so a future tightening of eligibility
# (e.g. a minimum data-recency bar) has a single place to land.
ELIGIBLE_IDS: tuple[str, ...] = tuple(sorted(LOCALITIES.keys()))


# ── Numeric metrics: the raw inputs the scorer normalizes ────────────────
# Keys here MUST match DEFAULT_CRITERIA's names in tools.py. Any locality
# missing one of these simply omits the key -- scoring.py renormalizes
# each item's weights over what IS present, so a coverage gap (see
# LOCALITIES_META's join_coverage / field-coverage notes -- e.g.
# socioeconomic_cluster is 102/104, rental_yield 98/104) costs that
# locality the one criterion, not the whole ranking.
_METRIC_FIELD_MAP: dict[str, str] = {
    "price_level": "median_price_4room",
    "socioeconomic_level": "socioeconomic_cluster",
    "accessibility": "km_to_nearest_core",
    "price_momentum_vs_country": "price_momentum_vs_country",
    "rental_yield": "rental_yield",
}


def _metrics(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for criterion_name, field in _METRIC_FIELD_MAP.items():
        value = row.get(field)
        if value is not None:
            out[criterion_name] = float(value)
    return out


METRICS: dict[str, dict[str, float]] = {k: _metrics(v) for k, v in LOCALITIES.items()}


# ── Categorical attributes: what you FILTER on, never what you score on ──
#
# SUBDISTRICT_NAMES maps the CBS נפה (subdistrict) numeric code stored in
# each locality's `subdistrict` field to an English/Hebrew name and its
# coarser CBS district, so `find_items` can filter on a region a human
# would actually type ("north", "the Sharon subdistrict") instead of a
# bare code nobody knows offhand.
#
# Built from the 22 distinct codes actually present in the 104 eligible
# localities, cross-checked against which cities fall under each code
# (e.g. code 51 contains Tel Aviv-Yafo, Ramat Hasharon, Herzliyya --
# unambiguously the נפת תל אביב subdistrict) against standard published
# CBS נפה numbering. 18 of the 22 codes are confidently identified this
# way. The remaining 4 (73, 74, 76, 77) are all in the 70s block CBS
# reserves for the Judea and Samaria Area, which confidently places them
# in that DISTRICT, but no source available to this pipeline confirms
# which specific נפה each of those four codes names -- so, per
# instruction, they are LEFT OUT of this table rather than guessed. Their
# `district` attribute is still set (see `_district_of` below); their
# `subdistrict_name` attribute reads "Unknown".
SUBDISTRICT_NAMES: dict[int, dict[str, str]] = {
    11: {"name_en": "Jerusalem", "name_he": "ירושלים", "district": "Jerusalem"},
    21: {"name_en": "Zefat (Safed)", "name_he": "צפת", "district": "North"},
    22: {"name_en": "Kinneret", "name_he": "כנרת", "district": "North"},
    23: {"name_en": "Yizre'el", "name_he": "יזרעאל", "district": "North"},
    24: {"name_en": "Akko (Acre)", "name_he": "עכו", "district": "North"},
    25: {"name_en": "Nazareth", "name_he": "נצרת", "district": "North"},
    29: {"name_en": "Golan", "name_he": "גולן", "district": "North"},
    31: {"name_en": "Haifa", "name_he": "חיפה", "district": "Haifa"},
    32: {"name_en": "Hadera", "name_he": "חדרה", "district": "Haifa"},
    41: {"name_en": "Sharon", "name_he": "השרון", "district": "Center"},
    42: {"name_en": "Petah Tikva", "name_he": "פתח תקווה", "district": "Center"},
    43: {"name_en": "Ramla", "name_he": "רמלה", "district": "Center"},
    44: {"name_en": "Rehovot", "name_he": "רחובות", "district": "Center"},
    51: {"name_en": "Tel Aviv", "name_he": "תל אביב", "district": "Tel Aviv"},
    52: {"name_en": "Ramat Gan", "name_he": "רמת גן", "district": "Tel Aviv"},
    53: {"name_en": "Holon", "name_he": "חולון", "district": "Tel Aviv"},
    61: {"name_en": "Ashkelon", "name_he": "אשקלון", "district": "South"},
    62: {"name_en": "Be'er Sheva", "name_he": "באר שבע", "district": "South"},
}

# Codes present in the data whose SPECIFIC נפה identity is not confirmed --
# see SUBDISTRICT_NAMES' comment. Named here (not just absent) so a reader
# can tell "we looked and couldn't confirm this" apart from "this code
# doesn't occur in our data at all".
UNCONFIRMED_SUBDISTRICT_CODES: tuple[int, ...] = (73, 74, 76, 77)
_JUDEA_AND_SAMARIA_DISTRICT = "Judea and Samaria"


def _district_of(code: int | None) -> str:
    if code is None:
        return "Unknown"
    known = SUBDISTRICT_NAMES.get(code)
    if known is not None:
        return known["district"]
    if code in UNCONFIRMED_SUBDISTRICT_CODES:
        return _JUDEA_AND_SAMARIA_DISTRICT
    return "Unknown"


# Hebrew GLOBAL_TYPE value -> a short English filter value. Kept distinct
# from `global_type` itself (which is passed through verbatim in Hebrew)
# so a caller can filter in either language.
_LOCALITY_TYPE_EN: dict[str, str] = {
    "ישובים עירוניים": "urban",
    "ישובים ערביים": "Arab",
    "ישובים קהילתיים": "community",
}


def _attributes(row: dict[str, Any]) -> dict[str, str]:
    code = row.get("subdistrict")
    subdistrict = SUBDISTRICT_NAMES.get(code) if code is not None else None
    global_type = row.get("global_type") or ""
    return {
        "district": _district_of(code),
        "subdistrict_name": subdistrict["name_en"] if subdistrict else "Unknown",
        "subdistrict_code": str(code) if code is not None else "Unknown",
        "global_type": global_type,
        "locality_type": _LOCALITY_TYPE_EN.get(global_type, "Unknown"),
    }


ATTRIBUTES: dict[str, dict[str, str]] = {k: _attributes(v) for k, v in LOCALITIES.items()}


# ── Entity resolution catalog ────────────────────────────────────────────
# Built from the dataset's own name columns plus the hand-curated
# transliteration table in app/hebrew.py, never hand-written per-locality.
# Deliberately does NOT include a romanize() of every name_he as a
# separate alias: entity_resolution.score_pair already tries a romanized
# comparison live for any Hebrew alias against a Latin query (and vice
# versa) -- pre-romanizing here would just duplicate work score_pair
# already does, for zero extra recall.
def _aliases(item_id: str, row: dict[str, Any]) -> list[str]:
    name_he = str(row.get("name_he") or "")
    name_en = str(row.get("name_en") or "")
    name_en_raw = str(row.get("name_en_raw") or "")
    aliases = {item_id, name_he, name_en, name_en_raw}
    aliases.update(TRANSLITERATION_VARIANTS.get(name_he, ()))
    return sorted(a for a in aliases if a)


ENTITY_CATALOG: dict[str, list[str]] = {k: _aliases(k, v) for k, v in LOCALITIES.items()}


# ── Region groups ─────────────────────────────────────────────────────────
# app.hebrew.match_region maps a query like "Gush Dan" / "the Krayot" to a
# canonical name and a tuple of HEBREW LOCALITY NAMES -- but that module
# has no catalog to resolve those names against (zero I/O, by its own
# rules). This is that consuming layer, exactly as its docstring expects.
#
# Matching those curated Hebrew names against LOCALITIES' own name_he is
# NOT a simple string-equality lookup: קרית vs קריית is a genuine
# orthographic difference (a whole letter), not niqqud/hyphen noise
# normalize_hebrew() folds away, so an exact-match index would silently
# drop half the Krayot. Reusing entity_resolution.resolve() -- the SAME
# fuzzy machinery that resolves a user's free-text query -- handles this
# for free and was spot-checked against the real catalog (קרית אתא ->
# קריית אתא at confidence 0.9122, etc.).
#
# NOT using `decisive` here, and NOT simply taking the top candidate
# either -- both were tried and both are wrong for this specific use.
# `decisive` is calibrated for "is it safe to act on a USER's free-text
# query without asking" and demands a confidence GAP over the runner-up;
# it is the wrong bar for "does this curated, trusted Hebrew string name a
# real place in our 104-locality catalog at all". The real failure mode
# here is a FALSE POSITIVE: a REGION_GROUPS member that is a genuine place
# but did not survive this dataset's eligibility gate (e.g. Efrat, in
# Gush Etzion, below the population floor) still gets scored against
# every catalog entry, and the best of a bad set of options can clear
# MIN_RELEVANCE (0.63) on partial token overlap alone -- caught on real
# data: 'אפרת' (Efrat) top-scored against 'אורנית' (Oranit) at 0.72,
# a wrong match, not a hedge. Genuine matches in this catalog (including
# every קרית/קריית spelling-variant case) scored >= 0.88 without
# exception; every confirmed non-member scored <= 0.72. The threshold
# below sits in that measured gap -- see the log
# (_working/agent-logs/tools-surface.md) for the full per-member
# confidence table this was calibrated against.
_REGION_MEMBER_MATCH_MIN_CONFIDENCE = 0.85


def resolve_region(query: str) -> dict[str, Any] | None:
    """`query` names a REGION_GROUPS entry (Hebrew, English, or a curated
    nickname) -> {'region_name', 'item_ids', 'members_not_in_eligible_set'},
    or None if `query` does not name a region at all.

    `members_not_in_eligible_set` is non-empty when a REGION_GROUPS member
    is a real place that didn't survive this dataset's eligibility gate
    (too small, or a co-op-housing type) -- e.g. Gush Etzion's Efrat and
    its neighbors, all below the population floor. Reported, not silently
    dropped, for the same reason `_gate_ids` in tools.py reports set-aside
    items rather than shortening a list with no explanation.
    """
    match = match_region(query)
    if match is None:
        return None
    canonical_name, hebrew_names = match

    item_ids: list[str] = []
    not_found: list[str] = []
    for name in hebrew_names:
        result = _resolve_entity(name, ENTITY_CATALOG)
        top = result.candidates[0] if result.candidates else None
        if top is not None and top.confidence >= _REGION_MEMBER_MATCH_MIN_CONFIDENCE:
            item_ids.append(top.item_id)
        else:
            not_found.append(name)

    return {
        "region_name": canonical_name,
        "item_ids": item_ids,
        "members_not_in_eligible_set": not_found,
    }


# ── Neighborhood sub-records, grouped by parent locality ─────────────────
# See the module docstring: neighborhoods are NOT independently rankable
# items, they are sub-records of their parent locality, consumed by
# aggregate_records (app/tools.py) the same way the old airport-domain
# build's Anchorage departure-mix records were sub-records of ANC.
#
# `category` compares a neighborhood's own median 4-room price against its
# PARENT LOCALITY's own median 4-room price (LOCALITIES[parent]
# ["median_price_4room"], which is 104/104 present, so every priced
# neighborhood can be categorized). This never falls back to a
# neighborhood's parent price when the NEIGHBORHOOD's own price is
# missing -- that would misattribute a city-level number to one
# neighborhood, exactly the thing neighborhoods.json's own build already
# refused to do (see its `known_limitations`). A neighborhood with no
# price of its own simply has no record here at all; it is still counted
# in NEIGHBORHOOD_COUNTS' `total_neighborhoods` so the denominator stays
# honest.
_ABOVE_MEDIAN = "above_median"
_AT_OR_BELOW_MEDIAN = "at_or_below_median"
KNOWN_NEIGHBORHOOD_CATEGORIES: tuple[str, str] = (_ABOVE_MEDIAN, _AT_OR_BELOW_MEDIAN)


def _build_neighborhood_records() -> tuple[dict[str, dict[str, int]], dict[str, list[dict[str, Any]]]]:
    counts: dict[str, dict[str, int]] = {
        locality_id: {"total_neighborhoods": 0, "with_price_data": 0} for locality_id in LOCALITIES
    }
    records: dict[str, list[dict[str, Any]]] = {locality_id: [] for locality_id in LOCALITIES}

    for neighborhood_id, row in NEIGHBORHOODS.items():
        parent_id = row.get("parent_id")
        # Every neighborhood in this file belongs to an eligible locality
        # by construction (see neighborhoods.json's own _meta.description);
        # the `in counts` guard is defensive, not expected to ever be
        # False on real data, and costs nothing to keep.
        if parent_id not in counts:
            continue
        counts[parent_id]["total_neighborhoods"] += 1

        price = row.get("median_price_4room")
        if price is None:
            continue
        counts[parent_id]["with_price_data"] += 1

        city_price = LOCALITIES[parent_id].get("median_price_4room")
        if city_price is None:
            # Never observed in the real data (104/104 localities carry
            # this field) but guarded rather than assumed, since a KeyError
            # here would silently kill aggregate_records for every OTHER
            # locality if it ever did happen for one.
            continue

        category = _ABOVE_MEDIAN if price > city_price else _AT_OR_BELOW_MEDIAN
        records[parent_id].append(
            {
                "neighborhood_id": neighborhood_id,
                "name_he": row.get("name_he"),
                "median_price_4room": price,
                "room_bucket": row.get("median_price_room_bucket"),
                "category": category,
                # aggregate() sums this field; each record is one
                # neighborhood, so a constant 1.0 makes "share" a share BY
                # COUNT of neighborhoods, which is what "what share of X's
                # neighborhoods are above the city median" asks for -- not
                # a price-weighted share, which would answer a different
                # question nobody asked.
                "units": 1.0,
            }
        )

    return counts, records


NEIGHBORHOOD_COUNTS, RECORDS = _build_neighborhood_records()


def _category_semantics(locality_id: str) -> str:
    counts = NEIGHBORHOOD_COUNTS.get(locality_id, {"total_neighborhoods": 0, "with_price_data": 0})
    total = counts["total_neighborhoods"]
    priced = counts["with_price_data"]
    missing = total - priced
    name = LOCALITIES.get(locality_id, {}).get("name_en", locality_id)
    return (
        f"Categories are '{_ABOVE_MEDIAN}' and '{_AT_OR_BELOW_MEDIAN}', comparing each "
        f"neighborhood's own median 4-room price against {name}'s OWN city-wide median "
        "4-room price (not the national median). Some neighborhoods carry a "
        "median_price_room_bucket of 'all_rooms_fallback' rather than '4room' -- nadlan.gov.il "
        "had no 4-room-specific series for them, so their price reflects all room counts "
        "pooled together, a slightly different quantity than the city's own 4-room figure; "
        f"treat those comparisons as approximate. Of {name}'s {total} neighborhoods tracked "
        f"by nadlan.gov.il, {priced} have real price data and {missing} do not -- shares below "
        f"are computed over the {priced} priced neighborhoods ONLY; the {missing} without data "
        "are excluded from both the numerator and the denominator, never treated as zero or as "
        "'at or below'."
    )


CATEGORY_SEMANTICS: dict[str, str] = {locality_id: _category_semantics(locality_id) for locality_id in LOCALITIES}
