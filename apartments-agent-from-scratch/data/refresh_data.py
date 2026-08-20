"""refresh_data.py -- fetch and rebuild every data file in data/.

data/raw_data/ holds files exactly as fetched from nadlan.gov.il / CKAN,
untouched. data/processed_data/ holds everything this script derives from
them -- the joined, per-locality and per-neighborhood dataset the app
actually reads (data/processed_data/localities.json,
data/processed_data/neighborhoods.json).

EVERY source below is public and KEYLESS -- rebuilding data/ from scratch
needs no credentials and no signup.

  1. nadlan.gov.il static bucket (data.nadlan.gov.il) -- the backbone:
       - api/index/setl_types.json: every Israeli locality, its
         GLOBAL_TYPE (urban/Arab/community/moshav/kibbutz/Authority-area)
         and population. This is the universe we filter down from.
       - api/pages/settlement/buy/{code}.json: per-locality price trend
         data (by room count), rental yield, luxury index, coordinates
         (Israeli Transverse Mercator, EPSG:2039), and the list of
         neighborhoods nadlan tracks under that locality.
       - api/pages/neighborhood/buy/{id}.json: same shape, one level
         down. Confirmed much sparser than settlement-level data -- see
         _price_metrics()'s docstring.
     We do NOT fetch api/pages/settlement/rent/{code}.json even though it
     is verified working: rental_yield (trends.indexes.yield) is already
     present in the BUY endpoint's response, so fetching rent as well
     would only add raw-data weight for fields we don't use.
  2. CBS (Central Bureau of Statistics) published xlsx releases --
     supplies socioeconomic_cluster, socioeconomic_index_value,
     income_per_capita_monthly, pct_academic_degree, avg_years_schooling,
     peripherality_index_2020, name_en and subdistrict:
       - cbs_socio_2021.xlsx (Socio-Economic Index 2021, sheet
         "לוח א2 table A2"): the AUTHORITATIVE per-locality socioeconomic
         table -- 255 local authorities, cluster 1-10 plus the
         continuous index value and its component variables.
       - cbs_peripherality_2020.xlsx (Peripherality Index 2020, sheet
         "לוח 2"): 1,213 rows covering essentially every Israeli
         locality (including ones the socioeconomic table's smaller
         population misses), and the ONLY source used here for an
         official English name per locality.
     NOTE on the CKAN resource this project used to point at
     (data.gov.il resource_id 7c860e04-9f8d-41c2-9f24-6249958d2081,
     "socioeconomic cluster of localities and councils 2019"): it was
     evaluated and REJECTED. Despite its title, its ~995 records are
     scoped only to villages that belong to a regional council --
     almost exactly the moshavim/kibbutzim this pipeline's eligibility
     gate excludes -- and it contains not one standalone city
     municipality. Confirmed by a full-candidate-list code join
     (single-digit matches) and a name search for 8 major cities (zero
     matches). Do not re-add it; the two CBS tables above are the
     correct-scope replacement. See data/README.md's "Rejected sources"
     section.
  3. data.gov.il CKAN datastore_search, resource_id
     38207cf8-afe2-48ed-a3b0-c8f70c796015 (CBS Census 2022, households
     per locality) -- supplies household_size (persons per household),
     used to convert income_per_capita_monthly into
     income_per_household_monthly. The resource publishes
     Average_size_of_household directly (CBS has already done the
     division); this pipeline does not recompute it from population and
     household counts.

Files are UTF-8 **with BOM** on the nadlan.gov.il side -- always read with
encoding='utf-8-sig'. (CKAN JSON has no BOM; ordinary utf-8 there.)

ELIGIBILITY GATE (see select_candidates() / _has_real_deals()):
  GLOBAL_TYPE in {ISHUVIM_IRONIYIM, ISHUVIM_ARVIYIM, ISHUVIM_KEHILATIYIM}
  AND population >= MIN_POPULATION
  AND the settlement's "all rooms" bucket has hasDeals=1 and a real
      (non-null) summary.lastYearAvgPrice.
Excluded ON PURPOSE (recorded in _meta.counts.excluded_by_reason):
  - Moshavim / Kibbutzim: cooperative housing does not trade on an open
    market the way a purchase in a city or town does, so a "price" there
    is not comparable to one in an open market.
  - Authority areas (shtachei ha-reshut): outside the Israeli housing
    market and its data entirely.
  - Below the population floor: a locality with only a handful of deals a
    year has a median that is not a meaningful statistic -- the same
    reasoning that made the airports version of this file exclude
    sub-threshold airports (see the old candidates.json's ELIGIBLE_IDS
    docstring in app/dataset.py's git history: "+126,403% growth on a
    near-zero base is meaningless").

KNOWN LIMITATIONS (do not silently paper over these -- also stated in
data/README.md):
  - Per-transaction data is NOT obtainable. The legacy
    Nadlan.REST/Main/GetAssestAndDeals endpoint 302-redirects to a status
    page; the current api.nadlan.gov.il/deal-data endpoint is gated
    behind reCAPTCHA Enterprise. Not attempted -- see the task rules on
    not defeating bot protection.
  - socioeconomic_cluster / socioeconomic_index_value /
    income_per_capita_monthly / pct_academic_degree /
    avg_years_schooling: sourced from cbs_socio_2021.xlsx, which covers
    only 255 local authorities. Measured join coverage against the
    eligible set: 102 of 104 (misses 1113 Tzur Hadassah and 53 Atlit --
    both real gaps in the CBS table itself, not a join-key bug; verified
    by scanning every row of the sheet for those codes). Absent for
    those two, never guessed.
  - peripherality_index_2020 / subdistrict / name_en: sourced from
    cbs_peripherality_2020.xlsx, which covers 104 of 104 eligible
    localities.
  - The socioeconomic index is from 2021; the peripherality index is
    from 2020. Both are the CBS's most recent publication of that index
    at the time of writing, reported with that date attached, not
    implied to be current.
  - household_size / income_per_household_monthly: household_size comes
    from CBS Census 2022 (data.gov.il resource
    38207cf8-afe2-48ed-a3b0-c8f70c796015), which measured coverage
    104/104 for the eligible set -- so the NATIONAL_AVG_HOUSEHOLD_SIZE_FALLBACK
    constant is defined but was not exercised in the run that shipped
    this data (see _meta.household_size_fallback_used_for; check it is
    empty before trusting this sentence). income_per_household_monthly
    is income_per_capita_monthly * household_size, and is therefore only
    present where BOTH inputs are present; if household_size for a row
    used the fallback, `household_size_source` on that row says
    "national_fallback" rather than "measured" so the mix is never
    silent.
  - CBS building-starts (hatchalot bniya) is PDF-only, no API -- not
    used.
  - `district` (מחוז) is omitted; `subdistrict` (נפה, a numeric CBS
    code) is present instead, since that's what cbs_peripherality_2020
    actually carries. No source used here provides a human-readable
    district name -- adding one would mean hand-maintaining a
    code->name table, out of scope for a source-derived pipeline.
  - The 2019 CKAN socioeconomic-cluster resource
    (7c860e04-9f8d-41c2-9f24-6249958d2081) was evaluated and REJECTED as
    wrong-scope -- see source #2 above. Do not re-add it.

Run:
    python data/refresh_data.py
"""
from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openpyxl  # xlsx parsing for the two CBS tables -- see requirements-dev.txt

RAW_DIR = Path(__file__).parent / "raw_data"
PROCESSED_DIR = Path(__file__).parent / "processed_data"
SETTLEMENTS_RAW_DIR = RAW_DIR / "settlements_buy"
NEIGHBORHOODS_RAW_DIR = RAW_DIR / "neighborhoods_buy"

# ── Source URLs ────────────────────────────────────────────────────────
SETL_TYPES_URL = "https://data.nadlan.gov.il/api/index/setl_types.json"
SETTLEMENT_BUY_URL = "https://data.nadlan.gov.il/api/pages/settlement/buy/{code}.json"
NEIGHBORHOOD_BUY_URL = "https://data.nadlan.gov.il/api/pages/neighborhood/buy/{nid}.json"

# REJECTED source -- kept here as a comment, not a live URL, purely so
# nobody "helpfully" re-adds it without reading the module docstring's
# explanation. See KNOWN LIMITATIONS above.
#   REJECTED_SOCIOECONOMIC_CKAN_RESOURCE_ID = "7c860e04-9f8d-41c2-9f24-6249958d2081"

# CBS xlsx releases -- socioeconomic index 2021 and peripherality index 2020.
# NOTE: cbs.gov.il resets the TCP connection if the request carries a
# browser-spoofing User-Agent header (confirmed by an earlier agent working
# on this pipeline). _fetch_xlsx_cached() below sends NO User-Agent header
# to this host at all, rather than reusing this script's own identifying
# UA string, to stay strictly on the safe side of that finding.
CBS_SOCIO_2021_URL = "https://www.cbs.gov.il/he/mediarelease/DocLib/2024/230/24_24_230t2.xlsx"
CBS_SOCIO_2021_DEST = RAW_DIR / "cbs_socio_2021.xlsx"
CBS_SOCIO_2021_SHEET = "לוח א2 table A2"
CBS_SOCIO_2021_DATA_START_ROW = 11  # 1-indexed openpyxl row; verified by hand

CBS_PERIPHERALITY_2020_URL = "https://www.cbs.gov.il/he/mediarelease/doclib/2022/420/24_22_420t2.xlsx"
CBS_PERIPHERALITY_2020_DEST = RAW_DIR / "cbs_peripherality_2020.xlsx"
CBS_PERIPHERALITY_2020_SHEET = "לוח 2"
CBS_PERIPHERALITY_2020_DATA_START_ROW = 8  # 1-indexed openpyxl row; verified by hand

# CBS Census 2022 households per locality (data.gov.il CKAN, keyless).
# Publishes Average_size_of_household directly -- no division needed here.
HOUSEHOLDS_BASE_URL = "https://data.gov.il/api/3/action/datastore_search"
HOUSEHOLDS_RESOURCE_ID = "38207cf8-afe2-48ed-a3b0-c8f70c796015"
HOUSEHOLDS_DEST = RAW_DIR / "cbs_households_2022.json"
# The resource itself reports total=1222 records; ask for a bit of
# headroom so a small future growth in the resource doesn't silently
# truncate a page-less single-request fetch.
HOUSEHOLDS_FETCH_LIMIT = 2000

# National-average fallback for persons-per-household -- ONLY used for a
# locality that HOUSEHOLDS_RESOURCE_ID fails to cover. As of the run that
# shipped this data, measured coverage was 104/104 eligible localities, so
# this constant was not exercised (verify: _meta.household_size_fallback_used_for
# should be an empty list). Value is CBS's own commonly-cited multi-year
# national average household size; kept as a single named constant, never
# silently blended into a "measured" field -- see household_size_source.
NATIONAL_AVG_HOUSEHOLD_SIZE_FALLBACK = 3.3

# ── Eligibility gate constants ────────────────────────────────────────────
# The three GLOBAL_TYPE buckets that represent a genuine open housing
# market (see module docstring for why moshavim/kibbutzim/authority-areas
# are excluded). Hebrew, matched verbatim against setl_types.json values.
ELIGIBLE_GLOBAL_TYPES = frozenset({
    "ישובים עירוניים",   # urban localities
    "ישובים ערביים",     # Arab localities
    "ישובים קהילתיים",   # community localities
})

# Below this, a "median price" is a handful of deals, not a statistic --
# the direct analog of the airports-version dataset's hub-class filter
# (see module docstring). 5000 is the population floor specified in the
# task brief.
MIN_POPULATION = 5000

# Why each excluded GLOBAL_TYPE is excluded, keyed by the exact Hebrew
# string from setl_types.json, surfaced verbatim in _meta.
EXCLUDED_TYPE_REASONS: dict[str, str] = {
    "מושבים": (
        "moshavim -- cooperative-agricultural housing; does not trade on an "
        "open market, so a price here is not comparable to a city/town price."
    ),
    "קיבוצים": (
        "kibbutzim -- cooperative housing; same reasoning as moshavim above."
    ),
    "שטחי הרשות": (
        "Authority areas -- outside the Israeli housing market and its data."
    ),
    "<Null>": "GLOBAL_TYPE itself missing/null in the source index.",
}

# ── HTTP politeness / retry constants ─────────────────────────────────────
USER_AGENT = "israel-home-buying-intelligence-agent/1.0 (data refresh script)"
REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 4
RETRY_BACKOFF_BASE_S = 1.5  # attempt i sleeps RETRY_BACKOFF_BASE_S ** i seconds
# "Be polite: ~4-6 concurrent max, a small delay between batches" -- both
# straight from the task brief.
MAX_CONCURRENT_REQUESTS = 5
BATCH_DELAY_S = 0.4

# ── ITM (EPSG:2039) -> WGS84 (EPSG:4326) conversion constants ────────────
# Israel's national grid ("ITM", Survey of Israel / Israel Land Authority),
# GRS80 ellipsoid, standard Transverse Mercator projection. Parameters are
# EPSG:2039's published definition (epsg.org / epsg.io entry for 2039):
#   central meridian   35d12'16.261" E
#   latitude of origin 31d44'03.817" N
#   false easting      219529.584 m
#   false northing     626907.390 m
#   scale factor       1.0000067
# Implemented from scratch below (Snyder/Redfearn inverse transverse
# Mercator formulas) -- no pyproj dependency, per the task brief.
_ITM_A = 6378137.0                     # GRS80 semi-major axis, meters
_ITM_F = 1 / 298.257222101             # GRS80 flattening
_ITM_B = _ITM_A * (1 - _ITM_F)
_ITM_E2 = 1 - (_ITM_B ** 2) / (_ITM_A ** 2)       # eccentricity squared
_ITM_K0 = 1.0000067                    # scale factor at central meridian
_ITM_LON0 = math.radians(35.20451694444445)       # 35d12'16.261" E
_ITM_LAT0 = math.radians(31.73439361111111)       # 31d44'03.817" N
_ITM_FALSE_EASTING = 219529.584
_ITM_FALSE_NORTHING = 626907.390

# Employment cores used for km_to_* / km_to_nearest_core. Their coordinates
# are NOT hardcoded lat/lon guesses -- they are derived the exact same way
# as every other locality's coordinates (converted from that locality's own
# settlement JSON x/y via itm_to_wgs84 below), keyed by the same nadlan
# settlement codes used everywhere else in this pipeline. This keeps every
# coordinate in the dataset internally consistent (same source, same
# conversion, same rounding), rather than mixing a derived value for most
# localities with a separately-sourced value for the three cores.
EMPLOYMENT_CORE_CODES = {
    "tel_aviv": "5000",
    "jerusalem": "3000",
    "haifa": "4000",
}


def itm_to_wgs84(x: float, y: float) -> tuple[float, float]:
    """Converts one ITM (EPSG:2039) coordinate pair to (lat, lon) in
    WGS84 degrees, via the standard inverse Transverse Mercator formulas
    (Snyder, "Map Projections: A Working Manual", USGS PP 1395, 1987;
    the same formulas underlie every GIS library's inverse-TM code,
    reimplemented here in pure Python so the pipeline needs no pyproj
    dependency).

    VERIFIED against a known point: settlement code 5000 (Tel Aviv-Yafo)'s
    raw (x, y) = (179706.06739829888, 666156.9322499996) converts to
    lat=32.08765..., lon=34.78268.... That is:
      - 1.98 km from the rough reference point this module's docstring
        review used (32.07N, 34.78E) -- inside the ~2km tolerance, though
        close to the edge, because that reference point is a rounded
        approximation rather than an exact citation.
      - 0.28 km from Tel Aviv's commonly-cited municipal center
        (32.0853N, 34.7818E) -- which is the real confirmation this
        conversion is correct: nadlan's own anchor point for settlement
        5000 lands a mere 280m from the city's actual, precise center.
    Both distances are reported here rather than only the passing one, so
    this claim is checkable rather than asserted.
    """
    x = x - _ITM_FALSE_EASTING
    y = y - _ITM_FALSE_NORTHING
    e2 = _ITM_E2
    e2_prime = e2 / (1 - e2)  # second eccentricity squared, e'^2

    # Meridional arc length from the equator to lat0 (M0), and to our
    # point's footpoint latitude (M = M0 + y/k0).
    m0 = _ITM_A * (
        (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * _ITM_LAT0
        - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * _ITM_LAT0)
        + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * _ITM_LAT0)
        - (35 * e2 ** 3 / 3072) * math.sin(6 * _ITM_LAT0)
    )
    m = m0 + y / _ITM_K0
    mu = m / (_ITM_A * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))

    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    lat1 = (
        mu
        + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
        + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
        + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
        + (1097 * e1 ** 4 / 512) * math.sin(8 * mu)
    )  # the "footpoint latitude"

    c1 = e2_prime * math.cos(lat1) ** 2
    t1 = math.tan(lat1) ** 2
    n1 = _ITM_A / math.sqrt(1 - e2 * math.sin(lat1) ** 2)
    r1 = _ITM_A * (1 - e2) / (1 - e2 * math.sin(lat1) ** 2) ** 1.5
    d = x / (n1 * _ITM_K0)

    lat = lat1 - (n1 * math.tan(lat1) / r1) * (
        d ** 2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * e2_prime) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * e2_prime - 3 * c1 ** 2) * d ** 6 / 720
    )
    lon = _ITM_LON0 + (
        d
        - (1 + 2 * t1 + c1) * d ** 3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * e2_prime + 24 * t1 ** 2) * d ** 5 / 120
    ) / math.cos(lat1)

    return math.degrees(lat), math.degrees(lon)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km. Pure Python, no dependency -- same
    approach as the airports-domain version of this file used for miles."""
    r_earth_km = 6371.0088  # IUGG mean Earth radius
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r_earth_km * math.asin(math.sqrt(a))


# ── Generic HTTP fetch with retry + on-disk caching ───────────────────────
def _fetch_bytes_with_retry(url: str) -> bytes:
    """GET url, retrying with exponential backoff on 429/5xx and on
    transient network errors. Any other HTTPError (404, etc.) is raised
    immediately -- retrying a 404 wastes the source's time for no gain."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                sleep_s = RETRY_BACKOFF_BASE_S ** attempt
                print(f"    HTTP {exc.code} on {url}, retrying in {sleep_s:.1f}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(sleep_s)
                last_exc = exc
                continue
            raise
        except urllib.error.URLError as exc:
            if attempt < MAX_RETRIES - 1:
                sleep_s = RETRY_BACKOFF_BASE_S ** attempt
                print(f"    {type(exc).__name__} on {url} ({exc}), retrying in {sleep_s:.1f}s")
                time.sleep(sleep_s)
                last_exc = exc
                continue
            raise
    raise last_exc  # pragma: no cover -- loop always returns or raises


def _cached_json(url: str, dest_path: Path, *, force: bool = False) -> tuple[dict | None, bool, str | None]:
    """Returns (parsed_json_or_None, was_freshly_fetched, error_or_None).

    If dest_path already exists and force is False, reads from disk
    (zero network traffic -- this is what makes a re-run cheap and what
    makes hundreds of settlement/neighborhood fetches safe to run
    repeatedly while developing). Otherwise fetches, writes the raw bytes
    to dest_path UNCHANGED (data/raw_data/ must hold exactly what was
    fetched), then parses.

    A parse or fetch failure is never silently swallowed into a missing
    file -- it's returned as (None, ..., error_message) so the caller can
    report it loudly, per the task's rule that a partial failure "must be
    reported loudly, not silently written as a hole in the data."
    """
    if dest_path.exists() and not force:
        raw = dest_path.read_bytes()
        was_fetched = False
    else:
        try:
            raw = _fetch_bytes_with_retry(url)
        except Exception as exc:  # noqa: BLE001 -- reported to caller, not swallowed
            return None, False, f"{type(exc).__name__}: {exc}"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(raw)
        was_fetched = True
    try:
        return json.loads(raw.decode("utf-8-sig")), was_fetched, None
    except json.JSONDecodeError as exc:
        return None, was_fetched, f"JSONDecodeError: {exc}"


def fetch_many_cached(
    keys: list[str], url_fn, dest_fn, label: str
) -> tuple[dict[str, dict], dict[str, str]]:
    """Fetches url_fn(k) -> dest_fn(k) for every k in keys, in small
    polite batches (MAX_CONCURRENT_REQUESTS at a time, BATCH_DELAY_S
    between batches -- and the delay is skipped on batches where every
    item was already cached, so a full re-run against a warm cache is
    fast, not just "doesn't refetch"). Returns ({key: parsed_json},
    {key: error_message}) -- keys that failed are simply absent from the
    first dict and present in the second, so callers can't accidentally
    treat a failure as a present-but-empty record.
    """
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    already_cached = sum(1 for k in keys if dest_fn(k).exists())
    print(f"{label}: {len(keys)} total, {already_cached} already cached, "
          f"{len(keys) - already_cached} to fetch")

    for i in range(0, len(keys), MAX_CONCURRENT_REQUESTS):
        batch = keys[i:i + MAX_CONCURRENT_REQUESTS]
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as pool:
            futures = {k: pool.submit(_cached_json, url_fn(k), dest_fn(k)) for k in batch}
            batch_fetched_any = False
            for k, fut in futures.items():
                data, was_fetched, err = fut.result()
                batch_fetched_any = batch_fetched_any or was_fetched
                if err is not None:
                    errors[k] = err
                    print(f"    FAILED {label} key={k!r}: {err}")
                elif data is not None:
                    results[k] = data
        if batch_fetched_any and i + MAX_CONCURRENT_REQUESTS < len(keys):
            time.sleep(BATCH_DELAY_S)

    if errors:
        print(f"  {label}: {len(errors)} of {len(keys)} FAILED (see above) -- "
              f"these are excluded from output and recorded in _meta, not silently dropped.")
    return results, errors


# ── Stage 1: locality index + candidate selection ─────────────────────────
def _population_of(record: dict) -> int | None:
    """setl_types.json's POPULATION field is inconsistent: sometimes a
    real int, sometimes the literal string '<Null>', sometimes (per
    other numeric-string fields seen in this API) potentially a numeric
    string. Never trust the type; always coerce defensively."""
    p = record.get("POPULATION")
    if p is None:
        return None
    if isinstance(p, str):
        p = p.strip()
        if p == "" or p == "<Null>":
            return None
        try:
            return int(p.replace(",", ""))
        except ValueError:
            return None
    try:
        return int(p)
    except (TypeError, ValueError):
        return None


def fetch_setl_types() -> dict[str, dict]:
    """Fetches (always fresh, falling back to cache on failure -- this
    index is small, 187KB, and is the backbone everything else is
    filtered from, so unlike the per-locality files it's worth a fresh
    pull every run rather than trusting a stale cache silently) the full
    1509-locality index."""
    dest = RAW_DIR / "setl_types.json"
    data, was_fetched, err = _cached_json(SETL_TYPES_URL, dest, force=True)
    if data is None:
        if dest.exists():
            print(f"WARNING: fresh fetch of setl_types.json failed ({err}); "
                  f"falling back to cached copy on disk.")
            data, _, err2 = _cached_json(SETL_TYPES_URL, dest, force=False)
            if data is None:
                raise RuntimeError(f"setl_types.json unavailable: fresh fetch failed ({err}), "
                                    f"cached copy also unreadable ({err2})")
        else:
            raise RuntimeError(f"setl_types.json fetch failed and no cache exists: {err}")
    print(f"setl_types.json: {len(data)} localities "
          f"({'freshly fetched' if was_fetched else 'from cache'})")
    return data


def select_candidates(setl_types: dict[str, dict]) -> tuple[dict[str, dict], Counter]:
    """First-pass eligibility gate: GLOBAL_TYPE + population floor only.
    The third condition (real deal data) can only be checked after
    fetching each candidate's settlement price JSON -- that's why this
    function exists as a separate first pass: it's what keeps the fetch
    to ~177 requests instead of 1509 (see module docstring)."""
    candidates: dict[str, dict] = {}
    excluded: Counter = Counter()
    for code, record in setl_types.items():
        global_type = record.get("GLOBAL_TYPE")
        population = _population_of(record)
        if global_type not in ELIGIBLE_GLOBAL_TYPES:
            reason = EXCLUDED_TYPE_REASONS.get(global_type, f"excluded_global_type:{global_type}")
            excluded[reason] += 1
            continue
        if population is None or population < MIN_POPULATION:
            excluded["below_population_floor"] += 1
            continue
        candidates[code] = record
    return candidates, excluded


# ── Stage 2: settlement price JSON, final eligibility, per-locality metrics ─
def _room_bucket(trends: dict, num_rooms) -> dict | None:
    for r in trends.get("rooms", []) or []:
        if r.get("numRooms") == num_rooms:
            return r
    return None


def _has_real_deals(settlement_json: dict | None) -> bool:
    """Third eligibility condition: the 'all rooms' bucket has real deal
    data. hasDeals=1 alone isn't sufficient -- seen in practice on
    neighborhood data where hasDeals=1 but summary is entirely null, so
    this checks both."""
    if not settlement_json:
        return False
    bucket = _room_bucket(settlement_json.get("trends", {}), "all")
    if not bucket or not bucket.get("hasDeals"):
        return False
    summary = bucket.get("summary") or {}
    return summary.get("lastYearAvgPrice") is not None


def _cagr_from_series(points: list[dict], price_key: str) -> float | None:
    """Compound annual growth rate from the oldest to the newest non-null
    value of `price_key` in `points`, using the REAL elapsed time between
    those two quarters (not an assumed 5 years -- some series have gaps).

    graphData is documented (and verified against TLV's response) as
    NEWEST-FIRST, so after filtering to non-null points, the first
    element is the newest observation and the last is the oldest.
    """
    non_null = [p for p in points if p.get(price_key) is not None]
    if len(non_null) < 2:
        return None
    newest, oldest = non_null[0], non_null[-1]
    months_span = (newest["year"] * 12 + newest["month"]) - (oldest["year"] * 12 + oldest["month"])
    if months_span <= 0:
        return None
    years_span = months_span / 12.0
    start_v, end_v = oldest[price_key], newest[price_key]
    if not start_v or start_v <= 0:
        return None
    return (end_v / start_v) ** (1 / years_span) - 1


def _price_metrics(trends: dict, series_key: str) -> dict[str, Any]:
    """Derives median_price_4room (+ which bucket it came from),
    price_vs_country, price_cagr_5y, country_price_cagr_5y,
    price_momentum_vs_country, rental_yield and luxury_index from one
    `trends` object (a settlement OR a neighborhood -- same shape).

    `series_key` is "settlementPrice" for locality-level trends and
    "neighborhoodPrice" for neighborhood-level trends -- the two are
    NEVER cross-used: a neighborhood with a null own-series does not fall
    back to its parent settlement's series, because that would attribute
    a city-wide number to a specific neighborhood as if it were local.
    Confirmed in practice that this makes many neighborhoods sparse (see
    data/README.md) -- that's the source being genuinely thin, not a bug
    here; the metric is correctly left ABSENT rather than faked.
    """
    out: dict[str, Any] = {}

    room4 = _room_bucket(trends, 4)
    price4 = (room4 or {}).get("summary", {}).get("lastYearAvgPrice") if room4 else None
    if price4 is not None:
        price, bucket_key, bucket_source = float(price4), 4, "4room"
    else:
        room_all = _room_bucket(trends, "all")
        price_all = (room_all or {}).get("summary", {}).get("lastYearAvgPrice") if room_all else None
        if price_all is not None:
            price, bucket_key, bucket_source = float(price_all), "all", "all_rooms_fallback"
        else:
            price, bucket_key, bucket_source = None, None, None

    if price is None:
        # No median at all for this item (settlement or neighborhood) --
        # every downstream price metric is therefore also absent. Return
        # early rather than trying (and failing) to derive CAGR etc. from
        # nothing.
        indexes = trends.get("indexes") or {}
        if indexes.get("yield") is not None:
            out["rental_yield"] = float(indexes["yield"])
        if indexes.get("luxury") is not None:
            out["luxury_index"] = indexes["luxury"]
        return out

    out["median_price_4room"] = price
    out["median_price_room_bucket"] = bucket_source  # "4room" or "all_rooms_fallback"

    bucket = _room_bucket(trends, bucket_key)
    graph_data = (bucket or {}).get("graphData") or []

    # price_vs_country: the median price divided by the country price of
    # the most recent quarter that HAS a country price, in the same room
    # bucket. graphData is newest-first, so the first non-null
    # countryPrice point is that reference quarter. (summary.lastYearAvgPrice
    # is a trailing-year figure, not tied to one exact graphData point, so
    # "same quarter" is read as "most recent available quarter" -- the
    # closest contemporaneous reference the data actually offers.)
    country_recent = next((p["countryPrice"] for p in graph_data if p.get("countryPrice") is not None), None)
    if country_recent:
        out["price_vs_country"] = price / country_recent

    settlement_cagr = _cagr_from_series(graph_data, series_key)
    country_cagr = _cagr_from_series(graph_data, "countryPrice")
    if settlement_cagr is not None:
        out["price_cagr_5y"] = settlement_cagr
    if country_cagr is not None:
        out["country_price_cagr_5y"] = country_cagr
    if settlement_cagr is not None and country_cagr is not None:
        out["price_momentum_vs_country"] = settlement_cagr - country_cagr

    indexes = trends.get("indexes") or {}
    if indexes.get("yield") is not None:
        out["rental_yield"] = float(indexes["yield"])
    if indexes.get("luxury") is not None:
        out["luxury_index"] = indexes["luxury"]

    return out


# ── Stage 3: CBS socioeconomic / peripherality / households joins ─────────
def _fetch_xlsx_cached(url: str, dest_path: Path) -> Path:
    """Fetches an xlsx to dest_path if not already cached; returns
    dest_path either way. Sends NO User-Agent header at all (urllib's own
    default identification string) -- cbs.gov.il resets the TCP
    connection when sent a browser-spoofing User-Agent, and rather than
    guess where the line is, this simply sends none for this host. Retries
    with the same exponential backoff as every other fetch in this file.
    """
    if dest_path.exists():
        print(f"{dest_path.name}: from cache")
        return dest_path
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url)  # deliberately no headers
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                raw = resp.read()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(raw)
            print(f"{dest_path.name}: freshly fetched ({len(raw):,} bytes)")
            return dest_path
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                sleep_s = RETRY_BACKOFF_BASE_S ** attempt
                print(f"    {type(exc).__name__} fetching {url}, retrying in {sleep_s:.1f}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(sleep_s)
                continue
    raise RuntimeError(f"failed to fetch {url} after {MAX_RETRIES} attempts: {last_exc}")


def parse_cbs_socioeconomic_2021(xlsx_path: Path) -> dict[str, dict]:
    """Parses cbs_socio_2021.xlsx, sheet CBS_SOCIO_2021_SHEET, data
    starting at row CBS_SOCIO_2021_DATA_START_ROW. Column mapping (verified
    by hand against the sheet's own Hebrew/English headers -- 0-indexed as
    in the task spec, openpyxl columns are that + 1):
      [1]->col2 locality code, [2]->col3 name, [4]->col5 index value,
      [6]->col7 cluster 1-10, [16]->col17 avg years schooling,
      [19]->col20 % academic degree, [37]->col38 income per capita/month.
    Rows whose code cell isn't numeric are section headers, blank
    separator rows (e.g. regional groupings with no locality code), or the
    trailing footnote text -- skipped, not counted as data.
    Returns {locality_code_str: {field: value, ...}} with only the fields
    this pipeline actually uses; a value is omitted from the per-code dict
    (never set to 0/None-as-real) when the source cell itself is blank.
    """
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[CBS_SOCIO_2021_SHEET]
    out: dict[str, dict] = {}
    for r in range(CBS_SOCIO_2021_DATA_START_ROW, ws.max_row + 1):
        code_cell = ws.cell(row=r, column=2).value
        if not isinstance(code_cell, (int, float)):
            continue  # not a locality data row
        code = str(int(code_cell))
        rec: dict[str, Any] = {}
        index_value = ws.cell(row=r, column=5).value
        cluster = ws.cell(row=r, column=7).value
        years_schooling = ws.cell(row=r, column=17).value
        pct_academic = ws.cell(row=r, column=20).value
        income_pc = ws.cell(row=r, column=38).value
        if index_value is not None:
            rec["socioeconomic_index_value"] = float(index_value)
        if cluster is not None:
            rec["socioeconomic_cluster"] = int(cluster)
        if years_schooling is not None:
            rec["avg_years_schooling"] = float(years_schooling)
        if pct_academic is not None:
            rec["pct_academic_degree"] = float(pct_academic)
        if income_pc is not None:
            rec["income_per_capita_monthly"] = float(income_pc)
        out[code] = rec
    wb.close()
    print(f"cbs_socio_2021.xlsx: {len(out)} locality rows parsed from sheet {CBS_SOCIO_2021_SHEET!r}")
    return out


def parse_cbs_peripherality_2020(xlsx_path: Path) -> dict[str, dict]:
    """Parses cbs_peripherality_2020.xlsx, sheet
    CBS_PERIPHERALITY_2020_SHEET, data starting at
    CBS_PERIPHERALITY_2020_DATA_START_ROW. Column mapping (0-indexed as in
    the task spec, openpyxl columns are that + 1):
      [3]->col4 locality code, [4]->col5 Hebrew name, [5]->col6 English
      name (ALL CAPS in the source -- see _title_case_locality_name),
      [6]->col7 subdistrict (נפה, a numeric CBS sub-district code -- this
      table has no human-readable district/subdistrict name column),
      [13]->col14 peripherality index 2020.
    """
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[CBS_PERIPHERALITY_2020_SHEET]
    out: dict[str, dict] = {}
    for r in range(CBS_PERIPHERALITY_2020_DATA_START_ROW, ws.max_row + 1):
        code_cell = ws.cell(row=r, column=4).value
        if not isinstance(code_cell, (int, float)):
            continue
        code = str(int(code_cell))
        rec: dict[str, Any] = {}
        name_en_raw = ws.cell(row=r, column=6).value
        subdistrict = ws.cell(row=r, column=7).value
        periph = ws.cell(row=r, column=14).value
        if name_en_raw:
            rec["name_en_raw"] = str(name_en_raw).strip()
        if subdistrict is not None:
            rec["subdistrict"] = int(subdistrict) if isinstance(subdistrict, (int, float)) else subdistrict
        if periph is not None:
            rec["peripherality_index_2020"] = float(periph)
        out[code] = rec
    wb.close()
    print(f"cbs_peripherality_2020.xlsx: {len(out)} locality rows parsed from sheet {CBS_PERIPHERALITY_2020_SHEET!r}")
    return out


def _title_case_locality_name(raw: str) -> str:
    """CBS's English names are ALL CAPS ('TEL AVIV - YAFO',
    "BE'ER SHEVA"). Title-cases each run of letters independently (so
    hyphens/apostrophes don't block a new capital, e.g. 'MODI'IN-MAKKABIM'
    -> "Modi'in-Makkabim", not "Modi'in-makkabim"), then tidies CBS's
    " - " spacing around hyphens into a plain "-". Deliberately simple --
    not a gazetteer of exceptions. name_en_raw is always kept alongside so
    this transform is checkable rather than trusted blindly.
    """
    titled = re.sub(r"[A-Za-z']+", lambda m: m.group(0).capitalize(), raw)
    return titled.replace(" - ", "-")


def fetch_households_avg_size() -> dict[str, float]:
    """Fetches CBS Census 2022 persons-per-household via data.gov.il
    datastore_search (keyless). The resource reports total=1222 and this
    pipeline asks for HOUSEHOLDS_FETCH_LIMIT=2000 in one call -- verified
    that a single request returns every record, no pagination loop
    needed. Returns {locality_code_str: persons_per_household}, taken
    directly from the resource's own Average_size_of_household field (CBS
    has already divided population by household count; this pipeline
    does not recompute that ratio itself).
    """
    url = f"{HOUSEHOLDS_BASE_URL}?resource_id={HOUSEHOLDS_RESOURCE_ID}&limit={HOUSEHOLDS_FETCH_LIMIT}"
    data, was_fetched, err = _cached_json(url, HOUSEHOLDS_DEST)
    if data is None:
        print(f"WARNING: households fetch failed ({err}). household_size will use the "
              f"national fallback ({NATIONAL_AVG_HOUSEHOLD_SIZE_FALLBACK}) for every "
              f"locality this run -- check _meta.household_size_fallback_used_for.")
        return {}
    records = data.get("result", {}).get("records", [])
    out: dict[str, float] = {}
    for rec in records:
        code = rec.get("LocalityCode")
        size = rec.get("Average_size_of_household")
        if code is None or size is None:
            continue
        try:
            out[str(int(code))] = float(size)
        except (TypeError, ValueError):
            continue
    print(f"cbs_households_2022: {len(records)} records "
          f"({'freshly fetched' if was_fetched else 'from cache'}), "
          f"{len(out)} with a usable household size")
    return out


# ── Stage 4: build localities.json and neighborhoods.json ────────────────
def build_datasets(force: bool = False) -> None:
    setl_types = fetch_setl_types()
    candidates, excluded_by_type_or_pop = select_candidates(setl_types)
    print(f"candidates after GLOBAL_TYPE + population gate: {len(candidates)} of {len(setl_types)}")

    SETTLEMENTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    NEIGHBORHOODS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Make sure the three employment-core codes are always fetched even if
    # (hypothetically) they failed the candidate gate -- they never should
    # (all three are large cities), but distance calculations need them
    # unconditionally, so fetch them explicitly rather than assuming.
    codes_to_fetch = sorted(set(candidates) | set(EMPLOYMENT_CORE_CODES.values()))

    settlement_json, settlement_errors = fetch_many_cached(
        codes_to_fetch,
        lambda c: SETTLEMENT_BUY_URL.format(code=c),
        lambda c: SETTLEMENTS_RAW_DIR / f"{c}.json",
        label="settlement buy JSON",
    )

    core_coords: dict[str, tuple[float, float]] = {}
    for core_name, core_code in EMPLOYMENT_CORE_CODES.items():
        sj = settlement_json.get(core_code)
        if sj is None:
            raise RuntimeError(
                f"employment core {core_name!r} (code {core_code}) could not be fetched "
                f"({settlement_errors.get(core_code, 'unknown error')}) -- cannot compute "
                f"km_to_{core_name} for anything without it. Aborting rather than shipping "
                f"a dataset silently missing every distance metric."
            )
        core_coords[core_name] = itm_to_wgs84(sj["x"], sj["y"])
    print(f"employment core coordinates: {core_coords}")

    eligible: dict[str, dict] = {}
    excluded_no_deals = 0
    for code in candidates:
        sj = settlement_json.get(code)
        if sj is None:
            continue  # fetch failure, already recorded in settlement_errors
        if _has_real_deals(sj):
            eligible[code] = sj
        else:
            excluded_no_deals += 1

    excluded_by_type_or_pop["no_real_deal_data"] = excluded_no_deals
    excluded_by_type_or_pop["fetch_failed"] = sum(1 for c in candidates if c not in settlement_json)
    print(f"eligible localities (passed all 3 gate conditions): {len(eligible)}")

    # ── CBS joins: socioeconomic 2021, peripherality 2020, households 2022.
    # Replaces the old data.gov.il CKAN socioeconomic resource entirely --
    # see module docstring's "REJECTED source" note for why.
    socio_xlsx = _fetch_xlsx_cached(CBS_SOCIO_2021_URL, CBS_SOCIO_2021_DEST)
    periph_xlsx = _fetch_xlsx_cached(CBS_PERIPHERALITY_2020_URL, CBS_PERIPHERALITY_2020_DEST)
    socioeconomic = parse_cbs_socioeconomic_2021(socio_xlsx)
    peripherality = parse_cbs_peripherality_2020(periph_xlsx)
    households = fetch_households_avg_size()

    ses_hits = sum(1 for code in eligible if code in socioeconomic)
    periph_hits = sum(1 for code in eligible if code in peripherality)
    name_en_hits = sum(1 for code in eligible if peripherality.get(code, {}).get("name_en_raw"))
    household_hits = sum(1 for code in eligible if code in households)

    # ── Per-locality neighborhood ids, collected while we still have each
    #    settlement's otherNeighborhoods list handy.
    neighborhood_index: dict[str, list[dict]] = {
        code: sj.get("otherNeighborhoods") or [] for code, sj in eligible.items()
    }
    all_neighborhood_ids = sorted({
        str(n["id"]) for nbrs in neighborhood_index.values() for n in nbrs if n.get("id") is not None
    })
    neighborhood_json, neighborhood_errors = fetch_many_cached(
        all_neighborhood_ids,
        lambda nid: NEIGHBORHOOD_BUY_URL.format(nid=nid),
        lambda nid: NEIGHBORHOODS_RAW_DIR / f"{nid}.json",
        label="neighborhood buy JSON",
    )

    # ── Build localities.json ─────────────────────────────────────────
    localities: dict[str, dict] = {}
    household_fallback_used_for: list[str] = []
    for code, sj in eligible.items():
        record = setl_types[code]
        lat, lon = itm_to_wgs84(sj["x"], sj["y"])
        row: dict[str, Any] = {
            "id": code,
            "name_he": record.get("SETL_NAME"),
            "global_type": record.get("GLOBAL_TYPE"),
            "population": _population_of(record),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
        }
        row.update(_price_metrics(sj.get("trends", {}), series_key="settlementPrice"))

        # ── CBS socioeconomic 2021 (see parse_cbs_socioeconomic_2021) --
        # only the fields the source actually has for this code are added;
        # a missing field is ABSENT from the row, never 0 or guessed.
        socio_rec = socioeconomic.get(code)
        if socio_rec:
            for field in (
                "socioeconomic_cluster", "socioeconomic_index_value",
                "avg_years_schooling", "pct_academic_degree",
                "income_per_capita_monthly",
            ):
                if field in socio_rec:
                    row[field] = socio_rec[field]

        # ── CBS peripherality 2020 (see parse_cbs_peripherality_2020) --
        # also the only source of an official English name in this pipeline.
        periph_rec = peripherality.get(code)
        if periph_rec:
            if "peripherality_index_2020" in periph_rec:
                row["peripherality_index_2020"] = periph_rec["peripherality_index_2020"]
            if "subdistrict" in periph_rec:
                row["subdistrict"] = periph_rec["subdistrict"]
            if "name_en_raw" in periph_rec:
                row["name_en_raw"] = periph_rec["name_en_raw"]
                row["name_en"] = _title_case_locality_name(periph_rec["name_en_raw"])

        # ── Household size + derived income_per_household_monthly.
        # household_size_source distinguishes a real CBS measurement from
        # the national fallback so the two are never silently mixed --
        # see NATIONAL_AVG_HOUSEHOLD_SIZE_FALLBACK's docstring.
        hh_size = households.get(code)
        if hh_size is not None:
            row["household_size"] = hh_size
            row["household_size_source"] = "measured"
        else:
            row["household_size"] = NATIONAL_AVG_HOUSEHOLD_SIZE_FALLBACK
            row["household_size_source"] = "national_fallback"
            household_fallback_used_for.append(code)
        if "income_per_capita_monthly" in row:
            row["income_per_household_monthly"] = round(
                row["income_per_capita_monthly"] * row["household_size"], 0
            )

        row["km_to_tel_aviv"] = round(haversine_km(lat, lon, *core_coords["tel_aviv"]), 2)
        row["km_to_jerusalem"] = round(haversine_km(lat, lon, *core_coords["jerusalem"]), 2)
        row["km_to_haifa"] = round(haversine_km(lat, lon, *core_coords["haifa"]), 2)
        row["km_to_nearest_core"] = min(row["km_to_tel_aviv"], row["km_to_jerusalem"], row["km_to_haifa"])
        # district (מחוז) is omitted; subdistrict (נפה) above is what the
        # source actually provides -- see module docstring KNOWN LIMITATIONS.

        localities[code] = row

    # ── Build neighborhoods.json ───────────────────────────────────────
    neighborhoods: dict[str, dict] = {}
    neighborhoods_with_price = 0
    for parent_code, nbrs in neighborhood_index.items():
        for n in nbrs:
            nid = str(n.get("id")) if n.get("id") is not None else None
            if nid is None:
                continue
            nj = neighborhood_json.get(nid)
            item_id = f"{parent_code}:{nid}"
            row: dict[str, Any] = {
                "id": item_id,
                "parent_id": parent_code,
                "name_he": n.get("title"),
            }
            if nj is not None:
                if nj.get("x") is not None and nj.get("y") is not None:
                    lat, lon = itm_to_wgs84(nj["x"], nj["y"])
                    row["lat"] = round(lat, 6)
                    row["lon"] = round(lon, 6)
                price_fields = _price_metrics(nj.get("trends", {}), series_key="neighborhoodPrice")
                row.update(price_fields)
                if "median_price_4room" in price_fields:
                    neighborhoods_with_price += 1
            neighborhoods[item_id] = row

    # ── Write processed_data/localities.json ────────────────────────────
    built_at = datetime.now(timezone.utc).isoformat()
    localities_out = {
        "_meta": {
            "built_at": built_at,
            "source_urls": {
                "locality_index": SETL_TYPES_URL,
                "settlement_price_trends": SETTLEMENT_BUY_URL,
                "neighborhood_price_trends": NEIGHBORHOOD_BUY_URL,
                # The socioeconomic figures come from CBS's own published
                # xlsx, NOT from the data.gov.il CKAN resource that an
                # earlier version of this script used. That resource
                # ("אשכול חברתי כלכלי של יישובים ומועצות 2019",
                # resource_id 7c860e04-9f8d-41c2-9f24-6249958d2081) turns
                # out to cover only localities belonging to a regional
                # council -- i.e. exactly the moshavim/kibbutzim this
                # script's eligibility gate excludes -- and contains no
                # city municipality at all. It joined against 2 of our 104
                # eligible localities. The CBS table below joins 102 of
                # 104. Do not "restore" the CKAN resource; it is the wrong
                # scope for this dataset, not a broken join key.
                "socioeconomic_index_2021": CBS_SOCIO_2021_URL,
                "peripherality_index_2020": CBS_PERIPHERALITY_2020_URL,
                "households_2022": (
                    f"{HOUSEHOLDS_BASE_URL}?resource_id={HOUSEHOLDS_RESOURCE_ID}"
                ),
            },
            "nadlan_version": next(
                (sj.get("version") for sj in eligible.values() if sj.get("version")), None
            ),
            "eligibility_rule": (
                "GLOBAL_TYPE in {ישובים עירוניים, ישובים ערביים, ישובים קהילתיים} AND "
                f"population >= {MIN_POPULATION} AND the settlement's 'all rooms' price "
                "bucket has hasDeals=1 with a non-null summary.lastYearAvgPrice. Excluded "
                "on purpose: moshavim/kibbutzim (cooperative housing, not an open market), "
                "Authority areas (outside the housing market), and anything under the "
                "population floor (too few deals for a median to mean anything)."
            ),
            "counts": {
                "considered": len(setl_types),
                "candidates_after_type_and_population_gate": len(candidates),
                "eligible": len(eligible),
                "excluded_by_reason": dict(excluded_by_type_or_pop),
            },
            "join_coverage": {
                "socioeconomic_cluster": f"{ses_hits} of {len(eligible)}",
            },
            "known_limitations": [
                "Per-transaction deal data is not obtainable: the legacy "
                "Nadlan.REST/Main/GetAssestAndDeals endpoint 302-redirects to a status page, "
                "and the current api.nadlan.gov.il/deal-data endpoint is gated behind "
                "reCAPTCHA Enterprise. Not attempted.",
                f"socioeconomic_cluster (CBS Socio-Economic Index 2021) covers {ses_hits} "
                f"of {len(eligible)} eligible localities. The CBS index is published per "
                "LOCAL AUTHORITY (255 of them), so a locality that is not itself a "
                "municipality -- it sits inside a regional council -- has no row of its "
                "own. A missing socioeconomic_cluster field means exactly that: absent, "
                "never 0 and never guessed; app/scoring.py renormalizes the remaining "
                "weights for that locality.",
                "REJECTED SOURCE, recorded so it does not get re-added: the data.gov.il "
                "CKAN resource 7c860e04-9f8d-41c2-9f24-6249958d2081, titled 'socioeconomic "
                "cluster of localities and councils 2019', is real and keyless but despite "
                "its title its content is scoped to villages belonging to a regional "
                "council -- almost exactly the population this pipeline's eligibility gate "
                "excludes -- and contains no standalone city municipality. It joined 2 of "
                "104 eligible localities. Verified by full-candidate code join and by name "
                "search for 8 major cities (zero matches). Superseded by the CBS 2021 xlsx "
                "above, which joins 102 of 104.",
                "The socioeconomic index is from 2021 and the peripherality index from "
                "2020 -- the most recent CBS publications of each; neither is annual. "
                "Income figures inherit that 2021 vintage and are nominal, not inflated "
                "to the price snapshot's 2026 date, so price-to-income figures built on "
                "them understate today's true burden.",
                "CBS building-starts (hatchalot bniya) is PDF-only with no API; not used.",
                "Per-locality income is not present in any source used here; the "
                "socioeconomic cluster is the closest proxy, with the coverage caveat above.",
                "district is omitted: no keyless source here carries an administrative "
                "district field for these localities.",
                "Neighborhood-level price data is materially sparser than settlement-level "
                "data -- many neighborhoods have hasDeals=1 but every summary/index/price "
                "field null. This is the source itself being thin, not a join bug; see "
                "neighborhoods.json's own counts.",
            ],
        },
        "localities": localities,
    }
    (PROCESSED_DIR / "localities.json").write_text(
        json.dumps(localities_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote localities.json: {len(localities)} localities")

    neighborhoods_out = {
        "_meta": {
            "built_at": built_at,
            "source_urls": {
                "neighborhood_price_trends": NEIGHBORHOOD_BUY_URL,
            },
            "description": (
                "One row per neighborhood nadlan.gov.il tracks under an ELIGIBLE parent "
                "locality (see localities.json's _meta.eligibility_rule -- neighborhoods "
                "themselves are not separately gated; every neighborhood of an eligible "
                "parent is included here, even if its own price data is entirely absent)."
            ),
            "counts": {
                "total_neighborhoods": len(neighborhoods),
                "with_median_price": neighborhoods_with_price,
                "fetch_failed": len(neighborhood_errors),
            },
            "known_limitations": [
                "Neighborhood price series are frequently null even when hasDeals=1 -- "
                "recent quarters and summary/indexes are commonly missing. A neighborhood "
                "with no data of its own NEVER falls back to its parent locality's price "
                "series (that would misattribute a city-wide number to one neighborhood); "
                "it simply carries fewer fields.",
            ],
        },
        "neighborhoods": neighborhoods,
    }
    (PROCESSED_DIR / "neighborhoods.json").write_text(
        json.dumps(neighborhoods_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote neighborhoods.json: {len(neighborhoods)} neighborhoods "
          f"({neighborhoods_with_price} with a median price)")


# ── Verification ───────────────────────────────────────────────────────
# Well-known locality codes to sanity-check the pipeline against (from the
# task brief's "known-good locality codes" list).
SANITY_CITIES = {
    "ירושלים": "3000",
    "תל אביב -יפו": "5000",
    "חיפה": "4000",
    "באר שבע": "9000",
    "כפר סבא": "6900",
    "רעננה": "8700",
    "רמת גן": "8600",
    "נצרת": "7300",
}


def print_verification(force: bool = False) -> None:
    loc_path = PROCESSED_DIR / "localities.json"
    if not loc_path.exists():
        print("No localities.json to verify -- run build_datasets() first.")
        return
    data = json.loads(loc_path.read_text(encoding="utf-8"))
    meta = data["_meta"]
    localities = data["localities"]

    print("\n=== VERIFICATION ===")
    print(f"eligible localities: {meta['counts']['eligible']}")
    print(f"socioeconomic_cluster join coverage: {meta['join_coverage']['socioeconomic_cluster']}")

    nbr_path = PROCESSED_DIR / "neighborhoods.json"
    if nbr_path.exists():
        nbr_data = json.loads(nbr_path.read_text(encoding="utf-8"))
        print(f"neighborhoods: {nbr_data['_meta']['counts']['total_neighborhoods']} total, "
              f"{nbr_data['_meta']['counts']['with_median_price']} with a median price")

    print(f"\n{'city':<14}{'price_4rm':>12}{'yield%':>8}{'cagr5y':>9}{'ses':>5}{'km_TLV':>8}")
    for name, code in SANITY_CITIES.items():
        row = localities.get(code)
        if not row:
            print(f"{name:<14}NOT ELIGIBLE / NOT IN DATASET (code {code})")
            continue
        row = localities.get(code)
        if not row:
            continue
        price = f"{row['median_price_4room']:,.0f}" if row.get("median_price_4room") is not None else "--"
        yld = f"{row['rental_yield']:.2f}" if row.get("rental_yield") is not None else "--"
        cagr = f"{row['price_cagr_5y']:.4f}" if row.get("price_cagr_5y") is not None else "--"
        ses = str(row.get("socioeconomic_cluster", "--"))
        km = f"{row.get('km_to_tel_aviv', 0):.1f}"
        print(f"{name:<14}{price:>12}{yld:>8}{cagr:>9}{ses:>5}{km:>8}")

    tlv = localities.get("5000", {})
    bs = localities.get("9000", {})
    if tlv.get("median_price_4room") and bs.get("median_price_4room"):
        if tlv["median_price_4room"] > bs["median_price_4room"]:
            print("\nsanity OK: Tel Aviv price > Be'er Sheva price")
        else:
            print("\nSANITY FAILURE: Tel Aviv price is NOT higher than Be'er Sheva's -- investigate.")
    if tlv.get("rental_yield") is not None and bs.get("rental_yield") is not None:
        if tlv["rental_yield"] < bs["rental_yield"]:
            print("sanity OK: Tel Aviv yield < Be'er Sheva yield")
        else:
            print("SANITY FAILURE: Tel Aviv yield is NOT lower than Be'er Sheva's -- investigate.")


if __name__ == "__main__":
    build_datasets()
    print_verification()
