# Data

Everything here is built from **public, keyless Israeli government sources**.
No API key is needed to rebuild it, and none is needed to run the agent.

```bash
python data/refresh_data.py      # idempotent; re-run any time
```

`raw_data/` is exactly what was fetched, untouched. `processed_data/` is
everything derived from it, and **`app/dataset.py` only ever reads from
`processed_data/`** — that separation is what keeps every other module free of
disk access and testable without fixtures.

`raw_data/` is **gitignored**: it is ~24MB across ~1,570 upstream files, and it
is a cache, not source. `refresh_data.py` rebuilds it from the URLs below.
`processed_data/` **is** committed, so a fresh clone runs with no network.

---

## Sources

| Source | What it gives | Vintage |
|---|---|---|
| [`data.nadlan.gov.il/api/index/setl_types.json`](https://data.nadlan.gov.il/api/index/setl_types.json) | All 1,509 Israeli localities: name, settlement type, population | snapshot `20-08-2026` |
| `data.nadlan.gov.il/api/pages/settlement/buy/{code}.json` | Per locality: 22 quarterly price points (2021 Q1 → 2026 Q2) split by room count, local vs. national; rental yield; luxury index; centroid in ITM coordinates; the locality's neighborhood list | snapshot `20-08-2026` |
| `data.nadlan.gov.il/api/pages/neighborhood/buy/{nid}.json` | The same shape per neighborhood | snapshot `20-08-2026` |
| [CBS Socio-Economic Index, table A2](https://www.cbs.gov.il/he/mediarelease/DocLib/2024/230/24_24_230t2.xlsx) | Per local authority: socioeconomic cluster (1–10), index value, years of schooling, share with an academic degree, **average monthly income per capita** | **2021** |
| [CBS Peripherality Index, table 2](https://www.cbs.gov.il/he/mediarelease/doclib/2022/420/24_22_420t2.xlsx) | Per locality: peripherality index, sub-district (נפה), and the **official English locality name** | **2020** |
| [CBS Census 2022 via data.gov.il CKAN](https://data.gov.il/api/3/action/datastore_search?resource_id=38207cf8-afe2-48ed-a3b0-c8f70c796015) | Persons per household, per locality | **2022** |

All are Israeli government open data. nadlan.gov.il is the Israel Tax
Authority's public real-estate portal; the CBS tables are official published
statistical releases; data.gov.il is the national open-data platform. Check
each publisher's terms before redistributing the raw files.

### Two fetch quirks that will waste your afternoon

- **nadlan.gov.il JSON is UTF-8 *with BOM*.** Read it with
  `encoding='utf-8-sig'` or `json.load` fails on the first character.
- **`cbs.gov.il` resets the connection if the request carries a
  browser-spoofing `User-Agent`.** Send no custom UA to that host at all.
- **Do not use data.gov.il's `/download/*.csv` resource links** — they return
  403 from CloudFront, or 200 with a Google sign-in page. Use the CKAN
  `datastore_search` API instead, which works reliably and keylessly.

---

## Who gets ranked, and why

Of 1,509 localities, **104 are ranked.**

| Filter | Dropped |
|---|---|
| מושבים — co-op agricultural housing | 502 |
| שטחי הרשות — outside the Israeli housing market and its data | 298 |
| קיבוצים — co-op housing | 267 |
| Population below 5,000 | 264 |
| No real transaction data in the price feed | 73 |
| Settlement type missing in the source index | 1 |

Moshavim and kibbutzim are excluded because **co-op housing does not trade on
an open market** — membership committees, land-lease structures and association
rules mean a recorded price there is not comparable to a price in a town, so
ranking them side by side would be arithmetic on incommensurable numbers.

The population floor exists because a median built on a handful of deals a year
is noise wearing the costume of a statistic.

The remaining 1,405 localities are still queryable by name — you just cannot
get a *ranking* of them, which is the honest position given the data.

---

## Field coverage in `localities.json`

| Field | Coverage |
|---|---|
| `name_en`, `peripherality_index_2020`, `subdistrict` | 104 / 104 |
| `median_price_4room`, `price_momentum_vs_country`, distances | 104 / 104 |
| `socioeconomic_cluster`, `income_per_capita_monthly`, `income_per_household_monthly` | 102 / 104 |
| `rental_yield` | 98 / 104 |
| `household_size` | 104 / 104, **all measured** — the national-average fallback was never used |

**A missing value is absent from the row — never `0`, never guessed.**
`app/scoring.py` renormalizes each locality's weights across the criteria it
actually has, so a gap costs that locality one criterion instead of poisoning
the whole ranking. `household_size_source` marks measured vs. fallback per row
precisely so a real and an assumed value can never be silently blended.

`neighborhoods.json` holds 1,394 neighborhoods, of which **621 have a median
price**. Neighborhood-level data is genuinely much sparser than locality-level;
tools that read it report the real denominator rather than quietly ranking the
621 as if they were all of them.

---

## Limitations — read these before trusting a number

**There is no per-transaction data here, and that is not an oversight.** Every
price in this dataset is an **area-level median**, never a specific home. The
legacy transaction endpoint (`Nadlan.REST/Main/GetAssestAndDeals`) that most
public projects still reference is **dead** — it 302-redirects to a status
page. The current replacement (`api.nadlan.gov.il/deal-data`) is gated behind
reCAPTCHA Enterprise; prior art has reproduced its request signing exactly and
still fails at the token step. No attempt was made to defeat it.

**Vintage mismatch.** Prices are from August 2026. Income and socioeconomic
figures are from **2021**, peripherality from **2020**, household size from
**2022**. Income is nominal and is *not* inflated to 2026, so every
price-to-income figure this dataset produces **understates today's real
burden.** The affordability tool surfaces this in its own output rather than
burying it here.

**Income is per capita**, converted to a household figure by multiplying by
measured persons-per-household. That is an approximation: it assumes income
scales linearly with household size, which is wrong at the margins (children
earn nothing).

**No feasibility, no quality, no legal status.** Nothing here models whether a
specific home is worth buying — condition, floor, parking, building rights,
planning disputes, tax exposure. This ranks *areas* to narrow a search.

**A rejected source, recorded so it does not get re-added.** The data.gov.il
CKAN resource `7c860e04-9f8d-41c2-9f24-6249958d2081`
("אשכול חברתי כלכלי של יישובים ומועצות 2019") is real, keyless and working —
but despite its title it covers only localities belonging to a *regional
council*, i.e. almost exactly the moshavim and kibbutzim the gate excludes, and
contains no standalone city municipality at all. It joined **2 of 104**
eligible localities. It was replaced by the CBS 2021 table, which joins 102.

**Not used:** CBS building-starts (התחלות בנייה) is published only as monthly
PDF press releases with per-locality tables inside them — no API, no feed.
Extracting it would mean PDF table parsing for a signal this ranking does not
currently need.
