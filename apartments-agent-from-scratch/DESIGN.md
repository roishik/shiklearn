# Design

Scoring methodology, where AI is used and where it isn't, and the tradeoffs.
For the file-by-file map see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md);
for data provenance see [`data/README.md`](data/README.md).

---

## 1. What the agent does

It answers questions about where in Israel a home is good value to buy and live
in, over 104 localities and 1,394 neighborhoods.

The framing problem this had to solve: **"attractive" is not a measurable
quantity.** A place can be attractive because it's cheap, because it's nice,
because it's close to work, or because it's about to become any of those. Those
pull in different directions, and any single number that claims to combine them
is making a judgement call. The design response is not to hide the judgement
call but to make it explicit, weighted, inspectable, and — the part that
matters — **testable for how much it actually depends on those weights**.

So the score answers one specific question: *what does a place cost, against
what you get for it?* It is deliberately **not** an investment-return score and
**not** a price forecast.

### The four question shapes

Only two of the four are rankings. That is why the tool surface isn't just a
scorer:

| Shape | Primitive | Why it needed its own path |
|---|---|---|
| Filtered ranking | `find_items` → `compare_items` | The general case |
| Comparison with an ambiguous name | `resolve_entity` → `compare_items` | "Modi'in" is two towns |
| Single-entity aggregate | `aggregate_records` | A composite score cannot tell you what share of one city's neighborhoods beat its own median |
| Derived quantity + cause | `estimate_derived_metric` | "The income you need" exists in no dataset — it must be modelled |

---

## 2. Scoring methodology

### Criteria and weights

| Criterion | Weight | Raw value | Direction |
|---|---|---|---|
| `price_level` | 30 | Median 4-room price | lower is better |
| `accessibility` | 20 | km to nearest of Tel Aviv / Jerusalem / Haifa | lower is better |
| `price_momentum_vs_country` | 20 | Local 5y CAGR − national 5y CAGR | higher is better |
| `socioeconomic_level` | 15 | CBS cluster 1–10 | higher is better |
| `rental_yield` | 15 | nadlan rental yield % | higher is better |

### The weights were set by the measured correlations, not by taste

Measured pairwise Pearson *r* across the eligible 104 (pairwise-complete):

|  | price_level | socioecon | accessibility | momentum | rental_yield |
|---|---|---|---|---|---|
| **price_level** | 1.000 | **0.666** | −0.371 | −0.003 | −0.391 |
| **socioecon** | 0.666 | 1.000 | −0.168 | −0.097 | −0.158 |
| **accessibility** | −0.371 | −0.168 | 1.000 | 0.020 | 0.405 |
| **momentum** | −0.003 | −0.097 | 0.020 | 1.000 | −0.195 |
| **rental_yield** | −0.391 | −0.158 | 0.405 | −0.195 | 1.000 |

My first draft had `socioeconomic_level` at 20 and `price_momentum` at 15. The
matrix said that was wrong, and the weights moved:

- **`socioeconomic_level` 20 → 15.** It correlates **r = 0.666** with
  `price_level`. Expensive places have richer residents — that is close to a
  tautology. At weight 20 alongside price's 30, half the total weight would
  have rested on two substantially overlapping criteria, and the score would
  have been measuring "expensive" twice while appearing to measure two things.
  It is held at 15 **and disclosed**, not dropped: the residual variance after
  that 0.666 is exactly where the useful answers live — somewhere cheap that is
  *also* a good place to live is precisely the thing worth finding.
- **`price_momentum_vs_country` 15 → 20.** It is the only genuinely
  decorrelated dimension in the set (|r| ≤ 0.10 against every other criterion).
  Weight moved from the criterion that duplicates price to the one that adds
  independent information.
- **`rental_yield` stays 15.** It does partly re-express `price_level`
  (r = −0.391). It earns its place as the opportunity-cost check — whether the
  price is supported by what the home is worth to *use* — but not more than 15,
  because it pushes toward "cheap" in the same direction `price_level` already
  does.

`price_momentum` carries an honest tension worth naming: sustained relative
appreciation is the market's own verdict that people with choices are moving
in, but it also means you are buying in later. It is included as evidence about
the place, not as a promise about the price.

### Normalization and missing data

Each raw value is min-max normalized into [0,1] against the **5th/95th
percentile of the eligible 104**, measured 2026-08-20 and **frozen as
constants**. Bounds that recompute per query would make two runs silently
incomparable — a ranking has to be reproducible. Values outside the window
clamp rather than escaping [0,1].

| Criterion | p5 | p95 | n |
|---|---|---|---|
| `price_level` | 757,500 | 3,467,262.5 | 104 |
| `accessibility` | 4.617 | 69.7465 | 104 |
| `price_momentum_vs_country` | −0.038353 | 0.053857 | 104 |
| `socioeconomic_level` | 2.0 | 9.0 | 102 |
| `rental_yield` | 2.1325 | 3.776 | 98 |

Real public data is ragged. `socioeconomic_cluster` is absent for 2 localities
and `rental_yield` for 6. Rather than failing the whole locality on any single
gap, `score_item()` **drops the absent criterion and renormalizes the weights
over the ones actually present**, reporting `covered_weight` so a partial score
is never mistaken for a complete one. It still fails loudly below a coverage
floor: a score built on scraps is worse than no score. The floor must be
*exceeded*, not merely met — enough evidence means most of the evidence, and
half is not most.

### Where the eligibility gate actually lives

Two places, and the distinction matters:

- **In the pipeline (build time).** `data/refresh_data.py` decides who is
  rankable at all: urban/Arab/community localities, population ≥ 5,000, with
  real deal data. `localities.json` therefore contains *only* the 104 eligible.
- **In the tools (query time).** All four ranking tools —`compare_items`,
  `rank_by_priorities`, `analyze_weight_sensitivity`, `weight_robustness_report`
  — refuse neighborhood ids and unknown ids, with the reason. A neighborhood has
  no socioeconomic or income data of its own, so ranking one against a city
  would either fabricate city-level data as the neighborhood's own or score it
  on a sliver of the criteria.

The gate is enforced in **all four** ranking tools, not just `compare_items`,
and there is a parametrized test proving it. In the project this architecture
came from, that gate lived only in `compare_items` and the three sibling tools
reproduced the exact bug it had been written to prevent.

---

## 3. How much do the weights actually matter? — the fragility finding

Full report: [`evals/results/sensitivity_20260823T102830Z.md`](evals/results/sensitivity_20260823T102830Z.md),
regenerate with `scripts/sensitivity_report.py`.

**The famous expensive cities lose, decisively.** Tel Aviv-Yafo ranks **63rd of
104** (0.5088), Herzliya 76th, Ra'anana 85th, Bnei Brak 98th, **Jerusalem 102nd**
(0.3790). The top is Qiryat Motzkin (0.7257), Judeide-Maker, Qiryat Bialik,
Shelomi, Qiryat Atta — the Krayot and the northern periphery. That is the score
working as designed: it rewards what you get for what you pay, and Tel Aviv's
price is not remotely offset by its socioeconomic cluster or its 2.72% yield.

**And the ranking is fragile. All five criteria flip the #1 spot on a weight
change of 15% or less — four of them at 5%:**

| Criterion | Weight | Flips #1 at | To |
|---|---|---|---|
| `price_level` | 30 | +5% | Judeide-Maker |
| `accessibility` | 20 | −5% | Judeide-Maker |
| `socioeconomic_level` | 15 | −5% | Judeide-Maker |
| `rental_yield` | 15 | +5% | Judeide-Maker |
| `price_momentum_vs_country` | 20 | −15% | Judeide-Maker |

Under a fixed ±50% probe applied to each criterion in turn, Kendall's tau
against the default ranking falls as low as **0.631** and averages **0.809** —
a plausible weight error doesn't nudge the ranking, it reshuffles it.

And the top two are **0.17% apart** (0.7257 vs 0.7245) — well inside
`DECISIVE_SCORE_GAP`. That is not a difference; it is noise relative to the
judgement baked into the weights.

**This is a finding, not a defect, and the agent is built to say it.**
`compare_items` returns `decisive: false` with a `tied_at_top` list, and system
prompt rule 6 requires the model to present tied places as tied and explain
what separates them qualitatively rather than crowning a winner. An honest
"these two are effectively tied, and here's what actually distinguishes them"
is a better answer than a confident ranking the arithmetic doesn't support.

---

## 4. Where and how AI is used

| Decision | Made by | Enforced how |
|---|---|---|
| Which tool to call, with what arguments | LLM | Tool schemas |
| What an ambiguous name means | Resolver proposes with confidence; LLM picks or asks | `decisive` flag + `NEVER_INVENT_IDS_RULE` |
| Every score, weight, normalization, rank, aggregate, mortgage figure | **Python only** | `NEVER_COMPUTE_RULE`; every tool returns a full breakdown so there is nothing left to compute |
| Wording of the explanation | LLM | Constrained to tool-returned numbers |

Both load-bearing rules are **named constants** in `app/system_prompt.py`, so
tests assert they actually reach the model instead of trusting a README.

**Model choice:** `gpt-4o-mini` by default. The model here does tool selection,
argument extraction and constrained explanation — not arithmetic and not
judgement. That is well within a small model's competence, and the cost
difference is large. `anthropic` and `groq` are drop-in via one env var.

### Voice

Two paths, chosen by whether a key exists. **Without a key**: browser-native
Web Speech dictation and read-aloud — no server, no cost, and the controls
disable themselves with an explanation on Firefox rather than failing on click.
**With `OPENAI_API_KEY`**: conversation mode — open mic, local energy-based
endpointing, server-side STT/TTS, and real barge-in.

A spoken turn goes through the **same `/chat/stream` endpoint as a typed one**:
same tools, same guardrails, same scoring, same tool-call log. Voice is a
modality, not a second agent.

Barge-in does three things and the third is the one that matters: it stops
playback, abandons what wasn't played, and then **rewrites the stored reply to
only the sentences actually heard** (`app/conversation.py`). Without that last
step the model believes it said things you never heard, and every later answer
is built on a conversation that didn't happen.

Known limits, stated: transcription is per-utterance, not streaming per token;
endpointing is energy-based, so it can't distinguish an interruption from a
backchannel — an "mm-hmm" will stop it. **Voice is English-only** while Hebrew
text is fully supported; wiring Hebrew speech was a scope decision, not an
oversight.

---

## 5. Key tradeoffs

**Area-level medians, not transactions.** Not a choice — the legacy nadlan
transaction endpoint is dead and the current one sits behind reCAPTCHA
Enterprise. Defeating it wasn't attempted. The consequence is real: this tool
can compare places, never homes.

**Snapshotted data with stated staleness, not live.** Prices are August 2026,
income and socioeconomic 2021, peripherality 2020. The mismatch is
one-directional and disclosed: nominal 2021 income against 2026 prices
**understates** today's burden. The one live call, mortgage rates, is
deliberately kept **outside** the scored path — a rate change moves what every
buyer can afford everywhere at once, so it says nothing about whether one
locality beats another.

**Income per capita scaled to household.** CBS publishes per capita; mortgage
underwriting works on household. Multiplying by measured persons-per-household
assumes income scales linearly with household size, which is wrong at the
margins. Stated rather than smoothed over.

**Socioeconomic cluster as an income proxy in the score.** The score uses the
1–10 cluster rather than raw income, because the cluster is a composite of
education, employment and standard of living that is more stable than a single
nominal income figure five years stale. The raw income is used where it belongs
— in the affordability calculation.

**Deterministic scoring, not an LLM judgement.** An LLM asked to rank 104
localities would produce a plausible ordering that cannot be reproduced,
audited, or sensitivity-tested. Everything in §3 is only possible because the
scorer is a pure function.

**Regex guardrails, not a classifier.** Catches the common phrasings, is fast
and testable, and misses paraphrased attacks. A production version adds a
model-based detector, an allow-listed tool surface, and output-side checks.

**Hand-rolled loop, no framework.** See the README.

### Deliberately not built

Property-level valuation; tax modelling (mas rechisha, betterment tax);
commute times by actual transit rather than great-circle distance; school
quality per locality; crime data; planning-pipeline supply, which exists as an
Israel Land Authority feed and would be the most valuable *next* criterion;
multi-user auth or persistent history; Hebrew speech.

The most defensible improvement is not another criterion. It is **user-weight
elicitation** — given §3, what the ranking most needs is not better weights but
the user's own, and `rank_by_priorities` is the beginning of that rather than
the end.
