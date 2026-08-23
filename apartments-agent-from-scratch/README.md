# Israel Home-Buying Intelligence Agent

A worked example of a **"simple" tool-calling AI agent built without a
framework**. It helps you work out where in Israel a home is good value to buy
and live in — ranking localities and neighborhoods on **deterministic,
inspectable scoring logic**, and using an LLM only to choose tools, work out
what you meant, and explain numbers it never computed.

The domain is real and the data is real: 104 Israeli localities and 1,394
neighborhoods, built from the Tax Authority's own price feed and CBS's
published statistics.

**The headline finding is an unflattering one, which is why it's worth
stating.** Under the default weights the famous expensive cities lose badly —
**Tel Aviv ranks 63rd of 104, Jerusalem 102nd** — and the ranking is
**fragile**: all five criteria flip the top spot on a weight change of 15% or
less, four of them at 5%. The top two localities are 0.17% apart, which is
noise. The agent is built to say that out loud rather than present a confident
winner. See [`DESIGN.md`](DESIGN.md) §3.

## Run it

```bash
git clone <this repo>
cd apartments-agent-from-scratch
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env
```

Add your key to `.env`: `OPENAI_API_KEY=sk-...`. `LLM_PROVIDER=openai` is
already set — that's the only required edit. Then:

```bash
.venv/bin/uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000**.

> **No key, or want to see it run first?** `LLM_PROVIDER` defaults to `mock`,
> so `.venv/bin/pytest -q` (391 tests) and `.venv/bin/python -m app.cli` both
> work with zero setup, no network and no key. `anthropic` and `groq` (free
> key, no credit card) are drop-in alternatives — set the provider and its key,
> nothing else changes.

See the four question shapes end to end without opening the UI:

```bash
.venv/bin/python -m scripts.run_example_questions
```

## Layout

```
app/        the agent: loop, tools, scoring, Hebrew, entity resolution, web + voice
data/       the committed dataset app/dataset.py reads — rebuild with refresh_data.py
evals/      runnable eval harness — 24 seeded failure-mode tasks
tests/      pytest suite (391 tests), runs with zero keys
static/     the web UI — chat, live tool-call log, voice controls
scripts/    sensitivity report, resolver calibration, example questions, smoke test
```

- [`DESIGN.md`](DESIGN.md) — scoring methodology, where AI is used and where it
  isn't, and the tradeoffs.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — file-by-file map and how a
  request flows.
- [`data/README.md`](data/README.md) — provenance, the eligibility gate, and
  every known gap.

## The four question shapes

The tool surface looks the way it does because these are four genuinely
different questions, and only two of them are rankings:

| Shape | Example | Primitive |
|---|---|---|
| Filtered ranking | "Which places within 30 km of Tel Aviv are good value on a 2.5M ₪ budget?" | `find_items` → `compare_items` |
| Comparison, ambiguous name | "Compare Modi'in and Ramat Gan" | `resolve_entity` → `compare_items` |
| Single-entity aggregate | "What share of Tel Aviv's neighborhoods are above the city median?" | `aggregate_records` |
| Derived quantity + cause | "What income do I need to buy a 4-room flat in Be'er Sheva, and why?" | `estimate_derived_metric` |

The third and fourth are the interesting ones. A composite score cannot tell
you what share of one city's neighborhoods beat its own median, and "the income
you need" exists in no dataset — it has to be modelled from price, the Bank of
Israel's LTV caps, a mortgage term and a repayment ceiling. Both needed their
own code paths, deliberately separate from `scoring.py`.

## Hebrew

The data is Hebrew; the conversation usually isn't. Israeli place names arrive
in Hebrew, in English, or in one of several competing transliterations, and the
official spelling is frequently not the one anyone types.

`app/hebrew.py` strips niqqud, folds final letters (ם→מ, ן→נ, ץ→צ, ף→פ, ך→כ),
folds geresh/gershayim/maqaf, and **romanizes** — so the existing
Jaro-Winkler + Soundex machinery, which only works on Latin text, can be reused
instead of writing a Hebrew phonetic algorithm. `Kfar Saba`, `Kefar Sava` and
`כפר סבא` all land on locality 6900.

Ambiguity is preserved rather than papered over. `resolve_entity("Modiin")`
returns **`decisive: false`** with both Modi'in-Makkabbim-Re'ut and Modi'in
Illit — two genuinely different towns — and the system prompt forbids the model
from quietly picking one.

The UI is English-first but handles bidirectional text properly: a
predominantly-Hebrew message flips only its own bubble to RTL, inline Hebrew
names inside English prose are wrapped in `<bdi>` so trailing punctuation stays
put, and the tool-call log stays LTR so JSON structure remains readable.

**Voice is English only.** Hebrew is supported as typed text throughout; no
Hebrew speech model is wired up. That's a scope line, stated rather than hidden.

## Where the LLM is, and where it is not

| Decision | Made by |
|---|---|
| Which tool to call, with what arguments | **LLM** |
| What the user meant by an ambiguous name | Deterministic resolver proposes with confidence; **LLM** picks or asks |
| Every score, weight, normalization, ranking, aggregate, mortgage figure | **Python** — never the LLM |
| The wording of the explanation | **LLM**, constrained to numbers the tools returned |

The system prompt forbids the model from computing any number itself, and every
tool returns a per-component breakdown (raw value, normalized score, weight,
contribution) so it always has the arithmetic in hand and never needs to invent
it. Both rules are named constants (`NEVER_COMPUTE_RULE`,
`NEVER_INVENT_IDS_RULE`) so tests can assert they actually reach the model,
rather than living in a README where they can drift.

## Why no framework

The agent loop is a plain `while` loop over one provider-agnostic `chat()`
call, with tool dispatch, guardrail wrapping and turn-limiting done by hand in
about seventy lines.

- **Control.** The three things most worth being able to point at — guaranteed
  termination, treating tool output as untrusted, and the reasoning trace — all
  live in that loop. Under a framework they'd be configuration spread across
  someone else's abstractions.
- **Debuggability.** When a tool call goes wrong, the stack trace is seventy
  lines of local code.
- **Honesty.** At this size a framework buys orchestration features this agent
  doesn't use, in exchange for a dependency whose behaviour I'd be guessing at.

This would change with multi-agent handoff, durable state, or human review
queues. It hasn't yet.

## Guardrails — what this is and isn't

Every tool result is wrapped in an explicit `<untrusted_data>` fence before it
enters the message history, and scanned by a deterministic regex pre-filter for
common injection phrasings. Both are tested.

**What it is:** the minimum "tool output is data, not instructions" discipline,
made visible, logged and testable. **What it isn't:** a classifier, an
allow-listed tool surface, or output-side data-loss checks. A production
version would add all three.

## Evals

```bash
.venv/bin/python -m evals.run_evals                    # mock provider, no key
.venv/bin/python -m evals.run_evals --provider openai  # real key from .env
```

24 seeded tasks across correctness, self-computation, entity resolution,
missing data, injection, explanation quality, uncertainty honesty and scope,
with deterministic graders plus an LLM judge.

**The mock-provider pass rate is 38% (9/24) and that number is not a quality
signal.** The mock is a scripted stub with fixed tool-routing; the failures
trace to that scripting, or to the judge correctly reporting "skipped" when no
key is present. The six pure-code tasks with no model in the loop pass
unconditionally. A meaningful rate needs a real provider. The tasks were
deliberately *not* tuned until the mock passed — that would have produced a
flattering number that measured nothing.

## Scope and limitations

[`data/README.md`](data/README.md) is the canonical list. The ones that matter
most:

- **This is not financial advice, and the agent says so.** It is an area-level
  screening tool: it narrows a search, it does not decide a purchase.
- **No per-transaction data.** Every price is an area-level median, never a
  specific home. The legacy nadlan transaction endpoint is dead and the current
  one is behind reCAPTCHA Enterprise; defeating that was not attempted.
- **Vintage mismatch.** Prices are August 2026; income and socioeconomic
  figures are 2021, peripherality 2020. Income is nominal, so every
  price-to-income figure **understates today's real burden**. The affordability
  tool surfaces this in its own output.
- **The ranking is fragile.** See the top of this file and `DESIGN.md` §3.
- Moshavim and kibbutzim are excluded: co-op housing doesn't trade on an open
  market, so its prices aren't comparable. So are localities under 5,000 people.
- Nothing models a specific property — condition, floor, parking, building
  rights, legal status, or tax exposure.
- Neighborhood data is much sparser than locality data: 621 of 1,394 have a
  price. Tools report the real denominator.
- No persistent multi-user history or auth — in-memory, single session.

## Tests

```bash
.venv/bin/pytest -q     # 391 passing
```

No network, no API key, and no mocking of the pure modules — `scoring.py`,
`analytics.py`, `affordability.py`, `hebrew.py` and `entity_resolution.py` are
pure functions and are tested as such. Two test files exercise the shipped
JavaScript under `node` and **skip cleanly if `node` isn't installed**, so
`pytest` stays green on a bare clone with zero setup.
