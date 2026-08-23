# Architecture

A file-by-file map. Navigation, not persuasion — if you want to know why a
decision was made, see `DESIGN.md`. If you want to know what a file does
and where to find a given piece of behaviour, it's here.

---

## 1. How a request flows

### A typed question

```
browser (static/index.html)
  -> POST /chat/stream                         app/main.py
  -> run_agent()                                app/agent_loop.py
       -> provider.chat(messages, tools)         app/providers/llm/*
       -> tool dispatch via TOOL_REGISTRY         app/tools.py
            -> data lookup                         app/dataset.py
            -> pure computation                     app/scoring.py / analytics.py /
                                                      affordability.py
            -> per-component breakdown returned    (back through tools.py)
       -> guardrails.wrap_untrusted() on every tool result   app/guardrails.py
  -> SSE push to the browser, one event per finished tool call   app/main.py
```

Concretely: `POST /chat/stream` (`app/main.py`) runs `run_agent()`
(`app/agent_loop.py`) on a worker thread, so the asyncio event loop stays
free while the (synchronous, network-blocking) agent loop runs. `run_agent`
sends the running message list plus `TOOL_SCHEMAS` to whichever LLM
provider is configured. If the model asks for a tool call, the loop looks
it up in `TOOL_REGISTRY` (`app/tools.py`) and calls it. Every tool function
in `app/tools.py` reads its raw data from `app/dataset.py` (the only module
that touches disk) and hands it to a pure module — `app/scoring.py` for any
ranking, `app/analytics.py` for a filter/aggregate/derived-metric shape,
`app/affordability.py` for mortgage arithmetic — which returns a
per-component breakdown (raw value, normalized score, weight, contribution
for a ranking; factors/assumptions/confidence/caveat for a derived metric).
`app/tools.py` shapes that into the JSON the model sees.

Before that JSON re-enters the conversation, `agent_loop.py` calls
`guardrails.wrap_untrusted()` on it unconditionally — every tool result,
no exceptions — fencing it as `<untrusted_data>` and flagging it if a
cheap regex pre-filter catches a likely prompt-injection phrasing. The loop
repeats (provider -> tool calls -> guardrail-wrapped results) until the
model returns a plain text answer, or `max_turns` (default 6) is hit, in
which case `MaxTurnsExceeded` carries the partial `tool_log` and
transcript back to the caller rather than losing the work.

Each finished `ToolLogEntry` is also pushed onto a queue the instant it
completes, via `run_agent`'s `on_tool_call` hook; `chat_stream`'s async
generator drains that queue and turns each entry into an SSE `tool_call`
event, so the browser's live tool-call log updates turn-by-turn instead of
dumping everything at once at the end. A final SSE `done` event carries the
reply text.

### A spoken question

```
mic (static/voice.js, local endpointing)
  -> POST /voice/transcribe                     app/voice_api.py
  -> [identical to a typed question from here — same /chat/stream path]
  -> reply text, synthesized per sentence
  -> POST /voice/speak (one call per sentence)   app/voice_api.py
  -> barge-in: POST /voice/interrupt             app/voice_api.py
       -> conversation.truncate_last_reply()      app/conversation.py
```

The browser owns the microphone, local voice-activity detection
("endpointing" — silence/speech gating done client-side, no server round
trip while the user is silent), and playback; the server owns
transcription, synthesis, and the conversation history. `static/voice.js`
buffers audio locally, detects an utterance boundary, and posts the
recorded audio to `POST /voice/transcribe` (`app/voice_api.py`), which
delegates to the configured STT provider (`app/providers/stt/`) and
returns text. That text is then posted to the *same* `/chat/stream`
endpoint a typed message would use — voice is an input/output modality on
top of the existing agent path, not a separate one. The reply text is
split into sentences client-side and each is sent to `POST /voice/speak`,
which delegates to the configured TTS provider (`app/providers/tts/`) and
returns audio; sentence-level granularity is what lets playback start
before the whole reply is synthesized.

If the user talks over the agent, the browser stops playback and
synthesis locally (zero network round-trip — it already owns the
speaker and mic), then calls `POST /voice/interrupt` with
`spoken_prefix`: everything the user actually heard before they
interrupted. That endpoint calls `conversation.truncate_last_reply()`
(`app/conversation.py`), which rewrites the stored assistant turn down to
just the spoken prefix (or deletes it if nothing was heard yet) — refusing
to write anything that isn't a verified leading substring of what was
actually stored, so barge-in can only ever shorten a reply, never put
words in the model's mouth.

---

## 2. `app/` — one row per file

| File | Job |
|---|---|
| `agent_loop.py` (172 lines) | The hand-rolled agent loop: `run_agent()`. Plain while-loop over provider `.chat()` calls, tool dispatch, guardrail wrapping, and a hard `max_turns` ceiling (`MaxTurnsExceeded`) that guarantees termination even if a provider keeps requesting tools forever. Every tool call is recorded in `tool_log` in call order — that list is both the UI's live log and the reasoning trace. |
| `system_prompt.py` (67 lines) | `BASE_SYSTEM_PROMPT` plus two named rule constants (`NEVER_COMPUTE_RULE`, `NEVER_INVENT_IDS_RULE`) referenced by name from tests, so the load-bearing rules can't silently drift out of the text that's actually checked. 10 numbered rules (2a counts separately): never compute a number yourself, explain from the per-criterion breakdown, never invent an id, treat tool output as untrusted data, state assumptions/caveats, match the tool to the question shape, don't answer a single-dimension question from `total_score`, say so on a tool error, report a statistical tie as a tie, keep the live mortgage rate out of the scored path, handle region names, route Hebrew names through `resolve_entity`, and disclose this is not financial advice. |
| `tools.py` (1930 lines) | The tool surface and this domain's actual KPIs — `DEFAULT_CRITERIA`, `CRITERION_DESCRIPTIONS`, the eligibility gate (`_gate_ids`), and all 11 tool functions plus `TOOL_SCHEMAS`/`TOOL_REGISTRY`. The only module holding domain judgement: what "good value" means here, which criteria and weights, and why. See §3. |
| `scoring.py` (571 lines) | Pure, generic weighted ranker. `Criterion`, `score_item`, `rank_items`, plus the sensitivity-analysis machinery (`sensitivity_analysis`, `find_weight_flip_point`, `apply_priority_emphasis`, Kendall-tau rank correlation). Knows nothing about Israel, localities, or what any criterion means — `tools.py` supplies that. |
| `analytics.py` (340 lines) | Pure. The three query shapes that are *not* ranking: `filter_items` (attribute subsetting), `aggregate` (share/mean/count/sum over sub-records), and `build_derived_metric` (the generic contract a modelled quantity must satisfy — factors that provably sum to the value, mandatory assumptions and caveat, a validated confidence level). |
| `affordability.py` (330 lines) | Pure. Israeli mortgage/affordability arithmetic: `monthly_payment` (amortization), `max_affordable_price` (its inverse), `years_of_income_to_buy`, `mortgage_burden`, `affordability_breakdown`. Documented Bank-of-Israel-Directive-329 LTV ceilings and a stated repayment-cap convention, all overridable arguments rather than hidden constants. |
| `hebrew.py` (583 lines) | Pure. Makes Hebrew a first-class input to entity resolution: `strip_niqqud`, `normalize_hebrew`, `is_hebrew`, `romanize` (hand-rolled, no third-party transliteration library), plus `REGION_GROUPS`/`TRANSLITERATION_VARIANTS` tables and `match_region()` for area names ("the Krayot", "Gush Dan") that name several localities, not one. |
| `entity_resolution.py` (679 lines) | Pure. Free-text name -> candidate locality ids with a confidence and a `decisive` flag, via hand-rolled Jaro-Winkler + Soundex (no `rapidfuzz`/`jellyfish`). `resolve()` is the entry point; `score_pair` folds in Hebrew-normalized and romanized comparisons alongside the plain-Latin one. |
| `dataset.py` (396 lines) | **The only module that touches disk.** Loads `data/processed_data/{localities,neighborhoods}.json` once at import, and exposes `LOCALITIES`, `NEIGHBORHOODS`, `METRICS`, `ATTRIBUTES`, `RECORDS`, `ELIGIBLE_IDS`, `ENTITY_CATALOG`, `SUBDISTRICT_NAMES`, `CATEGORY_SEMANTICS`, `NEIGHBORHOOD_COUNTS`, `LOCALITIES_META`, `NEIGHBORHOODS_META`, and `resolve_region()`. Also splits the two item kinds: 104 LOCALITIES (independently rankable) vs. 1,394 NEIGHBORHOODS (sub-records of a locality, reachable only through `aggregate_records`). Imports `REGION_GROUPS`/`TRANSLITERATION_VARIANTS`/`match_region` from `hebrew.py` — those are *defined* in `hebrew.py`, not in this file; `dataset.py` consumes them to build `resolve_region()` and `ENTITY_CATALOG`. |
| `guardrails.py` (124 lines) | `wrap_untrusted(text, source)` fences any text re-entering the prompt as `<untrusted_data>`, so the model can't mistake tool output for developer instructions. `scan_for_injection()` is a cheap regex pre-filter that flags (not blocks) common injection phrasings. `agent_loop.py` calls `wrap_untrusted` on every tool result, with no exceptions. |
| `conversation.py` (134 lines) | The in-memory, process-wide chat history, shared by the text and voice paths. `snapshot()`/`replace(since=...)`/`clear()` use a generation counter so a stale write (e.g. a Reset that happened mid-turn) is dropped rather than resurrecting a cleared conversation. `truncate_last_reply()` is the barge-in operation (see §1). |
| `config.py` (192 lines) | Env/config loading. The `*_PROVIDER` selector pattern (`LLM_PROVIDER`, `STT_PROVIDER`, `TTS_PROVIDER`) plus model-name and key constants. `_find_shared_env()` walks upward from the repo root looking for a workspace-level `.env` (bounded at 5 levels) as a fallback after the project-local one. Default `LLM_PROVIDER` is `"mock"` so the app runs with zero credentials. Never logs a key value; pins `httpx`'s logger to WARNING as defense against credential leakage into logs. |
| `main.py` (248 lines) | FastAPI app. Routes: `GET /`, `GET /health`, `POST /chat` (non-streaming), `POST /chat/stream` (SSE), `POST /reset`. Mounts `app/voice_api.py`'s router. `/chat` is a plain `def`, not `async def`, on purpose — `run_agent` blocks on the network, and an `async def` handler would freeze the whole event loop for the duration of one turn (measured: a concurrent `/health` took 1.71s instead of 0.3s). `/chat/stream` runs the agent on a worker thread instead. |
| `voice_api.py` (250 lines) | Routes: `GET /voice/health`, `POST /voice/transcribe`, `POST /voice/speak`, `POST /voice/interrupt` (router mounted with `prefix="/voice"`). See §1 for the flow. Everything here is inert — and the app starts, serves, and passes its tests — with no speech credentials configured; `/voice/health` reports why voice is unavailable rather than letting a button fail silently on click. |
| `cli.py` (55 lines) | Terminal chat loop (`python -m app.cli`). Zero-dependency alternative to the web UI; prints each tool call as it happens. Deliberately boring — no features beyond what the loop itself needs. |
| `providers/llm/` | `base.py` — the provider-agnostic `LLMProvider` protocol (`ToolCall`, `LLMResponse`). `__init__.py` — the factory (`get_llm_provider()`), env-selected, memoized. `mock_llm.py` — scripted two-phase provider (tool call, then explain-from-result) that needs no network or key; this is what makes the whole app runnable with zero setup. `openai_llm.py`, `anthropic_llm.py` — real REST implementations (httpx directly, not vendor SDKs). `groq_llm.py` — thin subclass of `OpenAILLMProvider` (Groq's Chat Completions API is wire-compatible with OpenAI's); the free-tier option for a reviewer with no paid key. |
| `providers/stt/` | `base.py` — `STTProvider` protocol, `TranscriptResult`. `__init__.py` — factory, no mock (voice is a bonus mode, not a hard zero-setup requirement, so an unconfigured key correctly reports voice as unavailable rather than faking transcription). `openai_stt.py` — `gpt-4o-mini-transcribe` with a `whisper-1` fallback for accounts lacking the newer model. |
| `providers/tts/` | `base.py` — `TTSProvider` protocol, `SynthesisResult`. `__init__.py` — factory. `openai_tts.py` — default, WAV output. `google_tts.py` — second implementation (Neural2 voices via the REST `text:synthesize` endpoint, API-key auth in a header, never a query param), included specifically so the provider abstraction is a demonstrated property rather than a claim with one implementation. |

### The purity boundary

`app/scoring.py`, `app/analytics.py`, `app/affordability.py`, `app/hebrew.py`,
and `app/entity_resolution.py` do **zero I/O and zero LLM calls** — no
network, no file reads, no env var reads, no wall-clock, no randomness.
Every function in them is pure: same inputs, same outputs, every time.
`app/dataset.py` is the **only** module in `app/` that reads from disk
(`data/processed_data/*.json`, loaded once at import). Everything else —
`tools.py`, `agent_loop.py`, `main.py`, `voice_api.py` — is handed data
rather than fetching it itself.

This is why the five pure modules are testable with no fixtures, no
network, and no API key: `tests/test_scoring.py`,
`tests/test_analytics.py`, `tests/test_affordability.py`,
`tests/test_hebrew.py`, and `tests/test_entity_resolution.py` construct
their own inputs directly and assert on outputs, with nothing to mock.

---

## 3. The tool surface

`app/tools.py` defines `TOOL_SCHEMAS` (what the LLM sees) and
`TOOL_REGISTRY` (name -> callable), both with exactly 11 entries:

| Tool | Answers | Reads from |
|---|---|---|
| `find_items` | "Which localities match these filters?" (a described group, e.g. "urban localities in the North") | `app/analytics.py` (`filter_items`) over `dataset.ATTRIBUTES` |
| `compare_items` † | "Rank/compare these specific localities." | `app/scoring.py` (`rank_items`) over `dataset.METRICS` |
| `rank_by_priorities` † | "Rank these, but weighted to what I said I care about." | `app/scoring.py` (`apply_priority_emphasis` + `rank_items`) |
| `analyze_weight_sensitivity` † | "What happens to the ranking if this weight were different?" | `app/scoring.py` (`scale_criterion_weight` + `sensitivity_analysis`) |
| `weight_robustness_report` † | "How confident are you in this ranking — which weight is it hanging on?" | `app/scoring.py` (`find_weight_flip_point` per criterion) |
| `list_criteria` | "What criteria/weights do you use? How is the score calculated?" | `DEFAULT_CRITERIA` + `CRITERION_DESCRIPTIONS` (static, in `tools.py`) |
| `resolve_entity` | "Which locality/localities does this name refer to?" | `app/entity_resolution.py` (`resolve`) + `dataset.resolve_region` |
| `get_item_metrics` | "Raw facts about one locality" (no scoring) | `dataset.METRICS` / `dataset.LOCALITIES` |
| `aggregate_records` | "What share of [locality]'s neighborhoods are above/below its own median?" | `app/analytics.py` (`aggregate`) over `dataset.RECORDS` |
| `estimate_derived_metric` | "What income do I need to buy here, and why?" (modelled, not looked up) | `app/affordability.py` (`monthly_payment`, `years_of_income_to_buy`) via `app/analytics.py`'s `build_derived_metric` contract |
| `get_current_mortgage_rates` | "What's today's mortgage rate?" | Live HTTP call to `boi.org.il/PublicApi/GetInterest` (see below) |

† = one of the **four ranking tools** that enforce the eligibility gate
(`_gate_ids`) before scoring anything: `compare_items`,
`rank_by_priorities`, `analyze_weight_sensitivity`,
`weight_robustness_report`. `_gate_ids` sets aside any NEIGHBORHOOD id
handed to a ranking tool (with a reason, in `ineligible`) rather than
scoring it on a sliver of the criteria a neighborhood has no data for.
This is duplicated across all four call sites deliberately — a prior
version of this same architecture had the gate live only inside
`compare_items`, and three sibling ranking tools added later silently
skipped it. `tests/test_tools_domain.py`'s
`test_gate_enforced_in_all_four_ranking_tools` pins this.

`get_current_mortgage_rates` is the **one live call** in the whole tool
surface (`urllib.request` to the Bank of Israel's public rate API, no
key needed) and it is **deliberately outside the scored path**: a rate
change moves what every buyer can afford everywhere at once, so it says
nothing about whether one locality is better *value* than another
(system prompt rule 7). On a network failure it returns a clearly
labelled `available: False` stated-assumption fallback rather than
raising or silently substituting another source.

---

## 4. `data/`, `evals/`, `tests/`, `static/`, `scripts/`

### `data/`
See `data/README.md` for full provenance, fetch quirks, and known
limitations — not repeated here. In short: `data/refresh_data.py`
(1,166 lines) fetches from nadlan.gov.il and CBS and writes
`data/processed_data/{localities,neighborhoods}.json`, the only files
`app/dataset.py` reads. `data/raw_data/` is a **gitignored, rebuildable
cache** (~24MB, ~1,570 upstream files) — `refresh_data.py` rebuilds it
idempotently. `data/processed_data/` **is committed** so a bare clone
runs offline with no network and no key.

### `evals/`
See `evals/README.md` for the full harness description and how to run
it — not repeated here. Implements Task/Trial/Trace/Outcome/Grader/Suite
(`evals/types.py`), a runner (`evals/runner.py`), deterministic and
LLM-judge graders (`evals/graders/`), and domain-specific seed tasks
(`evals/tasks/seed_tasks.py`, 696 lines). Runnable with
`.venv/bin/python -m evals.run_evals --provider mock` for a zero-cost
run.

| File | Job |
|---|---|
| `types.py` (346) | `Task`, `Trial`, `Trace`, `Outcome`, `Grader`, `SuiteResult` dataclasses |
| `runner.py` (222) | Executes one Task's trials against the real agent loop |
| `suite.py` (37) | A list of Tasks, run and aggregated |
| `report.py` (274) | Renders a `SuiteResult` to Markdown/JSON |
| `run_evals.py` (121) | CLI entry point |
| `graders/deterministic.py` (497) | Rule-based graders (e.g. "did it call the right tool") |
| `graders/llm_judge.py` (288) | LLM-as-judge grader |
| `judge_calibration_data.py` (266) / `judge_validation.py` (152) | Hand-labeled examples used to validate the judge grader against human judgement |
| `tasks/seed_tasks.py` (696) / `tasks/fixtures.py` (93) | This domain's seeded eval tasks, including failure-mode/injection cases |

### `tests/`
Mirrors `app/` roughly one-to-one, not exactly: `app/scoring.py`,
`analytics.py`, `affordability.py`, `hebrew.py`, `entity_resolution.py`,
`agent_loop.py`, `conversation.py`, `config.py`, `guardrails.py`,
`voice_api.py` each have a matching `test_*.py`. `app/tools.py` is
covered by two files, `test_tools_domain.py` and
`test_tools_uncovered.py`. `app/dataset.py` has no dedicated test file
of its own — it's exercised indirectly through the tools/scoring/
entity-resolution tests that consume it. `app/cli.py` and
`app/system_prompt.py` (a plain-string-constant module) have no
dedicated test file either; `system_prompt.py`'s rule constants are
referenced by name from `test_agent_loop.py`. `test_graders.py` and
`test_markdown_renderer.py` cover `evals/graders/` and
`static/markdown.js` respectively.

**The whole suite runs with zero keys**, on `LLM_PROVIDER=mock` (the
default): `python -m pytest -q` from the repo root → **391 passed**
(verified 2026-08-23, 5.74s, one unrelated `httpx`/starlette deprecation
warning).

### `static/`
`index.html` (1,341 lines) — the single-page chat UI, zero build step.
`markdown.js` (235) — renders the model's markdown replies, and its
`stripMarkdown` is also what `voice_api.py`'s speak path uses so a
synthesizer doesn't read `**` and table pipes aloud. `voice.js` (605) —
conversation-mode voice: local mic capture, local voice-activity
endpointing (speech/silence gating done in-browser), sentence-batched
TTS playback with look-ahead buffering, and the barge-in sequence (stop
playback locally → stop synthesis locally → tell the server what was
actually heard via `POST /voice/interrupt`). Echo cancellation
(`getUserMedia`'s `echoCancellation` constraint) is load-bearing here,
not a nicety — without it the agent's own voice reaching the mic would
trip the barge-in detector continuously.

### `scripts/`
None of these are part of the pytest suite (`pytest.ini`'s `testpaths`
is `tests` only) — each is a manually-run, one-off utility:

| File | Job |
|---|---|
| `calibrate_resolver.py` (370) | Calibrates `entity_resolution.py`'s relevance threshold against real, labeled Israeli place-name data |
| `run_example_questions.py` (205) | Runs this domain's four question shapes end to end through the real `run_agent`, writes `artifacts/example_questions.json` |
| `sensitivity_report.py` (353) | Measures how much the `DEFAULT_CRITERIA` weights actually matter against the real 104-locality dataset |
| `smoke_test.py` (91) | Full-stack manual smoke test: config → provider factory → agent loop → tools → scoring → guardrails, in one process |

---

## 5. Which doc to open for what

| Doc | For |
|---|---|
| `README.md` | Quickstart, what this project is |
| `DESIGN.md` | Scoring methodology, where AI is (and isn't) used, tradeoffs |
| `docs/ARCHITECTURE.md` | This file — the file-by-file map and request flow |
| `data/README.md` | Data provenance, fetch quirks, known limitations |
| `evals/README.md` | The eval harness: how to run it, what it checks |
