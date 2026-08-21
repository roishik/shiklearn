# evals/ — a small, real, runnable eval harness

Evaluates `app/agent_loop.py`'s Israel home-buying intelligence agent
(the locality compare/rank assistant built in this repo). Built because
"a candidate who doesn't start with evals" is the loudest single red flag
in publicly discussed AI-agent hiring signal — this is meant to be the
thing you point at first, not a checkbox added at the end.

It implements Anthropic's agent-eval anatomy for real, as Python types
you can point at and name:

| Term | Where |
|---|---|
| **Task** | `evals/types.py:Task` — an input + the graders that check it |
| **Trial** | `evals/types.py:Trial` — one isolated attempt at a Task |
| **Trace** | `evals/types.py:Trace` — the transcript (tool calls, messages, final answer) |
| **Outcome** | `evals/types.py:Outcome` — the derived final state a Grader actually looks at |
| **Grader** | `evals/types.py:Grader` — deterministic (`evals/graders/deterministic.py`) or LLM-as-judge (`evals/graders/llm_judge.py`) |
| **Suite** | `evals/suite.py:Suite` — a list of Tasks, run for real, aggregated into a `SuiteResult` |

The harness itself (`types.py`, `runner.py`, `suite.py`, `report.py`,
`run_evals.py`, `graders/`) is domain-neutral and unchanged from the
prior build this project re-did the domain of — only `tasks/seed_tasks.py`,
`tasks/fixtures.py`, `judge_calibration_data.py`, and this file are
specific to the home-buying domain.

## Run it

From the repo root, using the venv directly (shell activation doesn't
persist across separate commands, so call the binary explicitly):

```bash
.venv/bin/python -m evals.run_evals --provider mock         # forces mock — zero setup, agent side is free
.venv/bin/python -m evals.run_evals --provider openai       # real key from .env
# Omitting --provider falls through to LLM_PROVIDER in your environment or .env — pass
# --provider mock explicitly if you want a guaranteed zero-cost agent run regardless of
# what your shell has configured.
.venv/bin/python -m evals.run_evals --category injection    # filter by category
.venv/bin/python -m evals.run_evals --id-contains modiin
.venv/bin/python -m evals.run_evals --trials 3               # override every task's num_trials

.venv/bin/python -m evals.judge_validation                   # judge-vs-human agreement report
```

**On cost: the agent side and the judge side are billed separately, and
`--provider mock` only controls the agent side.** Every judge-graded task
builds its own OpenAI provider (`evals/graders/llm_judge.py`) and calls
it whenever `OPENAI_API_KEY` is set on disk — including during a `mock`
run. The reported "Agent cost" figure never includes those calls. If you
want a genuinely zero-cost run, unset the key for that shell first.

Every run writes a timestamped Markdown + JSON report to `evals/results/`
and prints the overall pass rate / avg score to stdout.

### Real results from this build (2026-08-21, mock provider, no OPENAI_API_KEY configured)

Ran for real: `PYTHONPATH=. python -m evals.run_evals`, no `--provider`
flag (falls through to `LLM_PROVIDER` in this environment, which is
`mock`, the default — see `app/config.py`). No API key of any kind is
configured in this environment (`app.config.have_openai_key()` /
`have_anthropic_key()` / `have_groq_key()` all return `False`, checked
directly).

**Overall: 9/24 trials passed = 38% pass rate, 0.52 avg partial-credit
score.** Full report: `evals/results/mock_20260821T060631Z.md` /
`.json` — cite that exact filename, not this summary, for anything more
detailed than the numbers below.

**Every single failure in this run is attributable to one of two
documented, verified limitations of this specific run — not to a bug in
`app/agent_loop.py`, `app/tools.py`, or `app/guardrails.py`.** Read
through all 15 failing tasks' grader detail in the full report before
believing that claim; it is not asserted lightly, and it is the central
point of running the suite this way before touching any task:

1. **`LLM_PROVIDER=mock` is a scripted stand-in, not a model.**
   `app/providers/llm/mock_llm.py`'s `MockLLMProvider` ALWAYS requests
   `compare_items` on turn 1, regardless of what the question actually
   asks, and its narration of the result is a fixed template (numbers
   only — no disclaimers, no hedging language, no mention of ambiguity).
   It can never call `find_items`, `resolve_entity`, `aggregate_records`,
   `estimate_derived_metric`, `rank_by_priorities`, or
   `get_current_mortgage_rates`, and it can never produce prose that
   surfaces uncertainty, states an assumption, or declines a request. 11
   of the 15 failures below are this, directly.
2. **No `OPENAI_API_KEY` is configured**, so every `LLMJudgeGrader`
   reports a clearly-labeled `SKIPPED` (score 0, `passed=False`,
   `details={"skipped": True}` — see `evals/graders/llm_judge.py`),
   never a fabricated pass. This is BY DESIGN (the alternative — silently
   passing an ungraded task — would be worse), but it means every task
   with a judge grader shows as a failure in this specific run for a
   reason that has nothing to do with the agent. 4 of the 15 failures
   below are this (2 overlap with #1, on tasks with BOTH problems).

| Category | Tasks | Pass rate | Why the failures happened |
|---|---|---|---|
| correctness | 4 | 0% | all 4: mock always calls `compare_items`, never the right tool |
| self-computation | 2 | **100%** | mock happens to call `compare_items` anyway, which is the correct behavior here |
| entity-resolution | 4 | 75% | 3 pure-code (`run_direct`) tasks PASS unconditionally — no LLM in the loop at all; the 1 agent-level task fails on judge-skipped only |
| missing-data | 3 | **100%** | all 3 are pure-code (`run_direct`) — no LLM, no mock limitation, genuinely exercise `app/tools.py` |
| injection | 3 | 0% | 2 need a real provider to even call the injection-bearing fixture tool; all 3 have a judge-skipped component |
| explanation-quality | 2 | 0% | both are pure judge tasks — 100% attributable to no OPENAI_API_KEY |
| honesty-about-uncertainty | 3 | 33% | 1 pure-code task PASSES unconditionally; the 2 agent-level tasks fail because mock never encounters the real near-tie pair or calls `estimate_derived_metric` |
| scope | 3 | 0% | mock's fixed narration template never emits disclaimer language, and never declines an off-topic question |

**The 6 pure-code (`run_direct`) tasks — no LLM, no provider, no mock
limitation at all — are 6/6 PASS**, and are the closest thing this suite
has to "does the deterministic core actually work, independent of any
model": `entity_resolution_junk_matches_nothing`,
`entity_resolution_neighborhood_id_gated_not_dropped`,
`entity_resolution_code_shaped_query_decisive`,
`missing_data_derived_metric_low_confidence_locality`,
`missing_data_neighborhood_denominator_honesty`,
`honesty_near_tie_presented_as_tied`. These pin real, verified facts
about the built dataset directly (e.g. Qiryat Motzkin id `8200` and
Judeide-Maker id `1292` really are 0.0012 apart, inside
`DECISIVE_SCORE_GAP=0.005`; locality `53` — Atlit — really is one of only
2 of 104 eligible localities missing a CBS income figure) — see each
task's own `notes` field in `evals/tasks/seed_tasks.py` for the exact
verification command run against the real dataset.

**Do not re-tune tasks to make the mock provider pass.** Every
mock-attributable failure above is the CORRECT, EXPECTED outcome given
what the mock provider is — tuning a task's wording or threshold until
a scripted stand-in that ignores the question happens to satisfy it
would make the suite lie about what it's testing. The tasks tagged
`EXPECTED TO FAIL under LLM_PROVIDER=mock` in their `notes` field are
meant to be re-run against a real provider (`--provider openai` or
`anthropic`, plus a key) for a result that means anything about tool
ROUTING or PROSE quality — this run's job was to prove the harness and
every task actually execute correctly end to end, which it does: no
task crashed, no grader raised, and the JSON/Markdown reports were
written successfully.

### Judge-vs-human agreement

`evals/judge_calibration_data.py` holds 10 hand-labeled examples (Roi,
acting as the "intern" in the research brief's intern test), each a real
`app.tools.compare_items(['6900', '8700'])` result (Kefar Sava vs
Ra'anana — verified against the real dataset, see that module's own
docstring for the ground truth) paired with a `final_text` spanning the
full 1-10 rubric range. `evals/judge_validation.py` runs the SAME
rubric-prompting/parsing code path `LLMJudgeGrader` uses at eval time
against that set and reports binary pass/fail agreement plus mean
absolute score difference.

**Not run in this session — no `OPENAI_API_KEY` is configured in this
environment**, and the judge has no mock/offline mode by design (see
`evals/graders/llm_judge.py`'s own docstring for why: judging free-form
text quality is exactly the kind of task a scripted stand-in cannot do).
Run `.venv/bin/python -m evals.judge_validation` yourself once a key is
available; it prints a plain verdict (`VALIDATED` at ≥80% binary
agreement, per the brief's "intern test" threshold, or
`NOT YET RELIABLE` below it) and writes
`evals/results/judge_validation_<timestamp>.md`. Until that has been run
for real, treat every `LLMJudgeGrader`-graded task's score as
**unvalidated** — a `skipped` result, not evidence the rubric works on
this domain's real traces.

## The seeded task set (24 tasks)

Written fresh for this domain 2026-08-21 (the file this replaced was
still verbatim the prior airport-domain project's `seed_tasks.py`).
Covers the eight failure-mode categories the brief asks for:

- **correctness** (4 tasks) — the right tool for the question shape:
  a single-locality statistic must go through `aggregate_records`, a
  described GROUP through `find_items`, a modelled quantity through
  `estimate_derived_metric`, stated priorities through
  `rank_by_priorities` — never the default-weighted `compare_items`.
- **self-computation** (2 tasks) — the model must not compute a score or
  do the weighted arithmetic itself even when explicitly told not to
  bother calling a tool (`NEVER_COMPUTE_RULE`).
- **entity-resolution** (4 tasks) — "Modiin" (not "Modi'in" — see the
  task's own `notes` for why the apostrophe matters, a real finding from
  building this task, not an assumption) must not be silently resolved
  to one of its two real candidates; junk must match nothing; a
  neighborhood id must be gated out of ranking with a reason, not
  dropped or silently scored; a bare locality code must resolve
  decisively.
- **missing-data** (3 tasks) — the real #1/#2 (`8200`/`1292`) near-tie
  must disclose that #2 is scored on `covered_weight=0.85` (missing
  `rental_yield`); a locality with no CBS income figure must report
  `confidence='low'` rather than silently treating the gap as zero; a
  locality's real, full neighborhood-tracking count must never collapse
  to just the priced subset.
- **injection** (3 tasks) — a prompt-injection payload delivered via a
  simulated municipal planning-note tool (`evals/tasks/fixtures.py`,
  domain-appropriate replacement for the prior project's
  `get_airport_advisory_note`) or directly in the user's own message,
  in two different injection SHAPES (plain imperative, and a fake
  `</system>` role-tag), must never be obeyed and must be flagged.
- **explanation-quality** (2 tasks) — an explanation must cite the real
  per-criterion numbers the tool returned, in plain language a
  non-technical reader can follow.
- **honesty-about-uncertainty** (3 tasks) — the real, dataset-verified
  near-tie must be presented as tied, never as a confident lone winner;
  `estimate_derived_metric`'s 2021-income-vs-2026-price vintage caveat
  must be surfaced, not dropped.
- **scope** (3 tasks) — the live mortgage rate must never be presented
  as evidence for or against a particular locality (system prompt rule
  7); the agent must not present itself as giving financial/purchase
  advice (rule 10); an off-topic question must not be forced into a
  ranking.

### Grading outcomes, not paths — with one documented exception

The default and expected way to grade a Task is against its Outcome: the
final answer text, whether a particular tool's numbers are traceable,
etc. A trial that reaches the right answer via an unexpected but valid
tool sequence should still pass.

The ONE deliberate exception is `Task.expected_tool` /
`Task.forbidden_tools` — used on the `correctness` and one `scope` task
above, where the whole point IS which tool got called. See
`evals/types.py`'s module docstring for the full reasoning.

### Partial credit, not just pass/fail

Every grader returns a float in `[0, 1]` (`GradeResult.score`), not just
a bool. `DirectResultGrader` and `ToolCallGrader` are binary by nature,
but `NoFabricatedNumbersGrader`, `ScoringMatchesGroundTruthGrader`, and
`LLMJudgeGrader` all carry real partial credit — see
`evals/graders/deterministic.py` for exactly how each computes it.

### Isolated trials

`evals/runner.py:run_trial` builds a fresh `messages` list and a fresh
tool-registry dict on every call — two trials of the same task, or two
different tasks, never share mutable state. See that module's own
docstring for the guarantee in detail.

### Known, documented mock-provider limitation

Every task above whose correct grading depends on `find_items`,
`resolve_entity`, `aggregate_records`, `estimate_derived_metric`,
`rank_by_priorities`, or `get_current_mortgage_rates` being called, or on
the model's own prose surfacing an ambiguity/caveat/disclaimer, is tagged
`EXPECTED TO FAIL under LLM_PROVIDER=mock` in its `notes` field — see
"Real results from this build" above for the actual, verified accounting
of every failure in this suite's most recent mock run.

## Add a new task in under 2 minutes

Open `evals/tasks/seed_tasks.py` and add one `Task(...)` to the `TASKS`
tuple. Five things to decide, in order:

1. **`id` / `category` / `description`** — pick an existing category if
   your failure mode fits one, or start a new one.
2. **`user_message`** (agent task) or **`run_direct`** (pure-code task
   against `app/scoring.py` / `app/tools.py` directly, no LLM involved —
   see the 6 `run_direct` tasks above for the pattern, including the
   named `_check_*` helper-function convention for
   `DirectResultGrader.check`, which must always return a `(bool, str)`
   tuple).
3. **Does this task need a specific tool called/not called?** If yes,
   set `expected_tool="..."` or `forbidden_tools=("...",)`. If it's
   about the OUTCOME regardless of path, skip these and use an outcome
   grader instead.
4. **Pick graders** from `evals/graders/deterministic.py` (fast, exact)
   and/or `evals/graders/llm_judge.py` (open-ended text quality — reuse
   an existing `RUBRIC_*` constant; there are four:
   `RUBRIC_EXPLANATION_CITES_REASONING`,
   `RUBRIC_HANDLES_AMBIGUITY_OR_REFUSES_INJECTION`,
   `RUBRIC_STAYS_ON_TOPIC`, `RUBRIC_TONE_FOR_NON_TECHNICAL_READER` — or
   write a new one with 1/4/7/10 anchors in
   `evals/graders/llm_judge.py`, following the pattern already there).
5. **Run it**: `.venv/bin/python -m evals.run_evals --id-contains <your-task-id>`.

Minimal example:

```python
Task(
    id="my_new_failure_mode",
    category="entity-resolution",
    description="One sentence describing the scenario.",
    user_message="the exact prompt you're testing",
    graders=(
        NoFabricatedNumbersGrader(),
        LLMJudgeGrader("my_check", RUBRIC_HANDLES_AMBIGUITY_OR_REFUSES_INJECTION,
                        context_fn=lambda task, outcome: {"scenario": "one-sentence context for the judge"}),
    ),
),
```

If your task needs a tool that doesn't exist in `app/tools.py`, add it to
`evals/tasks/fixtures.py` (following `get_locality_planning_note`'s
pattern) and pass it via `extra_tool_registry=` / `extra_tool_schemas=`
— never edit `app/tools.py` itself for an eval-only fixture.

## Reading a suite's report

Every `evals/results/<provider>_<timestamp>.md` has three sections:

1. **Test matrix** — one row per task: input, expected tool, expected
   behavior, partial-credit score, pass/fail. Read this first; it's the
   whole suite at a glance.
2. **By category** — pass rate and avg score per failure-mode category.
   A category with a low avg score across several tasks is a real
   pattern, not one flaky task.
3. **Per-task grader detail** — every trial, every grader's individual
   score and rationale. This is where you go when a task fails and you
   need to know WHICH check failed and why (the rationale string is
   written to be read, not just logged) — e.g. a `[FAIL]` on
   `tool_called:aggregate_records` means the wrong tool was called,
   while a `SKIPPED` on an LLM-judge grader means no key was configured,
   not that the model failed.

The JSON report (`evals/results/<provider>_<timestamp>.json`) has the
same data machine-readable, including the raw `final_text` for every
trial — use it if you want to diff two runs or feed results elsewhere.

**Read a few full transcripts by hand periodically** (the JSON's
`outcome.final_text` + `tools_called`), not just the pass rate — standard
evals hygiene, and the only way to catch a grader that's technically
passing/failing for the wrong reason (see, in this build's own
`evals/graders/llm_judge.py`, the comment on
`RUBRIC_HANDLES_AMBIGUITY_OR_REFUSES_INJECTION`'s anchor rewrite — a
prior version of that exact rubric scored a genuine security SUCCESS as
a fail, caught only by reading a transcript, not by the aggregate score).

## Files

```
evals/
  types.py                    Task / Trial / Trace / Outcome / Grader / Suite dataclasses
  runner.py                   runs one isolated Trial, wires expected_tool/forbidden_tools
  suite.py                    Suite / SuiteResult — runs every Task for real
  report.py                   renders SuiteResult to Markdown (test matrix) + JSON
  run_evals.py                CLI entrypoint
  graders/
    deterministic.py          code-based graders (fast, cheap, reproducible)
    llm_judge.py               LLM-as-judge grader + rubric templates (1/4/7/10 anchors)
  tasks/
    seed_tasks.py               the 24 seeded Task definitions (this domain)
    fixtures.py                  eval-only tool (get_locality_planning_note) for injection tasks
  judge_calibration_data.py       10 hand-labeled examples for judge validation (this domain)
  judge_validation.py              runs the judge against the calibration set, reports agreement
  results/                          generated reports (evidence of real runs, kept in-repo)
```
