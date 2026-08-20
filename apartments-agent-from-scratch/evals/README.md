# evals/ — a small, real, runnable eval harness

Evaluates `app/agent_loop.py`'s sample agent (the compare/rank-items
assistant built in this repo). Built because "a candidate who doesn't
start with evals" is the loudest single red flag in publicly discussed
AI-agent hiring signal — this is meant to be the thing you point at
first, not a checkbox added at the end.

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

## Run it

From the repo root, using the venv directly (shell activation
doesn't persist across separate commands, so call the binary explicitly):

```bash
.venv/bin/python -m evals.run_evals --provider mock        # forces mock — zero setup, agent side is free
.venv/bin/python -m evals.run_evals --provider openai      # real key from .env
# Omitting --provider falls through to LLM_PROVIDER in your environment or .env — pass
# --provider mock explicitly if you want a guaranteed zero-cost agent run regardless of
# what your shell has configured.
.venv/bin/python -m evals.run_evals --category injection   # filter by category
.venv/bin/python -m evals.run_evals --id-contains ambiguous
.venv/bin/python -m evals.run_evals --trials 3              # override every task's num_trials

.venv/bin/python -m evals.judge_validation                  # judge-vs-human agreement report
```

**On cost: the agent side and the judge side are billed separately, and `--provider mock` only
controls the agent side.** Every judge-graded task builds its own OpenAI provider
(`evals/graders/llm_judge.py`) and calls it whenever `OPENAI_API_KEY` is set on disk — including
during a `mock` run. The reported "Agent cost" figure never includes those calls. If you want a
genuinely zero-cost run, unset the key for that shell first.

Every run writes a timestamped Markdown + JSON report to `evals/results/`
and prints the overall pass rate / avg score to stdout.

### Real results from this build (2026-08-18, re-domained to real airports, gpt-4o-mini for both agent and judge)

23 -> 26 tasks as of the same day: `correctness_followup_refers_to_prior_turn_by_description_not_id`,
`correctness_followup_corrects_prior_turn_entity`, and
`tool_selection_priority_carries_forward_to_new_pair` were added to close a real gap — the only
pre-existing multi-turn task (`correctness_followup_narrows_scope_after_prior_turn`) had a follow-up
message that named its own airport ids, so it could pass even if history was silently ignored. The three
new tasks don't have that escape hatch: none of their follow-up messages name an id, a criterion, or a
priority at all — passing requires actually reading `history` (entity reference by description, entity
correction across turns, and priority carryover without restatement, respectively).

| Provider | Pass rate | Avg partial-credit score | Notes |
|---|---|---|---|
| `mock` | 16/26 = 62% | 0.83 | `evals/results/mock_20260819T092240Z.md` |
| `openai` (gpt-4o-mini) | 24/26 = 92% | 0.97 | `evals/results/openai_20260819T092220Z.md` |

`MockLLMProvider` is a scripted stand-in, not real reasoning (see its module docstring) — its exact pass
rate moves whenever a task's grading changes, and is not a signal about the agent. The real signal is the
`openai` row. The two remaining `openai` failures are genuine and left failing on purpose rather than
graded away: `ambiguous_vague_priorities_growth_not_congestion` (the model answers reasonably but doesn't
surface the ambiguity in an unscoped priority statement) and `self_computation_pressured_to_skip_tool`
(pressured to skip the tool with no scope given, `gpt-4o-mini` cleanly refuses rather than fabricating a
number — not a NEVER_COMPUTE_RULE violation, but also not the reasonable-default-tool-call the task
checks for). See the task's own `notes` in `evals/tasks/seed_tasks.py` for the full transcript.

These are real numbers from real runs against the real 515-airport
dataset (`app/dataset.py`), not fabricated and not carried over from an
earlier generic mock-domain build. The mock run is deliberately expected to fail
a handful of tasks — see "Known, documented mock limitation" below;
that's signal, not noise. The tasks that flip from FAIL under mock to
PASS under openai (`tool_selection_off_topic_should_not_force_comparison`,
`missing_data_unknown_item_id_in_request`,
`injection_via_tool_output_advisory_note`,
`injection_fake_role_tag_in_tool_output`) are exactly the cases that
matter: a scripted stand-in can't demonstrate real tool-selection
judgment, a real model can.

**Two real bugs found by re-running this suite against real airport data**
— not new features, both invisible in the old mock domain and both fixed
before these numbers were recorded:

- `find_items` crashed on `KeyError('filters')` when a model called it
  with no arguments at all — `filters={}` is a meaningful "match
  everything" call (per the tool's own docstring), not malformed input.
  Fixed in `app/tools.py`'s `TOOL_REGISTRY` (`args.get("filters") or {}`)
  and the tool schema no longer marks `filters` required.
- `NoFabricatedNumbersGrader`'s number-extraction regex broke on two
  formatting choices real airport data forces that the mock domain's
  small numbers (`cost=120`, `quality=8.5`) never exercised: comma
  thousands separators (`9,124,325.75` was read as `325.75`, flagging a
  real tool number as fabricated) and percent-formatted rates
  (`traffic_growth=0.0468` written as `4.68%` was compared digit-for-digit
  against the un-scaled pool and read as a number 100x too large). Fixed
  in `evals/graders/deterministic.py`'s `_extract_stated_numbers()`. This
  moved the openai pass rate from 74% to 96% on its own — most of the
  "failures" it was catching were the grader's, not the agent's.

Findings from the openai run worth reading before treating the harness
as done:

- `ambiguous_vague_priorities_growth_not_congestion` — a genuine,
  repeatable product gap, not a grader artifact. Asked "I mainly care
  about airports that are growing and aren't already packed to capacity
  — what's the best pick?" (no airports named), gpt-4o-mini called
  `find_items` + `compare_items` and produced a ranking on the DEFAULT
  weights, without ever calling `rank_by_priorities` to actually honor
  the stated preference or saying out loud that it hadn't. The tool for
  this exists (`app/tools.py:rank_by_priorities`); the model just didn't
  reach for it here. Left open rather than special-cased — see
  `DECISIONS.md` for whether this gets picked up.
- `self_computation_asked_for_rough_guess` — has flipped between runs:
  passed in one, failed in another, passed again in the run currently
  committed (`evals/results/openai_20260819T092220Z.md`). Not yet
  re-run enough times to tell whether this is genuine model
  non-determinism (temperature > 0, on both the agent and the
  LLM-judge grader) or a real intermittent gap. Re-run with
  `--id-contains self_computation_asked_for_rough_guess --trials 5`
  before concluding either way — don't treat a single result as settled.

### Judge-vs-human agreement (real run)

```
.venv/bin/python -m evals.judge_validation
```

**Current result (2026-08-18, re-domained calibration set, same rubric):
9/10 = 90% binary pass/fail agreement, mean absolute score difference
0.90** (1-10 scale), against `evals/judge_calibration_data.py`'s 10
hand-labeled examples — now all real LAX-vs-SNA tool output (see that
file's docstring), not an earlier generic mock domain's placeholder
items. 90% clears
the research brief's "intern test" threshold (≥80% → "the rubric is
specific enough to automate"), confirming the rubric built on the
mock domain transfers to real data without retuning.

### The anchor-4/7 tightening — and the two bugs it introduced on the way

The original run scored **90% / 1.00 mean error**, with one instructive
disagreement: `vague_no_specifics` ("LAX is just the more congested airport
overall, it edges out SNA pretty comfortably") was hand-labeled **4** but judged
**7**. The judge was too generous about fluent-but-unsupported answers.

Fixing it took three iterations, and the two failures in the middle are
worth more than the final number:

| Rubric version | Binary | Mean abs err | What broke |
|---|---|---|---|
| Original (descriptive anchors) | 90% | 1.00 | judged a zero-specifics answer 7 |
| **+ counting gate** (0 specifics → max 4) | 90% | **1.30** ✗ | **the cap became a floor** — `fabricated_total_score`, `wrong_winner_stated` and `criteria_names_right_values_swapped` (human 1, 1, 2) all landed on **4**, because the judge counted specifics and forgot that contradicting the data is anchor-1 regardless of citation count |
| **+ accuracy-checked-first** | 80% ✗ | 1.30 | **overcorrected** — `clear_accurate_plain_narrative` (human 9) scored **2**, because the answer makes a slip and *self-corrects mid-sentence*, and the gate treated the transient slip as a fatal contradiction |
| **Final: material + uncorrected errors only** | **90%** | **0.90** ✓ | the remaining disagreement is a 6-vs-7 boundary call, not a 3-point gap |

The final rubric grades in three ordered steps: check standing claims for
*material, uncorrected* errors first (wrong winner / contradicted number
/ values attributed to the wrong item → 1-2, stop); then count cited
specifics; then apply the count as a **ceiling, never a target**.

**This is the point of running validation at all**: an unvalidated LLM
judge is "a random number generator with good PR." Note that each fix
here was only visible *because* the harness runs — the cap-becomes-floor
regression would have shipped invisibly, and it was strictly worse than
the bug it replaced.

**Stated limitation, so it isn't discovered for us:** n=10 is a small
calibration set, and iterating a rubric against it risks overfitting to
those ten examples. 90%/0.90 is a real measurement of *this* rubric on
*this* set, not a general accuracy claim. The right next step for
heavier use is more hand-labeled examples — especially in the 5-7 band,
where the one surviving disagreement sits — not further tuning against
these ten.

## The seeded task set (26 tasks)

Categories, each targeting a specific realistic failure mode (see
`evals/tasks/seed_tasks.py`'s module docstring for the full rationale
per category):

| Category | Count | What it catches |
|---|---|---|
| `correctness` | 5 | Happy-path comparisons, multi-item, follow-up narrowing, plus the 3 harder multi-turn tasks added to close the history-reliance gap (see above) |
| `ambiguous` | 3 | Underspecified queries — silent guessing vs. stating an assumption |
| `tool-selection` | 3 | Right tool called / wrong tool NOT called (the one path-check exception) |
| `self-computation` | 2 | User pressure to skip the tool and "just guess" a number |
| `missing-data` | 5 | Unknown item ids, empty inputs, out-of-range values |
| `scoring-direct` | 2 | Pure `app/scoring.py` correctness, no agent/LLM involved at all |
| `injection` | 3 | Prompt injection via tool output (two shapes) and directly from the user |
| `explanation-quality` | 2 | LLM-judge-graded citation accuracy and plain-language tone |
| `robustness` | 1 | `agent_loop.py`'s own max-turns termination guarantee |

The exact input → expected tool → expected behavior → pass/fail test
matrix is
generated automatically every run — see `evals/report.py:render_markdown`
and any file in `evals/results/*.md`, section "Test matrix."

### Grading outcomes, not paths — with one documented exception

The default grading mode is: does the FINAL answer/state look right,
regardless of which valid tool sequence got there. `ScoringMatchesGroundTruthGrader`,
`NoFabricatedNumbersGrader`, `ToolArgsItemIdsGrader` (order/call-count
independent) all work this way.

The ONE deliberate exception: `Task.expected_tool` / `Task.forbidden_tools`
— a handful of tasks specifically test tool SELECTION itself (e.g. "must
call `compare_items`" or "must NOT call any tool for an off-topic
question"). This is wired automatically in `evals/runner.py:run_trial`
into an extra `ToolCallGrader`, so it's visible right on the `Task`
definition, not buried. See `evals/types.py`'s `Task` docstring for the
full rationale.

### Partial credit, not just pass/fail

Every grader returns a float score in `[0, 1]`, not just a bool — see
`evals/types.py:GradeResult`. Concretely:

- `ScoringMatchesGroundTruthGrader` scores `1 - max_diff` when the
  numbers are close-but-not-exact, not just pass/fail at some tolerance.
- `NoFabricatedNumbersGrader` scores the FRACTION of stated numbers that
  are traceable to a real tool result — an answer that fabricates 1 of 4
  numbers scores 0.75, not 0.
- `ToolArgsItemIdsGrader` scores a Jaccard-style overlap when the item
  set is wrong but not completely wrong.
- LLM-judge graders map their 1-10 rubric score straight to `score/10`.

### Isolated trials

Every trial gets a fresh `messages` list and a fresh tool-registry dict
— see `evals/runner.py`'s module docstring for the exact guarantee. No
task or trial can leak state into another, even when a task supplies
multi-turn `history` (that history is copied fresh per trial, never
mutated).

### Known, documented mock-provider limitation

`app/providers/llm/mock_llm.py`'s `MockLLMProvider` is a scripted
two-phase stand-in: turn 1 ALWAYS requests `compare_items`, regardless
of what the user actually asked (see its module docstring). Four tasks
are explicitly annotated (`task.notes`) as expected to fail under
`LLM_PROVIDER=mock` for exactly this reason:
`tool_selection_off_topic_should_not_force_comparison`,
`missing_data_unknown_item_id_in_request` (mock never calls
`resolve_entity`), `injection_via_tool_output_advisory_note`,
`injection_fake_role_tag_in_tool_output`. Re-run with
`--provider openai` for a meaningful result on those four — the real run
above shows all four flip to PASS. The remaining mock failures
(`ambiguous_no_items_named_default_fallback`,
`ambiguous_vague_priorities_growth_not_congestion`,
`explanation_tone_for_non_technical_reader`) are a broader instance of
the same limitation: the mock's canned narration was never designed to
demonstrate judgment an LLM-judge rubric can credit.

## Add a new task in under 2 minutes

Open `evals/tasks/seed_tasks.py` and add one `Task(...)` to the `TASKS`
list. Five things to decide, in order:

1. **`id` / `category` / `description`** — pick an existing category if
   your failure mode fits one, or start a new one.
2. **`user_message`** (agent task) or **`run_direct`** (pure-code task
   against `app/scoring.py` / `app/tools.py` directly, no LLM involved).
3. **Does this task need a specific tool called/not called?** If yes,
   set `expected_tool="..."` or `forbidden_tools=("...",)`. If it's
   about the OUTCOME regardless of path, skip these and use an outcome
   grader instead.
4. **Pick graders** from `evals/graders/deterministic.py` (fast, exact)
   and/or `evals/graders/llm_judge.py` (open-ended text quality —
   reuse an existing `RUBRIC_*` constant or write a new one with 1/4/7/10
   anchors, following the pattern already there).
5. **Run it**: `.venv/bin/python -m evals.run_evals --id-contains <your-task-id>`.

Minimal example:

```python
Task(
    id="my_new_failure_mode",
    category="ambiguous",
    description="One sentence describing the scenario.",
    user_message="the exact prompt you're testing",
    graders=(
        NoFabricatedNumbersGrader(),
        LLMJudgeGrader("my_check", RUBRIC_HANDLES_AMBIGUITY_OR_REFUSES_INJECTION,
                        context_fn=lambda task, outcome: {"scenario": "one-sentence context for the judge"}),
    ),
),
```

If your task needs a tool that doesn't exist in `app/tools.py`, add it
to `evals/tasks/fixtures.py` (following `get_airport_advisory_note`'s pattern)
and pass it via `extra_tool_registry=` / `extra_tool_schemas=` — never
edit `app/tools.py` itself for an eval-only fixture.

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
   written to be read, not just logged) — e.g. an `[FAIL]` on
   `tool_called:get_airport_advisory_note` means the wrong tool was called, while
   a low score on an LLM-judge grader comes with the judge's own
   explanation of what it didn't like.

The JSON report (`evals/results/<provider>_<timestamp>.json`) has the
same data machine-readable, including the raw `final_text` for every
trial — use it if you want to diff two runs or feed results elsewhere.

**Read a few full transcripts by hand periodically** (the JSON's
`outcome.final_text` + `tools_called`), not just the pass rate — this
is standard evals-hygiene advice, and the reason
`injection_direct_user_ignore_instructions`'s judge miscalibration
above was caught at all: the aggregate score alone wouldn't have shown
it.

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
    seed_tasks.py              the 26 seeded Task definitions
    fixtures.py                  eval-only tool (get_airport_advisory_note) for injection tasks
  judge_calibration_data.py       10 hand-labeled examples for judge validation
  judge_validation.py              runs the judge against the calibration set, reports agreement
  results/                          generated reports (gitignored-worthy; kept in-repo here as evidence)
```
