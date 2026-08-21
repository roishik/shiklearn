"""
judge_calibration_data.py — the hand-labeled set evals/judge_validation.py
checks the LLM judge against.

10 examples, each a (tool_context, final_text) pair graded against
RUBRIC_EXPLANATION_CITES_REASONING (evals/graders/llm_judge.py) — the
central "does the explanation cite the right reasoning" rubric named in
the brief. `tool_context` in every example is the REAL, literal output of
`app.tools.compare_items(['6900', '8700'])` (Kefar Sava vs Ra'anana —
verified by running it directly against the real dataset, see the
docstring below) so every example is grounded in an actual tool result,
not an invented one; only `final_text` varies, deliberately spanning the
full 1-10 range so the judge is tested on hard cases, not just obvious
ones.

Ground truth (from `PYTHONPATH=. python -c "from app.tools import
compare_items; import json; print(json.dumps(compare_items(['6900',
'8700']), indent=2))"`, verified 2026-08-21 against the real, built
104-locality dataset):

  Kefar Sava (6900) total_score=0.5797  (price_level 2,753,850 ₪ -> 0.2633
    normalized, contributes 0.079; accessibility 16.28 km -> 0.8209,
    contributes 0.1642; price_momentum_vs_country +0.97% -> 0.5207,
    contributes 0.1041; socioeconomic_level cluster 8 -> 0.8571,
    contributes 0.1286; rental_yield 3.27% -> 0.6921, contributes 0.1038)

  Ra'anana (8700) total_score=0.4701  (price_level 3,381,700 ₪ -> 0.0316,
    contributes 0.0095; accessibility 13.86 km -> 0.8581, contributes
    0.1716; price_momentum_vs_country +1.97% -> 0.6298, contributes
    0.126; socioeconomic_level cluster 8 -> 0.8571, contributes 0.1286;
    rental_yield 2.51% -> 0.2297, contributes 0.0345)

  Kefar Sava wins by 0.1096 -- a clean, non-tied margin (well above
  compare_items' own DECISIVE_SCORE_GAP=0.005), chosen deliberately so
  citation ACCURACY is the only thing under test here. A near-tied pair
  (e.g. the real #1/#2, Qiryat Motzkin vs Judeide-Maker, 0.0012 apart —
  see _working/agent-logs/scripts-sensitivity.md) would conflate this
  rubric with the separate tie-handling behavior system_prompt.py rule 6
  governs; that near-tie is instead the subject of a dedicated task in
  seed_tasks.py's "honesty about uncertainty" category, not this rubric.

  The real story here is genuinely interesting and worth knowing when
  reading the examples below: Ra'anana actually has the BETTER raw value
  on 2 of the 5 criteria (closer to the core cities at 13.86km vs
  16.28km, and faster price growth vs the national average at +1.97% vs
  +0.97%) and TIES on a third (socioeconomic cluster 8 for both) — but
  Kefar Sava still wins comfortably, because it is far cheaper
  (2,753,850 ₪ vs 3,381,700 ₪, the single largest weight at 30) and has a
  meaningfully higher rental yield (3.27% vs 2.51%). A citation that
  narrates "Kefar Sava wins across the board" would therefore be
  factually wrong on two specific claims even though it names the right
  overall winner — exactly the kind of subtle, checkable error this
  rubric and this calibration set exist to catch.

Each example is hand-labeled by a human (Roi, acting as the "intern" in
the research brief's intern test: "hand your rubric plus traces to
someone unfamiliar with the project — if their pass/fail matches yours
>=80% of the time, the rubric is specific enough to automate"). The
human's own reasoning is recorded in `human_rationale` so a reviewer can
audit the labels themselves, not just trust them.

`human_pass` uses the SAME threshold (score >= 7) as LLMJudgeGrader's
default `pass_threshold` — see evals/graders/llm_judge.py — so the
binary-agreement number in judge_validation.py is comparing apples to
apples.
"""
from __future__ import annotations

from dataclasses import dataclass

# The real, literal tool output for Kefar Sava vs Ra'anana — see module
# docstring. Field names/values copied verbatim from a live run of
# app.tools.compare_items(['6900', '8700']) against the real dataset.
TOOL_CONTEXT_KEFAR_SAVA_VS_RAANANA = {
    "criteria": [
        {"name": "price_level", "weight": 30, "higher_is_better": False},
        {"name": "accessibility", "weight": 20, "higher_is_better": False},
        {"name": "price_momentum_vs_country", "weight": 20, "higher_is_better": True},
        {"name": "socioeconomic_level", "weight": 15, "higher_is_better": True},
        {"name": "rental_yield", "weight": 15, "higher_is_better": True},
    ],
    "decisive": True,
    "tied_at_top": [],
    "ranking": [
        {
            "rank": 1,
            "item_id": "6900",
            "name_en": "Kefar Sava",
            "total_score": 0.5797,
            "components": [
                {"criterion": "price_level", "raw_value": 2753850.0, "normalized_score": 0.2633, "weight": 0.3, "contribution": 0.079},
                {"criterion": "accessibility", "raw_value": 16.28, "normalized_score": 0.8209, "weight": 0.2, "contribution": 0.1642},
                {"criterion": "price_momentum_vs_country", "raw_value": 0.009663, "normalized_score": 0.5207, "weight": 0.2, "contribution": 0.1041},
                {"criterion": "socioeconomic_level", "raw_value": 8.0, "normalized_score": 0.8571, "weight": 0.15, "contribution": 0.1286},
                {"criterion": "rental_yield", "raw_value": 3.27, "normalized_score": 0.6921, "weight": 0.15, "contribution": 0.1038},
            ],
        },
        {
            "rank": 2,
            "item_id": "8700",
            "name_en": "Ra'annana",
            "total_score": 0.4701,
            "components": [
                {"criterion": "price_level", "raw_value": 3381700.0, "normalized_score": 0.0316, "weight": 0.3, "contribution": 0.0095},
                {"criterion": "accessibility", "raw_value": 13.86, "normalized_score": 0.8581, "weight": 0.2, "contribution": 0.1716},
                {"criterion": "price_momentum_vs_country", "raw_value": 0.019718, "normalized_score": 0.6298, "weight": 0.2, "contribution": 0.126},
                {"criterion": "socioeconomic_level", "raw_value": 8.0, "normalized_score": 0.8571, "weight": 0.15, "contribution": 0.1286},
                {"criterion": "rental_yield", "raw_value": 2.51, "normalized_score": 0.2297, "weight": 0.15, "contribution": 0.0345},
            ],
        },
    ],
}


@dataclass(frozen=True)
class CalibrationExample:
    id: str
    final_text: str
    human_score: int  # 1-10, hand-assigned against the same rubric anchors the judge uses
    human_rationale: str
    tool_context: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.tool_context is None:
            object.__setattr__(self, "tool_context", TOOL_CONTEXT_KEFAR_SAVA_VS_RAANANA)

    @property
    def human_pass(self) -> bool:
        return self.human_score >= 7


CALIBRATION_SET: list[CalibrationExample] = [
    CalibrationExample(
        id="perfect_citation",
        final_text=(
            "Kefar Sava comes out ahead here, with a total score of 0.5797 against Ra'anana's "
            "0.4701 — a clear, non-tied margin. The two biggest drivers are price and rental "
            "yield: Kefar Sava's median 4-room home is 2,753,850 ₪ versus Ra'anana's 3,381,700 ₪ "
            "(price_level is the single largest weight, 30, and Kefar Sava's normalized score "
            "there is 0.2633 against Ra'anana's 0.0316), and Kefar Sava's rental yield is 3.27% "
            "versus Ra'anana's 2.51%. Ra'anana actually has the edge on two other fronts — it's "
            "closer to the core cities (13.86km vs 16.28km) and its prices are growing faster "
            "than the national average (+1.97% vs +0.97%) — and the two are tied on "
            "socioeconomic level (cluster 8 for both), but those advantages aren't enough to "
            "overcome how much cheaper and higher-yielding Kefar Sava is."
        ),
        human_score=10,
        human_rationale="Correct winner, correct numbers for every criterion on both items, "
        "explains WHY Kefar Sava wins despite Ra'anana's better accessibility and momentum — "
        "exactly what the rubric's 10-anchor asks for, and it correctly avoids the tempting but "
        "false 'Kefar Sava wins across the board' narrative.",
    ),
    CalibrationExample(
        id="fabricated_total_score",
        final_text="Kefar Sava wins with a total score of 0.81, comfortably ahead of Ra'anana at 0.44.",
        human_score=1,
        human_rationale="Both numbers are fabricated — the real scores are 0.5797 and 0.4701. "
        "Directly contradicts the tool context, which is anchor-1 territory verbatim.",
    ),
    CalibrationExample(
        id="vague_no_specifics",
        final_text="Kefar Sava is just better value overall, it edges out Ra'anana pretty comfortably.",
        human_score=4,
        human_rationale="Correct winner, but zero specific criteria or values cited — textbook anchor-4 case.",
    ),
    CalibrationExample(
        id="two_criteria_cited_slightly_unclear",
        final_text=(
            "Kefar Sava comes out on top mainly because it's cheaper (2,753,850 ₪ vs 3,381,700 ₪) "
            "and it also does fine on rental yield. Ra'anana is a bit better on the other stuff "
            "but not enough to catch up."
        ),
        human_score=7,
        human_rationale="Cites two real numbers correctly (price_level values, rental_yield "
        "gestured at but not quantified), but the reasoning is a little muddled — price alone "
        "doesn't fully explain the win since accessibility and momentum together carry 40% "
        "weight — and Ra'anana's better criteria are hand-waved as 'the other stuff.' Solid "
        "anchor-7: right but a bit unclear.",
    ),
    CalibrationExample(
        id="wrong_winner_stated",
        final_text="Ra'anana wins this comparison with the higher total score, thanks to its faster price growth of 1.97%.",
        human_score=1,
        human_rationale="States the wrong winner outright (Kefar Sava actually won, 0.5797 > "
        "0.4701) — a real, consequential error a reader would act on incorrectly. Anchor-1: "
        "contradicts the data.",
    ),
    CalibrationExample(
        id="jargon_heavy_but_numerically_accurate",
        final_text=(
            "item_id=6900 total_score=0.5797 [price_level: raw=2753850.0 normalized_score=0.2633 "
            "weight=0.3 contribution=0.079; accessibility: raw=16.28 normalized_score=0.8209 "
            "weight=0.2 contribution=0.1642; price_momentum_vs_country: raw=0.009663 "
            "normalized_score=0.5207 weight=0.2 contribution=0.1041; socioeconomic_level: raw=8.0 "
            "normalized_score=0.8571 weight=0.15 contribution=0.1286; rental_yield: raw=3.27 "
            "normalized_score=0.6921 weight=0.15 contribution=0.1038]. item_id=8700 "
            "total_score=0.4701 [price_level: raw=3381700.0 normalized_score=0.0316 weight=0.3 "
            "contribution=0.0095; accessibility: raw=13.86 normalized_score=0.8581 weight=0.2 "
            "contribution=0.1716; price_momentum_vs_country: raw=0.019718 normalized_score=0.6298 "
            "weight=0.2 contribution=0.126; socioeconomic_level: raw=8.0 normalized_score=0.8571 "
            "weight=0.15 contribution=0.1286; rental_yield: raw=2.51 normalized_score=0.2297 "
            "weight=0.15 contribution=0.0345]."
        ),
        human_score=8,
        human_rationale="Every single number is correct and complete — this rubric grades "
        "ACCURACY of citation, not plain-language tone (that's a separate rubric, "
        "RUBRIC_TONE_FOR_NON_TECHNICAL_READER). Docked from 10 only because it never actually "
        "explains WHY Kefar Sava wins in words, just dumps the raw structure.",
    ),
    CalibrationExample(
        id="one_wrong_number_among_real_ones",
        final_text=(
            "Kefar Sava wins overall. Its median price is 2,753,850 ₪ and its rental yield is "
            "solid, both outweighing Ra'anana's numbers. Its accessibility contributes about 0.6 "
            "to the total, which is the deciding factor."
        ),
        human_score=4,
        human_rationale="Median price is correct, but the accessibility contribution (real value "
        "0.1642) is misstated as 0.6 — nearly 4x too large, and it isn't the deciding factor at "
        "all (price_level and rental_yield are, and accessibility actually favors Ra'anana, the "
        "loser). A real citation attempt corrupted by a wrong number and a wrong causal claim: "
        "anchor-4, not anchor-7.",
    ),
    CalibrationExample(
        id="off_topic_non_answer",
        final_text="I'm not sure what you're looking for — could you clarify what you'd like me to compare?",
        human_score=1,
        human_rationale="A real tool result was available and the question was unambiguous "
        "(compare Kefar Sava and Ra'anana) — this is a non-answer that ignores the data "
        "entirely. Anchor-1: unrelated to the data.",
    ),
    CalibrationExample(
        id="clear_accurate_plain_narrative",
        final_text=(
            "Kefar Sava comes out ahead here, mostly because it's meaningfully cheaper and rents "
            "for more relative to its price. Ra'anana is actually closer to the big job centers "
            "(13.86km vs 16.28km) and its prices are growing a bit faster than the national "
            "average — wait, growing faster than Kefar Sava's, I mean, both are growing faster "
            "than the country as a whole, Ra'anana just more so (1.97% vs 0.97%) — but those "
            "edges aren't enough to close a price gap this size. Final scores: Kefar Sava 0.5797, "
            "Ra'anana 0.4701 — a clear win for Kefar Sava."
        ),
        human_score=9,
        human_rationale="Correct winner, correct final numbers, cites specific criteria values "
        "accurately (accessibility 13.86km vs 16.28km, momentum 1.97% vs 0.97% — even "
        "self-corrects a slip mid-sentence rather than leaving a wrong claim standing), and it's "
        "readable narrative prose. Not a perfect 10 only because the self-correction reads a "
        "little clumsy and it never mentions rental_yield, one of the two criteria that actually "
        "decided the outcome.",
    ),
    CalibrationExample(
        id="criteria_names_right_values_swapped",
        final_text=(
            "Kefar Sava wins. Its price growth is 1.97%, the strongest of the two, and it's the "
            "closer of the two to the core cities at 13.86km. Ra'anana trails on both fronts."
        ),
        human_score=2,
        human_rationale="Uses the right criterion NAMES (price_momentum_vs_country, "
        "accessibility) but attributes Ra'anana's actual values (momentum 1.97%, accessibility "
        "13.86km) to Kefar Sava, and reverses the real comparison (Kefar Sava's real momentum is "
        "0.97% and accessibility is 16.28km — Ra'anana is the one growing faster and closer to "
        "the core, just far more expensive with lower yield overall). This is the "
        "subtle-fabrication case: looks well-cited at a glance, is wrong on inspection. "
        "Anchor-1/2 territory.",
    ),
]
