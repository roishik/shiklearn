"""
judge_calibration_data.py — the hand-labeled set evals/judge_validation.py
checks the LLM judge against.

10 examples, each a (tool_context, final_text) pair graded against
RUBRIC_EXPLANATION_CITES_REASONING (evals/graders/llm_judge.py) — the
central "does the explanation cite the right reasoning" rubric named in
the brief. `tool_context` in every example is the REAL, literal output of
`app.tools.compare_items(['LAX', 'SNA'])` (verified by running it
directly — see the docstring below) so every example is grounded in an
actual tool result, not an invented one; only `final_text` varies,
deliberately spanning the full 1-10 range so the judge is tested on hard
cases, not just obvious ones.

Ground truth (from `python -c "from app.tools import compare_items;
print(compare_items(['LAX','SNA']))"`, verified 2026-08-18):
  LAX total_score=0.3584  (traffic_growth -3.35%->0.21, regional_demand
    -0.18%->0.02, catchment_monopoly 4.4mi->0.0, capacity_pressure
    9.12M/runway->1.0, absolute_scale 36.5M->1.0)
  SNA total_score=0.2917  (traffic_growth +2.82%->0.49, regional_demand
    -0.07%->0.07, catchment_monopoly 18.9mi->0.09, capacity_pressure
    5.52M/runway->0.70, absolute_scale 5.52M->0.19)
  LAX wins by 0.0667 -- a clean, non-tied margin (well above
  compare_items' own DECISIVE_SCORE_GAP=0.005), chosen deliberately so
  citation ACCURACY is the only thing under test here. A near-tied pair
  (e.g. BNA vs DEN, found in P2) would conflate this rubric with the
  separate tie-handling behavior system_prompt.py rule 6 governs.

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

# The real, literal tool output for LAX vs SNA — see module docstring.
TOOL_CONTEXT_LAX_VS_SNA = {
    "criteria": [
        {"name": "traffic_growth", "weight": 25, "higher_is_better": True},
        {"name": "regional_demand_growth", "weight": 25, "higher_is_better": True},
        {"name": "catchment_monopoly", "weight": 20, "higher_is_better": True},
        {"name": "capacity_pressure", "weight": 15, "higher_is_better": True},
        {"name": "absolute_scale", "weight": 15, "higher_is_better": True},
    ],
    "ranking": [
        {
            "rank": 1,
            "item_id": "LAX",
            "total_score": 0.3584,
            "components": [
                {"criterion": "traffic_growth", "raw_value": -0.0335, "normalized_score": 0.2089, "weight": 0.25, "contribution": 0.0522},
                {"criterion": "regional_demand_growth", "raw_value": -0.00184, "normalized_score": 0.0248, "weight": 0.25, "contribution": 0.0062},
                {"criterion": "catchment_monopoly", "raw_value": 4.4, "normalized_score": 0.0, "weight": 0.2, "contribution": 0.0},
                {"criterion": "capacity_pressure", "raw_value": 9124325.75, "normalized_score": 1.0, "weight": 0.15, "contribution": 0.15},
                {"criterion": "absolute_scale", "raw_value": 36497303.0, "normalized_score": 1.0, "weight": 0.15, "contribution": 0.15},
            ],
        },
        {
            "rank": 2,
            "item_id": "SNA",
            "total_score": 0.2917,
            "components": [
                {"criterion": "traffic_growth", "raw_value": 0.0282, "normalized_score": 0.4932, "weight": 0.25, "contribution": 0.1233},
                {"criterion": "regional_demand_growth", "raw_value": -0.00071, "normalized_score": 0.0673, "weight": 0.25, "contribution": 0.0168},
                {"criterion": "catchment_monopoly", "raw_value": 18.9, "normalized_score": 0.0912, "weight": 0.2, "contribution": 0.0182},
                {"criterion": "capacity_pressure", "raw_value": 5521729.0, "normalized_score": 0.698, "weight": 0.15, "contribution": 0.1047},
                {"criterion": "absolute_scale", "raw_value": 5521729.0, "normalized_score": 0.191, "weight": 0.15, "contribution": 0.0286},
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
            object.__setattr__(self, "tool_context", TOOL_CONTEXT_LAX_VS_SNA)

    @property
    def human_pass(self) -> bool:
        return self.human_score >= 7


CALIBRATION_SET: list[CalibrationExample] = [
    CalibrationExample(
        id="perfect_citation",
        final_text=(
            "LAX ranks first with a total score of 0.3584, driven almost entirely by scale: it's operating at "
            "9.12M passengers per runway (normalized to the maximum, 1.0, contributing 0.15) and has 36.5M "
            "annual passengers overall (also maxed out, contributing another 0.15). SNA trails at 0.2917 despite "
            "actually beating LAX on the forward-looking signals — SNA's traffic is growing 2.82% year over year "
            "versus LAX's -3.35% decline, and SNA has a much more isolated catchment (18.9 miles to the nearest "
            "competitor vs. LAX's 4.4). But those two criteria are only 45% of the weight combined, and LAX's "
            "capacity pressure and absolute scale are enough to carry it regardless."
        ),
        human_score=10,
        human_rationale="Correct winner, correct numbers for every criterion on both items, explains WHY LAX "
        "wins despite SNA's better growth and catchment signals — exactly what the rubric's 10-anchor asks for.",
    ),
    CalibrationExample(
        id="fabricated_total_score",
        final_text="LAX wins with a total score of 0.81, comfortably ahead of SNA at 0.44.",
        human_score=1,
        human_rationale="Both numbers are fabricated — the real scores are 0.3584 and 0.2917. Directly "
        "contradicts the tool context, which is anchor-1 territory verbatim.",
    ),
    CalibrationExample(
        id="vague_no_specifics",
        final_text="LAX is just the more congested airport overall, it edges out SNA pretty comfortably.",
        human_score=4,
        human_rationale="Correct winner, but zero specific criteria or values cited — textbook anchor-4 case.",
    ),
    CalibrationExample(
        id="two_criteria_cited_slightly_unclear",
        final_text=(
            "LAX comes out on top mainly because of its passenger volume (36.5M vs 5.52M for SNA) and it also "
            "does fine on capacity pressure. SNA is a bit better on the other stuff but not enough to catch up."
        ),
        human_score=7,
        human_rationale="Cites two real numbers correctly (absolute_scale 36.5M/5.52M, capacity_pressure "
        "gestured at but not quantified), but the reasoning is a little muddled — scale alone doesn't fully "
        "explain the win since growth and catchment monopoly carry 45% combined weight — and SNA's better "
        "criteria are hand-waved as 'the other stuff.' Solid anchor-7: right but a bit unclear.",
    ),
    CalibrationExample(
        id="wrong_winner_stated",
        final_text="SNA wins this comparison with the higher total score, thanks to its strong traffic growth of 2.82%.",
        human_score=1,
        human_rationale="States the wrong winner outright (LAX actually won, 0.3584 > 0.2917) — a real, "
        "consequential error a reader would act on incorrectly. Anchor-1: contradicts the data.",
    ),
    CalibrationExample(
        id="jargon_heavy_but_numerically_accurate",
        final_text=(
            "item_id=LAX total_score=0.3584 [traffic_growth: raw=-0.0335 normalized_score=0.2089 weight=0.25 "
            "contribution=0.0522; regional_demand_growth: raw=-0.00184 normalized_score=0.0248 weight=0.25 "
            "contribution=0.0062; catchment_monopoly: raw=4.4 normalized_score=0.0 weight=0.2 contribution=0.0; "
            "capacity_pressure: raw=9124325.75 normalized_score=1.0 weight=0.15 contribution=0.15; "
            "absolute_scale: raw=36497303.0 normalized_score=1.0 weight=0.15 contribution=0.15]. "
            "item_id=SNA total_score=0.2917 [traffic_growth: raw=0.0282 normalized_score=0.4932 weight=0.25 "
            "contribution=0.1233; regional_demand_growth: raw=-0.00071 normalized_score=0.0673 weight=0.25 "
            "contribution=0.0168; catchment_monopoly: raw=18.9 normalized_score=0.0912 weight=0.2 "
            "contribution=0.0182; capacity_pressure: raw=5521729.0 normalized_score=0.698 weight=0.15 "
            "contribution=0.1047; absolute_scale: raw=5521729.0 normalized_score=0.191 weight=0.15 "
            "contribution=0.0286]."
        ),
        human_score=8,
        human_rationale="Every single number is correct and complete — this rubric grades ACCURACY of citation, "
        "not plain-language tone (that's a separate rubric, RUBRIC_TONE_FOR_NON_TECHNICAL_READER). Docked from "
        "10 only because it never actually explains WHY LAX wins in words, just dumps the raw structure.",
    ),
    CalibrationExample(
        id="one_wrong_number_among_real_ones",
        final_text=(
            "LAX wins overall. Its passenger volume is 36.5M and its capacity pressure is high, both "
            "outweighing SNA's numbers. Its regional demand growth contributes about 0.6 to the total, which is "
            "the deciding factor."
        ),
        human_score=4,
        human_rationale="Passenger volume is correct, but the regional_demand_growth contribution (real value "
        "0.0062, tiny) is misstated as 0.6 — nearly 100x too large, and it isn't the deciding factor at all "
        "(absolute_scale and capacity_pressure are). A real citation attempt corrupted by a wrong number and a "
        "wrong causal claim: anchor-4, not anchor-7.",
    ),
    CalibrationExample(
        id="off_topic_non_answer",
        final_text="I'm not sure what you're looking for — could you clarify what you'd like me to compare?",
        human_score=1,
        human_rationale="A real tool result was available and the question was unambiguous (compare LAX and "
        "SNA) — this is a non-answer that ignores the data entirely. Anchor-1: unrelated to the data.",
    ),
    CalibrationExample(
        id="clear_accurate_plain_narrative",
        final_text=(
            "LAX comes out ahead here, mostly because of sheer size and how much traffic it handles per runway "
            "— it's maxed out on both of those measures. SNA is actually growing faster (2.82% year over year, "
            "versus LAX which is shrinking about 3%) and has a bigger buffer from its nearest competitor — 4.4 "
            "miles for LAX, wait, that's backwards, SNA's competitor is 18.9 miles away, LAX's is only 4.4 — but "
            "those forward-looking numbers aren't enough to overcome how much bigger and busier LAX already is. "
            "Final scores: LAX 0.3584, SNA 0.2917 — LAX wins by a clear margin."
        ),
        human_score=9,
        human_rationale="Correct winner, correct final numbers, cites specific criteria values accurately "
        "(growth 2.82% vs -3.35%, catchment 4.4mi vs 18.9mi — even self-corrects a slip mid-sentence rather than "
        "leaving a wrong claim standing), and it's readable narrative prose. Not a perfect 10 only because the "
        "self-correction reads a little clumsy and it slightly undersells why the 45%-weighted forward signals "
        "still lost.",
    ),
    CalibrationExample(
        id="criteria_names_right_values_swapped",
        final_text=(
            "LAX wins. Its traffic growth is 2.82%, the strongest of the two, and its nearest competitor is "
            "18.9 miles away, the most isolated market position available. SNA trails on both fronts."
        ),
        human_score=2,
        human_rationale="Uses the right criterion NAMES (traffic_growth, catchment_monopoly) but attributes "
        "SNA's actual values (growth 2.82%, catchment 18.9mi) to LAX, and reverses the real comparison (LAX's "
        "real growth is -3.35% and catchment is 4.4mi — SNA is the one growing faster and more isolated, just "
        "far smaller overall). This is the subtle-fabrication case: looks well-cited at a glance, is wrong on "
        "inspection. Anchor-1/2 territory.",
    ),
]
