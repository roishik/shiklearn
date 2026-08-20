"""
scoring.py — deterministic scoring / ranking logic.

RULES FOR THIS FILE, enforced by construction, not just convention:
  1. ZERO I/O. No network calls, no file reads, no env var reads, no
     imports of requests/httpx/openai/anthropic/os (other than typing).
  2. ZERO LLM calls. This file must be understandable and fully testable
     without any API key, network access, or running agent.
  3. Every function is PURE: same inputs -> same outputs. No globals
     mutated, no wall-clock/randomness (unless a seed is threaded through
     explicitly as an argument, which nothing here needs).
  4. This is the single most defensible artifact in the whole project —
     it's the literal answer to "show me the deterministic scoring logic"
     for the Israel Home-Buying Intelligence Agent. Point to this file,
     and to test_scoring.py, when asked "what part of your ranking is NOT
     the LLM?"

The Criterion / ComponentBreakdown / ScoreResult / RankedItem shapes here
are the contract app/tools.py and every LLM-facing explanation depend on
— DEFAULT_CRITERIA in tools.py supplies this domain's real KPIs for
ranking Israeli localities and neighborhoods (e.g. affordability, price
trend, price-to-income, amenity access; see DESIGN.md), and nothing in
this file knows or cares what any of them mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Criterion:
    """One scoring dimension: a name, a relative weight, and a min-max
    window that turns a raw value into a 0..1 normalized score.

    Weights are relative, not required to sum to 1 — score_item() and
    rank_items() normalize them internally so callers never have to do
    that arithmetic by hand (and can't get it silently wrong).
    """

    name: str
    weight: float
    lower_bound: float
    upper_bound: float
    higher_is_better: bool = True

    def normalize(self, raw_value: float) -> float:
        """Min-max normalize `raw_value` into [0, 1], clamped at both ends
        so an out-of-range raw value degrades gracefully instead of
        producing a score outside [0, 1] or crashing.

        If lower_bound == upper_bound (no information from this
        criterion — e.g. every candidate has the identical value), every
        raw value normalizes to 0.5. This is a deliberate choice, not a
        bug: it's the "criterion has zero discriminating power right now"
        signal, and it avoids a divide-by-zero crash on a plausible real
        input (e.g. a single-item comparison, or a KPI that happens to
        tie across every option that was fetched).
        """
        if self.upper_bound == self.lower_bound:
            return 0.5
        span = self.upper_bound - self.lower_bound
        position = (raw_value - self.lower_bound) / span
        position = max(0.0, min(1.0, position))
        return position if self.higher_is_better else 1.0 - position


@dataclass(frozen=True)
class ComponentBreakdown:
    """Per-criterion contribution to a total score. This is what a tool
    (see tools.py) returns to the LLM so it can EXPLAIN a score it never
    calculated: raw input, normalized score, the weight actually used
    (already renormalized to sum to 1 across all criteria), and the
    resulting contribution to the total."""

    criterion: str
    raw_value: float
    normalized_score: float  # 0..1
    weight: float  # normalized weight; all components for one item sum to 1
    contribution: float  # normalized_score * weight


@dataclass(frozen=True)
class ScoreResult:
    item_id: str
    total_score: float  # 0..1, sum of all component contributions
    components: tuple[ComponentBreakdown, ...]
    covered_weight: float  # fraction of DECLARED weight actually backed by data, 0..1
    missing_criteria: tuple[str, ...]  # declared criteria absent from raw_values


@dataclass(frozen=True)
class RankedItem:
    rank: int  # 1-based
    item_id: str
    total_score: float
    components: tuple[ComponentBreakdown, ...]
    covered_weight: float
    missing_criteria: tuple[str, ...]


@dataclass(frozen=True)
class ExcludedItem:
    """An item that had SOME data but not enough of it — excluded from
    the ranking rather than scored on scraps. Distinct from an item that
    was never fetched at all: this one made it into score_item and was
    turned away at the coverage floor."""

    item_id: str
    covered_weight: float
    missing_criteria: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class RankingResult:
    ranked: tuple[RankedItem, ...]
    excluded: tuple[ExcludedItem, ...]


class InsufficientCoverageError(ValueError):
    """Raised by score_item when the fraction of declared weight backed by
    available raw_values fails to exceed coverage_threshold. Deliberately
    a distinct type from the plain ValueError raised for caller bugs
    (empty criteria list, non-positive weight sum) so rank_items can catch
    ONLY this one and route it to `excluded` — a real caller bug should
    still propagate and crash the batch, not get silently absorbed into
    "excluded".

    Zero coverage is not special-cased: 0.0 never exceeds any positive
    coverage_threshold, so "all criteria missing" and "some criteria
    missing" are the same code path here, just different degrees of it.

    Carries `covered_weight` and `missing_criteria` so rank_items can
    build its `excluded` entry from what score_item ACTUALLY measured,
    rather than recomputing the same arithmetic a second time — two
    copies of that formula could drift apart and make the excluded list
    report a coverage number that isn't the one the floor rejected.
    """

    def __init__(self, message: str, covered_weight: float, missing_criteria: tuple[str, ...]):
        super().__init__(message)
        self.covered_weight = covered_weight
        self.missing_criteria = missing_criteria


def score_item(
    item_id: str,
    raw_values: Mapping[str, float],
    criteria: list[Criterion],
    coverage_threshold: float = 0.5,
) -> ScoreResult:
    """Score a single item against a list of criteria.

    Real public datasets are ragged — not every item reports every KPI.
    Rather than fail the whole item on any single missing criterion,
    this drops absent criteria and renormalizes weight over the ones
    that ARE present. But it still fails loudly — via
    InsufficientCoverageError — when too little of the declared weight
    survives that drop, per `coverage_threshold`: a score built on
    scraps is worse than no score, same principle as the original
    all-or-nothing behavior, just measured in weight covered rather than
    criteria count.

    `coverage_threshold` must be EXCEEDED, not merely met: the default
    0.5 means "more than half the declared weight must be backed by real
    data," so an item sitting at exactly half is excluded — enough
    evidence means most of the evidence, and half is not most. It's a
    reachable case, not a theoretical one: any single criterion carrying
    half the weight puts an item exactly on the line the moment that one
    criterion is missing. Pinned from both sides by
    test_score_item_exactly_at_floor_is_excluded_not_scored and
    test_score_item_just_above_floor_is_scored.
    """
    if not criteria:
        raise ValueError("criteria must be a non-empty list")

    weight_sum = sum(c.weight for c in criteria)
    if weight_sum <= 0:
        raise ValueError("sum of criterion weights must be > 0")

    available = [c for c in criteria if c.name in raw_values]
    missing = tuple(c.name for c in criteria if c.name not in raw_values)

    available_weight_sum = sum(c.weight for c in available)
    covered_weight = available_weight_sum / weight_sum
    if covered_weight <= coverage_threshold:
        raise InsufficientCoverageError(
            f"{item_id!r}: covered_weight={covered_weight:.3f} <= "
            f"coverage_threshold={coverage_threshold:.3f}; missing criteria: {list(missing)}",
            covered_weight=covered_weight,
            missing_criteria=missing,
        )

    # Belt-and-braces against a ZeroDivisionError in the renormalization
    # loop below. With any sane coverage_threshold >= 0 this is
    # unreachable — zero available weight means covered_weight == 0.0,
    # which the check above already catches. It earns its place only for
    # a caller who passes a NEGATIVE coverage_threshold ("rank absolutely
    # everything"), where the check above lets zero coverage through.
    # Without it that splits two ways, both bad: a raw ZeroDivisionError
    # (which rank_items does NOT catch, so one item kills the whole
    # batch), or a total_score of 0.0 with no components, which reads as
    # "scored zero" when it means "never scored." Same recoverable class
    # as any other coverage failure, so rank_items routes it to
    # `excluded` like the rest.
    if available_weight_sum <= 0:
        raise InsufficientCoverageError(
            f"{item_id!r}: no available criterion carries positive weight "
            f"(covered_weight={covered_weight:.3f}); missing criteria: {list(missing)}",
            covered_weight=covered_weight,
            missing_criteria=missing,
        )

    components: list[ComponentBreakdown] = []
    total = 0.0
    for criterion in available:
        normalized_weight = criterion.weight / available_weight_sum
        normalized_score = criterion.normalize(raw_values[criterion.name])
        contribution = normalized_score * normalized_weight
        total += contribution
        components.append(
            ComponentBreakdown(
                criterion=criterion.name,
                raw_value=raw_values[criterion.name],
                normalized_score=normalized_score,
                weight=normalized_weight,
                contribution=contribution,
            )
        )

    return ScoreResult(
        item_id=item_id,
        total_score=total,
        components=tuple(components),
        covered_weight=covered_weight,
        missing_criteria=missing,
    )


@dataclass(frozen=True)
class RankChange:
    """How one item moved between the baseline and perturbed rankings.

    Ranks are Optional because a weight change can move an item IN OR OUT
    of the ranking entirely, not just up and down it: covered_weight is a
    fraction of TOTAL declared weight, so raising a criterion's weight
    lowers the coverage of every item missing that criterion, and can
    push one under the threshold. Reporting that as a rank of 0, or
    silently dropping it, would hide a real consequence of the weight
    choice.
    """

    item_id: str
    baseline_rank: int | None
    perturbed_rank: int | None
    rank_delta: int | None  # +ve = moved DOWN the ranking; None if it entered/left
    baseline_score: float | None
    perturbed_score: float | None
    score_delta: float | None


@dataclass(frozen=True)
class SensitivityResult:
    baseline_weights: tuple[tuple[str, float], ...]
    perturbed_weights: tuple[tuple[str, float], ...]
    changes: tuple[RankChange, ...]
    baseline_top: str | None
    perturbed_top: str | None
    top_changed: bool
    max_rank_movement: int
    items_moved: int
    items_entered: tuple[str, ...]  # excluded at baseline, ranked after
    items_left: tuple[str, ...]  # ranked at baseline, excluded after
    kendall_tau: float  # +1 identical order, -1 fully reversed, computed over items ranked in BOTH


def apply_weight_overrides(criteria: list[Criterion], overrides: Mapping[str, float]) -> list[Criterion]:
    """Return a new criteria list with the named weights replaced.

    Pure — the input list is untouched, so a caller can't accidentally
    poison the baseline it's comparing against. Raises on an unknown
    criterion name rather than silently ignoring it: a typo'd criterion in
    a sensitivity run would otherwise report "no change" and read as
    reassuring stability, which is the exact opposite of the truth.
    """
    known = {c.name for c in criteria}
    unknown = [name for name in overrides if name not in known]
    if unknown:
        raise ValueError(f"unknown criteria in weight overrides: {unknown}; known: {sorted(known)}")

    return [
        Criterion(
            name=c.name,
            weight=overrides.get(c.name, c.weight),
            lower_bound=c.lower_bound,
            upper_bound=c.upper_bound,
            higher_is_better=c.higher_is_better,
        )
        for c in criteria
    ]


def scale_criterion_weight(criteria: list[Criterion], name: str, factor: float) -> list[Criterion]:
    """Multiply one criterion's weight by `factor` ('what if I halve it?'
    is factor=0.5). Multiplicative rather than additive because weights
    are RELATIVE and renormalized internally — '+0.2' means nothing
    without knowing the other weights, while 'half as important' is
    scale-free and is how the question actually gets asked."""
    current = {c.name: c.weight for c in criteria}
    if name not in current:
        raise ValueError(f"unknown criterion {name!r}; known: {sorted(current)}")
    return apply_weight_overrides(criteria, {name: current[name] * factor})


# How much emphasising a criterion multiplies its weight (and
# de-emphasising divides it). Bounded and fixed ON PURPOSE — see
# apply_priority_emphasis.
PRIORITY_EMPHASIS_FACTOR = 2.0


def apply_priority_emphasis(
    criteria: list[Criterion],
    emphasize: Sequence[str] = (),
    deemphasize: Sequence[str] = (),
    factor: float = PRIORITY_EMPHASIS_FACTOR,
) -> list[Criterion]:
    """Re-weight criteria to reflect a user's stated priorities ("I care
    about fast and cheap"), by a FIXED multiplier rather than by numbers
    supplied from outside.

    WHY BOUNDED, AND WHY THIS ISN'T A HOLE IN NEVER_COMPUTE_RULE:
    The caller here is ultimately the LLM, translating "fast and cheap"
    into criterion NAMES. That translation is preference elicitation —
    mapping natural language onto a known vocabulary — which is exactly
    what a language model should do, and it is not the same thing as
    computing a score. Every number still comes from this module.

    But letting the model pass arbitrary weights WOULD be a hole: it
    could set quality to 0.001 and "discover" that its preferred item
    wins, producing motivated reasoning dressed as arithmetic. So the
    model gets a verb, not a value — it may say WHICH criteria matter
    more, never BY HOW MUCH. A criterion can at most double or halve in
    relative weight, so no criterion can be silently switched off.

    Naming a criterion in both lists is a contradiction, not a no-op, and
    raises — silently cancelling would hide a confused request.
    """
    known = {c.name for c in criteria}
    unknown = sorted((set(emphasize) | set(deemphasize)) - known)
    if unknown:
        raise ValueError(f"unknown criteria in priorities: {unknown}; known: {sorted(known)}")

    both = sorted(set(emphasize) & set(deemphasize))
    if both:
        raise ValueError(f"criteria named as BOTH emphasized and deemphasized: {both}")

    if factor <= 0:
        raise ValueError("emphasis factor must be > 0")

    return [
        Criterion(
            name=c.name,
            weight=(
                c.weight * factor
                if c.name in emphasize
                else c.weight / factor
                if c.name in deemphasize
                else c.weight
            ),
            lower_bound=c.lower_bound,
            upper_bound=c.upper_bound,
            higher_is_better=c.higher_is_better,
        )
        for c in criteria
    ]


def _kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    """Rank correlation over the items present in both orderings: +1 same
    order, -1 reversed, 0 unrelated. One number for "how much did the
    ranking actually churn", which is more honest than eyeballing a
    before/after list — two rankings can look similar and still have
    swapped the pair anyone cares about.

    Tau-a (no tie correction). Ties in total_score are already broken
    deterministically by item_id upstream, so no genuine ties reach here.
    """
    common = [item for item in order_a if item in set(order_b)]
    n = len(common)
    if n < 2:
        return 1.0  # 0 or 1 items cannot disagree about order

    position_b = {item: i for i, item in enumerate(order_b)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_before = i < j  # by construction, always True
            b_before = position_b[common[i]] < position_b[common[j]]
            if a_before == b_before:
                concordant += 1
            else:
                discordant += 1

    total_pairs = n * (n - 1) // 2
    return (concordant - discordant) / total_pairs


def sensitivity_analysis(
    items: Mapping[str, Mapping[str, float]],
    criteria: list[Criterion],
    weight_overrides: Mapping[str, float],
    coverage_threshold: float = 0.5,
) -> SensitivityResult:
    """Re-rank under changed weights and report exactly what moved.

    This is the answer to "why those weights?" — a question that has no
    good response from intuition. With this, the honest answer is
    empirical: "the ranking barely moves, so the exact figure isn't
    load-bearing," or "it flips at 1.8x, so here's why I chose the side
    I did." Both are defensible; asserting a weight is correct is not.
    """
    perturbed_criteria = apply_weight_overrides(criteria, weight_overrides)

    baseline = rank_items(items, criteria, coverage_threshold=coverage_threshold)
    perturbed = rank_items(items, perturbed_criteria, coverage_threshold=coverage_threshold)

    baseline_by_id = {r.item_id: r for r in baseline.ranked}
    perturbed_by_id = {r.item_id: r for r in perturbed.ranked}

    changes: list[RankChange] = []
    for item_id in sorted(set(baseline_by_id) | set(perturbed_by_id)):
        before = baseline_by_id.get(item_id)
        after = perturbed_by_id.get(item_id)
        both = before is not None and after is not None
        changes.append(
            RankChange(
                item_id=item_id,
                baseline_rank=before.rank if before else None,
                perturbed_rank=after.rank if after else None,
                rank_delta=(after.rank - before.rank) if both else None,  # type: ignore[union-attr]
                baseline_score=before.total_score if before else None,
                perturbed_score=after.total_score if after else None,
                score_delta=(after.total_score - before.total_score) if both else None,  # type: ignore[union-attr]
            )
        )

    moved_deltas = [c.rank_delta for c in changes if c.rank_delta is not None and c.rank_delta != 0]
    baseline_top = baseline.ranked[0].item_id if baseline.ranked else None
    perturbed_top = perturbed.ranked[0].item_id if perturbed.ranked else None

    return SensitivityResult(
        baseline_weights=tuple((c.name, c.weight) for c in criteria),
        perturbed_weights=tuple((c.name, c.weight) for c in perturbed_criteria),
        changes=tuple(changes),
        baseline_top=baseline_top,
        perturbed_top=perturbed_top,
        top_changed=baseline_top != perturbed_top,
        max_rank_movement=max((abs(d) for d in moved_deltas), default=0),
        items_moved=len(moved_deltas),
        items_entered=tuple(c.item_id for c in changes if c.baseline_rank is None and c.perturbed_rank is not None),
        items_left=tuple(c.item_id for c in changes if c.baseline_rank is not None and c.perturbed_rank is None),
        kendall_tau=_kendall_tau(
            [r.item_id for r in baseline.ranked], [r.item_id for r in perturbed.ranked]
        ),
    )


def find_weight_flip_point(
    items: Mapping[str, Mapping[str, float]],
    criteria: list[Criterion],
    criterion_name: str,
    max_factor: float = 10.0,
    steps: int = 200,
    coverage_threshold: float = 0.5,
) -> float | None:
    """Smallest multiplier on `criterion_name`'s weight at which the
    top-ranked item changes, or None if it never does within max_factor.

    THE defence number. "Why 2.0 and not 1.5?" answered with "the winner
    doesn't change until 2.6x, so that choice isn't carrying the result"
    is an empirical answer; "it felt right" is not.

    Method note, because the honest limitation matters more than the
    number: this is a LINEAR SWEEP, not a binary search, and it returns
    the FIRST crossing found. Binary search would be faster but assumes
    the winner changes monotonically with the weight, which is not
    guaranteed — with three or more items, one can overtake and later be
    overtaken again. A sweep at fixed resolution can only miss a flip
    narrower than its step size, which is a bounded, statable error;
    binary search can miss one of any width and report a confident wrong
    answer. Resolution is max_factor/steps (default 0.05x).

    Sweeps both directions from 1.0 — a weight can be too HIGH as easily
    as too low, and only checking upward would report "stable" for a
    ranking that flips the moment the weight is halved.
    """
    baseline = rank_items(items, criteria, coverage_threshold=coverage_threshold)
    if not baseline.ranked:
        return None
    baseline_top = baseline.ranked[0].item_id

    step = max_factor / steps
    # Interleave outward from 1.0 so the SMALLEST perturbation that flips
    # the winner is found first, whichever side it's on.
    for i in range(1, steps + 1):
        for factor in (1.0 + i * step, max(0.0, 1.0 - i * step)):
            perturbed = rank_items(
                items,
                apply_weight_overrides(
                    criteria, {criterion_name: _weight_of(criteria, criterion_name) * factor}
                ),
                coverage_threshold=coverage_threshold,
            )
            if perturbed.ranked and perturbed.ranked[0].item_id != baseline_top:
                return round(factor, 6)

    return None


def _weight_of(criteria: list[Criterion], name: str) -> float:
    for c in criteria:
        if c.name == name:
            return c.weight
    raise ValueError(f"unknown criterion {name!r}; known: {sorted(c.name for c in criteria)}")


def rank_items(
    items: Mapping[str, Mapping[str, float]],
    criteria: list[Criterion],
    coverage_threshold: float = 0.5,
) -> RankingResult:
    """Score every item and return them ranked descending by total score,
    alongside anything the coverage floor turned away.

    Deterministic tie-break: equal total_score ties are broken by item_id
    ascending (alphabetical), so re-running this function on the same
    input always produces the same order — no dependency on dict
    iteration order or any hidden randomness.
    """
    scored: list[ScoreResult] = []
    excluded: list[ExcludedItem] = []
    for item_id, raw_values in items.items():
        try:
            scored.append(
                score_item(item_id, raw_values, criteria, coverage_threshold=coverage_threshold)
            )
        except InsufficientCoverageError as exc:
            excluded.append(
                ExcludedItem(
                    item_id=item_id,
                    covered_weight=exc.covered_weight,
                    missing_criteria=exc.missing_criteria,
                    reason=str(exc),
                )
            )

    scored.sort(key=lambda r: (-r.total_score, r.item_id))
    ranked = tuple(
        RankedItem(
            rank=i + 1,
            item_id=r.item_id,
            total_score=r.total_score,
            components=r.components,
            covered_weight=r.covered_weight,
            missing_criteria=r.missing_criteria,
        )
        for i, r in enumerate(scored)
    )
    excluded.sort(key=lambda e: e.item_id)
    return RankingResult(ranked=ranked, excluded=tuple(excluded))
