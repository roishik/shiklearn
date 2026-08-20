"""
analytics.py — the three query shapes that are NOT ranking.

Same rules as scoring.py and entity_resolution.py: ZERO I/O, ZERO LLM,
pure functions, fully testable with no key and no network.

Why this file exists at all — this project originally had one hammer
(a weighted composite score) and the real brief asks four different
question shapes, only two of which are ranking:

  1. "which of these are strong candidates?"      -> scoring.py (rank)
  2. "compare X and Y"                            -> scoring.py (rank of 2)
  3. "what PERCENTAGE of Haifa's transactions are new-build?"  -> aggregate()
  4. "how much home can you AFFORD at X, and why?" -> build_derived_metric()

Answering (3) with a composite score is a category error: it's a count
over a subset of ONE entity's own records, with no ranking and no
weighting anywhere in it. Answering (4) with a stored column is
impossible — the quantity appears in no dataset and must be MODELLED,
which means the assumptions have to travel with the number.

filter_items() is the subsetting primitive both of the above need
("...in the Northern district", "...new-build only"), and which
compare_items also needed but never had — without it the model has to
produce ids from memory, which is exactly the hallucination path
entity_resolution.py exists to close.

WHAT IS DELIBERATELY *NOT* HERE: any particular domain's formula. This
module holds the generic contract a modelled metric must satisfy
(build_derived_metric: factors that provably sum to the value,
mandatory assumptions and caveat, a validated confidence level). The
model itself — what the terms are, what constant discounts what — is
domain-specific and lives in app/tools.py, exactly as scoring.py holds
the generic ranker while tools.py holds this assignment's actual KPIs.
Swapping domains should mean rewriting tools.py, never this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class UnsupportedOperationError(ValueError):
    """Raised for an aggregate operation this module doesn't implement.
    Named (not a bare ValueError) so a caller can distinguish "you asked
    for something I can't do" from "your inputs were malformed"."""


# ─────────────────────────────────────────────────────────────────────────
# A3 — filter / subset
# ─────────────────────────────────────────────────────────────────────────
def filter_items(
    attributes: Mapping[str, Mapping[str, str]],
    filters: Mapping[str, str],
) -> tuple[str, ...]:
    """Return the ids whose attributes match EVERY filter (AND semantics),
    sorted for determinism.

    Matching is case-insensitive on both key and value — a user asking for
    "North" and a dataset storing "north" is a formatting difference, not
    a semantic one, and making the caller normalize invites the bug.

    AND rather than OR is deliberate: "localities in the Northern
    district that are coastal" means both conditions, and a user wanting
    OR can ask twice. Silently unioning would return a superset that
    looks authoritative and is wrong.

    An unknown filter KEY matches nothing rather than being ignored.
    Ignoring it would answer a different, broader question than the one
    asked while looking like a successful result — the failure mode this
    whole module exists to avoid. Callers that need to tell "no matches"
    from "you asked about a field I don't have" should check
    known_attribute_keys() first.
    """
    normalized_filters = {k.casefold(): str(v).casefold() for k, v in filters.items()}

    matched: list[str] = []
    for item_id, attrs in attributes.items():
        normalized_attrs = {k.casefold(): str(v).casefold() for k, v in attrs.items()}
        if all(normalized_attrs.get(key) == value for key, value in normalized_filters.items()):
            matched.append(item_id)

    return tuple(sorted(matched))


def known_attribute_keys(attributes: Mapping[str, Mapping[str, str]]) -> tuple[str, ...]:
    """Every attribute key present on any item. Lets a tool tell the model
    "there is no 'colour' field, the fields are X/Y/Z" instead of silently
    returning zero matches, which reads as "none exist"."""
    keys: set[str] = set()
    for attrs in attributes.values():
        keys.update(k.casefold() for k in attrs)
    return tuple(sorted(keys))


def known_attribute_values(
    attributes: Mapping[str, Mapping[str, str]],
    keys: Sequence[str],
    *,
    max_listed: int = 12,
) -> dict[str, dict[str, Any]]:
    """For each requested key, the values that actually OCCUR in the data.

    known_attribute_keys() answers "is there such a field?". This answers
    the next question down: "that field exists -- but does anything in it
    ever equal what I asked for?" Those are different failures and only
    the first one was reported, so the second produced a confident
    falsehood: `district` holds the six official CBS districts (Jerusalem/
    North/Haifa/Center/Tel Aviv/South), so filtering district="Sharon" --
    a sub-region, not an official district -- is a real key with a value
    the data never uses. It matched nothing, and "nothing matched" was
    reported to the user as "there are no localities in the Sharon", which
    is false; there are dozens, reachable via {'district': 'center'}.

    This is the same guard aggregate_records already applies to an
    unknown category VALUE, applied here to a filter value. See that
    function's comment for the original instance of the bug.

    High-cardinality keys are summarized rather than listed -- locality
    name has hundreds of values, and the point is to show the caller the
    SHAPE of the value space so it can spot its own mistake, not to paste
    the dataset into a tool result. The shape is uniform whether truncated
    or not, so a reader never has to branch on which form it got.
    """
    out: dict[str, dict[str, Any]] = {}
    for key in keys:
        folded = key.casefold()
        values = sorted(
            {
                str(attrs[k])
                for attrs in attributes.values()
                for k in attrs
                if k.casefold() == folded
            }
        )
        if not values:
            continue
        out[folded] = {
            "distinct_count": len(values),
            "values": values[:max_listed],
            "truncated": len(values) > max_listed,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────
# A4 — aggregate over one entity's records
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AggregateResult:
    operation: str
    value: float
    # The arithmetic, exposed. Same principle as scoring.py's
    # ComponentBreakdown: a number the model can't decompose is a number
    # it will end up paraphrasing wrongly.
    matching_records: int
    total_records: int
    matching_units: float
    total_units: float
    group_by_field: str | None
    group_value: str | None


SUPPORTED_OPERATIONS = ("share", "mean", "count", "sum")


def aggregate(
    records: Sequence[Mapping[str, Any]],
    operation: str,
    value_field: str = "units",
    group_by_field: str | None = None,
    group_value: str | None = None,
) -> AggregateResult:
    """Deterministic aggregate over one entity's sub-records.

    `share` is the one worth reading carefully, because it's the actual
    question a user asks ("what percentage of Haifa's transactions are
    new-build?") and it has a real ambiguity: share BY RECORD COUNT or
    share BY UNITS? Three new-build sales out of ten is 30% by count; if
    those three account for 600 of 1000 ₪ of transaction value it's 60%
    by units. Both are defensible answers to "what percentage", and they
    differ.

    Decision: `share` is computed on UNITS (the value_field), because the
    magnitude is almost always what the question means in a market-size
    or value context. AggregateResult returns the count-based numerator
    and denominator too, so the model can state the other reading if the
    user meant it — the ambiguity is surfaced, not silently resolved.
    """
    if operation not in SUPPORTED_OPERATIONS:
        raise UnsupportedOperationError(
            f"operation={operation!r} not supported; supported: {list(SUPPORTED_OPERATIONS)}"
        )

    total_records = len(records)
    total_units = float(sum(r.get(value_field, 0.0) for r in records))

    if group_by_field is not None and group_value is not None:
        wanted = str(group_value).casefold()
        matching: list[Mapping[str, Any]] = [
            r for r in records if str(r.get(group_by_field, "")).casefold() == wanted
        ]
    else:
        matching = list(records)

    matching_records = len(matching)
    matching_units = float(sum(r.get(value_field, 0.0) for r in matching))

    if operation == "count":
        value = float(matching_records)
    elif operation == "sum":
        value = matching_units
    elif operation == "mean":
        # Mean of an empty subset is undefined, not 0.0 — returning 0.0
        # would be a real number the model would happily report as a fact.
        value = matching_units / matching_records if matching_records else float("nan")
    else:  # share
        value = matching_units / total_units if total_units else float("nan")

    return AggregateResult(
        operation=operation,
        value=value,
        matching_records=matching_records,
        total_records=total_records,
        matching_units=matching_units,
        total_units=total_units,
        group_by_field=group_by_field,
        group_value=group_value,
    )


# ─────────────────────────────────────────────────────────────────────────
# A5 — derived / modelled metric: the generic CONTRACT
#
# What lives here is the shape every modelled quantity must satisfy, not
# any particular model. The actual formula is domain-specific and belongs
# with the rest of the domain code in app/tools.py — see the affordability
# estimate tool there (built on app/affordability.py's amortization and
# price-to-income arithmetic) for the worked example. Keeping the two
# apart is the same split as scoring.py (generic ranker) vs. tools.py's
# DEFAULT_CRITERIA (this assignment's KPIs): swap the domain, keep the
# machinery.
# ─────────────────────────────────────────────────────────────────────────
VALID_CONFIDENCE_LEVELS = ("low", "medium", "high")


@dataclass(frozen=True)
class FactorInput:
    """What a domain model supplies for one contributing term. `magnitude`
    is in the metric's own units; share_of_total is computed for you, so
    a caller can't get the arithmetic subtly wrong."""

    name: str
    magnitude: float
    source_field: str
    explanation: str


@dataclass(frozen=True)
class Factor:
    """One contributing term in a modelled quantity: what it is, how big
    it is, and which observed input produced it. The 'why' half of a
    'what is X, and why?' question must be assembled from THESE, never
    invented by the model."""

    name: str
    magnitude: float
    share_of_total: float  # 0..1 of the summed contributions
    source_field: str
    explanation: str


@dataclass(frozen=True)
class DerivedMetricResult:
    metric: str
    value: float
    unit: str
    factors: tuple[Factor, ...]
    assumptions: tuple[str, ...]
    confidence: str  # one of VALID_CONFIDENCE_LEVELS
    caveat: str


def build_derived_metric(
    metric: str,
    unit: str,
    contributions: Sequence[FactorInput],
    assumptions: Sequence[str],
    confidence: str,
    caveat: str,
) -> DerivedMetricResult:
    """Assemble a modelled quantity from its contributing factors.

    The invariant this exists to enforce: **the factors sum to the
    value.** The total is computed here, from the contributions, rather
    than passed in beside them — so an explanation can never drift away
    from the number it claims to explain. A "why" that doesn't
    reconstruct the "what" is decorative, and that failure is invisible
    if the two are supplied independently.

    Also enforced: a modelled number never travels without `assumptions`
    and a `caveat`. A derived quantity presented bare is indistinguishable
    from a measured one, which is the misrepresentation this whole shape
    is designed to prevent.

    Negative magnitudes are allowed (a model may have offsetting terms),
    but `share_of_total` is then computed against the sum of ABSOLUTE
    magnitudes, so shares stay interpretable as "how much of the
    explanation is this term" rather than going wild near a zero net.
    """
    if confidence not in VALID_CONFIDENCE_LEVELS:
        raise ValueError(f"confidence must be one of {list(VALID_CONFIDENCE_LEVELS)}, got {confidence!r}")
    if not assumptions:
        raise ValueError("a modelled metric must state its assumptions — an empty list is not acceptable")
    if not caveat.strip():
        raise ValueError("a modelled metric must carry a caveat explaining what it is not")

    value = float(sum(c.magnitude for c in contributions))
    absolute_total = float(sum(abs(c.magnitude) for c in contributions))

    factors = tuple(
        Factor(
            name=c.name,
            magnitude=c.magnitude,
            share_of_total=(abs(c.magnitude) / absolute_total) if absolute_total else 0.0,
            source_field=c.source_field,
            explanation=c.explanation,
        )
        for c in contributions
    )

    return DerivedMetricResult(
        metric=metric,
        value=value,
        unit=unit,
        factors=factors,
        assumptions=tuple(assumptions),
        confidence=confidence,
        caveat=caveat,
    )
