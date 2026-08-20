"""
affordability.py — mortgage / affordability arithmetic for the Israel
Home-Buying Intelligence Agent.

Same rules as scoring.py and analytics.py, and for the same reason: this
is the mechanism behind the derived metric app/analytics.py's
build_derived_metric() assembles ("how much home can you actually afford
at X, and why?"), so it has to be a *mechanism*, not a correlation — see
runway_geometry.py's old header in this codebase's history for the
airport-domain version of the same argument. This file is:

  1. ZERO I/O. No network calls, no file reads, no env var reads, no
     imports of requests/httpx/openai/anthropic/os (other than typing).
  2. ZERO LLM calls. Understandable and fully testable without any API
     key, network access, or running agent.
  3. Every function is PURE: same inputs -> same outputs. No globals
     mutated, no wall-clock, no randomness. Every rate, term, and ratio
     is passed in as an argument — never read from a global or the
     environment — specifically so a caller can quote "as of today's
     Bank of Israel prime rate" or "assuming a 25-year term" and mean it
     literally, rather than the function silently reaching for some
     baked-in figure that drifts out of date.
  4. Same defensibility goal as scoring.py: point to this file, and to
     tests/test_affordability.py, when asked "what part of the
     affordability numbers is NOT the LLM?"

The Israeli-specific constants below (max loan-to-value by buyer
category, standard mortgage term, the repayment-to-income cap) are
DOCUMENTED, OVERRIDABLE assumptions, not hidden magic numbers — every one
carries a comment saying what it is and where it comes from, and every
function that uses one takes it as an argument with the constant only
supplying the default.
"""
from __future__ import annotations

from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────
# Documented constants — Israeli mortgage-market assumptions
# ─────────────────────────────────────────────────────────────────────────

MONTHS_PER_YEAR = 12

# Bank of Israel Proper Conduct of Banking Business Directive 329 caps the
# loan-to-value ratio a bank may extend, by buyer category. These are the
# REGULATOR's ceilings, not a market average and not our own judgment —
# a bank may lend less, never more, than these figures. Using the
# regulator's number is a citation; inventing our own would be a claim we
# would have to separately defend.
#   - a buyer with no other home (first home): up to 75% LTV.
FIRST_HOME_MAX_LTV = 0.75
#   - a buyer who owns a home and is upgrading/replacing it (committed to
#     selling the old one within an 18-month window, per the same
#     Directive): up to 70% LTV.
UPGRADER_MAX_LTV = 0.70
#   - a buyer acquiring an investment property (keeping an existing home):
#     up to 50% LTV. The tightest band, because this is the segment the
#     Directive is explicitly trying to cool.
INVESTMENT_MAX_LTV = 0.50

# Standard Israeli mortgage terms run 25-30 years; 30 is the common
# maximum most banks will offer and is used here as the default "how much
# could a household afford" assumption. Callers modelling a shorter term
# (which raises the monthly payment and lowers max_affordable_price)
# should pass `years` explicitly rather than rely on this default.
STANDARD_MORTGAGE_TERM_MIN_YEARS = 25
STANDARD_MORTGAGE_TERM_MAX_YEARS = 30
DEFAULT_MORTGAGE_TERM_YEARS = STANDARD_MORTGAGE_TERM_MAX_YEARS

# Israeli lenders, as a matter of standard underwriting practice (and
# regulatory guidance from the Bank of Israel on responsible lending),
# generally will not approve a mortgage whose total monthly repayment
# exceeds roughly one third of the borrowing household's net monthly
# income. This is a widely used rule of thumb rather than a single
# published numeric ceiling in the way LTV is, so it is treated here as a
# STATED, OVERRIDABLE assumption — a caller with a more specific bank
# policy in hand should pass its own `repayment_cap`.
DEFAULT_REPAYMENT_CAP = 1.0 / 3.0


# ─────────────────────────────────────────────────────────────────────────
# Core amortization arithmetic
# ─────────────────────────────────────────────────────────────────────────
def monthly_payment(principal: float, annual_rate: float, years: float) -> float:
    """Standard fixed-rate amortizing-loan monthly payment:

        payment = P * r / (1 - (1 + r) ** -n)

    where P is the principal, r is the MONTHLY interest rate
    (annual_rate / 12), and n is the number of monthly payments
    (years * 12). This is the textbook annuity formula — the payment that
    exactly zeroes the balance after n equal installments at rate r.

    `annual_rate == 0` is handled as its own branch rather than let
    through to the formula above, which divides by r: a real,
    reachable input (a zero-interest bridge loan, or just "what if rates
    were 0") would otherwise crash with a ZeroDivisionError. The
    zero-rate payment is simply the principal spread evenly across n
    months — the limit of the annuity formula as r -> 0, so the branch
    is a documented special case of the same idea, not a different one.
    """
    if principal < 0:
        raise ValueError(f"principal must be >= 0, got {principal!r}")
    if annual_rate < 0:
        raise ValueError(f"annual_rate must be >= 0, got {annual_rate!r}")
    if years <= 0:
        raise ValueError(f"years must be > 0, got {years!r}")

    n = years * MONTHS_PER_YEAR
    if annual_rate == 0.0:
        return principal / n

    r = annual_rate / MONTHS_PER_YEAR
    return principal * r / (1.0 - (1.0 + r) ** -n)


def max_affordable_price(
    monthly_budget: float,
    annual_rate: float,
    years: float,
    ltv: float,
) -> float:
    """Invert monthly_payment(): the largest loan principal serviceable by
    `monthly_budget`, then grossed up by `ltv` to a total purchase price
    (loan + the buyer's own equity implied by 1 - ltv).

    Algebraically this is monthly_payment() solved for principal:

        principal = monthly_budget * (1 - (1 + r) ** -n) / r      (r > 0)
        principal = monthly_budget * n                            (r == 0)

    then `price = principal / ltv`. Round-tripping —
    monthly_payment(max_affordable_price(...) * ltv, ...) —
    must reproduce monthly_budget; that invariant is pinned in
    tests/test_affordability.py rather than trusted by inspection, since
    it's exactly the kind of thing an algebra slip breaks silently.

    `ltv` must be in (0, 1]: zero would mean "afford a home with a zero
    loan", which degenerates to price == 0 for any positive budget and is
    almost certainly a caller passing the wrong number, not a real
    scenario worth supporting silently.
    """
    if monthly_budget <= 0:
        raise ValueError(f"monthly_budget must be > 0, got {monthly_budget!r}")
    if not (0.0 < ltv <= 1.0):
        raise ValueError(f"ltv must be in (0, 1], got {ltv!r}")
    if annual_rate < 0:
        raise ValueError(f"annual_rate must be >= 0, got {annual_rate!r}")
    if years <= 0:
        raise ValueError(f"years must be > 0, got {years!r}")

    n = years * MONTHS_PER_YEAR
    if annual_rate == 0.0:
        principal = monthly_budget * n
    else:
        r = annual_rate / MONTHS_PER_YEAR
        principal = monthly_budget * (1.0 - (1.0 + r) ** -n) / r

    return principal / ltv


def years_of_income_to_buy(median_price: float, median_annual_household_income: float) -> float:
    """The classic price-to-income ratio, and the single number most
    affordability commentary leads with: how many years of a household's
    entire (pre-tax, unspent-on-anything-else) income it would take to
    buy the median home outright, with no mortgage at all.

    Deliberately NOT mortgage-aware — it says nothing about interest
    rates, down payments, or lending caps, which is exactly why it is
    useful alongside mortgage_burden(): this is the headline "is this
    place expensive" number, comparable across time and across places
    even when mortgage terms differ; mortgage_burden() is the "could a
    real household actually finance it" number underneath.
    """
    if median_annual_household_income <= 0:
        raise ValueError(
            f"median_annual_household_income must be > 0, got {median_annual_household_income!r}"
        )
    return median_price / median_annual_household_income


@dataclass(frozen=True)
class MortgageBurdenResult:
    """Payment as a share of income, plus the yes/no a lender actually
    asks: does it clear the bank's repayment cap? Carrying both, rather
    than just the boolean, is what lets a tool say "38% of income — 5
    points over the usual third" instead of a bare pass/fail."""

    monthly_payment: float
    monthly_income_share: float
    repayment_cap: float
    within_repayment_cap: bool


def mortgage_burden(
    median_price: float,
    median_monthly_household_income: float,
    annual_rate: float,
    years: float,
    ltv: float,
    repayment_cap: float = DEFAULT_REPAYMENT_CAP,
) -> MortgageBurdenResult:
    """Monthly mortgage payment on `median_price` at `ltv` financing, as a
    share of `median_monthly_household_income`, checked against
    `repayment_cap`.

    The cap check is INCLUSIVE (`<=`), deliberately the opposite
    convention from scoring.py's coverage floor: a lender's "no more than
    a third of income" is a MAXIMUM, and sitting exactly at the maximum
    is, by definition, still within it. scoring.py's floor is exclusive
    because "enough evidence" means MOST of it, a different kind of
    threshold — see that module's docstring. Pinned from both sides in
    tests/test_affordability.py so this distinction can't drift.
    """
    if median_monthly_household_income <= 0:
        raise ValueError(
            f"median_monthly_household_income must be > 0, got "
            f"{median_monthly_household_income!r}"
        )
    if repayment_cap <= 0:
        raise ValueError(f"repayment_cap must be > 0, got {repayment_cap!r}")
    if not (0.0 < ltv <= 1.0):
        raise ValueError(f"ltv must be in (0, 1], got {ltv!r}")

    principal = median_price * ltv
    payment = monthly_payment(principal, annual_rate, years)
    share = payment / median_monthly_household_income

    return MortgageBurdenResult(
        monthly_payment=payment,
        monthly_income_share=share,
        repayment_cap=repayment_cap,
        within_repayment_cap=share <= repayment_cap,
    )


@dataclass(frozen=True)
class AffordabilityBreakdown:
    """Every input, every intermediate, and every final number behind one
    affordability read on a place — the whole point being that a tool can
    hand this to the model and the model can quote any figure in it
    without ever having computed one itself.

    `max_affordable_price_at_cap` and `affordability_gap` answer a
    different, complementary question from `monthly_income_share`: not
    "what fraction of income does the median home cost", but "given the
    repayment cap, what's the most this household could actually finance
    — and is the median home above or below that line". A positive gap
    means the median home is within reach at the stated cap; a negative
    gap is a quantified "priced out by this much".
    """

    # Inputs, carried through verbatim so the breakdown is self-contained.
    median_price: float
    median_monthly_household_income: float
    median_annual_household_income: float
    annual_rate: float
    years: float
    ltv: float
    repayment_cap: float

    # Intermediates.
    loan_principal: float
    down_payment: float
    monthly_payment: float

    # Final numbers.
    monthly_income_share: float
    within_repayment_cap: bool
    years_of_income: float
    max_affordable_price_at_cap: float
    affordability_gap: float  # max_affordable_price_at_cap - median_price


def affordability_breakdown(
    median_price: float,
    median_monthly_household_income: float,
    median_annual_household_income: float,
    annual_rate: float,
    years: float = DEFAULT_MORTGAGE_TERM_YEARS,
    ltv: float = FIRST_HOME_MAX_LTV,
    repayment_cap: float = DEFAULT_REPAYMENT_CAP,
) -> AffordabilityBreakdown:
    """Assemble the full affordability picture for one locality/place at
    `median_price`, for a household earning the stated income.

    `median_monthly_household_income` and `median_annual_household_income`
    are taken as two SEPARATE inputs rather than one derived from the
    other by *12 or /12. Real income sources report different periods
    (a household-income survey typically reports annual; a mortgage
    affordability check needs monthly) and, in Israel specifically,
    monthly and annual income are not always related by a clean factor of
    12 — a 13th-month payment or seasonal bonus income breaks that
    assumption. Passing both from the actual data source is more honest
    than silently dividing one number by 12 and treating the result as
    measured; if a caller genuinely has only one, dividing/multiplying by
    12 before calling is a caller-visible choice, not a hidden one made
    in here.
    """
    principal = median_price * ltv
    down_payment = median_price - principal
    payment = monthly_payment(principal, annual_rate, years)
    burden = mortgage_burden(
        median_price, median_monthly_household_income, annual_rate, years, ltv, repayment_cap
    )
    income_years = years_of_income_to_buy(median_price, median_annual_household_income)

    # The most this household could finance without exceeding the cap,
    # at the SAME rate/term/ltv assumptions used above — i.e. an
    # apples-to-apples comparison point for median_price.
    affordable_budget = median_monthly_household_income * repayment_cap
    max_price_at_cap = max_affordable_price(affordable_budget, annual_rate, years, ltv)

    return AffordabilityBreakdown(
        median_price=median_price,
        median_monthly_household_income=median_monthly_household_income,
        median_annual_household_income=median_annual_household_income,
        annual_rate=annual_rate,
        years=years,
        ltv=ltv,
        repayment_cap=repayment_cap,
        loan_principal=principal,
        down_payment=down_payment,
        monthly_payment=payment,
        monthly_income_share=burden.monthly_income_share,
        within_repayment_cap=burden.within_repayment_cap,
        years_of_income=income_years,
        max_affordable_price_at_cap=max_price_at_cap,
        affordability_gap=max_price_at_cap - median_price,
    )
