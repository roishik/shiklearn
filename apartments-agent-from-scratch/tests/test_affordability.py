"""Unit tests for app/affordability.py — the pure, zero-I/O, zero-LLM
mortgage/affordability arithmetic. These tests should pass with no
network access, no API key, and no other part of the app imported.

Where the docstring on the module under test asks for it, expected
values here are worked out BY HAND in the comment above the assertion,
not copied from a run of the code — the point is to catch an algebra
slip in the implementation, which a test that merely echoes the
implementation's own output cannot do."""
from __future__ import annotations

import pytest

from app.affordability import (
    DEFAULT_REPAYMENT_CAP,
    AffordabilityBreakdown,
    MortgageBurdenResult,
    affordability_breakdown,
    max_affordable_price,
    mortgage_burden,
    monthly_payment,
    years_of_income_to_buy,
)


# ─────────────────────────────────────────────────────────────────────────
# monthly_payment — the annuity formula
# ─────────────────────────────────────────────────────────────────────────
def test_monthly_payment_matches_known_amortization_value():
    """The classic textbook example: a $10,000 loan at 12% annual interest
    (1% monthly), repaid over 12 months, has a monthly payment of $888.49.

    Hand-derivation:
      r = 0.12 / 12 = 0.01
      n = 12
      (1.01)^12 = 1.12682503... (a commonly tabulated compound-interest
        constant)
      payment = P * r * (1+r)^n / ((1+r)^n - 1)
              = 10000 * 0.01 * 1.12682503 / (1.12682503 - 1)
              = 112.682503 / 0.12682503
              ≈ 888.49
    """
    result = monthly_payment(principal=10_000.0, annual_rate=0.12, years=1)
    assert result == pytest.approx(888.49, abs=0.02)


def test_monthly_payment_zero_rate_branch():
    """Zero interest: the payment is just the principal spread evenly
    across the term, no compounding at all.

    Hand-derivation: principal=120,000, years=10 -> n=120 months.
      payment = 120,000 / 120 = 1,000 exactly.
    """
    result = monthly_payment(principal=120_000.0, annual_rate=0.0, years=10)
    assert result == pytest.approx(1_000.0, abs=1e-9)


def test_monthly_payment_rejects_non_positive_years():
    with pytest.raises(ValueError):
        monthly_payment(principal=10_000.0, annual_rate=0.05, years=0)


def test_monthly_payment_rejects_negative_principal():
    with pytest.raises(ValueError):
        monthly_payment(principal=-1.0, annual_rate=0.05, years=10)


def test_monthly_payment_rejects_negative_rate():
    with pytest.raises(ValueError):
        monthly_payment(principal=10_000.0, annual_rate=-0.01, years=10)


# ─────────────────────────────────────────────────────────────────────────
# max_affordable_price — inverse of monthly_payment, grossed up by LTV
# ─────────────────────────────────────────────────────────────────────────
def test_max_affordable_price_round_trips_through_monthly_payment():
    """Algebraic invariant, not a fixed hand-derived number: feeding the
    price back through the LTV and the annuity formula must reproduce the
    original monthly budget — this is what would break first if the
    inversion algebra were wrong in either function."""
    monthly_budget = 7_500.0
    annual_rate = 0.045
    years = 25
    ltv = 0.75

    price = max_affordable_price(monthly_budget, annual_rate, years, ltv)
    loan_principal = price * ltv
    recovered_budget = monthly_payment(loan_principal, annual_rate, years)

    assert recovered_budget == pytest.approx(monthly_budget, rel=1e-9)


def test_max_affordable_price_ltv_gross_up():
    """Zero-rate case, chosen so every intermediate is exact by hand.

    Hand-derivation: monthly_budget=1,000, years=10 -> n=120.
      loan_principal = 1,000 * 120 = 120,000  (zero-rate monthly_payment
        inverted is just budget * n)
      ltv=0.75 -> price = 120,000 / 0.75 = 160,000 exactly.
    """
    price = max_affordable_price(monthly_budget=1_000.0, annual_rate=0.0, years=10, ltv=0.75)
    assert price == pytest.approx(160_000.0, abs=1e-6)


def test_max_affordable_price_rejects_ltv_out_of_range():
    with pytest.raises(ValueError):
        max_affordable_price(monthly_budget=1_000.0, annual_rate=0.05, years=10, ltv=0.0)
    with pytest.raises(ValueError):
        max_affordable_price(monthly_budget=1_000.0, annual_rate=0.05, years=10, ltv=1.5)


def test_max_affordable_price_rejects_non_positive_budget():
    with pytest.raises(ValueError):
        max_affordable_price(monthly_budget=0.0, annual_rate=0.05, years=10, ltv=0.75)


# ─────────────────────────────────────────────────────────────────────────
# years_of_income_to_buy — the headline price-to-income ratio
# ─────────────────────────────────────────────────────────────────────────
def test_years_of_income_to_buy_hand_derived():
    """Hand-derivation: 1,000,000 / 200,000 = 5.0 years of gross income to
    buy the median home outright."""
    result = years_of_income_to_buy(median_price=1_000_000.0, median_annual_household_income=200_000.0)
    assert result == pytest.approx(5.0, abs=1e-9)


def test_years_of_income_to_buy_rejects_non_positive_income():
    with pytest.raises(ValueError):
        years_of_income_to_buy(median_price=1_000_000.0, median_annual_household_income=0.0)


# ─────────────────────────────────────────────────────────────────────────
# mortgage_burden — payment as a share of income, checked against the cap
# ─────────────────────────────────────────────────────────────────────────
def test_mortgage_burden_cap_boundary_from_both_sides():
    """The cap check is INCLUSIVE (share <= repayment_cap): a lender's
    maximum is still satisfied when you sit exactly on it. Pinned from
    both sides so that distinction — deliberately the opposite convention
    from scoring.py's EXCLUSIVE coverage floor — can't drift.

    Setup, all exact by hand: median_price=120,000, ltv=1.0, zero rate,
    years=10 -> n=120, so:
      monthly_payment = 120,000 / 120 = 1,000 exactly.
    repayment_cap = 1/3.

    - income = 3,000.0 -> share = 1000/3000 = 1/3 EXACTLY == cap
      -> within_repayment_cap must be True (at the cap still clears it).
    - income = 3,001.0 -> share = 1000/3001 ≈ 0.33322 < 1/3
      -> within_repayment_cap True (comfortably under).
    - income = 2,999.0 -> share = 1000/2999 ≈ 0.33344 > 1/3
      -> within_repayment_cap False (over the cap).
    """
    at_cap = mortgage_burden(
        median_price=120_000.0,
        median_monthly_household_income=3_000.0,
        annual_rate=0.0,
        years=10,
        ltv=1.0,
        repayment_cap=1.0 / 3.0,
    )
    assert at_cap.monthly_payment == pytest.approx(1_000.0, abs=1e-9)
    assert at_cap.monthly_income_share == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert at_cap.within_repayment_cap is True

    just_under = mortgage_burden(
        median_price=120_000.0,
        median_monthly_household_income=3_001.0,
        annual_rate=0.0,
        years=10,
        ltv=1.0,
        repayment_cap=1.0 / 3.0,
    )
    assert just_under.monthly_income_share < 1.0 / 3.0
    assert just_under.within_repayment_cap is True

    just_over = mortgage_burden(
        median_price=120_000.0,
        median_monthly_household_income=2_999.0,
        annual_rate=0.0,
        years=10,
        ltv=1.0,
        repayment_cap=1.0 / 3.0,
    )
    assert just_over.monthly_income_share > 1.0 / 3.0
    assert just_over.within_repayment_cap is False


def test_mortgage_burden_uses_default_repayment_cap_when_not_given():
    result = mortgage_burden(
        median_price=120_000.0,
        median_monthly_household_income=3_000.0,
        annual_rate=0.0,
        years=10,
        ltv=1.0,
    )
    assert result.repayment_cap == pytest.approx(DEFAULT_REPAYMENT_CAP)


def test_mortgage_burden_returns_a_mortgage_burden_result():
    result = mortgage_burden(
        median_price=1_000_000.0,
        median_monthly_household_income=20_000.0,
        annual_rate=0.05,
        years=30,
        ltv=0.75,
    )
    assert isinstance(result, MortgageBurdenResult)
    assert result.monthly_payment > 0


def test_mortgage_burden_rejects_non_positive_income():
    with pytest.raises(ValueError):
        mortgage_burden(
            median_price=1_000_000.0,
            median_monthly_household_income=0.0,
            annual_rate=0.05,
            years=30,
            ltv=0.75,
        )


# ─────────────────────────────────────────────────────────────────────────
# affordability_breakdown — the full per-factor breakdown
# ─────────────────────────────────────────────────────────────────────────
def test_affordability_breakdown_is_internally_consistent():
    """Every field is either a verbatim input or derivable from the
    functions already pinned above — this checks the assembly wiring,
    not new arithmetic."""
    result = affordability_breakdown(
        median_price=1_500_000.0,
        median_monthly_household_income=18_000.0,
        median_annual_household_income=216_000.0,
        annual_rate=0.045,
        years=30,
        ltv=0.75,
    )
    assert isinstance(result, AffordabilityBreakdown)
    assert result.loan_principal == pytest.approx(1_500_000.0 * 0.75)
    assert result.down_payment == pytest.approx(1_500_000.0 * 0.25)
    assert result.monthly_payment == pytest.approx(
        monthly_payment(result.loan_principal, 0.045, 30)
    )
    assert result.years_of_income == pytest.approx(1_500_000.0 / 216_000.0)
    # The gap is defined as max_affordable_price_at_cap - median_price —
    # checking the wiring, not re-deriving the value.
    assert result.affordability_gap == pytest.approx(
        result.max_affordable_price_at_cap - result.median_price
    )


def test_affordability_breakdown_gap_is_negative_when_priced_out():
    """A household earning far less than needed for the median home
    should show a NEGATIVE affordability_gap — the quantified 'priced out
    by this much', not just a low share-of-income number."""
    result = affordability_breakdown(
        median_price=5_000_000.0,
        median_monthly_household_income=8_000.0,
        median_annual_household_income=96_000.0,
        annual_rate=0.05,
        years=30,
        ltv=0.75,
    )
    assert result.affordability_gap < 0
    assert result.within_repayment_cap is False
