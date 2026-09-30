"""
Step 3b: Goal-based investment planning.

For each goal (e.g. "Buy a house: ₹30,00,000 in 5 years") this works out:
  - an asset mix (equity / debt / cash) based on risk tolerance and time horizon,
  - the expected annual return of that mix,
  - the monthly investment (SIP) needed to reach the target.
It then checks the total against the 20% "savings" share of the salary
from the 50/30/20 budgeting rule.

The expected returns below are ASSUMPTIONS for illustration, not forecasts.
"""

# Assumed long-run annual returns for each asset class (illustrative only)
ASSUMED_RETURNS = {"equity": 0.10, "debt": 0.065, "cash": 0.04}

# Allocation (% equity, debt, cash) by risk tolerance and time horizon.
# Short goals are kept safer because there is less time to recover from a fall.
ALLOCATIONS = {
    "low":    {"short": (10, 60, 30), "medium": (30, 60, 10), "long": (50, 45, 5)},
    "medium": {"short": (20, 60, 20), "medium": (50, 45, 5),  "long": (70, 25, 5)},
    "high":   {"short": (30, 55, 15), "medium": (70, 25, 5),  "long": (85, 15, 0)},
}
SAVINGS_SHARE = 0.20   # the "20" in the 50/30/20 rule
MAX_GOALS = 5


def horizon_band(years: float) -> str:
    if years < 3:
        return "short"
    if years <= 7:
        return "medium"
    return "long"


def allocation_for(risk: str, years: float):
    equity, debt, cash = ALLOCATIONS[risk][horizon_band(years)]
    return {"equity": equity, "debt": debt, "cash": cash}


def expected_return(allocation) -> float:
    return sum(allocation[k] / 100 * ASSUMED_RETURNS[k] for k in ASSUMED_RETURNS)


def monthly_investment(target: float, years: float, annual_return: float) -> float:
    """Monthly amount that grows to `target` after `years`, invested at the end of
    each month (future value of an ordinary annuity, solved for the payment)."""
    months = round(years * 12)
    r = (1 + annual_return) ** (1 / 12) - 1   # equivalent monthly rate
    if r == 0:
        return target / months
    return target * r / ((1 + r) ** months - 1)


def plan_goals(goals, risk: str, annual_salary: float):
    """Validate the goals and build the full plan returned to the dashboard."""
    risk = str(risk).lower()
    if risk not in ALLOCATIONS:
        raise ValueError("Risk tolerance must be low, medium or high")
    if annual_salary <= 0:
        raise ValueError("Salary must be greater than zero")
    if not goals:
        raise ValueError("Add at least one goal")
    if len(goals) > MAX_GOALS:
        raise ValueError(f"You can plan up to {MAX_GOALS} goals at a time")

    plans = []
    for g in goals:
        name = str(g.get("name", "")).strip()[:40] or "Goal"
        try:
            target = float(g.get("target"))
            years = float(g.get("years"))
        except (TypeError, ValueError):
            raise ValueError(f"'{name}': target amount and years must be numbers")
        if target <= 0:
            raise ValueError(f"'{name}': target amount must be greater than zero")
        if not 0.5 <= years <= 40:
            raise ValueError(f"'{name}': years must be between 0.5 and 40")

        alloc = allocation_for(risk, years)
        ret = expected_return(alloc)
        monthly = monthly_investment(target, years, ret)
        total_paid = monthly * round(years * 12)
        plans.append({
            "name": name,
            "target": round(target, 2),
            "years": years,
            "horizon": horizon_band(years),
            "allocation": alloc,
            "expected_return": round(ret * 100, 2),
            "monthly": round(monthly, 2),
            "total_invested": round(total_paid, 2),
            "growth": round(target - total_paid, 2),
        })

    monthly_salary = annual_salary / 12
    budget = monthly_salary * SAVINGS_SHARE
    total_monthly = sum(p["monthly"] for p in plans)
    return {
        "goals": plans,
        "monthly_salary": round(monthly_salary, 2),
        "savings_budget": round(budget, 2),
        "total_monthly": round(total_monthly, 2),
        "share_of_salary": round(total_monthly / monthly_salary * 100, 1),
        "fits_budget": total_monthly <= budget,
        "assumed_returns": {k: v * 100 for k, v in ASSUMED_RETURNS.items()},
    }