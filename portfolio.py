"""
Step 3a: Portfolio optimisation with Modern Portfolio Theory (mean-variance).

Given a few stocks, find the mix (weights) that gives:
  - the best return per unit of risk (maximum Sharpe ratio), and
  - the lowest possible risk (minimum volatility).
Uses 5 years of daily prices, annualised with 252 trading days per year.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data_utils import download_prices

TRADING_DAYS = 252
MIN_STOCKS, MAX_STOCKS = 2, 8


def load_prices(tickers, years=5):
    """Closing prices for several tickers, aligned on the dates they all traded."""
    series = {t: download_prices(t, years) for t in tickers}
    df = pd.concat(series, axis=1).dropna()
    if len(df) < TRADING_DAYS:
        raise ValueError("Not enough overlapping price history for these stocks")
    return df


def annualised_stats(prices: pd.DataFrame):
    """Expected annual return of each stock and the annual covariance matrix."""
    daily = prices.pct_change().dropna()
    mu = daily.mean().values * TRADING_DAYS
    cov = daily.cov().values * TRADING_DAYS
    return mu, cov


def portfolio_performance(w, mu, cov, rf):
    """Return (expected return, volatility, Sharpe ratio) for weights w."""
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _solve(objective, n, extra_constraints=()):
    """Minimise `objective` over long-only weights that sum to 1."""
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}, *extra_constraints]
    result = minimize(objective, np.full(n, 1 / n), method="SLSQP",
                      bounds=[(0.0, 1.0)] * n, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-12})
    if not result.success:
        raise ValueError(f"Optimisation did not converge: {result.message}")
    w = np.clip(result.x, 0, None)
    return w / w.sum()


def max_sharpe_weights(mu, cov, rf):
    return _solve(lambda w: -portfolio_performance(w, mu, cov, rf)[2], len(mu))


def min_volatility_weights(mu, cov):
    return _solve(lambda w: w @ cov @ w, len(mu))


def efficient_frontier(mu, cov, points=25):
    """Lowest-risk portfolio for a range of target returns (the frontier curve)."""
    w_min = min_volatility_weights(mu, cov)
    low, high = float(w_min @ mu), float(mu.max())
    curve = []
    for target in np.linspace(low, high, points):
        try:
            w = _solve(lambda w: w @ cov @ w, len(mu),
                       [{"type": "eq", "fun": lambda w, t=target: w @ mu - t}])
            curve.append({"ret": float(w @ mu), "vol": float(np.sqrt(w @ cov @ w))})
        except ValueError:
            continue
    return curve


def random_portfolios(mu, cov, rf, n=1500, seed=42):
    """Random long-only portfolios, drawn to show the cloud of possible mixes."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(mu)), size=n)
    rets = weights @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    return [{"ret": float(r), "vol": float(v), "sharpe": float((r - rf) / v)}
            for r, v in zip(rets, vols)]


def optimise_portfolio(prices: pd.DataFrame, rf: float, amount: float):
    """Run the full optimisation on a price table and return results for the API."""
    tickers = list(prices.columns)
    mu, cov = annualised_stats(prices)

    def describe(w):
        ret, vol, sharpe = portfolio_performance(w, mu, cov, rf)
        return {
            "weights": [{"ticker": t, "weight": round(float(x) * 100, 2),
                         "amount": round(float(x) * amount, 2)}
                        for t, x in zip(tickers, w)],
            "expected_return": round(ret * 100, 2),
            "volatility": round(vol * 100, 2),
            "sharpe": round(sharpe, 2),
        }

    equal = np.full(len(tickers), 1 / len(tickers))
    return {
        "tickers": tickers,
        "from_date": prices.index[0].strftime("%Y-%m-%d"),
        "to_date": prices.index[-1].strftime("%Y-%m-%d"),
        "risk_free_rate": round(rf * 100, 2),
        "stocks": [{"ticker": t, "expected_return": round(float(m) * 100, 2),
                    "volatility": round(float(np.sqrt(cov[i, i])) * 100, 2)}
                   for i, (t, m) in enumerate(zip(tickers, mu))],
        "max_sharpe": describe(max_sharpe_weights(mu, cov, rf)),
        "min_volatility": describe(min_volatility_weights(mu, cov)),
        "equal_weight": describe(equal),
        "frontier": [{"ret": round(p["ret"] * 100, 2), "vol": round(p["vol"] * 100, 2)}
                     for p in efficient_frontier(mu, cov)],
        "random": [{"ret": round(p["ret"] * 100, 2), "vol": round(p["vol"] * 100, 2),
                    "sharpe": round(p["sharpe"], 2)}
                   for p in random_portfolios(mu, cov, rf)],
    }