"""
Live market helpers: input validation, latest prices and currency conversion.
"""
import os
import re

import pandas as pd

from data_utils import DATA_DIR

TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")

# Stock exchange suffix -> currency the prices are quoted in
SUFFIX_CURRENCY = {".NS": "INR", ".BO": "INR"}
SALARY_CURRENCIES = {"INR", "GBP", "USD"}


def validate_ticker(raw: str) -> str:
    """Clean and check a ticker typed by the user (never trust user input)."""
    ticker = (raw or "").strip().upper()
    if not TICKER_PATTERN.match(ticker):
        raise ValueError("Please enter a valid stock symbol, e.g. AAPL, MSFT or RELIANCE.NS")
    return ticker


def stock_currency(ticker: str) -> str:
    """US stocks trade in USD; NSE/BSE stocks (.NS/.BO) trade in INR."""
    for suffix, currency in SUFFIX_CURRENCY.items():
        if ticker.endswith(suffix):
            return currency
    if "." in ticker:
        raise ValueError("Only US stocks (e.g. AAPL) and Indian stocks "
                         "(e.g. RELIANCE.NS) are supported at the moment")
    return "USD"


def latest_prices(ticker: str):
    """Fetch the most recent ~6 months of daily closing prices.

    Returns (prices, source, stale). Falls back to the saved training data if
    Yahoo Finance fails, and marks the result as stale so the dashboard
    can warn the user. Errors are printed to the terminal for debugging.
    """
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=True)
        if not df.empty:
            close = df["Close"].dropna()
            if close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            return close, "Yahoo Finance", False
        print(f"[latest_prices] Yahoo returned no data for {ticker}")
    except Exception as e:
        print(f"[latest_prices] Yahoo failed for {ticker}: {e!r}")

    cache = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(cache):
        close = pd.read_csv(cache, index_col=0, parse_dates=True)["Close"].dropna()
        return close, "saved data", True

    raise ValueError(f"Could not get current prices for {ticker}. Try again in a few minutes.")


def fx_rate(from_ccy: str, to_ccy: str) -> float:
    """How many units of `to_ccy` you get for 1 unit of `from_ccy`."""
    if from_ccy == to_ccy:
        return 1.0

    import requests
    for url in ("https://api.frankfurter.dev/v1/latest",
                "https://api.frankfurter.app/latest"):
        try:
            r = requests.get(url, params={"from": from_ccy, "to": to_ccy}, timeout=10)
            r.raise_for_status()
            return float(r.json()["rates"][to_ccy])
        except Exception:
            continue

    try:
        import yfinance as yf
        df = yf.Ticker(f"{from_ccy}{to_ccy}=X").history(period="5d")
        if not df.empty:
            return float(df["Close"].dropna().iloc[-1])
    except Exception:
        pass

    raise ValueError(f"Could not get the {from_ccy} to {to_ccy} exchange rate right now.")