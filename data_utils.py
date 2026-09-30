"""
Data preparation and evaluation helpers for the LSTM price forecaster.

Kept separate from the model code so each step can be tested on its own.
"""
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Folders live next to this file, so scripts work no matter where they are run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _last_n_years(close, years):
    """Keep only the most recent `years` years of a date-indexed series."""
    import pandas as pd
    cutoff = close.index.max() - pd.DateOffset(years=years)
    return close[close.index >= cutoff]


def _from_stooq(ticker):
    """Backup source: free daily prices from stooq.com (no API key needed).
    US stocks use the '.us' suffix on Stooq, e.g. AAPL -> aapl.us"""
    import pandas as pd
    symbol = ticker.lower() if "." in ticker else f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    df = pd.read_csv(url, index_col="Date", parse_dates=True)
    return df["Close"].dropna().sort_index()


def download_prices(ticker: str, years: int = 5, retries: int = 3, wait: int = 30):
    """Get daily closing prices for the last `years` years.

    Tries Yahoo Finance first, then Stooq as a backup.
    The first successful download is saved to data/<ticker>.csv, and later
    runs read from that file instead of calling the internet again.
    This avoids rate limits and makes results reproducible.
    Delete the CSV if you want fresh data.
    """
    import time
    import pandas as pd

    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"{ticker}.csv")

    if os.path.exists(cache):
        print(f"Loading saved data from {cache}")
        close = pd.read_csv(cache, index_col=0, parse_dates=True)["Close"].dropna()
        return _last_n_years(close, years)

    close = None

    # 1) Yahoo Finance
    try:
        import yfinance as yf
        for attempt in range(1, retries + 1):
            try:
                df = yf.Ticker(ticker).history(period=f"{years}y", interval="1d",
                                               auto_adjust=True)
                if not df.empty:
                    close = df["Close"].dropna()
                    close.index = close.index.tz_localize(None)  # plain dates
                    print("Downloaded from Yahoo Finance")
                    break
            except Exception as e:
                print(f"Yahoo attempt {attempt} failed ({e})")
            if attempt < retries:
                print(f"Waiting {wait}s before retrying...")
                time.sleep(wait)
    except ImportError:
        print("yfinance not installed, skipping Yahoo")

    # 2) Backup: Stooq
    if close is None:
        print("Trying backup source (Stooq)...")
        try:
            close = _from_stooq(ticker)
            print("Downloaded from Stooq")
        except Exception as e:
            print(f"Stooq failed ({e})")

    if close is None or close.empty:
        raise ValueError(
            f"Could not download {ticker} from Yahoo Finance or Stooq. "
            "Check your internet connection, wait 15-20 minutes and try again.")

    close.to_frame("Close").to_csv(cache)
    print(f"Saved to {cache}")
    return _last_n_years(close, years)


def chronological_split(prices: np.ndarray, train_ratio: float = 0.8):
    """Split in time order: first 80% for training, last 20% for testing.

    Never shuffle time-series data - that would let the model 'see the future'.
    """
    split = int(len(prices) * train_ratio)
    return prices[:split], prices[split:]


def scale(train: np.ndarray, test: np.ndarray):
    """Fit the scaler on TRAINING data only, then apply it to both sets.

    Fitting on the full dataset would leak information about future prices
    (e.g. the future maximum) into training.
    """
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1))
    test_scaled = scaler.transform(test.reshape(-1, 1))
    return train_scaled, test_scaled, scaler


def make_windows(series: np.ndarray, window: int = 60):
    """Turn a series into (X, y) pairs: X = previous `window` days, y = next day."""
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def test_windows(train_scaled: np.ndarray, test_scaled: np.ndarray, window: int = 60):
    """Build test windows. The first test windows need the last `window` days of
    training data as context, so we prepend them (this is not leakage: those days
    are in the past relative to every test target)."""
    combined = np.concatenate([train_scaled[-window:], test_scaled])
    return make_windows(combined, window)


def evaluate(actual: np.ndarray, predicted: np.ndarray, previous: np.ndarray):
    """Compare model predictions against actual prices.

    actual:    true price on day t+1
    predicted: model's forecast for day t+1
    previous:  true price on day t (used for the naive baseline and direction)
    """
    actual, predicted, previous = map(np.ravel, (actual, predicted, previous))

    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))

    # Naive baseline: "tomorrow's price = today's price"
    baseline_rmse = float(np.sqrt(np.mean((actual - previous) ** 2)))

    # Directional accuracy: did the model predict up/down correctly?
    true_dir = np.sign(actual - previous)
    pred_dir = np.sign(predicted - previous)
    directional_acc = float(np.mean(true_dir == pred_dir))

    return {
        "rmse": rmse,
        "mae": mae,
        "baseline_rmse": baseline_rmse,
        "beats_baseline": rmse < baseline_rmse,
        "directional_accuracy": directional_acc,
    }
