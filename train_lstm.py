"""
Step 1 (v2): Train an LSTM to forecast next-day prices by predicting DAILY
RETURNS, then converting back to prices. Evaluated against a naive baseline.

Why returns? Raw prices drift outside the range seen in training (distribution
shift), so a price-level LSTM lags and underestimates. Log returns stay in a
similar range over time, so the model generalises better.

Run:   python train_lstm.py AAPL
Output: models/AAPL_lstm.pt, models/AAPL_scaler.pkl, models/AAPL_results.png
"""
import os
import sys
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from data_utils import (download_prices, make_windows, test_windows, evaluate,
                        MODELS_DIR)

# ---------------- Settings ----------------
YEARS = 5          # years of daily history
WINDOW = 60        # days of past returns the model looks at
HIDDEN = 50        # LSTM hidden units
LAYERS = 2         # stacked LSTM layers
EPOCHS = 30
BATCH = 32
LR = 0.001
TRAIN_RATIO = 0.8
SEED = 42


class PriceLSTM(nn.Module):
    """Two-layer LSTM followed by a linear layer that outputs one value."""

    def __init__(self, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                            num_layers=layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # out: (batch, window, hidden)
        return self.fc(out[:, -1, :])  # use the last time step only


def prepare_returns(prices, window=WINDOW, train_ratio=TRAIN_RATIO):
    """Turn prices into scaled log-return windows, split chronologically.

    returns[k] = log(price[k+1]) - log(price[k])
    The price split is the same as before, so results are directly comparable.
    """
    returns = np.diff(np.log(prices)).astype(np.float32)
    split = int(len(prices) * train_ratio)

    train_r = returns[:split - 1]      # returns fully inside the training period
    test_r = returns[split - 1:]       # one return per test-day price

    scaler = StandardScaler()          # fitted on training returns only
    train_s = scaler.fit_transform(train_r.reshape(-1, 1))
    test_s = scaler.transform(test_r.reshape(-1, 1))

    X_train, y_train = make_windows(train_s, window)
    X_test, _ = test_windows(train_s, test_s, window)

    actual = prices[split:]                    # true prices on test days
    previous = prices[split - 1:-1]            # price the day before each one
    return X_train, y_train, X_test, actual, previous, scaler


def returns_to_prices(pred_scaled, scaler, previous):
    """Convert scaled predicted log returns back into predicted prices."""
    pred_r = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    return previous * np.exp(pred_r)


def train_model(ticker="AAPL", verbose=True):
    """Train and evaluate an LSTM for one ticker, save it, and return its test results.

    Used both from the command line and by the Flask API when a user asks
    about a stock that has not been trained yet.
    """
    log = print if verbose else (lambda *a, **k: None)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Data
    prices = download_prices(ticker, YEARS).values.astype(np.float64)
    log(f"{ticker}: {len(prices)} trading days")
    if len(prices) < WINDOW * 4:
        raise ValueError(f"Not enough price history for {ticker} to train a model")

    X_train, y_train, X_test, actual, previous, scaler = prepare_returns(prices)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # 2. Train
    model = PriceLSTM()
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH, shuffle=True)  # shuffling windows WITHIN train is fine

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(xb)
        if epoch % 5 == 0 or epoch == 1:
            log(f"Epoch {epoch:2d}/{EPOCHS}  train loss {total / len(X_train_t):.6f}")

    # 3. Evaluate on the unseen test period
    model.eval()
    with torch.no_grad():
        pred_s = model(X_test_t).numpy()
    predicted = returns_to_prices(pred_s, scaler, previous)

    results = evaluate(actual, predicted, previous)
    results["test_days"] = int(len(actual))
    log("\n--- Test results (returns model) ---")
    log(f"LSTM RMSE:             {results['rmse']:.2f}")
    log(f"LSTM MAE:              {results['mae']:.2f}")
    log(f"Naive baseline RMSE:   {results['baseline_rmse']:.2f}")
    log(f"Beats baseline:        {results['beats_baseline']}")
    log(f"Directional accuracy:  {results['directional_accuracy']:.1%}")

    # 4. Save model, scaler and results for the Flask API
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{ticker}_lstm.pt"))
    with open(os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, f"{ticker}_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 5. Plot actual vs predicted (only when run from the command line)
    if verbose:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(actual, label="Actual")
            plt.plot(predicted, label="LSTM prediction")
            plt.title(f"{ticker}: actual vs predicted (test period, returns model)")
            plt.xlabel("Trading day")
            plt.ylabel("Price")
            plt.legend()
            plt.tight_layout()
            path = os.path.join(MODELS_DIR, f"{ticker}_results.png")
            plt.savefig(path)
            plt.close()
            log(f"\nChart saved to {path}")
        except ImportError:
            pass

    return results


if __name__ == "__main__":
    train_model(sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL")
