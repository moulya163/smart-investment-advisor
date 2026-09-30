"""
Loads a trained LSTM for a stock (training one first if needed), forecasts the
next few days, and simulates investing part of the user's salary.
"""
import os
import json
import pickle

import numpy as np

from data_utils import MODELS_DIR

FORECAST_DAYS = 5
_loaded = {}  # ticker -> (model, scaler, metrics), kept in memory between requests


def forecast_from_prices(prices, predict_scaled, scaler, window, days=FORECAST_DAYS):
    """Forecast `days` future prices from recent prices.

    predict_scaled: function taking a (window,) array of scaled returns and
    returning the next scaled return. Each prediction is fed back in to
    predict the following day (recursive multi-step forecasting).
    """
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) < window + 1:
        raise ValueError("Not enough recent prices to make a forecast")

    returns = np.diff(np.log(prices))
    scaled = scaler.transform(returns.reshape(-1, 1)).ravel()
    history = list(scaled[-window:])

    preds = []
    for _ in range(days):
        nxt = float(predict_scaled(np.array(history[-window:], dtype=np.float32)))
        preds.append(nxt)
        history.append(nxt)

    pred_r = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
    return (prices[-1] * np.exp(np.cumsum(pred_r))).tolist()


def simulate_salary_investment(salary, invest_pct, salary_ccy, stock_ccy,
                               rate_salary_to_stock, current_price, predicted_price):
    """Invest `invest_pct`% of salary in the stock at today's price and value it
    at the predicted price. Money is converted into the stock's currency before
    buying shares, and back into the salary currency for the result."""
    if salary <= 0:
        raise ValueError("Salary must be greater than zero")
    if not 0 < invest_pct <= 100:
        raise ValueError("Investment percentage must be between 1 and 100")

    invest_salary_ccy = salary * invest_pct / 100
    invest_stock_ccy = invest_salary_ccy * rate_salary_to_stock
    shares = invest_stock_ccy / current_price
    value_stock_ccy = shares * predicted_price
    value_salary_ccy = value_stock_ccy / rate_salary_to_stock

    return {
        "salary_currency": salary_ccy,
        "stock_currency": stock_ccy,
        "exchange_rate": rate_salary_to_stock,
        "invested": round(invest_salary_ccy, 2),
        "invested_in_stock_currency": round(invest_stock_ccy, 2),
        "shares": round(shares, 4),
        "predicted_value": round(value_salary_ccy, 2),
        "profit_loss": round(value_salary_ccy - invest_salary_ccy, 2),
        "profit_loss_pct": round((predicted_price / current_price - 1) * 100, 2),
    }


def get_model(ticker):
    """Load the saved model for `ticker`, training it first if it doesn't exist."""
    if ticker in _loaded:
        return _loaded[ticker]

    import torch
    from train_lstm import PriceLSTM, train_model

    paths = {k: os.path.join(MODELS_DIR, f"{ticker}_{k}")
             for k in ("lstm.pt", "scaler.pkl", "metrics.json")}
    newly_trained = False
    if not all(os.path.exists(p) for p in paths.values()):
        train_model(ticker, verbose=False)
        newly_trained = True

    model = PriceLSTM()
    model.load_state_dict(torch.load(paths["lstm.pt"], weights_only=True))
    model.eval()
    with open(paths["scaler.pkl"], "rb") as f:
        scaler = pickle.load(f)
    with open(paths["metrics.json"]) as f:
        metrics = json.load(f)

    _loaded[ticker] = (model, scaler, metrics)
    return model, scaler, {**metrics, "newly_trained": newly_trained}


def torch_predictor(model):
    """Wrap the PyTorch model as a simple numpy -> float function."""
    import torch

    def predict(window_scaled):
        x = torch.tensor(window_scaled, dtype=torch.float32).reshape(1, -1, 1)
        with torch.no_grad():
            return model(x).item()
    return predict
