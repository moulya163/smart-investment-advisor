# Smart Investment Advisor

A full-stack web app that helps a user plan investments from their salary. It combines an **LSTM price forecaster**, **Modern Portfolio Theory optimisation** and **goal-based planning**, served through a **Flask** API with a **React** dashboard.

The project reports its results honestly: every forecast is compared against a naive baseline, so users can see how reliable the model really is.

> Educational project only. Not financial advice.

---

## Screenshots

| Stock forecast | Portfolio optimiser | Goal planner |
|---|---|---|
| ![Stock forecast](screenshots/forecast.png) | ![Portfolio optimiser](screenshots/portfolio.png) | ![Goal planner](screenshots/goals.png) |

---

## Features

### 1. Stock forecast and salary simulation
- Enter any US stock (e.g. `AAPL`, `MSFT`) or Indian stock (e.g. `RELIANCE.NS`), an annual salary and the percentage to invest.
- Fetches live prices from Yahoo Finance and forecasts the next 5 trading days with an LSTM.
- Simulates investing part of the salary: the amount is converted into the stock's currency at the **live exchange rate** before buying shares, then converted back to show the predicted profit or loss.
- If a stock has not been used before, a model is **trained automatically** on first request.
- Shows the model's test error next to a naive baseline, so the forecast is never presented as more reliable than it is.

### 2. Portfolio optimiser (Modern Portfolio Theory)
- Enter 2–8 stocks from one market and an amount to invest.
- Finds the **maximum Sharpe ratio** (best risk-adjusted) and **minimum volatility** (safest) portfolios, and compares them with an equal-weight portfolio.
- Plots 1,500 random portfolios and the **efficient frontier**, and shows how to split the amount across stocks.

### 3. Goal planner
- Add up to 5 goals (e.g. "Buy a house: ₹30,00,000 in 5 years").
- Suggests an equity / debt / cash mix for each goal based on risk tolerance and time horizon.
- Calculates the **monthly investment (SIP)** needed for each goal.
- Checks the total against 20% of monthly salary (the **50/30/20 budgeting rule**) and warns if the plan is unrealistic.

---

## Tech stack

| Area | Tools |
|---|---|
| Machine learning | PyTorch (LSTM), scikit-learn (scaling), NumPy, pandas |
| Optimisation | SciPy (SLSQP) |
| Backend | Python, Flask (REST API) |
| Frontend | React 18, SVG charts |
| Data | Yahoo Finance (`yfinance`), Frankfurter API (exchange rates) |

---

## Results

LSTM evaluated on the most recent 20% of 5 years of daily prices (unseen test period). The naive baseline predicts "tomorrow's price = today's price".

| Stock | Model | RMSE | Naive baseline RMSE | Directional accuracy |
|---|---|---|---|---|
| AAPL | v1: LSTM on raw prices | 16.37 | 4.50 | 48.2% |
| AAPL | **v2: LSTM on log returns** | **4.58** | 4.50 | 53.0% |
| MSFT | v2: LSTM on log returns | 8.92 | 8.81 | 46.2% |
| RELIANCE.NS | v2: LSTM on log returns | 18.13 | 18.06 | 49.2% |

**Key findings**
- The first model predicted raw prices and was **3.6× worse** than the naive baseline. Test-period prices were higher than most training prices (distribution shift), so the model lagged and underestimated.
- Reframing the task as predicting **log returns** fixed this and **cut RMSE by 72%**, bringing the model level with the baseline.
- Across three stocks in two markets, the LSTM consistently **matches but does not beat** the naive baseline, and its up/down accuracy stays close to 50%. This is consistent with daily stock prices behaving close to a random walk. The app therefore presents the forecast as one signal alongside portfolio and goal analysis, not as a guarantee.

---

## How it works

### LSTM forecasting (`train_lstm.py`, `data_utils.py`)
1. Downloads 5 years of daily closing prices and saves them locally, so later runs are reproducible and do not hit API rate limits.
2. Converts prices to daily **log returns**.
3. Splits the data **chronologically**: first 80% for training, last 20% for testing. Time series are never shuffled across the split.
4. Fits the scaler on **training data only**, to avoid leaking information from the test period.
5. Uses the previous **60 days** of returns to predict the next day's return.
6. Model: 2-layer LSTM (50 hidden units, dropout 0.2) with a linear output layer, trained with MSE loss and the Adam optimiser (learning rate 0.001, 30 epochs, batch size 32).
7. Evaluates with RMSE, MAE and directional accuracy against the naive baseline.
8. For the dashboard, forecasts 5 days ahead recursively, feeding each prediction back in as input.

### Portfolio optimisation (`portfolio.py`)
- Calculates annualised expected returns and the covariance matrix from daily returns (252 trading days per year).
- Solves mean-variance optimisation with SciPy (SLSQP), long-only, weights summing to 1:
  - **Max Sharpe:** maximise (return − risk-free rate) / volatility
  - **Min volatility:** minimise portfolio variance
- Traces the efficient frontier by minimising risk for a range of target returns.

### Goal planning (`goals.py`)
- Horizon bands: short (< 3 years), medium (3–7 years), long (> 7 years). Shorter goals get safer mixes.
- Assumed annual returns: equity 10%, debt 6.5%, cash 4% (clearly labelled as illustrative assumptions).
- Monthly investment uses the future value of an ordinary annuity:
  `payment = target × r / ((1 + r)^n − 1)`, where `r` is the monthly rate and `n` the number of months.

---

## Project structure

```
stock advisory/
├── app.py            Flask API and routes
├── train_lstm.py     LSTM model, training and evaluation
├── data_utils.py     Data download, caching, splitting, scaling, metrics
├── predictor.py      Loads models, forecasts, salary simulation
├── market.py         Input validation, live prices, exchange rates
├── portfolio.py      Modern Portfolio Theory optimisation
├── goals.py          Goal-based planning
├── static/
│   └── index.html    React dashboard
├── requirements.txt
└── README.md
```

`data/` (saved prices) and `models/` (trained models) are created automatically and are not stored in the repository.

---

## Setup and usage

**Requirements:** Python 3.10 or newer.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Train a model from the command line and see its test results
python train_lstm.py AAPL

# 3. Start the app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

If Yahoo Finance returns "Too Many Requests", upgrade the library with `pip install --upgrade yfinance` and wait a few minutes.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/analyze` | LSTM forecast and salary simulation for one stock |
| `POST` | `/api/optimize` | Portfolio optimisation for 2–8 stocks |
| `POST` | `/api/goals` | Monthly investment plan for up to 5 goals |

Example request:
```json
POST /api/analyze
{ "ticker": "AAPL", "salary": 1200000, "salary_currency": "INR", "invest_pct": 20 }
```

All inputs are validated on the server (ticker format, number ranges, supported currencies). Errors return clear messages without exposing internal details.

---

## Engineering notes

- **Rate limits:** Yahoo Finance frequently rate-limited older versions of `yfinance`. Downloaded data is cached locally, and the app falls back to saved data (with a visible warning) if live prices are unavailable.
- **Currency handling:** The original version of this idea mixed rupees and dollars when buying US shares. This version converts at the live exchange rate before buying and converts back for the result.
- **Data leakage:** Chronological splitting and training-only scaling ensure the model never sees future information.

---

## Limitations

- Forecasts use price history only, and do not beat a naive baseline on daily data.
- Portfolio optimisation is **in-sample**: weights are chosen and measured on the same historical period, so real future performance is likely lower. Max Sharpe portfolios often concentrate in a few stocks because they are sensitive to noisy return estimates.
- Goal-planner returns are fixed assumptions, not forecasts.
- Only US and Indian (NSE/BSE) stocks are supported.

---

## Future work

- Random Forest risk classification (low / moderate / high)
- Cryptocurrency support
- Maximum weight per stock in portfolio optimisation
- Out-of-sample backtesting of optimised portfolios
- Additional features for the forecaster (e.g. volume, technical indicators, news sentiment)

---

## Author

**Moulya Prasanna Kumar**, MSc Cybersecurity and Artificial Intelligence, University of Sheffield
[LinkedIn](http://www.linkedin.com/in/moulya-prasanna)