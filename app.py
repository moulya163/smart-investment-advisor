"""
Flask API + dashboard.
  /api/analyze   Step 2: LSTM forecast and salary simulation
  /api/optimize  Step 3: portfolio optimisation (Modern Portfolio Theory)
  /api/goals     Step 3: goal-based planning

Run:   python app.py
Then open http://127.0.0.1:5000 in your browser.
"""
import threading

from flask import Flask, jsonify, request, send_from_directory

from market import (validate_ticker, stock_currency, latest_prices, fx_rate,
                    SALARY_CURRENCIES)
from predictor import (get_model, torch_predictor, forecast_from_prices,
                       simulate_salary_investment, FORECAST_DAYS)

app = Flask(__name__, static_folder="static")
_training_lock = threading.Lock()  # only train one new model at a time


@app.get("/")
def dashboard():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    try:
        # 1. Validate input
        ticker = validate_ticker(data.get("ticker"))
        salary_ccy = str(data.get("salary_currency", "INR")).upper()
        if salary_ccy not in SALARY_CURRENCIES:
            raise ValueError("Salary currency must be INR, GBP or USD")
        try:
            salary = float(data.get("salary"))
            invest_pct = float(data.get("invest_pct", 20))
        except (TypeError, ValueError):
            raise ValueError("Salary and investment percentage must be numbers")
        stock_ccy = stock_currency(ticker)

        # 2. Latest prices (checked first, so an unknown symbol fails fast)
        prices, source, stale = latest_prices(ticker)

        # 3. Model (trains automatically the first time a stock is requested)
        with _training_lock:
            model, scaler, metrics = get_model(ticker)

        # 4. Forecast and salary simulation
        from train_lstm import WINDOW
        forecast = forecast_from_prices(prices.values, torch_predictor(model),
                                        scaler, WINDOW, FORECAST_DAYS)
        current = float(prices.iloc[-1])
        rate = fx_rate(salary_ccy, stock_ccy)
        sim = simulate_salary_investment(salary, invest_pct, salary_ccy, stock_ccy,
                                         rate, current, forecast[-1])

        history = prices.iloc[-60:]
        return jsonify({
            "ticker": ticker,
            "stock_currency": stock_ccy,
            "current_price": round(current, 2),
            "last_date": history.index[-1].strftime("%Y-%m-%d"),
            "price_source": source,
            "stale": stale,
            "history": [{"date": d.strftime("%Y-%m-%d"), "price": round(float(p), 2)}
                        for d, p in history.items()],
            "forecast": [round(p, 2) for p in forecast],
            "predicted_change_pct": round((forecast[-1] / current - 1) * 100, 2),
            "simulation": sim,
            "model": {
                "rmse": round(metrics["rmse"], 2),
                "baseline_rmse": round(metrics["baseline_rmse"], 2),
                "beats_baseline": metrics["beats_baseline"],
                "directional_accuracy": round(metrics["directional_accuracy"] * 100, 1),
                "newly_trained": metrics.get("newly_trained", False),
            },
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        app.logger.exception("Unexpected error in /api/analyze")
        return jsonify({"error": "Something went wrong on the server. Please try again."}), 500


# ---------------- Step 3: portfolio optimisation and goal planning ----------------

@app.post("/api/optimize")
def optimize():
    """Modern Portfolio Theory: best mix of the chosen stocks."""
    from portfolio import load_prices, optimise_portfolio, MIN_STOCKS, MAX_STOCKS
    data = request.get_json(silent=True) or {}
    try:
        raw = data.get("tickers", "")
        parts = raw.split(",") if isinstance(raw, str) else list(raw)
        tickers = list(dict.fromkeys(validate_ticker(t) for t in parts if str(t).strip()))
        if not MIN_STOCKS <= len(tickers) <= MAX_STOCKS:
            raise ValueError(f"Enter between {MIN_STOCKS} and {MAX_STOCKS} different stocks")
        currencies = {stock_currency(t) for t in tickers}
        if len(currencies) > 1:
            raise ValueError("Please choose stocks from one market only "
                             "(all US, or all Indian .NS/.BO)")
        stock_ccy = currencies.pop()

        salary_ccy = str(data.get("currency", "INR")).upper()
        if salary_ccy not in SALARY_CURRENCIES:
            raise ValueError("Currency must be INR, GBP or USD")
        try:
            amount = float(data.get("amount"))
            rf = float(data.get("risk_free_rate", 4)) / 100
        except (TypeError, ValueError):
            raise ValueError("Amount and risk-free rate must be numbers")
        if amount <= 0:
            raise ValueError("Amount must be greater than zero")
        if not 0 <= rf <= 0.2:
            raise ValueError("Risk-free rate must be between 0 and 20%")

        prices = load_prices(tickers)
        result = optimise_portfolio(prices, rf, amount)
        result["currency"] = salary_ccy
        result["stock_currency"] = stock_ccy
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        app.logger.exception("Unexpected error in /api/optimize")
        return jsonify({"error": "Something went wrong on the server. Please try again."}), 500


@app.post("/api/goals")
def goals():
    """Goal-based planning: monthly investment needed for each goal."""
    from goals import plan_goals
    data = request.get_json(silent=True) or {}
    try:
        currency = str(data.get("currency", "INR")).upper()
        if currency not in SALARY_CURRENCIES:
            raise ValueError("Currency must be INR, GBP or USD")
        try:
            salary = float(data.get("salary"))
        except (TypeError, ValueError):
            raise ValueError("Salary must be a number")
        goal_list = data.get("goals") or []
        if not isinstance(goal_list, list):
            raise ValueError("Goals must be a list")
        result = plan_goals(goal_list, data.get("risk", "medium"), salary)
        result["currency"] = currency
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        app.logger.exception("Unexpected error in /api/goals")
        return jsonify({"error": "Something went wrong on the server. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5000)