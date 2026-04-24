# 📈 StockScope — Stock Analysis & Recommendation System

> End-to-end stock analysis: technical indicators · directional ML prediction · strategy backtesting · news sentiment

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![ML](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-pytest-4CAF50?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://stockscope.streamlit.app)

🚀 **[Live Demo → stockscope.streamlit.app](https://stockscope.streamlit.app)**

---

## Dashboard Preview

| Price & Signals | Indicators |
|---|---|
| ![Price & Signals](docs/screenshots/Screenshot%202026-04-24%20191149.png) | ![Indicators](docs/screenshots/Screenshot%202026-04-24%20191207.png) |

| ML Direction Prediction | Backtest vs Buy & Hold |
|---|---|
| ![ML Prediction](docs/screenshots/Screenshot%202026-04-24%20191228.png) | ![Backtest](docs/screenshots/Screenshot%202026-04-24%20191503.png) |

| News Sentiment |
|---|
| ![Sentiment](docs/screenshots/Screenshot%202026-04-24%20191518.png) |

---

## Problem Statement

Retail investors often make decisions based on incomplete information or gut feeling. This project builds a data-driven pipeline that:

- Applies proven technical indicators (RSI, Bollinger Bands, Moving Averages)
- Predicts tomorrow's price **direction** (UP/DOWN) using classification — not raw price regression
- Backtests the trading strategy with realistic transaction costs vs a buy-and-hold benchmark
- Layers in news sentiment as a qualitative signal

---

## Dataset

**Source:** Real-time OHLCV data via [`yfinance`](https://github.com/ranaroussi/yfinance)

Default tickers: Reliance (`RELIANCE.NS`), TCS (`TCS.NS`), Infosys (`INFY.NS`)

The dashboard accepts **any Yahoo Finance ticker** — NSE, BSE, NYSE, crypto (e.g. `HDFCBANK.NS`, `AAPL`, `BTC-USD`).

> `data/` is auto-generated at runtime (cached CSVs + saved models) and is not committed — see `.gitignore`.

---

## ML Approach — Why Classification, Not Regression

Raw price regression gives R²~0.99 because the model learns "tomorrow ≈ today." This is a **data leakage artifact**, not a useful signal.

This project instead:
- Predicts **direction** (UP or DOWN) — the actionable output
- Uses only **return-based features** (no raw price levels) to prevent leakage
- Validates with **walk-forward (TimeSeriesSplit)** — no future data bleeds into training

Walk-forward accuracy of **52–56%** is realistic for financial data. 99% would be a red flag.

| Model | Features | Validation | Walk-Forward Acc | Test Acc |
|-------|----------|------------|-----------------|----------|
| Logistic Regression | Returns, RSI, volatility, MA distance, lagged returns | TimeSeriesSplit (5-fold) | ~0.53 ± 0.04 | ~0.54 |
| Random Forest | Same | TimeSeriesSplit (5-fold) | ~0.55 ± 0.03 | ~0.56 |

> Scores are on held-out test data only. Random seed fixed at `42` for reproducibility.

---

## Backtest Results (TCS.NS · 2-year period · ₹1,00,000 starting capital)

| Metric | MA Crossover Strategy | Buy & Hold |
|--------|-----------------------|------------|
| Total Return | ~14% | ~22% |
| Sharpe Ratio | ~0.95 | ~1.10 |
| Max Drawdown | ~-9% | ~-18% |
| Transaction Costs | ~₹320 | ₹0 |
| Total Trades | ~12 | 1 |

The strategy doesn't always beat buy-and-hold on raw return — that's expected and honest. Its value is in **reducing drawdown** during downtrends. Run the backtest tab yourself to see live numbers for any ticker and period.

---

## Features

**Technical Indicators** (`feature_engineering.py`)
- SMA / EMA — 7, 20, 50 day
- Bollinger Bands — 20-day ±2σ
- RSI — 14-day
- Daily returns, 7d/30d rolling volatility
- Lag features (previous 5 days' returns)

**Buy/Sell Signals** (`signals.py`)
- Golden Cross → BUY (short MA crosses above long MA)
- Death Cross → SELL (short MA crosses below long MA)
- RSI filter: suppresses BUY above 70, SELL below 30

**Backtester** (`backtester.py`)
- 0.1% transaction cost + 0.05% slippage per trade
- Tracks portfolio value, Sharpe Ratio, Max Drawdown
- Compares strategy vs buy-and-hold benchmark

**Sentiment** (`sentiment.py`)
- Google News RSS scraping
- TextBlob polarity scoring
- Limitation acknowledged: TextBlob is general-purpose NLP. FinBERT would be more accurate for financial text.

---

## Project Structure

```
stock-analysis-project/
├── data/                        # Auto-generated: cached CSVs + saved models (gitignored)
├── docs/screenshots/            # Dashboard screenshots for README
├── notebooks/EDA.ipynb          # Exploratory Data Analysis
├── src/
│   ├── config.py                # Central config — tickers, periods, model params
│   ├── data_loader.py           # yfinance + TTL cache
│   ├── feature_engineering.py   # Technical indicators
│   ├── model.py                 # Classification models + walk-forward validation
│   ├── signals.py               # Buy/Sell signal generation
│   ├── backtester.py            # Strategy backtesting with costs
│   ├── sentiment.py             # News sentiment
│   └── alerts.py                # Email alerts
├── dashboard/app.py             # Streamlit dashboard (5 tabs)
├── streamlit_app.py             # Streamlit Cloud entry point
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_signals.py
│   └── test_backtester.py
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/MAnkitkumar/Stock_Market.git
cd Stock_Market/stock-analysis-project
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard/app.py

# Run tests
pytest tests/ -v
```

---

## Design Decisions

**Why classification instead of regression?**
Raw price regression gives R²~0.99 because "tomorrow ≈ today" is trivially learnable from lagged prices. That's not a useful model — it can't tell you whether to buy or sell. Predicting direction (UP/DOWN) is the actual decision a trader needs.

**Why walk-forward validation instead of random split?**
Stock data is time-ordered. A random train/test split lets the model train on future data to predict the past — that's data leakage. `TimeSeriesSplit` ensures training always uses only past data relative to the test window.

**Why RSI confirmation on MA crossovers?**
Pure MA crossovers fire late and generate false signals in sideways markets. Adding an RSI filter (suppress BUY when RSI > 70, suppress SELL when RSI < 30) reduces entries at exhaustion points.

**Why include transaction costs in the backtester?**
A strategy that looks profitable without costs often breaks even or loses after brokerage + STT. Including 0.1% cost + 0.05% slippage gives a more realistic picture of actual returns.

**Why TextBlob and not FinBERT?**
TextBlob is lightweight with zero setup cost — good for a portfolio project. FinBERT is more accurate for financial text but requires a GPU or paid API. The limitation is explicitly documented in the dashboard.

---

## Limitations

- TextBlob sentiment is general-purpose — FinBERT would be more accurate for financial text
- MA crossover is a lagging indicator — signals fire after a portion of the move is done
- No short selling in the backtester
- Walk-forward accuracy of 52–56% is realistic; higher numbers would indicate overfitting

---

## Future Scope

- [ ] Replace TextBlob with FinBERT for finance-specific sentiment
- [ ] Add LSTM / Transformer for sequence modelling
- [ ] Portfolio optimization across multiple stocks (Markowitz / risk parity)
- [ ] SMS/WhatsApp alerts via Twilio
- [ ] Options chain data integration

---

## Interview Explanation

**"What does this project do?"**
It's a full-stack data pipeline for stock analysis. It pulls live OHLCV data from Yahoo Finance, computes technical indicators, generates buy/sell signals using MA crossover with RSI confirmation, backtests the strategy with realistic transaction costs, and predicts tomorrow's price direction using classification models — all visualized in a Streamlit dashboard.

**"Why not just predict the price directly?"**
Because raw price regression gives R²~0.99 due to autocorrelation — "tomorrow ≈ today" is trivially learnable. That's not useful for trading. Predicting direction (UP/DOWN) is the actual actionable signal, and it's evaluated honestly with walk-forward validation.

**"How did you prevent data leakage?"**
Two ways: (1) features use only return-based inputs, not raw price levels, and (2) validation uses `TimeSeriesSplit` so the model never trains on future data relative to the test window.

---

## License

MIT © [Ankit Kumar](https://github.com/MAnkitkumar)
