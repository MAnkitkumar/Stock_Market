# 📈 StockScope — Stock Analysis & Recommendation System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![ML](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-pytest-blue?logo=pytest)
![License](https://img.shields.io/badge/License-MIT-22c55e)

An end-to-end stock analysis system with technical indicators, directional ML prediction, strategy backtesting, and news sentiment — visualized in an interactive Streamlit dashboard.

---

## Problem Statement

Retail investors often make decisions based on incomplete information or gut feeling. This project builds a data-driven pipeline that:

- Applies proven technical indicators (RSI, Bollinger Bands, Moving Averages)
- Predicts tomorrow's price **direction** (UP/DOWN) using classification models
- Backtests the trading strategy with realistic transaction costs vs a buy-and-hold benchmark
- Layers in news sentiment as a qualitative signal

---

## Dataset

**Source:** Real-time OHLCV data via [`yfinance`](https://github.com/ranaroussi/yfinance)

Default stocks: Reliance (`RELIANCE.NS`), TCS (`TCS.NS`), Infosys (`INFY.NS`)

The dashboard also accepts **any custom ticker** — NSE, BSE, NYSE, crypto (e.g. `HDFCBANK.NS`, `AAPL`, `BTC-USD`).

Cache is auto-invalidated after 4 hours. Falls back to stale cache on network failure.

---

## ML Approach — Why Classification, Not Regression

Raw price regression (predicting tomorrow's exact price) gives R²~0.99 because the model learns "tomorrow ≈ today." This is a **data leakage artifact**, not a useful signal.

This project instead:
- Predicts **direction** (UP or DOWN) — the actionable output
- Uses only **return-based features** (no raw price levels) to avoid leakage
- Validates with **walk-forward (TimeSeriesSplit)** — no future data bleeds into training

Realistic walk-forward accuracy: **52–56%**. That's honest. 99% would be a red flag.

| Model | Features Used | Validation |
|-------|--------------|------------|
| Logistic Regression | Daily returns, RSI, volatility, MA distance, lagged returns | Walk-forward (5-fold) |
| Random Forest | Same | Walk-forward (5-fold) |

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
- Limitation acknowledged: TextBlob is general-purpose NLP, not finance-specific. FinBERT would be more accurate.

---

## Project Structure

```
stock-analysis-project/
├── data/                        # Cached CSVs + saved models (gitignored)
├── notebooks/EDA.ipynb          # Exploratory Data Analysis
├── src/
│   ├── config.py                # Central config
│   ├── data_loader.py           # yfinance + TTL cache
│   ├── feature_engineering.py   # Technical indicators
│   ├── model.py                 # Classification models + walk-forward validation
│   ├── signals.py               # Buy/Sell signal generation
│   ├── backtester.py            # Strategy backtesting with costs
│   ├── sentiment.py             # News sentiment
│   └── alerts.py                # Email alerts
├── dashboard/app.py             # Streamlit dashboard (5 tabs)
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

## Backtest Results (approximate, varies by period)

| Metric | TCS (2y) | Reliance (2y) | Buy & Hold (TCS) |
|--------|----------|---------------|-----------------|
| Total Return | ~12–20% | ~8–16% | ~15–25% |
| Sharpe Ratio | ~0.8–1.2 | ~0.7–1.0 | ~0.9–1.3 |
| Max Drawdown | ~-10% | ~-8% | ~-18% |

The strategy doesn't always beat buy-and-hold — that's expected and honest. Its value is in **reducing drawdown** during downtrends.

---

## Known Limitations

- TextBlob sentiment is general-purpose — FinBERT would be more accurate for financial text
- MA crossover is a lagging indicator — signals fire after a portion of the move is done
- No short selling in the backtester
- Walk-forward accuracy of 52–56% is realistic; higher numbers would indicate overfitting

---

## Future Scope

- [ ] Replace TextBlob with FinBERT
- [ ] Add LSTM for sequence modelling
- [ ] Portfolio optimization across multiple stocks
- [ ] Deploy on Streamlit Cloud
- [ ] SMS alerts via Twilio

---

## License

MIT © [Ankit Kumar](https://github.com/MAnkitkumar)
