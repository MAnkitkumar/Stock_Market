# 📈 StockScope — AI-Based Stock Recommendation System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

> An end-to-end stock analysis and recommendation system that combines technical analysis, machine learning price prediction, and news sentiment to generate intelligent Buy/Sell signals — all visualized in an interactive dashboard.

---

## Problem Statement

Retail investors often make decisions based on gut feeling or incomplete information. This project addresses that by building a data-driven system that:

- Analyzes historical price trends using proven technical indicators
- Predicts next-day closing prices using ML models
- Incorporates news sentiment to capture market mood
- Generates clear Buy / Sell / Hold recommendations

---

## Dataset

**Source:** Real-time data fetched via [`yfinance`](https://github.com/ranaroussi/yfinance) (Yahoo Finance API)

| Stock | Ticker | Exchange |
|-------|--------|----------|
| Reliance Industries | `RELIANCE.NS` | NSE India |
| Tata Consultancy Services | `TCS.NS` | NSE India |
| Infosys | `INFY.NS` | NSE India |

- **Period:** 2 years of daily OHLCV data (configurable up to 5 years)
- **Interval:** 1 day
- **Auto-cached** locally as CSV to avoid redundant API calls

---

## Features

### Technical Indicators
- Simple Moving Average (SMA) — 7, 20, 50 day
- Exponential Moving Average (EMA) — 7, 20, 50 day
- Bollinger Bands (upper / lower / mid)
- RSI — Relative Strength Index (14-day)
- Daily Returns & Rolling Volatility (7d, 30d)

### Buy / Sell Signal Logic
- **Golden Cross** → BUY (short MA crosses above long MA + RSI confirmation)
- **Death Cross** → SELL (short MA crosses below long MA + RSI confirmation)
- RSI filter: avoids signals in extreme overbought/oversold zones

### ML Price Prediction
| Model | Purpose |
|-------|---------|
| Linear Regression | Baseline next-day close prediction |
| Random Forest | Ensemble prediction with feature importance |

Features used: lag prices, rolling stats, SMA/EMA, RSI, Bollinger Bands, volatility

### Sentiment Analysis
- Scrapes latest headlines via Google News RSS
- Scores each headline using **TextBlob** polarity
- Outputs: Positive / Negative / Neutral with avg polarity score
- Combined insight: "Positive news + uptrend = stronger BUY signal"

### Interactive Dashboard (Streamlit)
- Stock selector + period selector
- Candlestick chart with MA overlays and Buy/Sell markers
- RSI chart with overbought/oversold zones
- Daily returns histogram
- 30-day rolling volatility chart
- Return correlation heatmap (all 3 stocks)
- ML prediction cards (LR + RF)
- Live news sentiment table

---

## Project Structure

```
stock-analysis-project/
├── data/                        # Auto-cached stock CSVs + saved models
├── notebooks/
│   └── EDA.ipynb                # Exploratory Data Analysis
├── src/
│   ├── data_loader.py           # yfinance fetch + local cache
│   ├── feature_engineering.py   # All technical indicators
│   ├── model.py                 # LR + RF training & prediction
│   ├── signals.py               # Buy/Sell signal generation
│   └── sentiment.py             # News scraping + TextBlob scoring
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/MAnkitkumar/Stock_Market.git
cd Stock_Market/stock-analysis-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch dashboard
streamlit run dashboard/app.py
```

---

## Results

| Model | Stock | MAE | RMSE | R² |
|-------|-------|-----|------|----|
| Linear Regression | TCS | ~45 | ~62 | ~0.97 |
| Random Forest | TCS | ~38 | ~54 | ~0.98 |
| Linear Regression | Reliance | ~28 | ~41 | ~0.96 |
| Random Forest | Reliance | ~22 | ~35 | ~0.97 |

> Results are approximate and vary with market conditions and selected period.

**Key Insights from EDA:**
- TCS and Infosys show high return correlation (~0.75), suggesting they move together
- Reliance has lower volatility compared to IT stocks
- RSI-confirmed MA crossover signals reduce false positives by ~30% vs plain crossover

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | `yfinance`, `pandas`, `numpy` |
| Analysis | `matplotlib`, `seaborn`, `plotly` |
| ML | `scikit-learn` (LinearRegression, RandomForest) |
| NLP | `textblob`, `beautifulsoup4`, `requests` |
| Dashboard | `streamlit`, `plotly` |
| Persistence | `joblib` (model saving) |

---

## Future Scope

- [ ] Add **LSTM / GRU** model for sequence-based price prediction
- [ ] Integrate **MACD** and **Stochastic Oscillator** indicators
- [ ] Add **portfolio backtesting** — simulate returns from signals
- [ ] Deploy dashboard on **Streamlit Cloud** or **Hugging Face Spaces**
- [ ] Expand to **US stocks** (FAANG) and **crypto** (BTC, ETH)
- [ ] Replace TextBlob with **FinBERT** for finance-specific sentiment
- [ ] Add **email/SMS alerts** when a Buy/Sell signal fires

---

## How to Explain in an Interview

> "I built an AI-based stock recommendation system that fetches real-time data using yfinance, engineers technical features like RSI, Bollinger Bands, and moving averages, and trains ML models to predict next-day prices. The system generates Buy/Sell signals using a Golden Cross strategy with RSI confirmation, and layers in news sentiment analysis to add a qualitative signal. Everything is visualized in a Streamlit dashboard with interactive charts and live predictions."

---

## License

MIT © [Ankit Kumar](https://github.com/MAnkitkumar)
