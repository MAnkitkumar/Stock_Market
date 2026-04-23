# StockScope: Data-Driven Stock Analysis & Insight System

A full end-to-end stock analysis dashboard built with Python, featuring:
- Real-time data via `yfinance`
- Technical indicators (SMA, EMA, Bollinger Bands)
- Buy/Sell signal generation
- Price prediction using Linear Regression & Random Forest
- Sentiment analysis from news headlines
- Interactive Streamlit dashboard

## Stocks Covered
- Reliance Industries (`RELIANCE.NS`)
- TCS (`TCS.NS`)
- Infosys (`INFY.NS`)

## Project Structure
```
stock-analysis-project/
├── data/                        # Cached stock data (CSV)
├── notebooks/EDA.ipynb          # Exploratory Data Analysis
├── src/
│   ├── data_loader.py           # Fetch & cache stock data
│   ├── feature_engineering.py   # Technical indicators & features
│   ├── model.py                 # Price prediction models
│   └── signals.py               # Buy/Sell signal logic
├── dashboard/app.py             # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Dashboard

```bash
streamlit run dashboard/app.py
```

## How It Works

1. **Data Collection** — Fetches historical OHLCV data using `yfinance`
2. **Feature Engineering** — Computes SMA, EMA, RSI, Bollinger Bands, daily returns, volatility
3. **Prediction** — Trains Linear Regression and Random Forest to predict next-day close
4. **Signals** — Golden Cross / Death Cross strategy for buy/sell signals
5. **Sentiment** — Scrapes news headlines and scores them with TextBlob
6. **Dashboard** — Visualizes everything interactively via Streamlit + Plotly
