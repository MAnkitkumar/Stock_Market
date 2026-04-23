"""
app.py — StockScope Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data_loader import DEFAULT_STOCKS, fetch_stock_data
from feature_engineering import build_features
from signals import generate_ma_signals, get_signal_summary, latest_signal
from model import train_models, predict_next_day
from sentiment import get_sentiment_summary

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockScope",
    page_icon="📈",
    layout="wide",
)

st.title("📈 StockScope: Stock Analysis & Insight Dashboard")
st.caption("Real-time data · Technical indicators · ML predictions · Sentiment analysis")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    stock_name = st.selectbox("Select Stock", list(DEFAULT_STOCKS.keys()))
    ticker = DEFAULT_STOCKS[stock_name]
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2)
    short_ma = st.slider("Short MA (days)", 5, 50, 20)
    long_ma = st.slider("Long MA (days)", 20, 200, 50)
    run_model = st.checkbox("Train / Refresh Prediction Model", value=False)
    st.markdown("---")
    st.caption("Data via Yahoo Finance (yfinance)")

# ── Load & process data ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(ticker, period):
    raw = fetch_stock_data(ticker, period=period)
    df = build_features(raw)
    return df

with st.spinner(f"Loading {stock_name} data..."):
    df = load_data(ticker, period)
    df = generate_ma_signals(df, short=short_ma, long=long_ma)

# ── KPI row ───────────────────────────────────────────────────────────────────
latest_close = df["Close"].iloc[-1]
prev_close = df["Close"].iloc[-2]
change_pct = (latest_close - prev_close) / prev_close * 100
avg_vol_30 = df["Volatility_30d"].iloc[-1] * 100
rsi_now = df["RSI"].iloc[-1]
signal_str = latest_signal(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Close", f"₹{latest_close:,.2f}", f"{change_pct:+.2f}%")
col2.metric("RSI (14)", f"{rsi_now:.1f}", help="<30 oversold · >70 overbought")
col3.metric("30d Volatility", f"{avg_vol_30:.2f}%")
col4.metric("Latest Signal", signal_str.split(" ")[0])

st.markdown("---")

# ── Candlestick + MA chart ────────────────────────────────────────────────────
st.subheader(f"{stock_name} — Price & Moving Averages")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name="OHLC"
))
for col, color in [(f"SMA_{short_ma}", "orange"), (f"SMA_{long_ma}", "blue"),
                   ("BB_upper", "rgba(150,150,150,0.4)"), ("BB_lower", "rgba(150,150,150,0.4)")]:
    if col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                 line=dict(color=color, width=1.2)))

# Overlay buy/sell signals
signals_df = get_signal_summary(df)
buys = signals_df[signals_df["Action"] == "BUY"]
sells = signals_df[signals_df["Action"] == "SELL"]
fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers",
                         marker=dict(symbol="triangle-up", color="green", size=10), name="BUY"))
fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers",
                         marker=dict(symbol="triangle-down", color="red", size=10), name="SELL"))

fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ── RSI chart ─────────────────────────────────────────────────────────────────
st.subheader("RSI (14)")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="violet")))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
fig_rsi.update_layout(height=250, template="plotly_dark", showlegend=False)
st.plotly_chart(fig_rsi, use_container_width=True)

# ── Returns & Volatility ──────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Daily Returns")
    fig_ret = px.histogram(df, x="Daily_Return", nbins=60,
                           color_discrete_sequence=["steelblue"], template="plotly_dark")
    fig_ret.update_layout(height=300)
    st.plotly_chart(fig_ret, use_container_width=True)

with col_b:
    st.subheader("30-Day Rolling Volatility")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=df.index, y=df["Volatility_30d"] * 100,
                                 fill="tozeroy", line=dict(color="tomato")))
    fig_vol.update_layout(height=300, template="plotly_dark",
                          yaxis_title="Volatility (%)", showlegend=False)
    st.plotly_chart(fig_vol, use_container_width=True)

# ── Correlation heatmap (all 3 stocks) ───────────────────────────────────────
st.subheader("Stock Correlation (Close Prices)")

@st.cache_data(ttl=3600)
def load_all_closes(period):
    closes = {}
    for name, sym in DEFAULT_STOCKS.items():
        try:
            raw = fetch_stock_data(sym, period=period)
            closes[name] = raw["Close"]
        except Exception:
            pass
    return pd.DataFrame(closes).dropna()

closes_df = load_all_closes(period)
corr = closes_df.pct_change().corr()
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                     template="plotly_dark", title="Return Correlation Matrix")
fig_corr.update_layout(height=350)
st.plotly_chart(fig_corr, use_container_width=True)

# ── ML Prediction ─────────────────────────────────────────────────────────────
st.subheader("Next-Day Price Prediction")

ticker_key = stock_name  # used as model filename key

if run_model:
    with st.spinner("Training models..."):
        results = train_models(df, ticker=ticker_key)
    st.success("Models trained and saved.")
    for mname, res in results.items():
        m = res["metrics"]
        st.write(f"**{mname}** — MAE: {m['MAE']} | RMSE: {m['RMSE']} | R²: {m['R2']}")

col_lr, col_rf = st.columns(2)
for col, mname in [(col_lr, "LinearRegression"), (col_rf, "RandomForest")]:
    try:
        pred = predict_next_day(df, model_name=mname, ticker=ticker_key)
        delta = pred - latest_close
        col.metric(f"{mname} Prediction", f"₹{pred:,.2f}", f"{delta:+.2f}")
    except Exception as e:
        col.warning(f"{mname}: Train model first. ({e})")

# ── Sentiment Analysis ────────────────────────────────────────────────────────
st.subheader("News Sentiment Analysis")

if st.button("Fetch Latest News Sentiment"):
    with st.spinner("Scraping headlines..."):
        sentiment = get_sentiment_summary(ticker)

    avg_pol = sentiment["avg_polarity"]
    dominant = sentiment["dominant_sentiment"]
    color = "green" if dominant == "Positive" else ("red" if dominant == "Negative" else "gray")

    st.markdown(f"**Dominant Sentiment:** :{color}[{dominant}]  |  Avg Polarity: `{avg_pol}`")

    if not sentiment["df"].empty:
        st.dataframe(
            sentiment["df"].style.applymap(
                lambda v: "color: green" if v == "Positive" else ("color: red" if v == "Negative" else ""),
                subset=["label"]
            ),
            use_container_width=True,
        )
    else:
        st.info("No headlines found. Check your internet connection.")

# ── Buy/Sell Signal Table ─────────────────────────────────────────────────────
st.subheader("Recent Buy/Sell Signals")
if not signals_df.empty:
    st.dataframe(signals_df.tail(15).style.applymap(
        lambda v: "color: green" if v == "BUY" else "color: red",
        subset=["Action"]
    ), use_container_width=True)
else:
    st.info("No crossover signals in the selected period.")

st.markdown("---")
st.caption("StockScope · Built with yfinance, scikit-learn, Streamlit & Plotly")
