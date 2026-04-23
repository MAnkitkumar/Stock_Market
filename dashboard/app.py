"""
app.py — StockScope Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data_loader import DEFAULT_STOCKS, fetch_stock_data
from feature_engineering import build_features
from signals import generate_ma_signals, get_signal_summary, latest_signal, get_recommendation
from model import train_models, predict_direction
from sentiment import get_sentiment_summary
from backtester import run_backtest, INITIAL_CAPITAL

st.set_page_config(page_title="StockScope", page_icon="📈", layout="wide")
st.title("📈 StockScope — Stock Analysis & Recommendation System")
st.caption("Real-time data · Technical analysis · Direction prediction · Backtesting · Sentiment")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    use_custom = st.checkbox("Enter custom ticker")
    if use_custom:
        custom_ticker = st.text_input("Ticker symbol (e.g. HDFCBANK.NS, AAPL, BTC-USD)", "HDFCBANK.NS")
        stock_name    = custom_ticker.upper()
        ticker        = custom_ticker.upper()
    else:
        stock_name = st.selectbox("Select Stock", list(DEFAULT_STOCKS.keys()))
        ticker     = DEFAULT_STOCKS[stock_name]

    period    = st.selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "2y", "5y"], index=5)
    interval  = "1m" if period == "1d" else ("5m" if period == "5d" else "1d")
    short_ma  = st.slider("Short MA (days)", 5, 50, 20)
    long_ma   = st.slider("Long MA (days)", 20, 200, 50)
    run_model = st.checkbox("Train / Refresh ML Models", value=False)

    st.markdown("---")
    live_mode = st.checkbox("🔴 Live refresh (60s)", value=False,
                            help="Auto-refreshes during market hours. Uses 1-min candles for 1d period.")
    st.caption("Supports any Yahoo Finance ticker")

# ── Load & process ────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data(ticker, period, interval):
    raw = fetch_stock_data(ticker, period=period, interval=interval, force_refresh=(interval != "1d"))
    return build_features(raw)

try:
    with st.spinner(f"Fetching {stock_name} data..."):
        df         = load_data(ticker, period, interval)
        df         = generate_ma_signals(df, short=short_ma, long=long_ma)
        signals_df = get_signal_summary(df)
        rec        = get_recommendation(df)
except Exception as e:
    st.error(f"Could not load data for '{ticker}': {e}")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
latest_close = df["Close"].iloc[-1]
prev_close   = df["Close"].iloc[-2]
change_pct   = (latest_close - prev_close) / prev_close * 100
rsi_now      = df["RSI"].iloc[-1]
vol_30       = df["Volatility_30d"].iloc[-1] * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Close",   f"₹{latest_close:,.2f}", f"{change_pct:+.2f}%")
c2.metric("RSI (14)",       f"{rsi_now:.1f}",         help="<30 oversold · >70 overbought")
c3.metric("30d Volatility", f"{vol_30:.2f}%")
c4.metric("Last Signal",    latest_signal(df).split(" ")[0])

# ── Recommendation Banner ─────────────────────────────────────────────────────
icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(rec["action"], "⚪")
st.markdown(f"### {icon} Signal: **{rec['action']}** &nbsp; Confidence: `{rec['confidence']}`")
st.caption(f"Based on: {rec['reason']}")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Price & Signals", "📉 Indicators", "🤖 ML Direction", "🔁 Backtest", "📰 Sentiment"]
)

# ── Tab 1: Price & Signals ────────────────────────────────────────────────────
with tab1:
    st.subheader(f"{stock_name} — Candlestick + Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC"
    ))
    for ma_col, color in [
        (f"SMA_{short_ma}", "orange"), (f"SMA_{long_ma}", "royalblue"),
        ("BB_upper", "rgba(180,180,180,0.3)"), ("BB_lower", "rgba(180,180,180,0.3)"),
    ]:
        if ma_col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ma_col], name=ma_col,
                                     line=dict(color=color, width=1.3)))
    buys  = signals_df[signals_df["Action"] == "BUY"]
    sells = signals_df[signals_df["Action"] == "SELL"]
    fig.add_trace(go.Scatter(x=buys.index,  y=buys["Close"],  mode="markers",
                             marker=dict(symbol="triangle-up",   color="lime", size=11), name="BUY"))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers",
                             marker=dict(symbol="triangle-down", color="red",  size=11), name="SELL"))
    fig.update_layout(xaxis_rangeslider_visible=False, height=520, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Signals (RSI-Confirmed MA Crossover)")
    if not signals_df.empty:
        st.dataframe(signals_df.tail(15).style.applymap(
            lambda v: "color: lime" if v == "BUY" else "color: red", subset=["Action"]
        ), use_container_width=True)
    else:
        st.info("No confirmed signals in this period.")

# ── Tab 2: Indicators ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("RSI (14)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="violet", width=1.5)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig_rsi.update_layout(height=280, template="plotly_dark", showlegend=False)
    st.plotly_chart(fig_rsi, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Daily Returns Distribution")
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

    st.subheader("Return Correlation (Default Stocks)")
    @st.cache_data(ttl=3600)
    def load_all_closes(period):
        closes = {}
        for name, sym in DEFAULT_STOCKS.items():
            try:
                closes[name] = fetch_stock_data(sym, period=period)["Close"]
            except Exception:
                pass
        return pd.DataFrame(closes).dropna()

    closes_df = load_all_closes(period)
    if not closes_df.empty:
        corr = closes_df.pct_change().corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                             template="plotly_dark")
        fig_corr.update_layout(height=350)
        st.plotly_chart(fig_corr, use_container_width=True)

# ── Tab 3: ML Direction Prediction ───────────────────────────────────────────
with tab3:
    st.subheader("Tomorrow's Price Direction — Classification Model")
    st.caption(
        "Predicts UP or DOWN (not raw price). Uses return-based features only. "
        "Validated with walk-forward (TimeSeriesSplit) — no data leakage."
    )

    if run_model:
        with st.spinner("Training models with walk-forward validation..."):
            results = train_models(df, ticker=stock_name)
        st.success("Models trained.")
        for mname, res in results.items():
            st.markdown(f"**{mname}**")
            for k, v in res["metrics"].items():
                st.write(f"  {k}: `{v}`")
            st.markdown("---")

    col_lr, col_rf = st.columns(2)
    for col, mname in [(col_lr, "LogisticRegression"), (col_rf, "RandomForest")]:
        try:
            pred = predict_direction(df, model_name=mname, ticker=stock_name)
            arrow = "⬆️" if pred["direction"] == "UP" else "⬇️"
            col.metric(
                f"{mname}",
                f"{arrow} {pred['direction']}",
                f"{pred['confidence']}% confidence"
            )
            col.caption(f"P(UP)={pred['prob_up']}%  P(DOWN)={pred['prob_down']}%")
        except Exception:
            col.warning(f"{mname}: Tick 'Train / Refresh ML Models' in sidebar first.")

    st.info(
        "Walk-forward accuracy of ~52–56% on financial data is realistic and meaningful. "
        "R²~0.99 on raw price regression is a data leakage artifact — not used here."
    )

# ── Tab 4: Backtest ───────────────────────────────────────────────────────────
with tab4:
    st.subheader(f"Strategy Backtest — {stock_name}")
    st.caption(
        f"Simulates MA crossover + RSI strategy. "
        f"Includes 0.1% transaction cost + 0.05% slippage per trade. "
        f"Starting capital: ₹{INITIAL_CAPITAL:,}"
    )

    if st.button("▶ Run Backtest"):
        with st.spinner("Running backtest..."):
            bt = run_backtest(df, initial_capital=INITIAL_CAPITAL)

        m  = bt["metrics"]
        bm = bt["benchmark_metrics"]

        st.subheader("Strategy vs Buy & Hold")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Strategy Return",  f"{m['Strategy Return (%)']}%",
                  f"{m['Strategy Return (%)'] - bm['Strategy Return (%)']:+.2f}% vs B&H")
        c2.metric("Sharpe Ratio",     str(m["Sharpe Ratio"]),
                  f"{m['Sharpe Ratio'] - bm['Sharpe Ratio']:+.3f} vs B&H")
        c3.metric("Max Drawdown",     f"{m['Max Drawdown (%)']}%")
        c4.metric("Transaction Costs",f"₹{m['Total Costs (₹)']:,}")

        # Portfolio vs benchmark chart
        st.subheader("Portfolio Value vs Buy & Hold")
        start_price = float(df["Close"].iloc[0])
        bh_value    = (df["Close"] / start_price) * INITIAL_CAPITAL

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt["portfolio_df"].index,
                                    y=bt["portfolio_df"]["total_value"],
                                    name="Strategy", line=dict(color="gold", width=2)))
        fig_bt.add_trace(go.Scatter(x=df.index, y=bh_value,
                                    name="Buy & Hold", line=dict(color="gray", width=1.5, dash="dash")))
        fig_bt.update_layout(height=380, template="plotly_dark", yaxis_title="Value (₹)")
        st.plotly_chart(fig_bt, use_container_width=True)

        st.subheader("Trade Log")
        if not bt["trade_log"].empty:
            st.dataframe(bt["trade_log"].style.applymap(
                lambda v: "color: lime" if v == "BUY" else "color: red", subset=["action"]
            ), use_container_width=True)
        else:
            st.info("No trades executed in this period.")

# ── Tab 5: Sentiment ──────────────────────────────────────────────────────────
with tab5:
    st.subheader("News Sentiment Analysis")
    st.caption(
        "Uses TextBlob for general polarity scoring. "
        "Note: TextBlob is a general-purpose NLP tool — treat scores as directional hints, "
        "not precise financial signals. FinBERT would be more accurate for financial text."
    )

    if st.button("🔍 Fetch Latest Headlines"):
        with st.spinner("Scraping headlines..."):
            sentiment = get_sentiment_summary(ticker)

        dominant = sentiment["dominant_sentiment"]
        avg_pol  = sentiment["avg_polarity"]
        color    = "green" if dominant == "Positive" else ("red" if dominant == "Negative" else "gray")
        icon_s   = "🟢" if dominant == "Positive" else ("🔴" if dominant == "Negative" else "🟡")

        st.markdown(f"### {icon_s} Dominant Sentiment: :{color}[**{dominant}**]")
        st.metric("Avg Polarity", avg_pol, help="-1 = very negative · 0 = neutral · +1 = very positive")

        if not sentiment["df"].empty:
            st.dataframe(sentiment["df"].style.applymap(
                lambda v: "color: lime" if v == "Positive" else ("color: red" if v == "Negative" else ""),
                subset=["label"],
            ), use_container_width=True)
        else:
            st.info("No headlines found. Check your internet connection.")

st.markdown("---")
st.caption("StockScope · Not financial advice · Data: Yahoo Finance")

# ── Live auto-refresh ─────────────────────────────────────────────────────────
if live_mode:
    import time
    st.toast("Live mode active — refreshing every 60s", icon="🔴")
    time.sleep(60)
    st.rerun()
