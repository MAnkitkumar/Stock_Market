"""
signals.py
Buy/Sell signal generation using MA crossover + RSI confirmation.

Strategy:
  Golden Cross → BUY  (short MA crosses above long MA, RSI not overbought)
  Death Cross  → SELL (short MA crosses below long MA, RSI not oversold)
  RSI filter reduces false signals in extreme zones.
"""

import pandas as pd


def generate_ma_signals(
    df: pd.DataFrame,
    short: int = 20,
    long: int = 50,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
) -> pd.DataFrame:
    """
    Generate Buy/Sell signals using SMA crossover with RSI confirmation.

    Args:
        df:              DataFrame with Close prices + RSI (from feature_engineering)
        short:           Short-term SMA window
        long:            Long-term SMA window
        rsi_overbought:  RSI threshold above which BUY signals are suppressed
        rsi_oversold:    RSI threshold below which SELL signals are suppressed

    Returns:
        DataFrame with Signal (-1 / 0 / 1) and Position columns added
    """
    df = df.copy()

    short_col = f"SMA_{short}"
    long_col = f"SMA_{long}"

    if short_col not in df.columns:
        df[short_col] = df["Close"].rolling(short).mean()
    if long_col not in df.columns:
        df[long_col] = df["Close"].rolling(long).mean()

    # Raw crossover position: 1 when short > long
    df["Position"] = (df[short_col] > df[long_col]).astype(int)
    raw_signal = df["Position"].diff()  # +1 = golden cross, -1 = death cross

    # RSI confirmation filter
    rsi = df.get("RSI", pd.Series(50, index=df.index))

    # BUY only if RSI < overbought threshold (not already overheated)
    # SELL only if RSI > oversold threshold (not already bottomed out)
    confirmed = raw_signal.copy()
    confirmed[(raw_signal == 1) & (rsi >= rsi_overbought)] = 0   # suppress BUY
    confirmed[(raw_signal == -1) & (rsi <= rsi_oversold)] = 0    # suppress SELL

    df["Signal"] = confirmed
    return df


def get_signal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where a confirmed buy or sell signal fired."""
    signals = df[df["Signal"] != 0][["Close", "Signal", "RSI"]].copy()
    signals["Action"] = signals["Signal"].map({1.0: "BUY", -1.0: "SELL"})
    return signals


def latest_signal(df: pd.DataFrame) -> str:
    """Return the most recent signal as a human-readable string."""
    summary = get_signal_summary(df)
    if summary.empty:
        return "HOLD"
    last = summary.iloc[-1]
    return f"{last['Action']} @ {last['Close']:.2f} on {last.name.date()}"


def get_recommendation(df: pd.DataFrame) -> dict:
    """
    Simple rule-based recommendation using latest price vs MAs + RSI.

    Returns:
        dict with action, reason, confidence
    """
    latest = df.iloc[-1]
    close = latest["Close"]
    sma20 = latest.get("SMA_20", close)
    sma50 = latest.get("SMA_50", close)
    rsi = latest.get("RSI", 50)

    if close > sma20 > sma50 and rsi < 70:
        action = "BUY"
        reason = f"Price above SMA20 & SMA50, RSI={rsi:.1f} (not overbought)"
        confidence = "High" if rsi < 60 else "Medium"
    elif close < sma20 < sma50 and rsi > 30:
        action = "SELL"
        reason = f"Price below SMA20 & SMA50, RSI={rsi:.1f} (not oversold)"
        confidence = "High" if rsi > 40 else "Medium"
    else:
        action = "HOLD"
        reason = f"No clear trend. RSI={rsi:.1f}"
        confidence = "Low"

    return {"action": action, "reason": reason, "confidence": confidence}


if __name__ == "__main__":
    from data_loader import fetch_stock_data
    from feature_engineering import build_features

    df = build_features(fetch_stock_data("TCS.NS"))
    df = generate_ma_signals(df)
    print(get_signal_summary(df).tail(10))
    print("\nLatest signal:", latest_signal(df))
    print("Recommendation:", get_recommendation(df))
