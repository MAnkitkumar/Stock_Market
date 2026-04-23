"""
signals.py
Buy/Sell signal generation using moving average crossover strategy.
Golden Cross  → Buy  (short MA crosses above long MA)
Death Cross   → Sell (short MA crosses below long MA)
"""

import pandas as pd


def generate_ma_signals(df: pd.DataFrame, short: int = 20, long: int = 50) -> pd.DataFrame:
    """
    Generate buy/sell signals based on SMA crossover.

    Args:
        df:    DataFrame with Close prices (feature_engineering already applied)
        short: Short-term SMA window
        long:  Long-term SMA window

    Returns:
        DataFrame with added Signal, Position columns
    """
    df = df.copy()

    short_col = f"SMA_{short}"
    long_col = f"SMA_{long}"

    # Compute MAs if not already present
    if short_col not in df.columns:
        df[short_col] = df["Close"].rolling(short).mean()
    if long_col not in df.columns:
        df[long_col] = df["Close"].rolling(long).mean()

    # 1 when short > long, 0 otherwise
    df["Position"] = (df[short_col] > df[long_col]).astype(int)

    # Signal fires only on the crossover day
    df["Signal"] = df["Position"].diff()
    # Signal:  1.0 = Buy,  -1.0 = Sell,  0 = Hold

    return df


def get_signal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the rows where a buy or sell signal fired."""
    signals = df[df["Signal"] != 0][["Close", "Signal"]].copy()
    signals["Action"] = signals["Signal"].map({1.0: "BUY", -1.0: "SELL"})
    return signals


def latest_signal(df: pd.DataFrame) -> str:
    """Return the most recent signal as a human-readable string."""
    summary = get_signal_summary(df)
    if summary.empty:
        return "HOLD"
    last = summary.iloc[-1]
    return f"{last['Action']} @ {last['Close']:.2f} on {last.name.date()}"


if __name__ == "__main__":
    from data_loader import fetch_stock_data
    from feature_engineering import build_features

    df = build_features(fetch_stock_data("TCS.NS"))
    df = generate_ma_signals(df)
    print(get_signal_summary(df).tail(10))
    print("\nLatest signal:", latest_signal(df))
