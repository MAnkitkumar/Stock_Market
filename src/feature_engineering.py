"""
feature_engineering.py
Computes technical indicators and ML-ready features from raw OHLCV data.
"""

import pandas as pd
import numpy as np


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """SMA and EMA for common windows."""
    df = df.copy()
    for window in [7, 20, 50]:
        df[f"SMA_{window}"] = df["Close"].rolling(window).mean()
        df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Upper / lower Bollinger Bands (±2 std from SMA)."""
    df = df.copy()
    sma = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df["BB_upper"] = sma + 2 * std
    df["BB_lower"] = sma - 2 * std
    df["BB_mid"] = sma
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Relative Strength Index."""
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_returns_and_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Daily returns and rolling volatility (std of returns)."""
    df = df.copy()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_7d"] = df["Daily_Return"].rolling(7).std()
    df["Volatility_30d"] = df["Daily_Return"].rolling(30).std()
    return df


def add_lag_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Lag features for ML model (previous N days' close prices)."""
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f"Lag_{lag}"] = df["Close"].shift(lag)
    return df


def add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean and std of close price."""
    df = df.copy()
    df["Rolling_Mean_7"] = df["Close"].rolling(7).mean()
    df["Rolling_Std_7"] = df["Close"].rolling(7).std()
    df["Rolling_Mean_30"] = df["Close"].rolling(30).mean()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature pipeline — applies all indicators.
    Drops rows with NaN (from rolling windows).
    """
    df = add_moving_averages(df)
    df = add_bollinger_bands(df)
    df = add_rsi(df)
    df = add_returns_and_volatility(df)
    df = add_lag_features(df)
    df = add_rolling_stats(df)
    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    from data_loader import fetch_stock_data
    raw = fetch_stock_data("TCS.NS")
    featured = build_features(raw)
    print(featured.tail(3))
    print("Features:", featured.columns.tolist())
