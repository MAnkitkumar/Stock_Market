"""
Tests for feature_engineering.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np
from feature_engineering import (
    add_moving_averages, add_bollinger_bands, add_rsi,
    add_returns_and_volatility, add_lag_features, build_features,
)


@pytest.fixture
def sample_df():
    """100 days of synthetic OHLCV data."""
    np.random.seed(42)
    n = 100
    close = 1000 + np.cumsum(np.random.randn(n) * 10)
    idx   = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open":   close * 0.99,
        "High":   close * 1.01,
        "Low":    close * 0.98,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=idx)


def test_sma_columns_created(sample_df):
    df = add_moving_averages(sample_df)
    for w in [7, 20, 50]:
        assert f"SMA_{w}" in df.columns
        assert f"EMA_{w}" in df.columns


def test_sma_values_correct(sample_df):
    df = add_moving_averages(sample_df)
    expected = sample_df["Close"].rolling(20).mean()
    pd.testing.assert_series_equal(df["SMA_20"], expected, check_names=False)


def test_bollinger_bands(sample_df):
    df = add_bollinger_bands(sample_df)
    assert "BB_upper" in df.columns
    assert "BB_lower" in df.columns
    # Upper must always be >= lower
    valid = df[["BB_upper", "BB_lower"]].dropna()
    assert (valid["BB_upper"] >= valid["BB_lower"]).all()


def test_rsi_range(sample_df):
    df = add_rsi(sample_df)
    rsi = df["RSI"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_daily_return_first_is_nan(sample_df):
    df = add_returns_and_volatility(sample_df)
    assert pd.isna(df["Daily_Return"].iloc[0])


def test_lag_features(sample_df):
    df = add_lag_features(sample_df, lags=3)
    assert "Lag_1" in df.columns
    assert "Lag_3" in df.columns
    # Lag_1 on row 5 should equal Close on row 4
    assert df["Lag_1"].iloc[5] == pytest.approx(sample_df["Close"].iloc[4])


def test_build_features_no_nan(sample_df):
    df = build_features(sample_df)
    assert df.isnull().sum().sum() == 0


def test_build_features_row_count(sample_df):
    df = build_features(sample_df)
    # Should have fewer rows than input due to rolling window dropna
    assert len(df) < len(sample_df)
