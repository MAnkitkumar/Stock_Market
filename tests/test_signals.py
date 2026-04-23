"""
Tests for signals.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np
from feature_engineering import build_features
from signals import generate_ma_signals, get_signal_summary, get_recommendation


@pytest.fixture
def featured_df():
    np.random.seed(0)
    n     = 200
    close = 1500 + np.cumsum(np.random.randn(n) * 15)
    idx   = pd.date_range("2022-01-01", periods=n, freq="B")
    raw   = pd.DataFrame({
        "Open": close * 0.99, "High": close * 1.01,
        "Low":  close * 0.98, "Close": close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=idx)
    return build_features(raw)


def test_signal_column_exists(featured_df):
    df = generate_ma_signals(featured_df)
    assert "Signal" in df.columns


def test_signal_values_valid(featured_df):
    df  = generate_ma_signals(featured_df)
    vals = df["Signal"].dropna().unique()
    for v in vals:
        assert v in (-1.0, 0.0, 1.0)


def test_no_buy_when_rsi_overbought(featured_df):
    df = generate_ma_signals(featured_df, rsi_overbought=70)
    buys = get_signal_summary(df)
    buys = buys[buys["Action"] == "BUY"]
    if not buys.empty:
        assert (buys["RSI"] < 70).all()


def test_recommendation_keys(featured_df):
    df  = generate_ma_signals(featured_df)
    rec = get_recommendation(df)
    assert "action" in rec
    assert "reason" in rec
    assert "confidence" in rec
    assert rec["action"] in ("BUY", "SELL", "HOLD")


def test_recommendation_confidence_values(featured_df):
    df  = generate_ma_signals(featured_df)
    rec = get_recommendation(df)
    assert rec["confidence"] in ("High", "Medium", "Low")
