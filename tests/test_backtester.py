"""
Tests for backtester.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np
from feature_engineering import build_features
from signals import generate_ma_signals
from backtester import run_backtest, INITIAL_CAPITAL


@pytest.fixture
def signal_df():
    np.random.seed(7)
    n     = 300
    close = 2000 + np.cumsum(np.random.randn(n) * 20)
    idx   = pd.date_range("2021-01-01", periods=n, freq="B")
    raw   = pd.DataFrame({
        "Open": close * 0.99, "High": close * 1.01,
        "Low":  close * 0.98, "Close": close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=idx)
    return generate_ma_signals(build_features(raw))


def test_backtest_returns_required_keys(signal_df):
    result = run_backtest(signal_df)
    assert "portfolio_df" in result
    assert "trade_log" in result
    assert "metrics" in result
    assert "benchmark_metrics" in result


def test_portfolio_value_always_positive(signal_df):
    result = run_backtest(signal_df)
    assert (result["portfolio_df"]["total_value"] > 0).all()


def test_portfolio_starts_near_initial_capital(signal_df):
    result = run_backtest(signal_df)
    first_val = result["portfolio_df"]["total_value"].iloc[0]
    assert abs(first_val - INITIAL_CAPITAL) / INITIAL_CAPITAL < 0.05


def test_trade_log_actions_valid(signal_df):
    result = run_backtest(signal_df)
    if not result["trade_log"].empty:
        assert set(result["trade_log"]["action"].unique()).issubset({"BUY", "SELL"})


def test_metrics_keys_present(signal_df):
    result  = run_backtest(signal_df)
    metrics = result["metrics"]
    for key in ["Strategy Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Total Trades"]:
        assert key in metrics


def test_benchmark_always_has_one_trade(signal_df):
    result = run_backtest(signal_df)
    assert result["benchmark_metrics"]["Total Trades"] == 1
