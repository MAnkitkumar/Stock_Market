"""
backtester.py
Simulates the MA crossover + RSI strategy on historical data.

Realistic assumptions:
  - Transaction cost: 0.1% per trade (brokerage + STT approximation)
  - Slippage: 0.05% per trade
  - No short selling
  - Compares against Buy-and-Hold benchmark
"""

import pandas as pd
import numpy as np

INITIAL_CAPITAL   = 100_000   # INR
POSITION_SIZE_PCT = 0.95      # invest 95% of available cash per trade
TRANSACTION_COST  = 0.001     # 0.1% per trade
SLIPPAGE          = 0.0005    # 0.05% per trade
TOTAL_COST        = TRANSACTION_COST + SLIPPAGE


def run_backtest(df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> dict:
    """
    Simulate Buy/Sell strategy with realistic transaction costs.

    Args:
        df:              DataFrame with 'Close' and 'Signal' columns
        initial_capital: Starting capital in INR

    Returns:
        dict with portfolio_df, trade_log, metrics, benchmark_metrics
    """
    cash    = initial_capital
    shares  = 0.0
    records = []
    trades  = []

    for date, row in df.iterrows():
        price  = float(row["Close"])
        signal = row.get("Signal", 0)

        if signal == 1.0 and cash > 0:
            # Apply slippage — buy at slightly higher price
            exec_price = price * (1 + SLIPPAGE)
            invest     = cash * POSITION_SIZE_PCT
            cost       = invest * TRANSACTION_COST
            shares     = (invest - cost) / exec_price
            cash      -= invest
            trades.append({"date": date, "action": "BUY", "price": round(exec_price, 2),
                           "shares": round(shares, 4), "cost": round(cost, 2),
                           "value": round(invest, 2)})

        elif signal == -1.0 and shares > 0:
            # Apply slippage — sell at slightly lower price
            exec_price = price * (1 - SLIPPAGE)
            proceeds   = shares * exec_price
            cost       = proceeds * TRANSACTION_COST
            cash      += proceeds - cost
            trades.append({"date": date, "action": "SELL", "price": round(exec_price, 2),
                           "shares": round(shares, 4), "cost": round(cost, 2),
                           "value": round(proceeds, 2)})
            shares = 0.0

        total_value = cash + shares * price
        records.append({"date": date, "cash": round(cash, 2),
                        "shares": round(shares, 4), "total_value": round(total_value, 2)})

    portfolio_df = pd.DataFrame(records).set_index("date")
    trade_log    = pd.DataFrame(trades)

    metrics           = _compute_metrics(portfolio_df, initial_capital, trade_log)
    benchmark_metrics = _buy_and_hold(df, initial_capital)

    return {
        "portfolio_df":      portfolio_df,
        "trade_log":         trade_log,
        "metrics":           metrics,
        "benchmark_metrics": benchmark_metrics,
    }


def _compute_metrics(portfolio_df: pd.DataFrame, initial_capital: float,
                     trade_log: pd.DataFrame) -> dict:
    final_value  = portfolio_df["total_value"].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100

    daily_ret    = portfolio_df["total_value"].pct_change().dropna()
    sharpe       = ((daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
                    if daily_ret.std() > 0 else 0.0)

    rolling_max  = portfolio_df["total_value"].cummax()
    drawdown     = (portfolio_df["total_value"] - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    total_costs  = trade_log["cost"].sum() if not trade_log.empty else 0

    return {
        "Strategy Return (%)": round(total_return, 2),
        "Final Value (₹)":     round(final_value, 2),
        "Sharpe Ratio":        round(sharpe, 3),
        "Max Drawdown (%)":    round(max_drawdown, 2),
        "Total Trades":        len(trade_log),
        "Total Costs (₹)":     round(total_costs, 2),
    }


def _buy_and_hold(df: pd.DataFrame, initial_capital: float) -> dict:
    """Benchmark: buy on day 1, hold until end."""
    start_price  = float(df["Close"].iloc[0])
    end_price    = float(df["Close"].iloc[-1])
    shares       = (initial_capital * POSITION_SIZE_PCT) / start_price
    final_value  = shares * end_price + (initial_capital * (1 - POSITION_SIZE_PCT))
    total_return = (final_value - initial_capital) / initial_capital * 100

    daily_ret    = df["Close"].pct_change().dropna()
    sharpe       = ((daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
                    if daily_ret.std() > 0 else 0.0)

    rolling_max  = df["Close"].cummax()
    drawdown     = (df["Close"] - rolling_max) / rolling_max * 100

    return {
        "Strategy Return (%)": round(total_return, 2),
        "Final Value (₹)":     round(final_value, 2),
        "Sharpe Ratio":        round(sharpe, 3),
        "Max Drawdown (%)":    round(float(drawdown.min()), 2),
        "Total Trades":        1,
        "Total Costs (₹)":     0,
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import fetch_stock_data
    from feature_engineering import build_features
    from signals import generate_ma_signals

    df     = generate_ma_signals(build_features(fetch_stock_data("TCS.NS")))
    result = run_backtest(df)

    print("\n=== Strategy ===")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")

    print("\n=== Buy & Hold Benchmark ===")
    for k, v in result["benchmark_metrics"].items():
        print(f"  {k}: {v}")
