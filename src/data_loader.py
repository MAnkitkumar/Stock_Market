"""
data_loader.py
Fetches historical stock data using yfinance with TTL-based cache invalidation.
Cache is considered stale after CACHE_TTL_HOURS and auto-refreshed.
"""

import os
import time
import yfinance as yf
import pandas as pd

DEFAULT_STOCKS = {
    "Reliance": "RELIANCE.NS",
    "TCS":      "TCS.NS",
    "Infosys":  "INFY.NS",
}

DATA_DIR       = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_TTL_HOURS = 4   # refresh cache if older than this


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _cache_path(ticker: str, period: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}_{period}.csv")


def _is_cache_stale(path: str) -> bool:
    """Returns True if cache file doesn't exist or is older than CACHE_TTL_HOURS."""
    if not os.path.exists(path):
        return True
    age_hours = (time.time() - os.path.getmtime(path)) / 3600
    return age_hours > CACHE_TTL_HOURS


def fetch_stock_data(ticker: str, period: str = "2y", interval: str = "1d",
                     force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given ticker from Yahoo Finance.
    - Daily data: cached to CSV with 4-hour TTL
    - Intraday (1m, 5m): never cached — always fetched live

    Args:
        ticker:        Yahoo Finance ticker symbol (e.g. 'TCS.NS')
        period:        Data period — '1y', '2y', '5y', etc.
        interval:      Data interval — '1d', '1m', '5m', etc.
        force_refresh: Bypass cache and re-fetch regardless of TTL.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    ensure_data_dir()

    # Intraday intervals — never cache, always live
    intraday = interval in ("1m", "2m", "5m", "15m", "30m", "60m", "90m")

    if not intraday:
        path = _cache_path(ticker, period)
        if not force_refresh and not _is_cache_stale(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            print(f"[cache] Loaded {ticker} (fresh cache)")
            return df

    print(f"[fetch] Downloading {ticker} ({period}, {interval}) ...")
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
    except Exception as e:
        if not intraday and os.path.exists(_cache_path(ticker, period)):
            print(f"[fetch] Network error ({e}). Using stale cache.")
            return pd.read_csv(_cache_path(ticker, period), index_col=0, parse_dates=True)
        raise

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # yfinance >=1.0 returns a MultiIndex (Price, Ticker) — flatten to single level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Drop Adj Close if present (auto_adjust=True makes it redundant)
    df = df.drop(columns=["Adj Close"], errors="ignore")

    # Ensure standard column names
    df.columns = [c.strip() for c in df.columns]

    # Only persist daily data to disk
    if not intraday:
        df.to_csv(_cache_path(ticker, period))
        print(f"[fetch] Saved {ticker} → {_cache_path(ticker, period)}")

    return df


def load_multiple(tickers: dict = None, period: str = "2y") -> dict:
    """Load data for multiple tickers. Accepts any {name: ticker} dict."""
    if tickers is None:
        tickers = DEFAULT_STOCKS
    return {name: fetch_stock_data(sym, period=period) for name, sym in tickers.items()}


def search_ticker(query: str) -> str:
    """
    Basic helper — returns the query as-is (yfinance accepts any valid ticker).
    Users can type any NSE/BSE/NYSE ticker directly.
    """
    return query.strip().upper()


if __name__ == "__main__":
    data = load_multiple()
    for name, df in data.items():
        print(f"{name}: {df.shape}  last={df.index[-1].date()}")
