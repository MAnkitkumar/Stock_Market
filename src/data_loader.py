"""
data_loader.py
Fetches historical stock data using yfinance and caches it locally.
"""

import os
import yfinance as yf
import pandas as pd

# Default stocks (NSE-listed Indian stocks)
DEFAULT_STOCKS = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def fetch_stock_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data for a given ticker from Yahoo Finance.
    Caches result as CSV to avoid repeated API calls.

    Args:
        ticker:   Yahoo Finance ticker symbol (e.g. 'TCS.NS')
        period:   Data period — '1y', '2y', '5y', etc.
        interval: Data interval — '1d', '1wk', etc.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    ensure_data_dir()
    cache_path = os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}_{period}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"[cache] Loaded {ticker} from {cache_path}")
        return df

    print(f"[fetch] Downloading {ticker} ({period}) ...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache_path)
    print(f"[fetch] Saved to {cache_path}")
    return df


def load_multiple(tickers: dict = None, period: str = "2y") -> dict:
    """
    Load data for multiple tickers.

    Args:
        tickers: dict of {name: ticker_symbol}. Defaults to DEFAULT_STOCKS.
        period:  Data period string.

    Returns:
        dict of {name: DataFrame}
    """
    if tickers is None:
        tickers = DEFAULT_STOCKS

    return {name: fetch_stock_data(symbol, period=period) for name, symbol in tickers.items()}


def refresh_cache(ticker: str, period: str = "2y"):
    """Delete cached file and re-fetch fresh data."""
    cache_path = os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}_{period}.csv")
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"[cache] Cleared cache for {ticker}")
    return fetch_stock_data(ticker, period=period)


if __name__ == "__main__":
    data = load_multiple()
    for name, df in data.items():
        print(f"\n{name}: {df.shape}")
        print(df.tail(3))
