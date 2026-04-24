"""
config.py
Central configuration for StockScope.
Edit this file to change stocks, periods, model params, and alert settings.
"""

# ── Stocks ────────────────────────────────────────────────────────────────────
STOCKS = {
    "Reliance": "RELIANCE.NS",
    "TCS":      "TCS.NS",
    "Infosys":  "INFY.NS",
}

# ── Data ──────────────────────────────────────────────────────────────────────
DEFAULT_PERIOD   = "2y"       # yfinance period string
DEFAULT_INTERVAL = "1d"       # daily candles
CACHE_TTL_HOURS  = 1          # how long Streamlit caches data

# ── Technical Indicators ──────────────────────────────────────────────────────
SMA_WINDOWS  = [7, 20, 50]
EMA_WINDOWS  = [7, 20, 50]
RSI_WINDOW   = 14
BB_WINDOW    = 20
LAG_FEATURES = 5

# ── Signal Strategy ───────────────────────────────────────────────────────────
SIGNAL_SHORT_MA      = 20     # short-term SMA for crossover
SIGNAL_LONG_MA       = 50     # long-term SMA for crossover
RSI_OVERBOUGHT       = 70     # suppress BUY above this
RSI_OVERSOLD         = 30     # suppress SELL below this

# ── ML Models ─────────────────────────────────────────────────────────────────
TEST_SIZE            = 0.2
RF_N_ESTIMATORS      = 100
RF_RANDOM_STATE      = 42

# ── Backtesting ───────────────────────────────────────────────────────────────
INITIAL_CAPITAL      = 100_000   # INR
POSITION_SIZE_PCT    = 0.95      # invest 95% of available capital per trade

# ── Alerts (optional — set your email credentials in .env) ───────────────────
ALERT_EMAIL_SENDER   = ""        # e.g. your_email@gmail.com
ALERT_EMAIL_PASSWORD = ""        # use App Password for Gmail
ALERT_EMAIL_RECEIVER = ""        # recipient email
SMTP_HOST            = "smtp.gmail.com"
SMTP_PORT            = 587
