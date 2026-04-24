"""
alerts.py
Sends email alerts when a Buy/Sell signal is detected.
Uses Gmail SMTP — set credentials in config.py or via environment variables.

Usage:
    python alerts.py --ticker TCS.NS
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    from config import (ALERT_EMAIL_SENDER, ALERT_EMAIL_PASSWORD,
                        ALERT_EMAIL_RECEIVER, SMTP_HOST, SMTP_PORT)
except ImportError:
    ALERT_EMAIL_SENDER   = os.getenv("ALERT_EMAIL_SENDER", "")
    ALERT_EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "")
    ALERT_EMAIL_RECEIVER = os.getenv("ALERT_EMAIL_RECEIVER", "")
    SMTP_HOST            = "smtp.gmail.com"
    SMTP_PORT            = 587


def send_alert(subject: str, body: str) -> bool:
    """
    Send an email alert via Gmail SMTP.

    Returns True on success, False on failure.
    """
    if not all([ALERT_EMAIL_SENDER, ALERT_EMAIL_PASSWORD, ALERT_EMAIL_RECEIVER]):
        print("[alerts] Email credentials not configured. Skipping alert.")
        return False

    msg = MIMEMultipart()
    msg["From"]    = ALERT_EMAIL_SENDER
    msg["To"]      = ALERT_EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(ALERT_EMAIL_SENDER, ALERT_EMAIL_PASSWORD)
            server.sendmail(ALERT_EMAIL_SENDER, ALERT_EMAIL_RECEIVER, msg.as_string())
        print(f"[alerts] Alert sent: {subject}")
        return True
    except Exception as e:
        print(f"[alerts] Failed to send alert: {e}")
        return False


def check_and_alert(ticker: str, stock_name: str) -> None:
    """
    Full pipeline: fetch data → compute signals → send alert if signal fired today.
    Designed to be run as a scheduled job (cron / Task Scheduler).
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import fetch_stock_data
    from feature_engineering import build_features
    from signals import generate_ma_signals, get_signal_summary, get_recommendation

    df  = generate_ma_signals(build_features(fetch_stock_data(ticker, period="3mo")))
    sig = get_signal_summary(df)
    rec = get_recommendation(df)

    if sig.empty:
        print(f"[alerts] No signal for {stock_name} today.")
        return

    last       = sig.iloc[-1]
    last_date  = last.name.date()
    from datetime import date
    if last_date != date.today():
        print(f"[alerts] Last signal was {last_date}, not today. No alert sent.")
        return

    action = last["Action"]
    price  = last["Close"]
    rsi    = last.get("RSI", "N/A")

    subject = f"StockScope Alert: {action} signal for {stock_name}"
    body = f"""
    <h2>📈 StockScope — {action} Signal Detected</h2>
    <table>
      <tr><td><b>Stock</b></td><td>{stock_name} ({ticker})</td></tr>
      <tr><td><b>Action</b></td><td><b style="color:{'green' if action=='BUY' else 'red'}">{action}</b></td></tr>
      <tr><td><b>Price</b></td><td>₹{price:,.2f}</td></tr>
      <tr><td><b>RSI</b></td><td>{rsi}</td></tr>
      <tr><td><b>Recommendation</b></td><td>{rec['action']} ({rec['confidence']} confidence)</td></tr>
      <tr><td><b>Reason</b></td><td>{rec['reason']}</td></tr>
    </table>
    <br><small>Sent by StockScope · Not financial advice</small>
    """
    send_alert(subject, body)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="TCS.NS", help="Yahoo Finance ticker")
    parser.add_argument("--name",   default="TCS",    help="Stock display name")
    args = parser.parse_args()
    check_and_alert(args.ticker, args.name)
