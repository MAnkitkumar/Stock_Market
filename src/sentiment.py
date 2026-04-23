"""
sentiment.py
Scrapes recent news headlines for a stock and scores sentiment using TextBlob.
"""

import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import pandas as pd

# Map ticker → search query
TICKER_QUERY_MAP = {
    "RELIANCE.NS": "Reliance Industries stock news",
    "TCS.NS": "TCS Tata Consultancy stock news",
    "INFY.NS": "Infosys stock news",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_headlines(ticker: str, max_results: int = 10) -> list[str]:
    """
    Fetch recent news headlines for a stock ticker via Google News RSS.

    Args:
        ticker:      Yahoo Finance ticker symbol
        max_results: Max number of headlines to return

    Returns:
        List of headline strings
    """
    query = TICKER_QUERY_MAP.get(ticker, ticker.replace(".NS", "") + " stock news")
    query_encoded = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query_encoded}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "xml")
        items = soup.find_all("item", limit=max_results)
        headlines = [item.find("title").text for item in items if item.find("title")]
        return headlines
    except Exception as e:
        print(f"[sentiment] Could not fetch headlines: {e}")
        return []


def score_sentiment(headlines: list[str]) -> pd.DataFrame:
    """
    Score each headline using TextBlob polarity.

    Returns:
        DataFrame with columns: headline, polarity, subjectivity, label
    """
    records = []
    for h in headlines:
        blob = TextBlob(h)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        label = "Positive" if polarity > 0.05 else ("Negative" if polarity < -0.05 else "Neutral")
        records.append({
            "headline": h,
            "polarity": round(polarity, 4),
            "subjectivity": round(subjectivity, 4),
            "label": label,
        })
    return pd.DataFrame(records)


def get_sentiment_summary(ticker: str) -> dict:
    """
    Full pipeline: fetch headlines → score → summarize.

    Returns:
        dict with avg_polarity, dominant_sentiment, headlines_df
    """
    headlines = fetch_headlines(ticker)
    if not headlines:
        return {"avg_polarity": 0.0, "dominant_sentiment": "Neutral", "df": pd.DataFrame()}

    df = score_sentiment(headlines)
    avg_polarity = df["polarity"].mean()
    dominant = df["label"].value_counts().idxmax()

    return {
        "avg_polarity": round(avg_polarity, 4),
        "dominant_sentiment": dominant,
        "df": df,
    }


if __name__ == "__main__":
    result = get_sentiment_summary("TCS.NS")
    print(f"Avg Polarity: {result['avg_polarity']}")
    print(f"Dominant Sentiment: {result['dominant_sentiment']}")
    print(result["df"])
