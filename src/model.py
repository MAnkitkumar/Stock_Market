"""
model.py
Trains Linear Regression and Random Forest models to predict next-day closing price.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")

FEATURE_COLS = [
    "SMA_7", "SMA_20", "SMA_50",
    "EMA_7", "EMA_20", "EMA_50",
    "RSI",
    "BB_upper", "BB_lower",
    "Daily_Return", "Volatility_7d", "Volatility_30d",
    "Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_5",
    "Rolling_Mean_7", "Rolling_Std_7", "Rolling_Mean_30",
]


def prepare_xy(df: pd.DataFrame):
    """Extract feature matrix X and target y (next-day close)."""
    df = df.copy()
    df["Target"] = df["Close"].shift(-1)  # next day close
    df.dropna(inplace=True)

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df["Target"].values
    return X, y, df.index[:-1] if len(df) > 1 else df.index


def train_models(df: pd.DataFrame, ticker: str = "stock"):
    """
    Train LR and RF models, save them, return evaluation metrics.

    Returns:
        dict with model objects and metrics
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y, _ = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # time-series: no shuffle
    )

    results = {}

    for name, model in [
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "MAE": round(mean_absolute_error(y_test, preds), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
            "R2": round(r2_score(y_test, preds), 4),
        }

        model_path = os.path.join(MODELS_DIR, f"{ticker}_{name}.pkl")
        joblib.dump(model, model_path)

        results[name] = {"model": model, "metrics": metrics, "path": model_path}
        print(f"[{name}] MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  R²={metrics['R2']}")

    return results


def predict_next_day(df: pd.DataFrame, model_name: str = "RandomForest", ticker: str = "stock") -> float:
    """
    Load saved model and predict next-day closing price using latest row.

    Returns:
        Predicted price as float
    """
    model_path = os.path.join(MODELS_DIR, f"{ticker}_{model_name}.pkl")

    if not os.path.exists(model_path):
        print(f"[model] No saved model found at {model_path}. Training now...")
        train_models(df, ticker=ticker)

    model = joblib.load(model_path)
    available = [c for c in FEATURE_COLS if c in df.columns]
    latest = df[available].iloc[-1].values.reshape(1, -1)
    return float(model.predict(latest)[0])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import fetch_stock_data
    from feature_engineering import build_features

    df = build_features(fetch_stock_data("TCS.NS"))
    results = train_models(df, ticker="TCS")
    pred = predict_next_day(df, ticker="TCS")
    print(f"\nPredicted next-day close for TCS: ₹{pred:.2f}")
