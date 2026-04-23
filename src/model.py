"""
model.py
Directional classification: predicts whether tomorrow's close will be UP or DOWN.

Why classification instead of raw price regression?
- Raw price prediction (regression) is trivially easy — "tomorrow ≈ today" gives R²~0.99
  but is completely useless for trading decisions.
- Predicting direction (up/down) is the actual useful signal.
- Evaluated with accuracy, precision, recall, F1 — not R².

Validation: Walk-forward (time-series split) — no data leakage.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")

# Only return-based and indicator features — NO raw price, NO rolling mean of price.
# Raw price features (SMA, EMA, BB, lag close) cause data leakage and trivial R².
FEATURE_COLS = [
    "Daily_Return",
    "Volatility_7d",
    "Volatility_30d",
    "RSI",
    # Normalised distance from MAs (price-relative, not raw price)
    "Dist_SMA20",   # (Close - SMA20) / Close
    "Dist_SMA50",   # (Close - SMA50) / Close
    "BB_width",     # (BB_upper - BB_lower) / Close  — measures squeeze
    # Lagged returns (not lagged prices)
    "Ret_Lag1", "Ret_Lag2", "Ret_Lag3", "Ret_Lag4", "Ret_Lag5",
]


def _add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that are safe from data leakage:
    - All features are derived from past data only
    - No raw price levels used directly
    """
    df = df.copy()

    if "SMA_20" in df.columns and "Close" in df.columns:
        df["Dist_SMA20"] = (df["Close"] - df["SMA_20"]) / df["Close"]
    if "SMA_50" in df.columns and "Close" in df.columns:
        df["Dist_SMA50"] = (df["Close"] - df["SMA_50"]) / df["Close"]
    if "BB_upper" in df.columns and "BB_lower" in df.columns:
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["Close"]

    # Lagged daily returns (not lagged prices)
    if "Daily_Return" in df.columns:
        for lag in range(1, 6):
            df[f"Ret_Lag{lag}"] = df["Daily_Return"].shift(lag)

    return df


def prepare_classification_xy(df: pd.DataFrame):
    """
    Build feature matrix X and binary target y.
    Target: 1 if tomorrow's close > today's close, else 0.
    Features are computed on past data only — no leakage.
    """
    df = _add_model_features(df).copy()

    # Target: next-day direction
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df["Target"].values
    return X, y, available


def train_models(df: pd.DataFrame, ticker: str = "stock") -> dict:
    """
    Train Logistic Regression and Random Forest classifiers.
    Uses TimeSeriesSplit (walk-forward) for honest out-of-sample evaluation.

    Returns:
        dict with model objects, metrics (test-only), feature names
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y, feature_names = prepare_classification_xy(df)

    # Walk-forward validation — 5 folds, respects time order
    tscv    = TimeSeriesSplit(n_splits=5)
    results = {}

    models_to_train = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        ("RandomForest",       RandomForestClassifier(n_estimators=200, max_depth=6,
                                                      random_state=42, n_jobs=-1)),
    ]

    for name, model in models_to_train:
        fold_accs = []
        scaler    = StandardScaler()

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            fold_accs.append(accuracy_score(y_test, preds))

        # Final fit on full data for production use
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)

        # Final test metrics on last 20% (held-out)
        split     = int(len(X) * 0.8)
        X_tr_s    = scaler.fit_transform(X[:split])
        X_te_s    = scaler.transform(X[split:])
        model.fit(X_tr_s, y[:split])
        final_preds = model.predict(X_te_s)

        report = classification_report(y[split:], final_preds,
                                       target_names=["DOWN", "UP"], output_dict=True)

        metrics = {
            "Walk-Forward Accuracy": f"{np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}",
            "Test Accuracy":         f"{accuracy_score(y[split:], final_preds):.3f}",
            "Precision (UP)":        f"{report['UP']['precision']:.3f}",
            "Recall (UP)":           f"{report['UP']['recall']:.3f}",
            "F1 (UP)":               f"{report['UP']['f1-score']:.3f}",
            "Note": "Scores on held-out test set only — no data leakage",
        }

        # Refit on full data for prediction
        X_full_s = scaler.fit_transform(X)
        model.fit(X_full_s, y)

        model_path  = os.path.join(MODELS_DIR, f"{ticker}_{name}.pkl")
        scaler_path = os.path.join(MODELS_DIR, f"{ticker}_{name}_scaler.pkl")
        joblib.dump(model,  model_path)
        joblib.dump(scaler, scaler_path)

        results[name] = {
            "model":         model,
            "scaler":        scaler,
            "metrics":       metrics,
            "feature_names": feature_names,
        }
        print(f"[{name}] Walk-forward acc: {metrics['Walk-Forward Accuracy']}  "
              f"Test acc: {metrics['Test Accuracy']}")

    return results


def predict_direction(df: pd.DataFrame, model_name: str = "RandomForest",
                      ticker: str = "stock") -> dict:
    """
    Predict tomorrow's price direction using the latest row.

    Returns:
        dict with direction ('UP'/'DOWN'), probability, and confidence
    """
    model_path  = os.path.join(MODELS_DIR, f"{ticker}_{model_name}.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_{model_name}_scaler.pkl")

    if not os.path.exists(model_path):
        print(f"[model] No saved model for {ticker}. Training now...")
        train_models(df, ticker=ticker)

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    df_feat   = _add_model_features(df)
    available = [c for c in FEATURE_COLS if c in df_feat.columns]
    latest    = df_feat[available].dropna().iloc[-1].values.reshape(1, -1)
    latest_s  = scaler.transform(latest)

    pred      = model.predict(latest_s)[0]
    proba     = model.predict_proba(latest_s)[0]
    direction = "UP" if pred == 1 else "DOWN"
    confidence = round(float(max(proba)) * 100, 1)

    return {
        "direction":  direction,
        "confidence": confidence,
        "prob_up":    round(float(proba[1]) * 100, 1),
        "prob_down":  round(float(proba[0]) * 100, 1),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import fetch_stock_data
    from feature_engineering import build_features

    df = build_features(fetch_stock_data("TCS.NS"))
    results = train_models(df, ticker="TCS")
    pred = predict_direction(df, ticker="TCS")
    print(f"\nTomorrow's direction: {pred['direction']} ({pred['confidence']}% confidence)")
    print(f"  P(UP)={pred['prob_up']}%  P(DOWN)={pred['prob_down']}%")
