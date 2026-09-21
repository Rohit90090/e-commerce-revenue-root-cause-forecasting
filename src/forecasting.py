from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data" / "models" / "best_revenue_model.pkl"
META_PATH = ROOT / "data" / "models" / "model_metadata.json"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "test_predictions.csv"
WEEKLY_PATH = ROOT / "data" / "processed" / "weekly_kpis.csv"


def load_model_bundle():
    return joblib.load(MODEL_PATH)


def model_metadata():
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def test_predictions():
    return pd.read_csv(PREDICTIONS_PATH, parse_dates=["week"])


def next_week_forecast():
    bundle = load_model_bundle()
    model = bundle["model"]
    features = bundle["features"]

    weekly = pd.read_csv(WEEKLY_PATH, parse_dates=["week"]).sort_values("week").reset_index(drop=True)
    last = weekly.iloc[-1]
    next_week = last["week"] + pd.Timedelta(days=7)

    revenue = weekly["net_revenue"]
    row = {
        "trend": len(weekly),
        "week_sin": np.sin(2 * np.pi * next_week.isocalendar().week / 52),
        "week_cos": np.cos(2 * np.pi * next_week.isocalendar().week / 52),
        "revenue_lag_1": revenue.iloc[-1],
        "revenue_lag_2": revenue.iloc[-2],
        "revenue_lag_4": revenue.iloc[-4],
        "revenue_roll_4": revenue.iloc[-4:].mean(),
        "revenue_roll_8": revenue.iloc[-8:].mean(),
        "sessions_lag_1": weekly["sessions"].iloc[-1],
        "orders_lag_1": weekly["orders"].iloc[-1],
        "conversion_lag_1": weekly["conversion_rate"].iloc[-1],
        "aov_lag_1": weekly["aov"].iloc[-1],
        "refunds_lag_1": weekly["refund_amount"].iloc[-1],
    }
    X = pd.DataFrame([row])[features]
    prediction = max(float(model.predict(X)[0]), 0.0)

    meta = model_metadata()
    mae = float(meta.get("best_model_mae", 0))
    return {
        "week": next_week,
        "prediction": prediction,
        "lower": max(prediction - mae, 0),
        "upper": prediction + mae,
    }
