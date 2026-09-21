from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
WEEKLY = ROOT / "data" / "processed" / "weekly_kpis.csv"
MODEL_DIR = ROOT / "data" / "models"
PROCESSED = ROOT / "data" / "processed"

FEATURES = [
    "trend",
    "week_sin",
    "week_cos",
    "revenue_lag_1",
    "revenue_lag_2",
    "revenue_lag_4",
    "revenue_roll_4",
    "revenue_roll_8",
    "sessions_lag_1",
    "orders_lag_1",
    "conversion_lag_1",
    "aov_lag_1",
    "refunds_lag_1",
]


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def make_features(df):
    data = df.copy().sort_values("week").reset_index(drop=True)
    data["week"] = pd.to_datetime(data["week"])
    data["trend"] = np.arange(len(data))
    week_num = data["week"].dt.isocalendar().week.astype(int)
    data["week_sin"] = np.sin(2 * np.pi * week_num / 52)
    data["week_cos"] = np.cos(2 * np.pi * week_num / 52)
    data["revenue_lag_1"] = data["net_revenue"].shift(1)
    data["revenue_lag_2"] = data["net_revenue"].shift(2)
    data["revenue_lag_4"] = data["net_revenue"].shift(4)
    data["revenue_roll_4"] = data["net_revenue"].shift(1).rolling(4).mean()
    data["revenue_roll_8"] = data["net_revenue"].shift(1).rolling(8).mean()
    data["sessions_lag_1"] = data["sessions"].shift(1)
    data["orders_lag_1"] = data["orders"].shift(1)
    data["conversion_lag_1"] = data["conversion_rate"].shift(1)
    data["aov_lag_1"] = data["aov"].shift(1)
    data["refunds_lag_1"] = data["refund_amount"].shift(1)
    return data.dropna(subset=FEATURES + ["net_revenue"]).copy()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(WEEKLY, parse_dates=["week"])
    data = make_features(df)

    split = int(len(data) * 0.80)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    X_train, y_train = train[FEATURES], train["net_revenue"]
    X_test, y_test = test[FEATURES], test["net_revenue"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.04, max_depth=2, random_state=42, loss="huber"
        ),
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = np.maximum(model.predict(X_test), 0)
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred),
            "MAPE": mape(y_test, pred),
        })
        fitted[name] = (model, pred)

    metrics = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    best_name = metrics.iloc[0]["model"]
    best_model, best_pred = fitted[best_name]

    # Refit the selected model on all historical rows for the next-week forecast.
    best_model.fit(data[FEATURES], data["net_revenue"])
    joblib.dump({"model": best_model, "features": FEATURES, "name": best_name}, MODEL_DIR / "best_revenue_model.pkl")

    test_out = test[["week", "net_revenue"]].copy()
    test_out["predicted_revenue"] = fitted[best_name][1]
    test_out["residual"] = test_out["net_revenue"] - test_out["predicted_revenue"]
    test_out.to_csv(PROCESSED / "test_predictions.csv", index=False)
    metrics.to_csv(MODEL_DIR / "model_metrics.csv", index=False)

    meta = {
        "best_model": best_name,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_end": str(train["week"].max().date()),
        "test_start": str(test["week"].min().date()),
        "best_model_mae": float(metrics.iloc[0]["MAE"]),
        "best_model_rmse": float(metrics.iloc[0]["RMSE"]),
        "best_model_r2": float(metrics.iloc[0]["R2"]),
        "best_model_mape": float(metrics.iloc[0]["MAPE"]),
        "features": FEATURES,
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(metrics.to_string(index=False))
    print(f"\nBest model: {best_name}")


if __name__ == "__main__":
    main()
