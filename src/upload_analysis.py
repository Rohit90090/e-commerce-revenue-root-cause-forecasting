from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class UploadConfig:
    date_col: str
    revenue_col: str
    sessions_col: Optional[str] = None
    orders_col: Optional[str] = None
    refunds_col: Optional[str] = None
    source_col: Optional[str] = None
    product_col: Optional[str] = None
    device_col: Optional[str] = None
    revenue_includes_refunds: bool = True


def prepare_uploaded_data(df: pd.DataFrame, cfg: UploadConfig) -> pd.DataFrame:
    data = df.copy()
    data[cfg.date_col] = pd.to_datetime(data[cfg.date_col], errors="coerce")
    data[cfg.revenue_col] = pd.to_numeric(data[cfg.revenue_col], errors="coerce")

    numeric_optional = [cfg.sessions_col, cfg.orders_col, cfg.refunds_col]
    for col in [c for c in numeric_optional if c]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    data = data.dropna(subset=[cfg.date_col, cfg.revenue_col]).copy()
    data = data.sort_values(cfg.date_col).reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid rows remain after parsing the date and revenue columns.")

    data["_gross_revenue"] = data[cfg.revenue_col].astype(float)
    if cfg.refunds_col and not cfg.revenue_includes_refunds:
        data["_net_revenue"] = data["_gross_revenue"] - data[cfg.refunds_col].astype(float)
    else:
        data["_net_revenue"] = data["_gross_revenue"]

    data["_refund_amount"] = data[cfg.refunds_col].astype(float) if cfg.refunds_col else 0.0
    return data


def _period_aggregate(data: pd.DataFrame, cfg: UploadConfig, freq: str) -> pd.DataFrame:
    frame = data.copy()
    frame["period"] = frame[cfg.date_col].dt.to_period(freq).dt.start_time

    agg = {
        "_gross_revenue": "sum",
        "_net_revenue": "sum",
        "_refund_amount": "sum",
    }
    if cfg.sessions_col:
        agg[cfg.sessions_col] = "sum"
    if cfg.orders_col:
        agg[cfg.orders_col] = "sum"

    out = frame.groupby("period", as_index=False).agg(agg)
    out = out.rename(
        columns={
            "_gross_revenue": "gross_revenue",
            "_net_revenue": "net_revenue",
            "_refund_amount": "refund_amount",
            cfg.sessions_col: "sessions" if cfg.sessions_col else cfg.sessions_col,
            cfg.orders_col: "orders" if cfg.orders_col else cfg.orders_col,
        }
    )

    if "sessions" not in out:
        out["sessions"] = np.nan
    if "orders" not in out:
        out["orders"] = np.nan

    out["conversion_rate"] = np.where(
        out["sessions"].notna() & (out["sessions"] != 0) & out["orders"].notna(),
        out["orders"] / out["sessions"] * 100,
        np.nan,
    )
    out["aov"] = np.where(
        out["orders"].notna() & (out["orders"] != 0),
        out["gross_revenue"] / out["orders"],
        np.nan,
    )
    return out.sort_values("period").reset_index(drop=True)


def weekly_kpis(data: pd.DataFrame, cfg: UploadConfig) -> pd.DataFrame:
    # W-SUN keeps complete calendar-week buckets and is easy to explain in interviews.
    frame = data.copy()
    frame["period"] = frame[cfg.date_col].dt.to_period("W-SUN").dt.start_time

    agg = {"_gross_revenue": "sum", "_net_revenue": "sum", "_refund_amount": "sum"}
    if cfg.sessions_col:
        agg[cfg.sessions_col] = "sum"
    if cfg.orders_col:
        agg[cfg.orders_col] = "sum"

    out = frame.groupby("period", as_index=False).agg(agg).rename(
        columns={
            "period": "week",
            "_gross_revenue": "gross_revenue",
            "_net_revenue": "net_revenue",
            "_refund_amount": "refund_amount",
        }
    )
    if cfg.sessions_col:
        out = out.rename(columns={cfg.sessions_col: "sessions"})
    else:
        out["sessions"] = np.nan
    if cfg.orders_col:
        out = out.rename(columns={cfg.orders_col: "orders"})
    else:
        out["orders"] = np.nan

    out["conversion_rate"] = np.where(
        out["sessions"].notna() & (out["sessions"] != 0) & out["orders"].notna(),
        out["orders"] / out["sessions"] * 100,
        np.nan,
    )
    out["aov"] = np.where(
        out["orders"].notna() & (out["orders"] != 0),
        out["gross_revenue"] / out["orders"],
        np.nan,
    )
    return out.sort_values("week").reset_index(drop=True)


def monthly_kpis(data: pd.DataFrame, cfg: UploadConfig) -> pd.DataFrame:
    out = _period_aggregate(data, cfg, "M")
    return out.rename(columns={"period": "month"})


def executive_kpis(data: pd.DataFrame, cfg: UploadConfig) -> dict:
    gross = float(data["_gross_revenue"].sum())
    net = float(data["_net_revenue"].sum())
    refunds = float(data["_refund_amount"].sum())
    sessions = float(data[cfg.sessions_col].sum()) if cfg.sessions_col else np.nan
    orders = float(data[cfg.orders_col].sum()) if cfg.orders_col else np.nan
    conversion = orders / sessions * 100 if cfg.sessions_col and cfg.orders_col and sessions else np.nan
    aov = gross / orders if cfg.orders_col and orders else np.nan
    return {
        "gross_revenue": gross,
        "net_revenue": net,
        "refund_amount": refunds,
        "sessions": sessions,
        "orders": orders,
        "conversion_rate": conversion,
        "aov": aov,
        "rows": len(data),
        "start_date": data[cfg.date_col].min(),
        "end_date": data[cfg.date_col].max(),
    }


def _feature_frame(weekly: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = weekly.copy().sort_values("week").reset_index(drop=True)
    revenue = d["net_revenue"]
    iso_week = d["week"].dt.isocalendar().week.astype(int)

    d["trend"] = np.arange(len(d))
    d["week_sin"] = np.sin(2 * np.pi * iso_week / 52)
    d["week_cos"] = np.cos(2 * np.pi * iso_week / 52)
    d["revenue_lag_1"] = revenue.shift(1)
    d["revenue_lag_2"] = revenue.shift(2)
    d["revenue_lag_4"] = revenue.shift(4)
    d["revenue_roll_4"] = revenue.shift(1).rolling(4).mean()
    d["revenue_roll_8"] = revenue.shift(1).rolling(8).mean()

    features = [
        "trend", "week_sin", "week_cos", "revenue_lag_1", "revenue_lag_2",
        "revenue_lag_4", "revenue_roll_4", "revenue_roll_8",
    ]

    optional_pairs = [
        ("sessions", "sessions_lag_1"),
        ("orders", "orders_lag_1"),
        ("conversion_rate", "conversion_lag_1"),
        ("aov", "aov_lag_1"),
        ("refund_amount", "refunds_lag_1"),
    ]
    for col, feature in optional_pairs:
        if col in d and d[col].notna().sum() >= 8:
            d[feature] = d[col].shift(1)
            features.append(feature)

    d = d.dropna(subset=features + ["net_revenue"]).reset_index(drop=True)
    return d, features


def train_forecast(weekly: pd.DataFrame) -> dict:
    feature_data, features = _feature_frame(weekly)
    if len(feature_data) < 12:
        raise ValueError(
            "Not enough weekly history for a reliable train/test forecast. "
            "Upload data covering at least about 20 weeks; 30+ weeks is recommended."
        )

    test_size = max(4, int(round(len(feature_data) * 0.2)))
    if len(feature_data) - test_size < 8:
        test_size = len(feature_data) - 8
    train = feature_data.iloc[:-test_size].copy()
    test = feature_data.iloc[-test_size:].copy()

    X_train, y_train = train[features], train["net_revenue"]
    X_test, y_test = test[features], test["net_revenue"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = np.maximum(model.predict(X_test), 0)
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred) if len(y_test) > 1 else np.nan
        nonzero = y_test != 0
        mape = float((np.abs((y_test[nonzero] - pred[nonzero]) / y_test[nonzero])).mean() * 100) if nonzero.any() else np.nan
        results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape})
        fitted[name] = (model, pred)

    metrics = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    best_name = metrics.iloc[0]["model"]
    best_model, best_pred = fitted[best_name]

    predictions = test[["week", "net_revenue"]].copy()
    predictions["predicted_revenue"] = best_pred

    # Refit the selected model on all available feature rows before forecasting the next week.
    best_model.fit(feature_data[features], feature_data["net_revenue"])
    next_week = weekly["week"].max() + pd.Timedelta(days=7)
    revenue = weekly["net_revenue"].reset_index(drop=True)
    row = {
        "trend": len(weekly),
        "week_sin": np.sin(2 * np.pi * int(next_week.isocalendar().week) / 52),
        "week_cos": np.cos(2 * np.pi * int(next_week.isocalendar().week) / 52),
        "revenue_lag_1": revenue.iloc[-1],
        "revenue_lag_2": revenue.iloc[-2],
        "revenue_lag_4": revenue.iloc[-4],
        "revenue_roll_4": revenue.iloc[-4:].mean(),
        "revenue_roll_8": revenue.iloc[-8:].mean(),
    }
    last_feature_map = {
        "sessions_lag_1": "sessions",
        "orders_lag_1": "orders",
        "conversion_lag_1": "conversion_rate",
        "aov_lag_1": "aov",
        "refunds_lag_1": "refund_amount",
    }
    for feature, col in last_feature_map.items():
        if feature in features:
            row[feature] = float(weekly[col].dropna().iloc[-1])

    prediction = max(float(best_model.predict(pd.DataFrame([row])[features])[0]), 0.0)
    best_mae = float(metrics.iloc[0]["MAE"])
    return {
        "best_model": best_name,
        "metrics": metrics,
        "predictions": predictions,
        "next_week": next_week,
        "next_prediction": prediction,
        "lower": max(prediction - best_mae, 0),
        "upper": prediction + best_mae,
    }


def detect_anomalies(weekly: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    d = weekly[["week", "net_revenue"]].copy().sort_values("week").reset_index(drop=True)
    d["baseline"] = d["net_revenue"].shift(1).rolling(8, min_periods=4).mean()
    d["rolling_std"] = d["net_revenue"].shift(1).rolling(8, min_periods=4).std()
    d["z_score"] = (d["net_revenue"] - d["baseline"]) / d["rolling_std"].replace(0, np.nan)
    d["deviation_pct"] = np.where(d["baseline"] != 0, (d["net_revenue"] - d["baseline"]) / d["baseline"] * 100, np.nan)
    d["is_anomaly"] = d["z_score"].abs() >= threshold
    d["severity"] = np.select(
        [d["z_score"].abs() >= 4, d["z_score"].abs() >= threshold],
        ["High", "Medium"],
        default="Normal",
    )
    return d


def _dimension_change(data: pd.DataFrame, cfg: UploadConfig, current_month: str, previous_month: str, col: Optional[str]) -> pd.DataFrame:
    if not col:
        return pd.DataFrame(columns=["dimension", "previous", "current", "change", "change_pct"])

    temp = data.copy()
    temp["month_key"] = temp[cfg.date_col].dt.strftime("%Y-%m")
    temp["dimension"] = temp[col].fillna("Unknown").astype(str)
    grouped = temp.groupby(["month_key", "dimension"], as_index=False)["_net_revenue"].sum()
    prev = grouped[grouped["month_key"] == previous_month][["dimension", "_net_revenue"]].rename(columns={"_net_revenue": "previous"})
    curr = grouped[grouped["month_key"] == current_month][["dimension", "_net_revenue"]].rename(columns={"_net_revenue": "current"})
    out = prev.merge(curr, on="dimension", how="outer").fillna(0)
    out["change"] = out["current"] - out["previous"]
    out["change_pct"] = np.where(out["previous"] != 0, out["change"] / out["previous"].abs() * 100, np.nan)
    return out.sort_values("change").reset_index(drop=True)


def compare_months(data: pd.DataFrame, cfg: UploadConfig, current_month: str) -> dict:
    monthly = monthly_kpis(data, cfg).copy()
    monthly["month_key"] = monthly["month"].dt.strftime("%Y-%m")
    matches = monthly.index[monthly["month_key"] == current_month].tolist()
    if not matches:
        raise ValueError("Selected month is not available.")
    idx = matches[0]
    if idx == 0:
        raise ValueError("The first month has no previous month for comparison.")

    current = monthly.loc[idx]
    previous = monthly.loc[idx - 1]
    prev_key = previous["month_key"]
    delta = float(current["net_revenue"] - previous["net_revenue"])
    pct = float(delta / previous["net_revenue"] * 100) if previous["net_revenue"] else 0.0

    decomposition = []
    if pd.notna(previous["sessions"]) and pd.notna(current["sessions"]) and pd.notna(previous["orders"]) and pd.notna(current["orders"]):
        s0, s1 = float(previous["sessions"]), float(current["sessions"])
        cr0, cr1 = float(previous["conversion_rate"] / 100), float(current["conversion_rate"] / 100)
        a0, a1 = float(previous["aov"]), float(current["aov"])
        r0, r1 = float(previous["refund_amount"]), float(current["refund_amount"])
        decomposition = [
            ("Traffic", (s1 - s0) * cr0 * a0),
            ("Conversion rate", s1 * (cr1 - cr0) * a0),
            ("Average order value", s1 * cr1 * (a1 - a0)),
        ]
        if cfg.refunds_col and not cfg.revenue_includes_refunds:
            decomposition.append(("Refunds", -(r1 - r0)))
    else:
        decomposition = [("Total revenue change", delta)]

    dec = pd.DataFrame(decomposition, columns=["driver", "impact"])
    denom = dec["impact"].abs().sum()
    dec["share_abs_pct"] = dec["impact"].abs() / denom * 100 if denom else 0

    return {
        "current_month": current_month,
        "previous_month": prev_key,
        "current": current,
        "previous": previous,
        "net_revenue_change": delta,
        "net_revenue_change_pct": pct,
        "decomposition": dec.sort_values("impact"),
        "source_drivers": _dimension_change(data, cfg, current_month, prev_key, cfg.source_col),
        "product_drivers": _dimension_change(data, cfg, current_month, prev_key, cfg.product_col),
        "device_drivers": _dimension_change(data, cfg, current_month, prev_key, cfg.device_col),
    }
