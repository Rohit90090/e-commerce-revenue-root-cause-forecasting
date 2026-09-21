from __future__ import annotations

import numpy as np
import pandas as pd
from .analytics import weekly_kpis


def detect_weekly_anomalies(window: int = 8, threshold: float = 2.0) -> pd.DataFrame:
    df = weekly_kpis().copy().sort_values("week")
    df["baseline"] = df["net_revenue"].rolling(window, min_periods=4).mean().shift(1)
    df["baseline_std"] = df["net_revenue"].rolling(window, min_periods=4).std().shift(1)
    df["z_score"] = (df["net_revenue"] - df["baseline"]) / df["baseline_std"].replace(0, np.nan)
    df["deviation_pct"] = np.where(
        df["baseline"] != 0,
        (df["net_revenue"] - df["baseline"]) / df["baseline"] * 100,
        np.nan,
    )
    df["is_anomaly"] = df["z_score"].abs() >= threshold
    df["severity"] = pd.cut(
        df["z_score"].abs(),
        bins=[-np.inf, 2, 3, np.inf],
        labels=["Normal", "Medium", "High"],
    ).astype(str)
    return df
