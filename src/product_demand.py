from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


REQUIRED_PRODUCT_COLUMNS = {
    "StockCode",
    "Month",
    "Quantity",
    "Revenue",
    "Transactions",
    "Customers",
    "Avg_Price",
}


def is_product_monthly_dataset(
    df: pd.DataFrame,
) -> bool:

    normalized = {
        str(c)
        .strip()
        .lower()
        .replace(
            " ",
            "_",
        ): c
        for c in df.columns
    }

    required = {
        c.lower()
        for c
        in REQUIRED_PRODUCT_COLUMNS
    }

    return required.issubset(
        set(
            normalized.keys()
        )
    )


def _safe_mode(
    series: pd.Series,
) -> str:

    s = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    if s.empty:

        return (
            "Unknown product"
        )

    mode = s.mode()

    return (
        str(
            mode.iloc[0]
        )
        if not mode.empty
        else str(
            s.iloc[-1]
        )
    )


def prepare_product_monthly(
    df: pd.DataFrame,
) -> pd.DataFrame:

    d = df.copy()

    rename_map = {}

    for col in d.columns:

        key = (
            str(col)
            .strip()
            .lower()
            .replace(
                " ",
                "_",
            )
        )

        canonical = {
            "stockcode":
                "StockCode",
            "description":
                "Description",
            "month":
                "Month",
            "quantity":
                "Quantity",
            "revenue":
                "Revenue",
            "transactions":
                "Transactions",
            "customers":
                "Customers",
            "avg_price":
                "Avg_Price",
            "min_price":
                "Min_Price",
            "max_price":
                "Max_Price",
            "revenue_per_unit":
                "Revenue_Per_Unit",
        }.get(
            key
        )

        if canonical:

            rename_map[
                col
            ] = canonical

    d = d.rename(
        columns=rename_map
    )

    missing = [
        c
        for c
        in REQUIRED_PRODUCT_COLUMNS
        if c not in d.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if "Description" not in d.columns:

        d[
            "Description"
        ] = (
            d[
                "StockCode"
            ]
            .astype(str)
        )

    d[
        "StockCode"
    ] = (
        d[
            "StockCode"
        ]
        .astype(str)
        .str.strip()
    )

    d[
        "Description"
    ] = (
        d[
            "Description"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove service and fee lines.
    service_codes = {
        "M",
        "POST",
        "DOT",
        "D",
        "C2",
        "BANK CHARGES",
        "AMAZONFEE",
        "CRUK",
    }

    service_description_terms = [
        "POSTAGE",
        "DOTCOM POSTAGE",
        "MANUAL",
        "BANK CHARGES",
        "AMAZON FEE",
        "CRUK COMMISSION",
        "CARRIAGE",
        "ADJUST BAD DEBT",
    ]

    stock_upper = (
        d[
            "StockCode"
        ]
        .str.upper()
    )

    description_upper = (
        d[
            "Description"
        ]
        .str.upper()
    )

    service_mask = (
        stock_upper
        .isin(
            service_codes
        )
    )

    for term in service_description_terms:

        service_mask = (
            service_mask
            | description_upper
            .str.contains(
                term,
                regex=False,
                na=False,
            )
        )

    d = (
        d.loc[
            ~service_mask
        ]
        .copy()
    )

    d[
        "Month"
    ] = (
        pd.to_datetime(
            d[
                "Month"
            ],
            errors="coerce",
        )
        .dt.to_period(
            "M"
        )
        .dt.to_timestamp()
    )

    for col in [
        "Quantity",
        "Revenue",
        "Transactions",
        "Customers",
        "Avg_Price",
    ]:

        d[
            col
        ] = pd.to_numeric(
            d[
                col
            ],
            errors="coerce",
        )

    d = d.dropna(
        subset=[
            "StockCode",
            "Month",
            "Quantity",
            "Revenue",
        ]
    )

    d = d[
        (
            d[
                "StockCode"
            ] != ""
        )
        & (
            d[
                "Quantity"
            ] >= 0
        )
        & (
            d[
                "Revenue"
            ] >= 0
        )
    ]

    d = d.sort_values(
        [
            "StockCode",
            "Month",
            "Description",
        ]
    )

    agg = (
        d
        .groupby(
            [
                "StockCode",
                "Month",
            ],
            as_index=False,
        )
        .agg(
            Description=(
                "Description",
                "last",
            ),
            Quantity=(
                "Quantity",
                "sum",
            ),
            Revenue=(
                "Revenue",
                "sum",
            ),
            Transactions=(
                "Transactions",
                "sum",
            ),
            Customers=(
                "Customers",
                "sum",
            ),
            Avg_Price=(
                "Avg_Price",
                "mean",
            ),
        )
    )

    agg[
        "Revenue_Per_Unit"
    ] = np.where(
        agg[
            "Quantity"
        ] > 0,
        (
            agg[
                "Revenue"
            ]
            / agg[
                "Quantity"
            ]
        ),
        agg[
            "Avg_Price"
        ],
    )

    meta = (
        agg
        .groupby(
            "StockCode",
            as_index=False,
        )
        .agg(
            First_Month=(
                "Month",
                "min",
            ),
            Description=(
                "Description",
                "last",
            ),
        )
    )

    months = pd.DataFrame(
        {
            "Month":
                pd.date_range(
                    agg[
                        "Month"
                    ].min(),
                    agg[
                        "Month"
                    ].max(),
                    freq="MS",
                )
        }
    )

    meta[
        "_key"
    ] = 1

    months[
        "_key"
    ] = 1

    full = (
        meta
        .merge(
            months,
            on="_key",
        )
        .drop(
            columns="_key"
        )
    )

    full = (
        full[
            full[
                "Month"
            ]
            >= full[
                "First_Month"
            ]
        ]
        .drop(
            columns="First_Month"
        )
    )

    full = full.merge(
        agg.drop(
            columns="Description"
        ),
        on=[
            "StockCode",
            "Month",
        ],
        how="left",
    )

    for col in [
        "Quantity",
        "Revenue",
        "Transactions",
        "Customers",
    ]:

        full[
            col
        ] = (
            full[
                col
            ]
            .fillna(
                0.0
            )
        )

    full = (
        full
        .sort_values(
            [
                "StockCode",
                "Month",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    full[
        "Avg_Price"
    ] = (
        full
        .groupby(
            "StockCode"
        )[
            "Avg_Price"
        ]
        .ffill()
        .bfill()
        .fillna(
            0.0
        )
    )

    full[
        "Revenue_Per_Unit"
    ] = (
        full
        .groupby(
            "StockCode"
        )[
            "Revenue_Per_Unit"
        ]
        .ffill()
        .bfill()
    )

    full[
        "Revenue_Per_Unit"
    ] = (
        full[
            "Revenue_Per_Unit"
        ]
        .fillna(
            full[
                "Avg_Price"
            ]
        )
        .fillna(
            0.0
        )
    )

    full[
        "Product"
    ] = (
        full[
            "StockCode"
        ]
        + " — "
        + full[
            "Description"
        ].astype(str)
    )

    full[
        "Months_History"
    ] = (
        full
        .groupby(
            "StockCode"
        )[
            "Month"
        ]
        .transform(
            "count"
        )
    )

    return full


def portfolio_monthly(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    grouped = (
        panel
        .groupby(
            "Month",
            as_index=False,
        )
        .agg(
            Revenue=(
                "Revenue",
                "sum",
            ),
            Quantity=(
                "Quantity",
                "sum",
            ),
            Transactions=(
                "Transactions",
                "sum",
            ),
        )
    )

    active = (
        panel[
            panel[
                "Quantity"
            ] > 0
        ]
        .groupby(
            "Month"
        )[
            "StockCode"
        ]
        .nunique()
        .rename(
            "Active_Products"
        )
        .reset_index()
    )

    return (
        grouped
        .merge(
            active,
            on="Month",
            how="left",
        )
        .fillna(
            {
                "Active_Products":
                    0
            }
        )
        .sort_values(
            "Month"
        )
        .reset_index(
            drop=True
        )
    )


def eligible_products(
    panel: pd.DataFrame,
    min_months: int = 8,
) -> pd.DataFrame:

    stats = (
        panel
        .groupby(
            [
                "StockCode",
                "Description",
            ],
            as_index=False,
        )
        .agg(
            Months=(
                "Month",
                "count",
            ),
            Active_Months=(
                "Quantity",
                lambda s: int(
                    (
                        s > 0
                    ).sum()
                ),
            ),
            Total_Quantity=(
                "Quantity",
                "sum",
            ),
            Total_Revenue=(
                "Revenue",
                "sum",
            ),
        )
    )

    stats = stats[
        (
            stats[
                "Months"
            ] >= min_months
        )
        & (
            stats[
                "Active_Months"
            ] >= 4
        )
    ]

    stats[
        "Product"
    ] = (
        stats[
            "StockCode"
        ]
        + " — "
        + stats[
            "Description"
        ]
    )

    return (
        stats
        .sort_values(
            [
                "Total_Revenue",
                "Total_Quantity",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def _build_feature_frame(
    panel: pd.DataFrame,
    target: str,
):

    if target not in {
        "Quantity",
        "Revenue",
    }:

        raise ValueError(
            "Target must be Quantity or Revenue."
        )

    d = (
        panel
        .copy()
        .sort_values(
            [
                "StockCode",
                "Month",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    g = d.groupby(
        "StockCode",
        group_keys=False,
    )

    d[
        "Units_Per_Transaction"
    ] = np.where(
        d[
            "Transactions"
        ] > 0,
        (
            d[
                "Quantity"
            ]
            / d[
                "Transactions"
            ]
        ),
        0.0,
    )

    d[
        "Transactions_Per_Customer"
    ] = np.where(
        d[
            "Customers"
        ] > 0,
        (
            d[
                "Transactions"
            ]
            / d[
                "Customers"
            ]
        ),
        0.0,
    )

    d[
        "Revenue_Per_Customer"
    ] = np.where(
        d[
            "Customers"
        ] > 0,
        (
            d[
                "Revenue"
            ]
            / d[
                "Customers"
            ]
        ),
        0.0,
    )

    for base, prefix in [
        (
            "Quantity",
            "qty",
        ),
        (
            "Revenue",
            "rev",
        ),
        (
            "Transactions",
            "txn",
        ),
        (
            "Customers",
            "cust",
        ),
        (
            "Avg_Price",
            "price",
        ),
    ]:

        d[
            f"{prefix}_lag_1"
        ] = (
            g[
                base
            ]
            .shift(
                1
            )
        )

        d[
            f"{prefix}_lag_2"
        ] = (
            g[
                base
            ]
            .shift(
                2
            )
        )

        if base in {
            "Quantity",
            "Revenue",
        }:

            d[
                f"{prefix}_lag_3"
            ] = (
                g[
                    base
                ]
                .shift(
                    3
                )
            )

            d[
                f"{prefix}_lag_12"
            ] = (
                g[
                    base
                ]
                .shift(
                    12
                )
            )

            d[
                f"{prefix}_roll_3"
            ] = (
                g[
                    base
                ]
                .transform(
                    lambda s:
                    s.rolling(
                        3,
                        min_periods=1,
                    )
                    .mean()
                )
            )

            d[
                f"{prefix}_roll_6"
            ] = (
                g[
                    base
                ]
                .transform(
                    lambda s:
                    s.rolling(
                        6,
                        min_periods=2,
                    )
                    .mean()
                )
            )

            d[
                f"{prefix}_roll_12"
            ] = (
                g[
                    base
                ]
                .transform(
                    lambda s:
                    s.rolling(
                        12,
                        min_periods=3,
                    )
                    .mean()
                )
            )

    d[
        "price_change_pct"
    ] = (
        g[
            "Avg_Price"
        ]
        .pct_change()
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .clip(
            -1,
            5,
        )
    )

    d[
        "target_month"
    ] = (
        d[
            "Month"
        ]
        + pd.offsets.MonthBegin(
            1
        )
    )

    target_month_num = (
        d[
            "target_month"
        ]
        .dt.month
    )

    d[
        "month_sin"
    ] = np.sin(
        2
        * np.pi
        * target_month_num
        / 12
    )

    d[
        "month_cos"
    ] = np.cos(
        2
        * np.pi
        * target_month_num
        / 12
    )

    d[
        "product_age"
    ] = (
        g.cumcount()
        + 1
    )

    d[
        "Target"
    ] = (
        g[
            target
        ]
        .shift(
            -1
        )
    )

    raw_features = [
        "Quantity",
        "Revenue",
        "Transactions",
        "Customers",
        "Avg_Price",
        "Revenue_Per_Unit",
        "Units_Per_Transaction",
        "Transactions_Per_Customer",
        "Revenue_Per_Customer",
        "qty_lag_1",
        "qty_lag_2",
        "qty_lag_3",
        "qty_lag_12",
        "qty_roll_3",
        "qty_roll_6",
        "qty_roll_12",
        "rev_lag_1",
        "rev_lag_2",
        "rev_lag_3",
        "rev_lag_12",
        "rev_roll_3",
        "rev_roll_6",
        "rev_roll_12",
        "txn_lag_1",
        "txn_lag_2",
        "cust_lag_1",
        "cust_lag_2",
        "price_lag_1",
        "price_lag_2",
        "price_change_pct",
        "month_sin",
        "month_cos",
        "product_age",
    ]

    log_features = []

    keep_raw = {
        "price_change_pct",
        "month_sin",
        "month_cos",
        "product_age",
    }

    for col in raw_features:

        if col in keep_raw:
            continue

        new_col = (
            f"log_{col}"
        )

        d[
            new_col
        ] = np.log1p(
            d[
                col
            ]
            .clip(
                lower=0
            )
        )

        log_features.append(
            new_col
        )

    features = (
        log_features
        + [
            "price_change_pct",
            "month_sin",
            "month_cos",
            "product_age",
        ]
    )

    d[
        features
    ] = (
        d[
            features
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
    )

    return (
        d,
        features,
        target,
    )


def _metrics(
    y_true,
    pred,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    pred = np.asarray(
        pred,
        dtype=float,
    )

    mae = mean_absolute_error(
        y_true,
        pred,
    )

    rmse = (
        mean_squared_error(
            y_true,
            pred,
        )
        ** 0.5
    )

    r2 = (
        r2_score(
            y_true,
            pred,
        )
        if len(
            y_true
        ) > 1
        else np.nan
    )

    nz = (
        y_true > 0
    )

    mape = (
        float(
            np.mean(
                np.abs(
                    (
                        y_true[
                            nz
                        ]
                        - pred[
                            nz
                        ]
                    )
                    / y_true[
                        nz
                    ]
                )
            )
            * 100
        )
        if nz.any()
        else np.nan
    )

    denom = (
        np.abs(
            y_true
        )
        .sum()
    )

    wmape = (
        float(
            np.abs(
                y_true
                - pred
            )
            .sum()
            / denom
            * 100
        )
        if denom > 0
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "WMAPE": wmape,
    }


def train_product_forecast(
    panel: pd.DataFrame,
    target: str = "Quantity",
) -> dict:

    (
        d,
        features,
        target_name,
    ) = _build_feature_frame(
        panel,
        target,
    )

    model_data = (
        d[
            d[
                "Target"
            ].notna()
        ]
        .copy()
    )

    months = sorted(
        model_data[
            "target_month"
        ]
        .dropna()
        .unique()
    )

    if len(
        months
    ) < 10:

        raise ValueError(
            "At least 10 monthly periods "
            "are required for product forecasting."
        )

    test_month_count = max(
        4,
        int(
            round(
                len(
                    months
                )
                * 0.2
            )
        ),
    )

    test_months = set(
        months[
            -test_month_count:
        ]
    )

    train = (
        model_data[
            ~model_data[
                "target_month"
            ].isin(
                test_months
            )
        ]
        .copy()
    )

    test = (
        model_data[
            model_data[
                "target_month"
            ].isin(
                test_months
            )
        ]
        .copy()
    )

    if (
        len(
            train
        ) < 200
        or len(
            test
        ) < 50
    ):

        raise ValueError(
            "Not enough product-month history "
            "after feature engineering."
        )

    X_train = train[
        features
    ]

    X_test = test[
        features
    ]

    y_train_log = np.log1p(
        train[
            "Target"
        ]
        .clip(
            lower=0
        )
    )

    y_test = (
        test[
            "Target"
        ]
        .clip(
            lower=0
        )
        .to_numpy()
    )

    baseline_pred = (
        test[
            target
        ]
        .clip(
            lower=0
        )
        .to_numpy()
    )

    rows = [
        {
            "Model":
                "Last-Month Baseline",
            **_metrics(
                y_test,
                baseline_pred,
            ),
            "Type":
                "Baseline",
        }
    ]

    models = {
        "Linear Regression":
            LinearRegression(),

        "Histogram Gradient Boosting":
            HistGradientBoostingRegressor(
                learning_rate=0.07,
                max_iter=140,
                max_leaf_nodes=31,
                min_samples_leaf=25,
                l2_regularization=0.2,
                random_state=42,
            ),
    }

    fitted = {}

    predictions = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train_log,
        )

        pred = np.expm1(
            model.predict(
                X_test
            )
        )

        pred = np.maximum(
            pred,
            0.0,
        )

        rows.append(
            {
                "Model": name,
                **_metrics(
                    y_test,
                    pred,
                ),
                "Type": "ML",
            }
        )

        fitted[
            name
        ] = model

        predictions[
            name
        ] = pred

    metrics = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "WMAPE",
                "MAE",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    ml_metrics = (
        metrics[
            metrics[
                "Type"
            ] == "ML"
        ]
        .sort_values(
            [
                "WMAPE",
                "MAE",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    best_name = str(
        ml_metrics
        .iloc[0][
            "Model"
        ]
    )

    best_model = fitted[
        best_name
    ]

    best_pred = predictions[
        best_name
    ]

    pred_df = (
        test[
            [
                "StockCode",
                "Description",
                "Month",
                "target_month",
                "Target",
            ]
        ]
        .copy()
    )

    pred_df[
        "Predicted"
    ] = best_pred

    pred_df[
        "Baseline"
    ] = baseline_pred

    best_model.fit(
        model_data[
            features
        ],
        np.log1p(
            model_data[
                "Target"
            ]
            .clip(
                lower=0
            )
        ),
    )

    latest_month = d[
        "Month"
    ].max()

    latest = (
        d[
            d[
                "Month"
            ]
            == latest_month
        ]
        .copy()
    )

    next_pred = np.expm1(
        best_model.predict(
            latest[
                features
            ]
        )
    )

    latest[
        "Forecast"
    ] = np.maximum(
        next_pred,
        0.0,
    )

    latest[
        "Forecast_Month"
    ] = (
        latest_month
        + pd.offsets.MonthBegin(
            1
        )
    )

    latest[
        "Current_Value"
    ] = latest[
        target
    ]

    latest[
        "Forecast_Change"
    ] = (
        latest[
            "Forecast"
        ]
        - latest[
            "Current_Value"
        ]
    )

    latest[
        "Forecast_Change_Pct"
    ] = np.where(
        latest[
            "Current_Value"
        ] > 0,
        (
            latest[
                "Forecast_Change"
            ]
            / latest[
                "Current_Value"
            ]
            * 100
        ),
        np.nan,
    )

    return {
        "target":
            target_name,

        "metrics":
            metrics,

        "best_model":
            best_name,

        "predictions":
            pred_df,

        "portfolio_forecasts":
            latest[
                [
                    "StockCode",
                    "Description",
                    "Product",
                    "Forecast_Month",
                    "Current_Value",
                    "Forecast",
                    "Forecast_Change",
                    "Forecast_Change_Pct",
                    "Avg_Price",
                    "Revenue_Per_Unit",
                ]
            ]
            .copy(),

        "test_months":
            [
                pd.Timestamp(
                    m
                )
                for m
                in months[
                    -test_month_count:
                ]
            ],

        "latest_month":
            latest_month,

        "next_month":
            (
                latest_month
                + pd.offsets.MonthBegin(
                    1
                )
            ),
    }


def product_history(
    panel: pd.DataFrame,
    stock_code: str,
) -> pd.DataFrame:

    return (
        panel[
            panel[
                "StockCode"
            ].astype(str)
            == str(
                stock_code
            )
        ]
        .copy()
        .sort_values(
            "Month"
        )
        .reset_index(
            drop=True
        )
    )


def product_anomalies(
    panel: pd.DataFrame,
    stock_code: str,
    target: str = "Quantity",
) -> pd.DataFrame:

    s = (
        product_history(
            panel,
            stock_code,
        )[
            [
                "Month",
                target,
            ]
        ]
        .copy()
    )

    s[
        "Baseline"
    ] = (
        s[
            target
        ]
        .shift(
            1
        )
        .rolling(
            6,
            min_periods=3,
        )
        .mean()
    )

    s[
        "Rolling_Std"
    ] = (
        s[
            target
        ]
        .shift(
            1
        )
        .rolling(
            6,
            min_periods=3,
        )
        .std()
    )

    s[
        "Z_Score"
    ] = (
        (
            s[
                target
            ]
            - s[
                "Baseline"
            ]
        )
        / s[
            "Rolling_Std"
        ]
        .replace(
            0,
            np.nan,
        )
    )

    s[
        "Deviation_Pct"
    ] = np.where(
        s[
            "Baseline"
        ] > 0,
        (
            (
                s[
                    target
                ]
                - s[
                    "Baseline"
                ]
            )
            / s[
                "Baseline"
            ]
            * 100
        ),
        np.nan,
    )

    s[
        "Is_Anomaly"
    ] = (
        (
            s[
                "Z_Score"
            ]
            .abs()
            >= 2.0
        )
        | (
            s[
                "Deviation_Pct"
            ]
            .abs()
            >= 60
        )
    )

    s[
        "Severity"
    ] = np.select(
        [
            (
                s[
                    "Z_Score"
                ]
                .abs()
                >= 3.5
            ),
            s[
                "Is_Anomaly"
            ],
        ],
        [
            "High",
            "Medium",
        ],
        default="Normal",
    )

    return s


def product_root_cause(
    panel: pd.DataFrame,
    stock_code: str,
    current_month: Optional[
        pd.Timestamp
    ] = None,
) -> dict:

    s = product_history(
        panel,
        stock_code,
    )

    if len(
        s
    ) < 2:

        raise ValueError(
            "At least two monthly observations are required."
        )

    if current_month is None:

        candidates = []

        for i in range(
            1,
            len(
                s
            ),
        ):

            if (
                s.loc[
                    i,
                    "Quantity",
                ] > 0
                or s.loc[
                    i - 1,
                    "Quantity",
                ] > 0
            ):

                candidates.append(
                    i
                )

        idx = (
            candidates[-1]
            if candidates
            else len(
                s
            ) - 1
        )

    else:

        current_month = (
            pd.Timestamp(
                current_month
            )
            .to_period(
                "M"
            )
            .to_timestamp()
        )

        matches = (
            s.index[
                s[
                    "Month"
                ]
                == current_month
            ]
            .tolist()
        )

        if (
            not matches
            or matches[0]
            == 0
        ):

            raise ValueError(
                "Selected month does not have "
                "a prior comparison month."
            )

        idx = matches[0]

    prev = (
        s.loc[
            idx - 1
        ]
        .copy()
    )

    curr = (
        s.loc[
            idx
        ]
        .copy()
    )

    t0 = float(
        prev[
            "Transactions"
        ]
    )

    t1 = float(
        curr[
            "Transactions"
        ]
    )

    q0 = float(
        prev[
            "Quantity"
        ]
    )

    q1 = float(
        curr[
            "Quantity"
        ]
    )

    r0 = float(
        prev[
            "Revenue"
        ]
    )

    r1 = float(
        curr[
            "Revenue"
        ]
    )

    c0 = float(
        prev[
            "Customers"
        ]
    )

    c1 = float(
        curr[
            "Customers"
        ]
    )

    upt0 = (
        q0 / t0
        if t0 > 0
        else 0.0
    )

    upt1 = (
        q1 / t1
        if t1 > 0
        else 0.0
    )

    rpu0 = (
        r0 / q0
        if q0 > 0
        else (
            float(
                prev[
                    "Revenue_Per_Unit"
                ]
            )
            if pd.notna(
                prev[
                    "Revenue_Per_Unit"
                ]
            )
            else 0.0
        )
    )

    rpu1 = (
        r1 / q1
        if q1 > 0
        else (
            float(
                curr[
                    "Revenue_Per_Unit"
                ]
            )
            if pd.notna(
                curr[
                    "Revenue_Per_Unit"
                ]
            )
            else 0.0
        )
    )

    demand_txn_impact = (
        (
            t1 - t0
        )
        * upt0
    )

    demand_basket_impact = (
        t1
        * (
            upt1
            - upt0
        )
    )

    revenue_txn_impact = (
        (
            t1 - t0
        )
        * upt0
        * rpu0
    )

    revenue_basket_impact = (
        t1
        * (
            upt1
            - upt0
        )
        * rpu0
    )

    revenue_unit_value_impact = (
        t1
        * upt1
        * (
            rpu1
            - rpu0
        )
    )

    demand_dec = pd.DataFrame(
        [
            {
                "Driver":
                    "Transaction volume",
                "Impact":
                    demand_txn_impact,
            },
            {
                "Driver":
                    "Units per transaction",
                "Impact":
                    demand_basket_impact,
            },
        ]
    )

    revenue_dec = pd.DataFrame(
        [
            {
                "Driver":
                    "Transaction volume",
                "Impact":
                    revenue_txn_impact,
            },
            {
                "Driver":
                    "Units per transaction",
                "Impact":
                    revenue_basket_impact,
            },
            {
                "Driver":
                    "Revenue per unit",
                "Impact":
                    revenue_unit_value_impact,
            },
        ]
    )

    return {
        "previous":
            prev,

        "current":
            curr,

        "quantity_change":
            (
                q1
                - q0
            ),

        "quantity_change_pct":
            (
                (
                    q1 - q0
                )
                / q0
                * 100
                if q0 > 0
                else np.nan
            ),

        "revenue_change":
            (
                r1
                - r0
            ),

        "revenue_change_pct":
            (
                (
                    r1 - r0
                )
                / r0
                * 100
                if r0 > 0
                else np.nan
            ),

        "customer_change":
            (
                c1
                - c0
            ),

        "transaction_change":
            (
                t1
                - t0
            ),

        "avg_price_change":
            float(
                curr[
                    "Avg_Price"
                ]
                - prev[
                    "Avg_Price"
                ]
            ),

        "demand_decomposition":
            demand_dec,

        "revenue_decomposition":
            revenue_dec,

        "units_per_transaction_previous":
            upt0,

        "units_per_transaction_current":
            upt1,

        "transactions_per_customer_previous":
            (
                t0 / c0
                if c0 > 0
                else np.nan
            ),

        "transactions_per_customer_current":
            (
                t1 / c1
                if c1 > 0
                else np.nan
            ),
    }


def local_product_summary(
    result: dict,
    forecast_row: Optional[
        pd.Series
    ] = None,
    target: str = "Quantity",
) -> str:

    curr = result[
        "current"
    ]

    qchg = result[
        "quantity_change"
    ]

    rchg = result[
        "revenue_change"
    ]

    parts = [
        (
            f"For {pd.Timestamp(curr['Month']).strftime('%b %Y')}, "
            f"demand "
            f"{'increased' if qchg >= 0 else 'decreased'} "
            f"by {abs(qchg):,.0f} units "
            f"versus the previous month."
        ),
        (
            f"Revenue "
            f"{'increased' if rchg >= 0 else 'decreased'} "
            f"by ${abs(rchg):,.0f}."
        ),
    ]

    dec = (
        result[
            "revenue_decomposition"
        ]
        .copy()
    )

    if not dec.empty:

        neg = (
            dec
            .sort_values(
                "Impact"
            )
            .iloc[0]
        )

        pos = (
            dec
            .sort_values(
                "Impact",
                ascending=False,
            )
            .iloc[0]
        )

        if float(
            neg[
                "Impact"
            ]
        ) < 0:

            parts.append(
                "The largest negative measured revenue driver was "
                f"{str(neg['Driver']).lower()} "
                f"(${float(neg['Impact']):,.0f})."
            )

        if float(
            pos[
                "Impact"
            ]
        ) > 0:

            parts.append(
                "The strongest positive offset was "
                f"{str(pos['Driver']).lower()} "
                f"(${float(pos['Impact']):,.0f})."
            )

    if forecast_row is not None:

        next_month = (
            pd.Timestamp(
                forecast_row[
                    "Forecast_Month"
                ]
            )
            .strftime(
                "%b %Y"
            )
        )

        if target == "Quantity":

            parts.append(
                "The selected model forecasts approximately "
                f"{float(forecast_row['Forecast']):,.0f} units "
                f"for {next_month}."
            )

        else:

            parts.append(
                "The selected model forecasts approximately "
                f"${float(forecast_row['Forecast']):,.0f} revenue "
                f"for {next_month}."
            )

    return " ".join(
        parts
    )


def product_ai_summary(
    result: dict,
    forecast_row: Optional[
        pd.Series
    ] = None,
    target: str = "Quantity",
):

    api_key = (
        os.getenv(
            "OPENAI_API_KEY",
            "",
        )
        .strip()
    )

    local = (
        local_product_summary(
            result,
            forecast_row,
            target,
        )
    )

    if not api_key:

        return (
            local,
            "Local diagnostic summary",
        )

    payload = {
        "previous_month":
            str(
                pd.Timestamp(
                    result[
                        "previous"
                    ][
                        "Month"
                    ]
                )
                .date()
            ),

        "current_month":
            str(
                pd.Timestamp(
                    result[
                        "current"
                    ][
                        "Month"
                    ]
                )
                .date()
            ),

        "quantity_change":
            result[
                "quantity_change"
            ],

        "quantity_change_pct":
            result[
                "quantity_change_pct"
            ],

        "revenue_change":
            result[
                "revenue_change"
            ],

        "revenue_change_pct":
            result[
                "revenue_change_pct"
            ],

        "customer_change":
            result[
                "customer_change"
            ],

        "transaction_change":
            result[
                "transaction_change"
            ],

        "avg_price_change":
            result[
                "avg_price_change"
            ],

        "demand_decomposition":
            result[
                "demand_decomposition"
            ]
            .to_dict(
                orient="records"
            ),

        "revenue_decomposition":
            result[
                "revenue_decomposition"
            ]
            .to_dict(
                orient="records"
            ),

        "forecast_target":
            target,

        "forecast":
            (
                None
                if forecast_row is None
                else {
                    "month":
                        str(
                            pd.Timestamp(
                                forecast_row[
                                    "Forecast_Month"
                                ]
                            )
                            .date()
                        ),
                    "value":
                        float(
                            forecast_row[
                                "Forecast"
                            ]
                        ),
                }
            ),
    }

    try:

        from openai import OpenAI

        client = OpenAI(
            api_key=api_key
        )

        model = os.getenv(
            "OPENAI_MODEL",
            "gpt-5.6-luna",
        )

        response = (
            client.responses.create(
                model=model,
                input=[
                    {
                        "role":
                            "system",

                        "content":
                            (
                                "You are a business analyst. "
                                "Use only the supplied calculated evidence. "
                                "Do not invent causes. "
                                "Write one short paragraph followed by "
                                "three concise bullets: "
                                "main demand driver, main revenue driver, "
                                "and one practical investigation action."
                            ),
                    },
                    {
                        "role":
                            "user",

                        "content":
                            json.dumps(
                                payload,
                                default=str,
                            ),
                    },
                ],
            )
        )

        return (
            response
            .output_text
            .strip(),
            f"OpenAI ({model})",
        )

    except Exception:

        return (
            local,
            "Local diagnostic summary (AI unavailable)",
        )