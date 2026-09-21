from __future__ import annotations

import pandas as pd
import numpy as np
from .analytics import monthly_kpis
from .database import query


def _month_bounds(month_str: str):
    start = pd.Timestamp(month_str + "-01")
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def funnel_decomposition(previous: pd.Series, current: pd.Series) -> pd.DataFrame:
    s0, s1 = float(previous["sessions"]), float(current["sessions"])
    cr0 = float(previous["conversion_rate"] / 100)
    cr1 = float(current["conversion_rate"] / 100)
    aov0, aov1 = float(previous["aov"]), float(current["aov"])
    r0, r1 = float(previous["refund_amount"]), float(current["refund_amount"])

    impacts = [
        ("Traffic", (s1 - s0) * cr0 * aov0),
        ("Conversion rate", s1 * (cr1 - cr0) * aov0),
        ("Average order value", s1 * cr1 * (aov1 - aov0)),
        ("Refunds", -(r1 - r0)),
    ]
    out = pd.DataFrame(impacts, columns=["driver", "impact"])
    total = out["impact"].sum()
    out["share_abs_pct"] = np.where(
        out["impact"].abs().sum() > 0,
        out["impact"].abs() / out["impact"].abs().sum() * 100,
        0,
    )
    return out.sort_values("impact")


def compare_months(current_month: str) -> dict:
    monthly = monthly_kpis().copy()
    monthly["month_key"] = monthly["month"].dt.strftime("%Y-%m")
    if current_month not in set(monthly["month_key"]):
        raise ValueError("Selected month is not available.")

    current_idx = monthly.index[monthly["month_key"] == current_month][0]
    if current_idx == 0:
        raise ValueError("The first month has no previous month for comparison.")

    current = monthly.loc[current_idx]
    previous = monthly.loc[current_idx - 1]
    previous_month = previous["month_key"]

    delta = float(current["net_revenue"] - previous["net_revenue"])
    pct = float(delta / previous["net_revenue"] * 100) if previous["net_revenue"] else 0

    return {
        "current_month": current_month,
        "previous_month": previous_month,
        "current": current,
        "previous": previous,
        "net_revenue_change": delta,
        "net_revenue_change_pct": pct,
        "decomposition": funnel_decomposition(previous, current),
        "source_drivers": dimension_drivers(current_month, previous_month, "source"),
        "campaign_drivers": dimension_drivers(current_month, previous_month, "campaign"),
        "device_drivers": dimension_drivers(current_month, previous_month, "device"),
        "product_drivers": product_drivers(current_month, previous_month),
    }


def dimension_drivers(current_month: str, previous_month: str, dimension: str) -> pd.DataFrame:
    colmap = {
        "source": "COALESCE(s.utm_source, 'direct/other')",
        "campaign": "COALESCE(s.utm_campaign, 'none')",
        "device": "COALESCE(s.device_type, 'unknown')",
    }
    expr = colmap[dimension]

    def period_frame(month):
        start, end = _month_bounds(month)
        sales = query(
            f"""
            SELECT
                {expr} AS dimension,
                COALESCE(SUM(o.price_usd), 0) AS gross_revenue
            FROM orders o
            JOIN website_sessions s ON o.website_session_id = s.website_session_id
            WHERE datetime(o.created_at) >= datetime(?)
              AND datetime(o.created_at) < datetime(?)
            GROUP BY {expr}
            """,
            [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
        )
        refunds = query(
            f"""
            SELECT
                {expr} AS dimension,
                COALESCE(SUM(r.refund_amount_usd), 0) AS refunds
            FROM order_item_refunds r
            JOIN orders o ON r.order_id = o.order_id
            JOIN website_sessions s ON o.website_session_id = s.website_session_id
            WHERE datetime(r.created_at) >= datetime(?)
              AND datetime(r.created_at) < datetime(?)
            GROUP BY {expr}
            """,
            [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
        )
        d = sales.merge(refunds, on="dimension", how="outer").fillna(0)
        d["net_revenue"] = d["gross_revenue"] - d["refunds"]
        return d[["dimension", "net_revenue"]]

    prev = period_frame(previous_month).rename(columns={"net_revenue":"previous"})
    curr = period_frame(current_month).rename(columns={"net_revenue":"current"})
    out = prev.merge(curr, on="dimension", how="outer").fillna(0)
    out["change"] = out["current"] - out["previous"]
    out["change_pct"] = np.where(out["previous"] != 0, out["change"] / out["previous"].abs() * 100, np.nan)
    return out.sort_values("change")


def product_drivers(current_month: str, previous_month: str) -> pd.DataFrame:
    def period_frame(month):
        start, end = _month_bounds(month)
        sales = query(
            """
            SELECT
                p.product_name AS dimension,
                SUM(oi.price_usd) AS gross_revenue
            FROM order_items oi
            JOIN products p ON oi.product_id = p.product_id
            WHERE datetime(oi.created_at) >= datetime(?)
              AND datetime(oi.created_at) < datetime(?)
            GROUP BY p.product_name
            """,
            [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
        )
        refunds = query(
            """
            SELECT
                p.product_name AS dimension,
                SUM(r.refund_amount_usd) AS refunds
            FROM order_item_refunds r
            JOIN order_items oi ON r.order_item_id = oi.order_item_id
            JOIN products p ON oi.product_id = p.product_id
            WHERE datetime(r.created_at) >= datetime(?)
              AND datetime(r.created_at) < datetime(?)
            GROUP BY p.product_name
            """,
            [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
        )
        d = sales.merge(refunds, on="dimension", how="outer").fillna(0)
        d["net_revenue"] = d["gross_revenue"] - d["refunds"]
        return d[["dimension", "net_revenue"]]

    prev = period_frame(previous_month).rename(columns={"net_revenue":"previous"})
    curr = period_frame(current_month).rename(columns={"net_revenue":"current"})
    out = prev.merge(curr, on="dimension", how="outer").fillna(0)
    out["change"] = out["current"] - out["previous"]
    out["change_pct"] = np.where(out["previous"] != 0, out["change"] / out["previous"].abs() * 100, np.nan)
    return out.sort_values("change")
