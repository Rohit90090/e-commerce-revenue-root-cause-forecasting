from __future__ import annotations

import pandas as pd
from .database import query


def executive_kpis() -> dict:
    sessions = query("SELECT COUNT(*) AS n FROM website_sessions").iloc[0]["n"]
    orders = query("SELECT COUNT(*) AS n FROM orders").iloc[0]["n"]
    financial = query(
        """
        SELECT
            SUM(price_usd) AS gross_revenue,
            SUM(price_usd - cogs_usd) AS gross_profit,
            AVG(price_usd) AS aov
        FROM orders
        """
    ).iloc[0]
    refunds = query("SELECT SUM(refund_amount_usd) AS refunds FROM order_item_refunds").iloc[0]["refunds"] or 0

    gross_revenue = float(financial["gross_revenue"] or 0)
    return {
        "sessions": int(sessions),
        "orders": int(orders),
        "gross_revenue": gross_revenue,
        "refunds": float(refunds),
        "net_revenue": gross_revenue - float(refunds),
        "gross_profit": float(financial["gross_profit"] or 0),
        "conversion_rate": float(orders / sessions * 100) if sessions else 0,
        "aov": float(financial["aov"] or 0),
        "refund_pct_revenue": float(refunds / gross_revenue * 100) if gross_revenue else 0,
    }


def monthly_kpis() -> pd.DataFrame:
    return pd.read_csv(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "processed" / "monthly_kpis.csv",
        parse_dates=["month"],
    )


def weekly_kpis() -> pd.DataFrame:
    return pd.read_csv(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "processed" / "weekly_kpis.csv",
        parse_dates=["week"],
    )


def marketing_performance(start_date=None, end_date=None) -> pd.DataFrame:
    where = ""
    params = []
    if start_date and end_date:
        where = "WHERE date(s.created_at) BETWEEN date(?) AND date(?)"
        params = [str(start_date), str(end_date)]

    sql = f"""
    SELECT
        COALESCE(s.utm_source, 'direct/other') AS source,
        COALESCE(s.utm_campaign, 'none') AS campaign,
        s.device_type,
        COUNT(DISTINCT s.website_session_id) AS sessions,
        COUNT(DISTINCT o.order_id) AS orders,
        COALESCE(SUM(o.price_usd), 0) AS gross_revenue
    FROM website_sessions s
    LEFT JOIN orders o
        ON s.website_session_id = o.website_session_id
    {where}
    GROUP BY
        COALESCE(s.utm_source, 'direct/other'),
        COALESCE(s.utm_campaign, 'none'),
        s.device_type
    """
    df = query(sql, params)
    df["conversion_rate"] = (df["orders"] / df["sessions"] * 100).fillna(0)
    df["revenue_per_session"] = (df["gross_revenue"] / df["sessions"]).fillna(0)
    return df


def product_performance(start_date=None, end_date=None) -> pd.DataFrame:
    order_where = ""
    refund_where = ""
    order_params = []
    refund_params = []
    if start_date and end_date:
        order_where = "WHERE date(oi.created_at) BETWEEN date(?) AND date(?)"
        refund_where = "WHERE date(r.created_at) BETWEEN date(?) AND date(?)"
        order_params = [str(start_date), str(end_date)]
        refund_params = [str(start_date), str(end_date)]

    sales = query(
        f"""
        SELECT
            p.product_id,
            p.product_name,
            COUNT(*) AS units,
            SUM(oi.price_usd) AS gross_revenue,
            SUM(oi.price_usd - oi.cogs_usd) AS gross_profit
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        {order_where}
        GROUP BY p.product_id, p.product_name
        """,
        order_params,
    )

    refunds = query(
        f"""
        SELECT
            oi.product_id,
            COUNT(*) AS refunded_items,
            SUM(r.refund_amount_usd) AS refund_amount
        FROM order_item_refunds r
        JOIN order_items oi ON r.order_item_id = oi.order_item_id
        {refund_where}
        GROUP BY oi.product_id
        """,
        refund_params,
    )

    df = sales.merge(refunds, on="product_id", how="left")
    df[["refunded_items", "refund_amount"]] = df[["refunded_items", "refund_amount"]].fillna(0)
    df["net_revenue"] = df["gross_revenue"] - df["refund_amount"]
    df["refund_rate"] = (df["refunded_items"] / df["units"] * 100).fillna(0)
    return df
