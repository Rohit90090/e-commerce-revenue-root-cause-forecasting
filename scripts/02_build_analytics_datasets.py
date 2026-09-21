from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "database" / "ecommerce.db"
OUT = ROOT / "data" / "processed"


def load_table(conn, table, cols):
    return pd.read_sql_query(f"SELECT {', '.join(cols)} FROM {table}", conn)


def aggregate_period(sessions, orders, refunds, freq, period_name):
    s = sessions.copy()
    o = orders.copy()
    r = refunds.copy()

    s["created_at"] = pd.to_datetime(s["created_at"])
    o["created_at"] = pd.to_datetime(o["created_at"])
    r["created_at"] = pd.to_datetime(r["created_at"])

    s_period = s.set_index("created_at").resample(freq).agg(
        sessions=("website_session_id", "nunique")
    )

    o["gross_profit"] = o["price_usd"] - o["cogs_usd"]
    o_period = o.set_index("created_at").resample(freq).agg(
        orders=("order_id", "nunique"),
        gross_revenue=("price_usd", "sum"),
        gross_profit=("gross_profit", "sum"),
    )

    r_period = r.set_index("created_at").resample(freq).agg(
        refund_amount=("refund_amount_usd", "sum"),
        refunded_items=("order_item_refund_id", "count"),
    )

    df = s_period.join(o_period, how="outer").join(r_period, how="outer").fillna(0)
    df["net_revenue"] = df["gross_revenue"] - df["refund_amount"]
    df["conversion_rate"] = (df["orders"] / df["sessions"] * 100).fillna(0)
    df["aov"] = (df["gross_revenue"] / df["orders"]).replace([float("inf"), -float("inf")], 0).fillna(0)
    df["refund_pct_revenue"] = (df["refund_amount"] / df["gross_revenue"] * 100).replace([float("inf"), -float("inf")], 0).fillna(0)
    df = df.reset_index().rename(columns={"created_at": period_name})
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    try:
        sessions = load_table(conn, "website_sessions", ["website_session_id", "created_at"])
        orders = load_table(conn, "orders", ["order_id", "created_at", "price_usd", "cogs_usd"])
        refunds = load_table(conn, "order_item_refunds", ["order_item_refund_id", "created_at", "refund_amount_usd"])

        weekly = aggregate_period(sessions, orders, refunds, "W-SUN", "week")
        monthly = aggregate_period(sessions, orders, refunds, "MS", "month")

        # Keep only complete periods for fair time comparisons.
        min_session_date = pd.to_datetime(sessions["created_at"]).min()
        max_session_date = pd.to_datetime(sessions["created_at"]).max()

        complete_week_end = max_session_date.normalize() - pd.Timedelta(days=(max_session_date.weekday() + 1) % 7)
        weekly_complete = weekly[weekly["week"] <= complete_week_end].copy()

        first_complete_month = (min_session_date.to_period("M") + 1).start_time
        last_complete_month_exclusive = max_session_date.to_period("M").start_time
        monthly_complete = monthly[
            (monthly["month"] >= first_complete_month)
            & (monthly["month"] < last_complete_month_exclusive)
        ].copy()

        weekly_complete.to_csv(OUT / "weekly_kpis.csv", index=False)
        monthly_complete.to_csv(OUT / "monthly_kpis.csv", index=False)

        print(f"Weekly rows: {len(weekly_complete):,}")
        print(f"Monthly rows: {len(monthly_complete):,}")
        print(f"Weekly range: {weekly_complete['week'].min()} to {weekly_complete['week'].max()}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
