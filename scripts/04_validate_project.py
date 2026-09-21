from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "database" / "ecommerce.db"

EXPECTED = {
    "website_sessions": 472871,
    "website_pageviews": 1188124,
    "orders": 32313,
    "order_items": 40025,
    "order_item_refunds": 1731,
    "products": 4,
}


def main():
    with sqlite3.connect(DB) as conn:
        print("TABLE COUNTS")
        for table, expected in EXPECTED.items():
            actual = pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {table}", conn).iloc[0]["n"]
            status = "OK" if int(actual) == expected else "CHECK"
            print(f"{table:<22} {int(actual):>10,}  {status}")

    monthly = pd.read_csv(ROOT / "data" / "processed" / "monthly_kpis.csv")
    weekly = pd.read_csv(ROOT / "data" / "processed" / "weekly_kpis.csv")
    metrics = pd.read_csv(ROOT / "data" / "models" / "model_metrics.csv")

    print(f"\nMonthly KPI rows: {len(monthly)}")
    print(f"Weekly KPI rows:  {len(weekly)}")
    print("\nMODEL METRICS")
    print(metrics.to_string(index=False))

    required = [
        ROOT / "data" / "models" / "best_revenue_model.pkl",
        ROOT / "data" / "models" / "model_metadata.json",
        ROOT / "data" / "processed" / "test_predictions.csv",
    ]
    for p in required:
        print(f"{p.name}: {'OK' if p.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
