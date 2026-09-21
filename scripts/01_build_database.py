from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "database" / "ecommerce.db"

TABLE_FILES = {
    "website_sessions": "website_sessions.csv",
    "website_pageviews": "website_pageviews.csv",
    "products": "products.csv",
    "orders": "orders.csv",
    "order_items": "order_items.csv",
    "order_item_refunds": "order_item_refunds.csv",
}


def main():
    if DB.exists():
        DB.unlink()

    conn = sqlite3.connect(DB)
    try:
        conn.executescript((ROOT / "sql" / "create_tables.sql").read_text(encoding="utf-8"))

        for table, filename in TABLE_FILES.items():
            path = RAW / filename
            rows = 0
            for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
                chunk.to_sql(table, conn, if_exists="append", index=False, chunksize=5_000)
                rows += len(chunk)
            print(f"{table:<22} {rows:>10,} rows")

        conn.executescript((ROOT / "sql" / "create_indexes.sql").read_text(encoding="utf-8"))
        conn.commit()
        print(f"Database created: {DB}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
