from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "database" / "ecommerce.db"


def connect():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Database not found. Run: python scripts/01_build_database.py"
        )
    return sqlite3.connect(DB_PATH)


def query(sql: str, params=None) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params or ())
