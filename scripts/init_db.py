"""
scripts/init_db.py
------------------
Bootstraps the database by creating all tables defined in src/database/models.py.

Run from the project root:
    python scripts/init_db.py

No Alembic migrations are used at this stage — this is intentional for the
prototype.  Alembic can be wired up later when the schema stabilises.
"""

import io
import sys
from pathlib import Path

# Force UTF-8 stdout so Unicode chars don't crash on Windows cp1252 consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# -- make sure the project root is on PYTHONPATH so 'src' is importable ------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.database.db import engine, ping_db, settings
from src.database.models import Base  # imports all ORM models


def main() -> None:
    print("=" * 60)
    print("  Pothole Detection - Database Initialiser")
    print("=" * 60)
    print(f"\n  Target: {settings.DATABASE_URL}\n")

    # 1. Connectivity check
    print("[1/3] Checking database connectivity ...")
    if not ping_db():
        print("\n  [FAIL]  Cannot reach the database.  Is Postgres running?\n")
        sys.exit(1)
    print("       [OK]  Connected\n")

    # 2. Create all tables (safe to re-run -- CREATE TABLE IF NOT EXISTS)
    print("[2/3] Running Base.metadata.create_all() ...")
    Base.metadata.create_all(bind=engine)
    print("       [OK]  Done\n")

    # 3. Report what tables now exist
    print("[3/3] Tables present in the database:")
    from sqlalchemy import inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    if not tables:
        print("       (none found -- something may have gone wrong)")
    else:
        for tbl in sorted(tables):
            cols = inspector.get_columns(tbl)
            col_names = ", ".join(c["name"] for c in cols)
            print(f"       [OK]  {tbl:<25} columns: [{col_names}]")

    print("\n  All done. Database is ready.\n")


if __name__ == "__main__":
    main()
