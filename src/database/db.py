"""
Database engine and session setup for the Pothole Detection backend.

Configuration is loaded from the .env file via pydantic-settings.

Usage:
    from src.database.db import get_db, engine

    # In FastAPI route / dependency:
    def some_route(db: Session = Depends(get_db)):
        ...
"""

from collections.abc import Generator

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker


# ---------------------------------------------------------------------------
# Settings  (reads DATABASE_URL from .env automatically)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application settings sourced from the .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # ignore unrecognised env vars
    )

    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/pothole_db"


settings = Settings()


# ---------------------------------------------------------------------------
# Engine  (pool_pre_ping keeps stale connections from breaking long-lived apps)
# ---------------------------------------------------------------------------

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=False,  # set True to log every SQL statement (useful for debugging)
)


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# FastAPI dependency  (yields a DB session, always closes it on exit)
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a SQLAlchemy session.

    Example::

        @router.get("/cameras")
        def list_cameras(db: Session = Depends(get_db)):
            return db.query(Camera).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Quick connectivity test  (called by scripts/init_db.py)
# ---------------------------------------------------------------------------

def ping_db() -> bool:
    """Return True if the database is reachable, False otherwise."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[db] Connection failed: {exc}")
        return False
