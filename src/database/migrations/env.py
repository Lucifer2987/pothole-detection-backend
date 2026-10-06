"""
Alembic migration environment.

Reads DATABASE_URL from the project's .env file (via src.database.db.settings)
so credentials are never hard-coded here.  Uses the 'online' migration mode
(connects to the live database to run migrations).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the project root is on the path so `src.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.database.db import settings
from src.database.models import Base  # import all models so Alembic can see them

# ---------------------------------------------------------------------------
# Alembic Config object
# ---------------------------------------------------------------------------

config = context.config

# Inject the real DATABASE_URL from our settings (overrides the placeholder in alembic.ini)
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Set up loggers from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Give Alembic the metadata so it can autogenerate migrations
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Online migration (connects to a running database)
# ---------------------------------------------------------------------------

def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,          # detect column type changes
            compare_server_default=True,  # detect server default changes
        )
        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------------------------------
# Offline migration (generates SQL without connecting)
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """Generate migration SQL without a live database connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
