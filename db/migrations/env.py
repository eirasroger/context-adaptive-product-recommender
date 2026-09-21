"""Alembic environment.

The database URL comes from the same place the application gets it, so a
migration can never be run against a different file than the one the code uses.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import event

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from db.models import Base  # noqa: E402
from db.session import create_db_engine, db_path  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=f"sqlite:///{db_path()}",
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _suspend_foreign_keys(dbapi_connection, _connection_record) -> None:
    """Turn foreign key enforcement off for the migration connection.

    Batch mode rebuilds a table by copying it and dropping the original, and any
    table holding a foreign key into it blocks that drop while enforcement is
    live. This has to happen at connect time: ``PRAGMA foreign_keys`` is
    silently ignored inside a transaction, and by the time a statement has run
    there is one open.

    Suspending enforcement is safe here and only here -- a migration is the one
    moment the schema is legitimately inconsistent with itself -- and the result
    is checked before it is kept.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = OFF")
    cursor.close()


def run_migrations_online() -> None:
    connectable = create_db_engine()
    # Registered after the application's own pragmas, so it wins.
    event.listen(connectable, "connect", _suspend_foreign_keys)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite cannot ALTER most things in place; batch mode rebuilds the
            # table instead, which is the only way a constraint change is
            # applicable at all.
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()

    # Reconnect with enforcement back on and make the database prove itself.
    verifier = create_db_engine()
    with verifier.connect() as connection:
        violations = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(
            f"migration left {len(violations)} foreign key violation(s), "
            f"first: {violations[0]}"
        )


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
