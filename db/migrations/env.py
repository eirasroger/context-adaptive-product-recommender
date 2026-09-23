"""Alembic environment, reading the database path the application uses."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import event

from db.models import Base
from db.session import create_db_engine, db_path

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
    """Let batch mode drop a table that others reference; checked after the migration."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = OFF")  # ignored inside a transaction, so set on connect
    cursor.close()


def run_migrations_online() -> None:
    connectable = create_db_engine()
    # Registered after the application's pragmas, so it wins.
    event.listen(connectable, "connect", _suspend_foreign_keys)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite cannot alter a constraint in place; batch mode rebuilds the table.
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()

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
