"""Engine and session setup.

A single SQLite file is the system of record, so the database can be deposited
alongside a dataset release and reproduced without provisioning anything.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DB_PATH = Path("data/corpus.db")

ENV_DB_PATH = "RECOMMENDER_DB"


def db_path() -> Path:
    """Where the database lives, overridable for tests and alternate corpora."""
    return Path(os.environ.get(ENV_DB_PATH, DEFAULT_DB_PATH))


def _apply_pragmas(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    # Off by default in SQLite: without this, foreign key constraints silently
    # do not exist.
    cursor.execute("PRAGMA foreign_keys = ON")
    # Concurrent reads while the inference API is serving.
    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute("PRAGMA synchronous = NORMAL")
    cursor.close()


def create_db_engine(path: Path | str | None = None, echo: bool = False) -> Engine:
    target = Path(path) if path is not None else db_path()
    if str(target) != ":memory:":
        target.parent.mkdir(parents=True, exist_ok=True)
    url = "sqlite://" if str(target) == ":memory:" else f"sqlite:///{target}"
    engine = create_engine(url, echo=echo, future=True)
    event.listen(engine, "connect", _apply_pragmas)
    return engine


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """A transactional scope that commits on success and rolls back on error."""
    session = session_factory(engine)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all(engine: Engine) -> None:
    """Create the schema directly, bypassing migrations.

    For tests and throwaway databases only.  A real database is built by
    Alembic, so that every schema change is a reviewable commit.
    """
    from db.models import Base

    Base.metadata.create_all(engine)
