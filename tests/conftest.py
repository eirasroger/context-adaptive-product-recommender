"""Shared fixtures.

Tests build a registry from the seed files into a throwaway database, so they
exercise the same path a real deployment takes rather than a hand-built object
that could drift from it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import registry as registry_module  # noqa: E402
from db.seed import seed  # noqa: E402
from db.session import create_all, create_db_engine, session_factory  # noqa: E402


@pytest.fixture(scope="session")
def engine(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "test.db"
    engine = create_db_engine(path)
    create_all(engine)
    return engine


@pytest.fixture(scope="session")
def seeded(engine):
    session = session_factory(engine)()
    seed(session)
    session.commit()
    return session


@pytest.fixture(scope="session")
def registry(seeded):
    return registry_module.from_session(seeded, version="test")


@pytest.fixture(scope="session")
def category_key(registry):
    """The first seeded category.

    Named by lookup rather than by literal, because no test should need to know
    which category happens to be seeded first.
    """
    return sorted(registry.categories)[0]
