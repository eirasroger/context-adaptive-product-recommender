"""Shared fixtures.

Tests build a registry from the seed files into a throwaway database, so they
exercise the same path a real deployment takes rather than a hand-built object
that could drift from it.
"""

from __future__ import annotations

import pytest

from core import registry as registry_module
from db.seed import seed
from db.session import create_all, create_db_engine, session_factory


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


@pytest.fixture(scope="session")
def checkpoint(registry, seeded, tmp_path_factory):
    """A small untrained model saved with the seeded registry."""
    import torch

    from db import release as release_module
    from model import checkpoint as checkpoint_module
    from model.recommender import ModelConfig, Recommender

    blob = release_module.canonical_yaml(release_module.export(seeded))
    torch.manual_seed(0)
    model = Recommender(
        ModelConfig.for_registry(registry, dim=32, encoder_blocks=1, comparator_blocks=1)
    )
    path = tmp_path_factory.mktemp("checkpoint") / "model.pt"
    checkpoint_module.save(
        path,
        model,
        registry,
        blob,
        checkpoint_module.CheckpointMeta(
            registry_version="test",
            registry_content_hash=registry.content_hash,
            snapshot_hash="deadbeef",
            split_key="default",
            created_at=checkpoint_module.now(),
            metrics={},
        ),
    )
    return path


@pytest.fixture(scope="session")
def served(checkpoint):
    """That model exported to the ONNX file the service runs."""
    from model.export import export

    return export(checkpoint, checkpoint.with_name("model.onnx"))
