"""Which registry a training run trains under."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from db import release as release_module
from db.models import RegistryRelease
from db.session import session_factory
from train.config import TrainConfig
from train.run import resolve_registry

CONFIG_DIR = Path(__file__).resolve().parents[1] / "train" / "configs"


@pytest.fixture
def released(tmp_path):
    """Its own database with two releases, the second cut after the first.

    Built from scratch rather than from the shared seeded session, because
    cutting a release and editing a row would leak into every other test.
    """
    from db.models import Stakeholder
    from db.seed import seed
    from db.session import create_all, create_db_engine

    engine = create_db_engine(tmp_path / "registry.db")
    create_all(engine)
    session = session_factory(engine)()
    seed(session)
    session.flush()

    release_module.create_release(session, "9.0.0", "", mirror_dir=tmp_path / "m0")
    # The content hash is unique, so the second release has to say something
    # different from the first.
    session.get(Stakeholder, "balanced_optimizer").display_name = "Renamed"
    session.flush()
    release_module.create_release(session, "9.0.1", "", mirror_dir=tmp_path / "m1")

    for version, when in (("9.0.0", "2020-01-01T00:00:00+00:00"),
                          ("9.0.1", "2020-01-02T00:00:00+00:00")):
        session.get(RegistryRelease, version).created_at = when
    session.commit()

    yield session
    session.close()
    engine.dispose()


def test_a_config_without_a_version_follows_the_newest_release(released):
    config = TrainConfig()
    assert config.registry_version is None

    registry, blob = resolve_registry(config, released)

    assert config.registry_version == "9.0.1", "the resolved version is written back"
    assert registry.version == "9.0.1"
    assert blob


def test_a_pinned_config_gets_the_release_it_names(released):
    """Pinning is how an old run is reproduced, so it has to keep working."""
    config = TrainConfig(registry_version="9.0.0")

    registry, _ = resolve_registry(config, released)

    assert registry.version == "9.0.0"


def test_a_pin_nobody_released_is_refused(released):
    config = TrainConfig(registry_version="does-not-exist")

    with pytest.raises(SystemExit, match="does-not-exist"):
        resolve_registry(config, released)


def test_the_shipped_configs_pin_nothing():
    """A checked-in config is for retraining, so it names no version and cannot
    fall behind the registry."""
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "registry_version" not in config, (
            f"{path.name} pins a version, which goes stale at the next release"
        )
