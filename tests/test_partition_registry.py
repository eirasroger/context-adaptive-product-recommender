"""The partition seed in the live registry, against the registry without it."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core import registry as registry_module
from db import validate
from db.seed import SEED_DIR, seed
from db.session import create_all, create_db_engine, session_factory

PARTITION_SEED = SEED_DIR / "60_category_partition.yaml"
PARTITION = "indoor_partition"


def _session(path: Path, seed_dir: Path = SEED_DIR):
    engine = create_db_engine(path)
    create_all(engine)
    session = session_factory(engine)()
    seed(session, seed_dir)
    session.commit()
    return session


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    return _session(tmp_path_factory.mktemp("seeded") / "registry.db")


@pytest.fixture(scope="module")
def without_partitions(tmp_path_factory):
    seeds = tmp_path_factory.mktemp("seeds")
    for path in SEED_DIR.glob("*.yaml"):
        if path != PARTITION_SEED:
            shutil.copy(path, seeds / path.name)
    return _session(tmp_path_factory.mktemp("without") / "registry.db", seeds)


@pytest.fixture(scope="module")
def registry(seeded):
    return registry_module.from_session(seeded, version="seeded")


def test_the_seeds_with_partitions_pass_every_registry_check(seeded):
    assert validate.check(seeded) == []


def test_partitions_get_their_own_baseline_and_the_seven_confirmed_contexts(seeded):
    assert validate.available_contexts(seeded, PARTITION) == {
        "partition_within_home",
        "unit_separation",
        "noisy_neighbour",
        "unheated_neighbour",
        "heavy_use",
        "limited_support",
        "hot_summer",
        "seismic",
    }


def test_adding_partitions_leaves_the_other_categories_contexts_unchanged(seeded, without_partitions):
    for category in ("concrete", "facade_system"):
        assert validate.available_contexts(seeded, category) == validate.available_contexts(without_partitions, category)


def test_u_carries_no_preference_between_heated_rooms_and_matters_next_to_a_cold_space(registry):
    assert registry.resolve_direction(PARTITION, "u_value", ["partition_within_home"])[0] == 0
    assert registry.resolve_direction(PARTITION, "u_value", ["unheated_neighbour"])[0] == -1
    assert registry.resolve_direction("facade_system", "u_value", ["new_build_mild"])[0] == -1


def test_each_partition_context_pushes_its_indicator_the_confirmed_way(registry):
    expected = {
        "unit_separation": ("acoustic_reduction_pink", 1),
        "noisy_neighbour": ("acoustic_reduction_pink", 1),
        "heavy_use": ("impact_resistance", 1),
        "limited_support": ("surface_mass", -1),
        "seismic": ("surface_mass", -1),
        "hot_summer": ("heat_capacity", 1),
    }
    for context, (indicator, direction) in expected.items():
        assert registry.resolve_direction(PARTITION, indicator, [context])[0] == direction, context


def test_the_model_builds_from_the_registry_with_partitions(registry):
    from model.recommender import ModelConfig, Recommender

    model = Recommender(ModelConfig.for_registry(registry, dim=32, encoder_blocks=1, comparator_blocks=1))
    assert "impact_resistance" in registry.category(PARTITION).token_order
    assert sum(p.numel() for p in model.parameters()) > 0
