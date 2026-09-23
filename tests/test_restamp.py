from __future__ import annotations

import pytest
import torch
import yaml

from db import release as db_release
from model import checkpoint as checkpoint_module, restamp as restamp_module
from model.recommender import ModelConfig, Recommender


@pytest.fixture
def checkpoint(tmp_path, seeded, registry):
    blob = db_release.canonical_yaml(db_release.export(seeded))
    torch.manual_seed(0)
    model = Recommender(
        ModelConfig.for_registry(registry, dim=16, encoder_blocks=1, comparator_blocks=1)
    )
    path = checkpoint_module.save(
        tmp_path / "model.pt",
        model,
        registry,
        registry_blob=blob,
        meta=checkpoint_module.CheckpointMeta(
            registry_version="0.0.1",
            registry_content_hash="old",
            snapshot_hash="abc123",
            split_key="default",
            created_at=checkpoint_module.now(),
            metrics={},
        ),
    )
    return path, blob


def _edit(blob: str, table: str, column: str, value):
    document = yaml.safe_load(blob)
    document[table][0][column] = value
    return db_release.canonical_yaml(document)


def test_a_renamed_level_reaches_the_checkpoint(checkpoint):
    path, blob = checkpoint
    renamed = _edit(blob, "indicator_level", "display_name", "Something else")

    restamp_module.restamp(path, renamed, "0.0.2", "new")

    recovered = checkpoint_module.load_registry(path)
    indicator_key = yaml.safe_load(blob)["indicator_level"][0]["indicator_key"]
    assert recovered.indicator(indicator_key).levels[0].display_name == "Something else"
    assert recovered.version == "0.0.2"


def test_a_moved_normalised_position_is_refused(checkpoint):
    """Level positions are a model input."""
    path, blob = checkpoint
    moved = _edit(blob, "indicator_level", "normalised_position", 0.123)

    with pytest.raises(restamp_module.Drift, match="indicator_level"):
        restamp_module.restamp(path, moved, "0.0.2", "new")

    assert checkpoint_module.load_registry(path).version == "0.0.1"


def test_withdrawing_a_disqualifying_flag_is_prose(checkpoint):
    path, blob = checkpoint
    assert restamp_module.drift(blob, _edit(blob, "indicator_level", "is_disqualifying", 0)) == []
