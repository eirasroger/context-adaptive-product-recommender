"""What gets shipped, and what refuses to ship."""

from __future__ import annotations

import json

import pytest
import torch

from model import checkpoint as checkpoint_module
from model.recommender import ModelConfig, Recommender
from serve import release as release_module
from serve.background import Background


def _run(tmp_path, registry, passed: bool | None):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    torch.manual_seed(0)
    model = Recommender(
        ModelConfig.for_registry(registry, dim=16, encoder_blocks=1, comparator_blocks=1)
    )
    checkpoint_module.save(
        run_dir / "model.pt",
        model,
        registry,
        registry_blob="indicator: []\n",
        meta=checkpoint_module.CheckpointMeta(
            registry_version="test",
            registry_content_hash=registry.content_hash,
            snapshot_hash="abc123",
            split_key="default",
            created_at=checkpoint_module.now(),
            metrics={},
        ),
    )
    if passed is not None:
        (run_dir / "metrics.json").write_text(
            json.dumps({
                "overall": {"gap_fidelity": 0.05},
                "behavioural": {"passed": passed, "failures": [] if passed else ["x"]},
            }),
            encoding="utf-8",
        )
    return run_dir


def test_a_failing_run_is_refused(tmp_path, registry):
    """The gate has to bite at the point where a model becomes deployable.

    Somewhere to record that a build failed is worth nothing if the failing
    build ships anyway.
    """
    run_dir = _run(tmp_path, registry, passed=False)
    with pytest.raises(release_module.GateFailed, match="behavioural gate"):
        release_module.promote(run_dir, database=None, release_dir=tmp_path / "release")
    assert not (tmp_path / "release" / "model.pt").exists()


def test_an_unevaluated_run_is_refused(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=None)
    with pytest.raises(release_module.GateFailed, match="no evaluation"):
        release_module.promote(run_dir, database=None, release_dir=tmp_path / "release")


def test_forcing_records_that_it_was_forced(tmp_path, registry):
    """Shipping a failing model stays possible and stays on the record."""
    run_dir = _run(tmp_path, registry, passed=False)
    manifest = release_module.promote(
        run_dir, database=None, release_dir=tmp_path / "release",
        force=True, notes="known bad, for a reproduction",
    )
    assert manifest["evaluation"]["behavioural_passed"] is False
    assert manifest["notes"]


def test_a_passing_run_ships_with_its_provenance(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=True)
    release_dir = tmp_path / "release"
    manifest = release_module.promote(run_dir, database=None, release_dir=release_dir)

    assert (release_dir / "model.pt").exists()
    assert manifest["snapshot_hash"] == "abc123"
    assert manifest["registry_version"] == "test"
    assert len(manifest["model_sha256"]) == 64
    assert manifest["model_sha256"] == release_module.digest(release_dir / "model.pt")

    on_disk = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_background_falls_back_when_nothing_is_attached():
    """A deployment with no database still answers, and says what it used."""
    background = Background({}, "reference ranges")
    assert not background.available
    assert background.size("anything") == 0
    assert background.rows("anything", ["gwp"]) == []


def test_background_round_trips_through_a_file(tmp_path):
    path = tmp_path / "background.json"
    path.write_text(json.dumps({
        "origin": "corpus",
        "categories": {"concrete": [{"gwp": 0.2, "health": "h4", "wdp": None}]},
    }), encoding="utf-8")

    background = Background.from_file(path)
    assert background.origin == "corpus"
    assert background.size("concrete") == 1

    rows = background.rows("concrete", ["gwp", "health"])
    assert rows == [{"gwp": 0.2, "health": "h4"}]


def test_the_released_artefacts_stay_small():
    """A checkpoint belongs in the repository. A corpus does not."""
    from serve.background import RELEASE_DIR

    if not (RELEASE_DIR / "model.pt").exists():
        pytest.skip("nothing has been promoted yet")

    total = sum(f.stat().st_size for f in RELEASE_DIR.iterdir() if f.is_file())
    assert total < release_module.SIZE_WARNING_MB * 1e6, (
        f"release is {total / 1e6:.1f} MB"
    )


def test_serving_does_not_import_the_data_stack():
    """The serving bundle has a size limit, and these are not needed to score.

    pandas, pyarrow and SQLAlchemy belong to building data and training. Letting
    them back into the import path silently adds about 175 MB to a deployment.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, serve.api;"
         "heavy = {'pandas', 'pyarrow', 'sqlalchemy', 'alembic'};"
         "found = sorted(m for m in sys.modules if m.split('.')[0] in heavy);"
         "print(','.join(found))"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"serving imports {result.stdout.strip()}"
