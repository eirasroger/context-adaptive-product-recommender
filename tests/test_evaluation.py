
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "snapshot"
SHIPPED = ROOT / "serve" / "release" / "model.pt"
BASELINE = ROOT / "serve" / "release" / "metrics.json"


TOLERANCE = 1e-3

REBUILD = (
    "python -c \"from pathlib import Path; from serve.release import "
    "refresh_fixture; refresh_fixture(Path('data/snapshots/<hash>'))\""
)


@pytest.fixture(scope="module")
def scored():
    if not SHIPPED.exists() or not BASELINE.exists():
        pytest.skip("nothing has been promoted yet")

    from core.prepare import load_or_prepare
    from eval import report as report_module
    from model import checkpoint as checkpoint_module

    registry = checkpoint_module.load_registry(SHIPPED)
    model, _, _ = checkpoint_module.load(SHIPPED, device="cpu")
    prepared = load_or_prepare(registry, FIXTURE)
    return report_module.evaluate_all(
        model, registry, prepared, device="cpu", fold="test"
    )


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE.exists():
        pytest.skip("nothing has been promoted yet")
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_the_fixture_came_from_the_snapshot_the_shipped_model_was_scored_on():

    import torch

    if not SHIPPED.exists():
        pytest.skip("nothing has been promoted yet")

    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    meta = torch.load(SHIPPED, map_location="cpu", weights_only=False)["meta"]

    assert manifest.get("derived_from") == meta["snapshot_hash"], (
        f"the fixture came from snapshot {manifest.get('derived_from')!r} and the "
        f"shipped model from {meta['snapshot_hash']!r}; rebuild it: {REBUILD}"
    )
    assert manifest["filter_spec"]["folds"] == ["test"]

 
def test_the_shipped_model_reproduces_its_recorded_metrics(scored, baseline):
    for name, recorded in baseline["overall"].items():
        assert scored["overall"][name] == pytest.approx(recorded, abs=TOLERANCE), name


def test_no_stratum_drifts_from_what_was_recorded(scored, baseline):
    """An overall average hides one context or one stakeholder moving."""
    for kind, cells in baseline["stratified"].items():
        assert set(scored["stratified"][kind]) == set(cells), kind
        for label, cell in cells.items():
            for name, recorded in cell.items():
                assert scored["stratified"][kind][label][name] == pytest.approx(
                    recorded, abs=TOLERANCE
                ), f"{kind}/{label}/{name}"
