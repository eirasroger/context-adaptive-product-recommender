from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from db import release as db_release
from eval import behavioural
from model import checkpoint as checkpoint_module
from serve import release as serve_release

RELEASE_DIR = Path(__file__).resolve().parents[1] / "serve" / "release"
MIRROR_DIR = Path(__file__).resolve().parents[1] / "registry"
SHIPPED = RELEASE_DIR / "model.pt"

RELEASE_STEPS = (
    "python -m db.seed && python -m db.release <version> && "
    "python -m model.restamp serve/release/model.pt <version>"
)


@pytest.fixture(scope="module")
def document(seeded):
    return db_release.export(seeded)


@pytest.fixture(scope="module")
def shipped():
    if not SHIPPED.exists():
        pytest.skip("nothing has been promoted yet")
    return torch.load(SHIPPED, map_location="cpu", weights_only=False)


def test_the_registry_mirror_is_an_export_of_the_seeds(document):
    """The mirror is an export, never an input.

    Editing it by hand would put a reviewed diff in front of someone that the
    database never agreed to.
    """
    mirror: dict[str, list[dict]] = {}
    for path in sorted(MIRROR_DIR.glob("*.yaml")):
        mirror.update(yaml.safe_load(path.read_text(encoding="utf-8")))

    assert set(mirror) == set(document), "the mirror is missing or has extra tables"
    for table in document:
        assert mirror[table] == document[table], (
            f"registry/{table}.yaml is stale; re-run: {RELEASE_STEPS}"
        )


def test_the_shipped_checkpoint_carries_the_seeded_registry(shipped, document):
    """Serving reads the blob inside the checkpoint, so a seed edit reaches a
    deployment only once it has been released and stamped in."""
    assert shipped["registry_blob"] == db_release.canonical_yaml(document), (
        f"the shipped checkpoint is stamped with an older registry; {RELEASE_STEPS}"
    )
    assert shipped["meta"]["registry_content_hash"] == db_release.content_hash(document)


def test_the_manifest_describes_the_checkpoint_beside_it(shipped):
    manifest = serve_release.read_manifest(RELEASE_DIR)
    assert manifest is not None, "serve/release holds a model with no manifest"
    assert manifest["registry_content_hash"] == shipped["meta"]["registry_content_hash"]
    assert manifest["registry_version"] == shipped["meta"]["registry_version"]
    assert manifest["model_sha256"] == serve_release.digest(SHIPPED)


def test_the_release_carries_the_baseline_the_next_promotion_needs(shipped):
    metrics = RELEASE_DIR / serve_release.METRICS_FILE
    assert metrics.exists(), "the release ships no metrics.json"

    recorded = json.loads(metrics.read_text(encoding="utf-8"))
    assert recorded.get("stratified"), "the baseline holds no strata to compare"
    assert recorded["behavioural"]["passed"] is True


def test_the_readme_describes_the_shipped_release(shipped):
    text = serve_release.README.read_text(encoding="utf-8")
    start = text.index(serve_release.RESULTS_START) + len(serve_release.RESULTS_START)
    end = text.index(serve_release.RESULTS_END)
    assert text[start:end].strip() == serve_release.results_section(RELEASE_DIR), (
        "the README results table is stale; run "
        "python -c \"from serve import release; release.refresh_readme()\""
    )


def test_the_shipped_checkpoint_passes_the_behavioural_suite():

    if not SHIPPED.exists():
        pytest.skip("nothing has been promoted yet")

    model, _, _ = checkpoint_module.load(SHIPPED, device="cpu")
    registry = checkpoint_module.load_registry(SHIPPED)
    suite = behavioural.run(model, registry, device="cpu")

    assert suite.assertions, "the suite made no assertions at all"
    assert suite.passed, "\n".join(str(failure) for failure in suite.failures)
    assert not suite.flat, "\n".join(str(flat) for flat in suite.flat)
