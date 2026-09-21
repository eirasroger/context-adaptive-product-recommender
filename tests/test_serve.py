"""The API says what it assumes, and refuses what it cannot answer."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from model import checkpoint as checkpoint_module  # noqa: E402
from model.recommender import ModelConfig, Recommender  # noqa: E402


@pytest.fixture(scope="module")
def client(registry, seeded, tmp_path_factory):
    from db import release as release_module
    from serve.api import create_app

    document = release_module.export(seeded)
    blob = release_module.canonical_yaml(document)

    torch.manual_seed(0)
    model = Recommender(
        ModelConfig.for_registry(registry, dim=32, encoder_blocks=1, comparator_blocks=1)
    )
    path = tmp_path_factory.mktemp("serve") / "model.pt"
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
    with TestClient(create_app(path)) as client:
        yield client


def _payload(registry, category_key, n=3):
    from ingest.generators.parametric import ideal_alternative

    category = registry.category(category_key)
    context = category.default_context_key
    base = ideal_alternative(registry, category_key, [context])
    return {
        "category": category_key,
        "context": [context],
        "stakeholders": sorted(registry.stakeholders)[:1],
        "alternatives": [
            {"id": f"a{i}", "values": dict(base.values), "levels": dict(base.levels)}
            for i in range(n)
        ],
    }


def test_health_reports_what_is_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["categories"]


def test_scoring_echoes_the_eligibility_precondition(client, registry, category_key):
    """The API must never let a score be mistaken for a compliance statement."""
    response = client.post("/score", json=_payload(registry, category_key))
    assert response.status_code == 200
    body = response.json()

    declared = registry.category(category_key).eligibility_precondition_text
    assert body["eligibility_precondition"] == declared
    assert body["functional_unit"]
    assert len(body["results"]) == 3
    assert sorted(r["rank"] for r in body["results"]) == [1, 2, 3]


def test_unavailable_context_is_refused(client, registry, category_key):
    category = registry.category(category_key)
    unavailable = [k for k in registry.contexts if k not in category.available_contexts]
    if not unavailable:
        pytest.skip("every context is available for this category")

    payload = _payload(registry, category_key)
    payload["context"] = [unavailable[0]]
    response = client.post("/score", json=payload)
    assert response.status_code == 400
    assert "not available" in response.json()["detail"]


def test_missing_values_are_reported_not_imputed(client, registry, category_key):
    payload = _payload(registry, category_key, n=2)
    dropped = sorted(payload["alternatives"][0]["values"])[0]
    payload["alternatives"][0]["values"].pop(dropped)

    body = client.post("/score", json=payload).json()
    assert dropped in body["results"][0]["missing_indicators"]
    assert dropped not in body["results"][1]["missing_indicators"]


def test_a_disqualifying_level_is_surfaced(client, registry, category_key):
    category = registry.category(category_key)
    target = None
    for key in category.token_order:
        indicator = registry.indicator(key)
        for level in indicator.levels:
            if level.is_disqualifying:
                target = (key, level.key)
                break
        if target:
            break
    if target is None:
        pytest.skip("no disqualifying level in the seeded registry")

    key, level_key = target
    payload = _payload(registry, category_key, n=2)
    payload["alternatives"][0]["levels"][key] = level_key

    body = client.post("/score", json=payload).json()
    assert f"{key}={level_key}" in body["results"][0]["disqualifying_levels"]
    assert body["notes"]


def test_indicator_listing_exposes_the_definitions(client, category_key):
    """A caller has to be able to find out what it is being asked for."""
    body = client.get(f"/categories/{category_key}/indicators").json()
    assert body
    for row in body:
        assert row["definition"].strip()
        assert row["value_type"] in ("continuous", "ordinal", "nominal", "boolean")


def test_unknown_category_is_a_404(client):
    response = client.post(
        "/score",
        json={"category": "not_a_category", "alternatives": [{"id": "a"}]},
    )
    assert response.status_code == 404
