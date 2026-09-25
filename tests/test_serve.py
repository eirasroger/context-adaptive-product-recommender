from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def frontend(tmp_path_factory):
    """A stand-in build, so the Python suite never needs Node."""
    directory = tmp_path_factory.mktemp("frontend")
    (directory / "index.html").write_text("<!doctype html><title>page</title>")
    return directory


@pytest.fixture(scope="module")
def client(served, frontend):
    from serve.api import create_app

    with TestClient(create_app(served, frontend=frontend)) as client:
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
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["categories"]


def test_health_names_the_deployed_commit(client, monkeypatch):
    """The smoke test waits for this commit."""
    from serve.api import COMMIT_ENV

    monkeypatch.setenv(COMMIT_ENV, "b11d5ea3")
    assert client.get("/api/health").json()["commit"] == "b11d5ea3"


def test_scoring_echoes_the_eligibility_precondition(client, registry, category_key):
    response = client.post("/api/score", json=_payload(registry, category_key))
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
    response = client.post("/api/score", json=payload)
    assert response.status_code == 400
    assert "not available" in response.json()["detail"]


def test_missing_values_are_reported_not_imputed(client, registry, category_key):
    payload = _payload(registry, category_key, n=2)
    dropped = sorted(payload["alternatives"][0]["values"])[0]
    payload["alternatives"][0]["values"].pop(dropped)

    body = client.post("/api/score", json=payload).json()
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

    body = client.post("/api/score", json=payload).json()
    assert f"{key}={level_key}" in body["results"][0]["disqualifying_levels"]
    assert body["notes"]


def test_indicator_listing_exposes_the_definitions(client, category_key):
    body = client.get(f"/api/categories/{category_key}/indicators").json()
    assert body
    for row in body:
        assert row["definition"].strip()
        assert row["value_type"] in ("continuous", "ordinal", "nominal", "boolean")


def test_unknown_category_is_a_404(client):
    response = client.post(
        "/api/score",
        json={"category": "not_a_category", "alternatives": [{"id": "a"}]},
    )
    assert response.status_code == 404
    assert "unknown category" in response.json()["detail"]


def test_the_bare_domain_reaches_the_tool(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_a_missing_frontend_build_fails_at_startup(served, tmp_path):
    from serve.api import create_app

    with pytest.raises(RuntimeError):
        create_app(served, frontend=tmp_path / "dist")


def test_the_page_endpoints_answer_in_their_declared_shape(client, registry, category_key):
    form = client.get("/api/explore/form").json()
    assert {c["key"] for c in form["categories"]} == set(registry.categories)

    payload = _payload(registry, category_key)
    payload["alternatives"][1]["values"] = {}
    compared = client.post("/api/explore/compare", json=payload)
    assert compared.status_code == 200
    assert len(compared.json()["scores"]) == len(payload["alternatives"])


def test_a_category_trained_on_synthetic_labels_is_shown_as_a_preview(client, registry):
    from core.registry import SYNTHETIC_PROVENANCE

    form = client.get("/api/explore/form").json()
    shown = {c["key"]: c["preview"] for c in form["categories"]}
    assert shown == {
        key: category.provenance_weights.get(SYNTHETIC_PROVENANCE, 0.0) > 0.0
        for key, category in registry.categories.items()
    }


def test_the_sensitivity_endpoints_name_their_rows(client, registry, category_key):
    payload = _payload(registry, category_key)
    by_context = client.post("/api/explore/context-sensitivity", json=payload).json()
    by_stakeholder = client.post("/api/explore/stakeholder-sensitivity", json=payload).json()

    assert [row["label"] for row in by_context["rows"]] == sorted(
        registry.category(category_key).available_contexts
    )
    assert [row["label"] for row in by_stakeholder["rows"]] == sorted(registry.stakeholders)
    assert all(len(row["scores"]) == len(payload["alternatives"]) for row in by_context["rows"])


def test_a_page_path_serves_the_frontend_and_an_api_path_never_does(client):
    browser = {"Accept": "text/html"}
    assert client.get("/a/page/path", headers=browser).headers["content-type"].startswith(
        "text/html"
    )
    missing = client.get("/api/no-such-endpoint", headers=browser)
    assert missing.status_code == 404
    assert missing.headers["content-type"] == "application/json"


def test_the_former_page_address_still_reaches_the_tool(client):
    for old in ("/explore/", "/explore"):
        response = client.get(old, follow_redirects=True)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")


def test_a_caller_past_the_rate_limit_is_refused(served, registry, category_key, monkeypatch):
    from serve.api import create_app

    monkeypatch.setenv("RECOMMENDER_RATE_LIMIT", "2")
    monkeypatch.setenv("RECOMMENDER_RATE_LIMIT_TOTAL", "0")
    payload = _payload(registry, category_key)
    headers = {"x-forwarded-for": "198.51.100.4"}

    with TestClient(create_app(served, frontend=None)) as limited:
        assert limited.post("/api/score", json=payload, headers=headers).status_code == 200
        assert limited.post("/api/score", json=payload, headers=headers).status_code == 200
        refused = limited.post("/api/score", json=payload, headers=headers)

    assert refused.status_code == 429
    assert int(refused.headers["retry-after"]) >= 1


def test_the_rate_limit_is_per_caller(served, registry, category_key, monkeypatch):
    from serve.api import create_app

    monkeypatch.setenv("RECOMMENDER_RATE_LIMIT", "1")
    monkeypatch.setenv("RECOMMENDER_RATE_LIMIT_TOTAL", "0")
    payload = _payload(registry, category_key)

    with TestClient(create_app(served, frontend=None)) as limited:
        first = {"x-forwarded-for": "198.51.100.4"}
        assert limited.post("/api/score", json=payload, headers=first).status_code == 200
        assert limited.post("/api/score", json=payload, headers=first).status_code == 429
        second = {"x-forwarded-for": "198.51.100.9"}
        assert limited.post("/api/score", json=payload, headers=second).status_code == 200


def test_the_page_stays_reachable_when_scoring_is_rate_limited(
    served, registry, category_key, monkeypatch
):
    from serve.api import create_app

    monkeypatch.setenv("RECOMMENDER_RATE_LIMIT", "1")
    monkeypatch.setenv("RECOMMENDER_RATE_LIMIT_TOTAL", "0")
    payload = _payload(registry, category_key)
    headers = {"x-forwarded-for": "198.51.100.4"}

    with TestClient(create_app(served, frontend=None)) as limited:
        limited.post("/api/score", json=payload, headers=headers)
        assert limited.post("/api/score", json=payload, headers=headers).status_code == 429
        assert limited.get("/api/health", headers=headers).status_code == 200
        assert limited.get("/api/explore/form", headers=headers).status_code == 200


def test_an_oversized_shortlist_is_refused(client, registry, category_key):
    from serve import limits

    payload = _payload(registry, category_key, n=limits.MAX_ALTERNATIVES + 1)
    assert client.post("/api/score", json=payload).status_code == 422


def test_a_shortlist_wider_than_the_corpus_is_refused(client, registry, category_key):
    payload = _payload(registry, category_key, n=6)
    assert client.post("/api/score", json=payload).status_code == 422
