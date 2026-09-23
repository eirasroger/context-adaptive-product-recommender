from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "openapi.json"

#: Set to rewrite the committed contract after a deliberate API change.
UPDATE_ENV = "RECOMMENDER_UPDATE_CONTRACT"


@pytest.fixture(scope="module")
def schema():
    from serve.api import create_app

    return TestClient(create_app(frontend=None)).get("/api/openapi.json").json()


def test_the_published_schema_matches_the_committed_contract(schema):
    """The frontend's types are generated from this contract."""
    rendered = json.dumps(schema, indent=2, sort_keys=True) + "\n"

    if os.environ.get(UPDATE_ENV):
        CONTRACT.write_text(rendered, encoding="utf-8")
        pytest.skip(f"contract rewritten ({UPDATE_ENV} was set)")

    assert CONTRACT.exists(), f"no committed contract; run with {UPDATE_ENV}=1"
    assert rendered == CONTRACT.read_text(encoding="utf-8"), (
        "the API surface moved. Review the diff, then re-record it with "
        f"{UPDATE_ENV}=1 python -m pytest tests/test_api_contract.py"
    )


def test_every_endpoint_lives_under_api(schema):
    """The frontend owns every other path."""
    stray = sorted(path for path in schema["paths"] if not path.startswith("/api/"))
    assert not stray, f"endpoints outside /api: {stray}"


def test_operation_ids_are_unique(schema):
    ids = [
        operation["operationId"]
        for path in schema["paths"].values()
        for operation in path.values()
    ]
    assert len(ids) == len(set(ids)), sorted(ids)
