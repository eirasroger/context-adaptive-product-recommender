from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Path(__file__).parent / "contracts" / "openapi.json"
PAGE = ROOT / "serve" / "explore" / "static" / "index.html"

#: Set to rewrite the committed contract after a deliberate API change.
UPDATE_ENV = "RECOMMENDER_UPDATE_CONTRACT"


@pytest.fixture(scope="module")
def schema():
    """The published schema. Reading it needs no checkpoint, only the routes."""
    from serve.api import create_app

    return TestClient(create_app()).get("/openapi.json").json()


def test_the_published_schema_matches_the_committed_contract(schema):
    rendered = json.dumps(schema, indent=2, sort_keys=True) + "\n"

    if os.environ.get(UPDATE_ENV):
        CONTRACT.write_text(rendered, encoding="utf-8")
        pytest.skip(f"contract rewritten ({UPDATE_ENV} was set)")

    assert CONTRACT.exists(), f"no committed contract; run with {UPDATE_ENV}=1"
    assert rendered == CONTRACT.read_text(encoding="utf-8"), (
        "the API surface moved. Review the diff, then re-record it with "
        f"{UPDATE_ENV}=1 python -m pytest tests/test_api_contract.py"
    )


def test_every_route_the_page_calls_exists(schema):
    """The three endpoints the tool fetches, read out of the page itself."""
    called = set(re.findall(r"""\bapi\(\s*["']([^"']+)["']""", PAGE.read_text(encoding="utf-8")))

    assert called, "found no API calls in the page; the pattern has changed"
    missing = sorted(path for path in called if path not in schema["paths"])
    assert not missing, f"the page calls {missing}, which the API does not serve"


def test_the_fields_the_page_reads_are_in_the_score_response(schema):
    scored = schema["components"]["schemas"]["ScoredAlternative"]["properties"]
    response = schema["components"]["schemas"]["ScoreResponse"]["properties"]

    assert {"id", "score", "rank", "missing_indicators", "disqualifying_levels"} <= set(scored)
    assert {"results", "notes", "eligibility_precondition", "functional_unit"} <= set(response)
