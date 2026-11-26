import json
import random

import pytest

from core import registry as registry_module
from db.seed import seed
from db.session import create_all, create_db_engine, session_factory
from ingest.labelling import fingerprint
from ingest.partition import label, synthesise


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    engine = create_db_engine(tmp_path_factory.mktemp("registry") / "registry.db")
    create_all(engine)
    session = session_factory(engine)()
    seed(session)
    session.commit()
    return registry_module.from_session(session, version="seeded")


def _product(key: str, typology: str) -> dict:
    return {"id_prod": key, "typology": typology, "sources": ["uuid"], "gwp": 20.0, "impact_resistance": "category_iii"}


POOLS = {t: [_product(f"{t}_{i}", t) for i in range(10)] for t in ("framed", "masonry", "double_leaf")}


def test_a_mixed_shortlist_always_holds_two_typologies_or_more():
    rng = random.Random(0)
    for _ in range(200):
        kind, alternatives = synthesise.shortlist(rng, POOLS)
        if kind == synthesise.SHORTLIST_ACROSS:
            assert len({a["typology"] for a in alternatives}) >= 2


def test_the_labeller_sees_letters_and_indicators_never_the_typology_or_sources(registry):
    scenario = {"id": "s", "contexts": ["heavy_use"], "stakeholders": ["balanced_optimizer"],
                "alternatives": [POOLS["framed"][0], POOLS["masonry"][0]]}
    shown, letters = label.ROUND.payload(registry, scenario)
    text = json.dumps(shown)
    assert "framed" not in text and "masonry" not in text and "uuid" not in text
    assert letters == {"A": "framed_0", "B": "masonry_0"}
    assert shown["alternatives"][0]["impact_resistance"] == "Category III"


def test_a_changed_value_changes_the_fingerprint():
    scenario = {"contexts": ["heavy_use"], "stakeholders": ["balanced_optimizer"], "alternatives": [_product("a", "framed")]}
    changed = json.loads(json.dumps(scenario))
    changed["alternatives"][0]["gwp"] = 21.0
    assert fingerprint(scenario) != fingerprint(changed)
