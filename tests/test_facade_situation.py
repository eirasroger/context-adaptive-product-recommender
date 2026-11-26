import random
from collections import Counter

from ingest.facade.synthesise import CONTEXT_COUNT_WEIGHTS, situation

BASELINE = "baseline"
OTHERS = ["cold", "hot", "noisy", "tall", "wildfire", "seismic", "boundary"]


def _draws(n: int = 20_000) -> list[list[str]]:
    rng = random.Random(0)
    return [situation(rng, BASELINE, OTHERS) for _ in range(n)]


def test_the_baseline_never_joins_another_context():
    for contexts in _draws():
        assert contexts == [BASELINE] or BASELINE not in contexts


def test_no_context_appears_twice_in_a_set():
    for contexts in _draws():
        assert len(contexts) == len(set(contexts))


def test_every_other_context_gets_the_same_volume():
    counts = Counter(key for contexts in _draws() for key in contexts if key != BASELINE)
    assert set(counts) == set(OTHERS)
    assert max(counts.values()) / min(counts.values()) < 1.1


def test_the_number_of_contexts_follows_the_declared_shares():
    draws = _draws()
    sizes = Counter(0 if c == [BASELINE] else len(c) for c in draws)
    for size, share in CONTEXT_COUNT_WEIGHTS.items():
        assert abs(sizes[size] / len(draws) - share) < 0.02


def test_extending_keeps_every_existing_set_and_appends_combined_contexts(registry, tmp_path):
    import json

    from ingest.facade import synthesise

    products = [
        {"id_prod": f"{typology}_{i:04d}", "typology": typology, "gwp": 40.0 + i}
        for typology in ("etics", "masonry") for i in range(6)
    ]
    existing = [{"id": "synthetic_000001", "shortlist": "across", "contexts": ["hot_summer"],
                 "stakeholders": ["balanced_optimizer"], "alternatives": products}]
    (tmp_path / synthesise.SCENARIOS_FILE).write_text(json.dumps(existing), encoding="utf-8")
    (tmp_path / synthesise.LABELS_FILE).write_text(json.dumps([{"id": "synthetic_000001"}]), encoding="utf-8")

    synthesise.extend(registry, tmp_path, sets=40, seed=0, counts=synthesise.COMBINED_CONTEXT_WEIGHTS)

    scenarios = json.loads((tmp_path / synthesise.SCENARIOS_FILE).read_text(encoding="utf-8"))
    labels = json.loads((tmp_path / synthesise.LABELS_FILE).read_text(encoding="utf-8"))
    assert scenarios[0] == existing[0]
    assert [s["id"] for s in scenarios[1:3]] == ["synthetic_000002", "synthetic_000003"]
    assert {len(s["contexts"]) for s in scenarios[1:]} == {3, 4}
    assert [s["id"] for s in scenarios] == [entry["id"] for entry in labels]


def test_a_changed_scenario_no_longer_matches_its_fingerprint():
    from ingest.labelling import fingerprint

    scenario = {"id": "s", "contexts": ["hot_summer"], "stakeholders": ["x"], "alternatives": [{"id_prod": "a", "gwp": 1.0}]}
    assert fingerprint(scenario) == fingerprint(dict(scenario, shortlist="within"))
    assert fingerprint(scenario) != fingerprint(dict(scenario, contexts=["seismic"]))
    assert fingerprint(scenario) != fingerprint(dict(scenario, alternatives=[{"id_prod": "a", "gwp": 2.0}]))
