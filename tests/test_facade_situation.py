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
