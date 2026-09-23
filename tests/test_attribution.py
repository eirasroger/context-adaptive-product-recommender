from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments import attribution
from model.recommender import ModelConfig, Recommender
from serve.explore import analysis


@pytest.fixture(scope="module")
def model(registry):
    torch.manual_seed(0)
    return Recommender(
        ModelConfig.for_registry(
            registry, dim=32, encoder_blocks=1, comparator_blocks=1, dropout=0.0
        )
    ).eval()


def _shortlist(registry, category_key, size=3):
    context = registry.category(category_key).default_context_key
    return attribution.Shortlist(
        category_key=category_key,
        alternatives=tuple(analysis.example_shortlist(registry, category_key, [context], size)),
        contexts=(context,),
        stakeholders=("balanced_optimizer",),
    )


def test_withholding_an_indicator_moves_some_score(registry, category_key, model):
    shifts = attribution.attribution(model, registry, _shortlist(registry, category_key))
    assert shifts
    assert all(len(delta) == 3 for delta in shifts.values())
    assert max(float(np.abs(delta).max()) for delta in shifts.values()) > 0.0


def test_derived_indicators_are_never_withheld(registry, category_key, model):
    shifts = attribution.attribution(model, registry, _shortlist(registry, category_key, 2))
    category = registry.category(category_key)
    derived = {k for k in category.token_order if registry.indicator(k).is_derived}
    assert not (set(shifts) & derived)


def test_batching_gives_the_same_shifts_as_one_shortlist_at_a_time(
    registry, category_key, model
):
    shortlists = [_shortlist(registry, category_key, n) for n in (2, 3, 4)]
    batched = attribution.attribute_many(model, registry, shortlists)
    for shortlist, together in zip(shortlists, batched):
        alone = attribution.attribution(model, registry, shortlist)
        assert together.keys() == alone.keys()
        for key in alone:
            np.testing.assert_allclose(together[key], alone[key], atol=1e-5)


def test_rank_correlation_reads_direction_and_ignores_scale():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert attribution.rank_correlation(x, x ** 3) == pytest.approx(1.0)
    assert attribution.rank_correlation(x, -x) == pytest.approx(-1.0)
    assert attribution.rank_correlation(x, np.zeros(4)) == 0.0


def test_the_committed_test_fold_loads_as_shortlists(registry):
    shortlists = attribution.load_shortlists(attribution.SNAPSHOT)
    assert shortlists
    first = shortlists[0]
    assert len(first.alternatives) >= 2
    assert first.contexts and set(first.contexts) <= set(registry.contexts)
    assert set(first.stakeholders) <= set(registry.stakeholders)
    assert any(a.values for a in first.alternatives)


def test_only_indicators_with_a_value_that_sinks_the_preference_to_zero_are_masked(registry):
    """A mask on any other indicator would hide a real effect."""
    import pandas as pd

    floors = attribution.preference_floors(registry, attribution.SNAPSHOT)
    assert floors

    sets = pd.read_parquet(attribution.SNAPSHOT / "sets.parquet")
    members = pd.read_parquet(attribution.SNAPSHOT / "members.parquet")
    control = members.merge(sets[sets["provenance_key"] == "control"], on="set_id")
    lowest = control.groupby("generator_key")["pref"].min()
    for key, floor in lowest.items():
        if key in registry.indicators:
            assert (key in floors) == (floor < attribution.FLOOR), key
