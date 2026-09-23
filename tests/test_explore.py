from __future__ import annotations

import numpy as np
import pytest
import torch

from core.encoding import AlternativeInput
from ingest.generators import parametric
from model.export import TorchScorer
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


@pytest.fixture(scope="module")
def scorer(model):
    return TorchScorer(model)


def _opposed(registry, category_key):
    category = registry.category(category_key)
    for key in category.token_order:
        directions = {
            ctx: registry.resolve_direction(category_key, key, [ctx])[0]
            for ctx in sorted(category.available_contexts)
        }
        live = {c: d for c, d in directions.items() if d != 0}
        if live and min(live.values()) < 0 < max(live.values()):
            positive = next(c for c, d in live.items() if d > 0)
            negative = next(c for c, d in live.items() if d < 0)
            return key, positive, negative
    pytest.skip("no indicator inverts between contexts")


def test_response_curve_reports_the_declared_direction(registry, category_key, scorer):
    indicator, positive, negative = _opposed(registry, category_key)

    up = analysis.response_curve(
        scorer, registry, category_key, indicator, [positive],
        sorted(registry.stakeholders)[:2], steps=7,
    )
    down = analysis.response_curve(
        scorer, registry, category_key, indicator, [negative],
        sorted(registry.stakeholders)[:2], steps=7,
    )
    assert up["direction"] > 0
    assert down["direction"] < 0
    assert len(up["series"]) == 2
    assert len(up["values"]) == 7
    assert len(up["expected"]) == 7


def test_response_curve_carries_the_registry_expectation(registry, category_key, scorer):
    indicator = registry.sweepable(category_key)[0]
    category = registry.category(category_key)
    data = analysis.response_curve(
        scorer, registry, category_key, indicator,
        [category.default_context_key], ["balanced_optimizer"], steps=9,
    )
    assert min(data["expected"]) >= 0.0
    assert max(data["expected"]) <= 1.0
    assert max(data["expected"]) > min(data["expected"])


def test_excluded_indicators_are_flagged_in_the_response(registry, category_key, scorer):
    category = registry.category(category_key)
    excluded = [
        key for key in category.token_order
        if not category.members[key].is_sweepable
        and not registry.indicator(key).is_derived
    ]
    if not excluded:
        pytest.skip("nothing is excluded in the seeded registry")

    data = analysis.response_curve(
        scorer, registry, category_key, excluded[0],
        [category.default_context_key], ["balanced_optimizer"], steps=5,
    )
    assert data["sweepable"] is False
    assert data["control_note"]


def test_sensitivity_covers_every_available_context(registry, category_key, scorer):
    category = registry.category(category_key)
    alternatives = analysis.example_shortlist(
        registry, category_key, [category.default_context_key], size=3
    )
    data = analysis.context_sensitivity(
        scorer, registry, category_key, alternatives, ["balanced_optimizer"]
    )
    assert {row["label"] for row in data["rows"]} == category.available_contexts
    assert all(len(row["scores"]) == 3 for row in data["rows"])


def test_sensitivity_covers_every_stakeholder(registry, category_key, scorer):
    category = registry.category(category_key)
    alternatives = analysis.example_shortlist(
        registry, category_key, [category.default_context_key], size=3
    )
    data = analysis.stakeholder_sensitivity(
        scorer, registry, category_key, alternatives, [category.default_context_key]
    )
    assert {row["label"] for row in data["rows"]} == set(registry.stakeholders)


def test_example_shortlist_varies_something(registry, category_key):
    category = registry.category(category_key)
    context = category.default_context_key
    alternatives = analysis.example_shortlist(registry, category_key, [context], size=4)
    assert len(alternatives) == 4

    varying = set()
    for key in category.token_order:
        seen = {a.values.get(key, a.levels.get(key)) for a in alternatives}
        if len(seen) > 1:
            varying.add(key)
    assert varying


def test_scoring_one_shortlist_matches_the_serving_path(
    registry, category_key, model, scorer
):
    from core.dataset import collate
    from core.encoding import encode_set

    context = registry.category(category_key).default_context_key
    base = parametric.ideal_alternative(registry, category_key, [context])
    alternatives = [
        AlternativeInput("a", dict(base.values), dict(base.levels)),
        AlternativeInput("b", dict(base.values), dict(base.levels)),
    ]

    through_explorer = analysis.score(
        scorer, registry, category_key, alternatives, [context], ["balanced_optimizer"]
    )

    encoded = encode_set(
        registry, category_key, alternatives, [context], ["balanced_optimizer"]
    )
    item = {
        "index": 0,
        "channels": encoded.channels,
        "level_slots": encoded.level_slots,
        "indicator_slots": encoded.indicator_slots[0],
        "family_slots": encoded.family_slots[0],
        "category_slot": encoded.category_slot,
        "category_key": category_key,
        "stakeholder_slots": encoded.stakeholder_slots,
        "context_slots": encoded.context_slots,
        "pref": np.full(2, np.nan, dtype=np.float32),
        "conf": np.full(2, np.nan, dtype=np.float32),
        "provenance": "control",
    }
    with torch.no_grad():
        direct = model(collate([item]))[0, :2].numpy()

    np.testing.assert_allclose(through_explorer, direct, atol=1e-6)
