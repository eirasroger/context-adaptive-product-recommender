from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from serve.shortlist import checked_contexts, checked_stakeholder_count
from train.run import trained_context_limits, trained_stakeholder_limits


def _offered(category):
    return [key for key in category.context_presentation if key != category.default_context_key]


def test_no_context_means_the_baseline(registry, category_key):
    category = registry.category(category_key)
    assert checked_contexts(category, [], limit=2) == [category.default_context_key]


def test_the_baseline_never_combines_with_another_context(registry, category_key):
    category = registry.category(category_key)
    with pytest.raises(HTTPException, match="stands alone"):
        checked_contexts(category, [category.default_context_key, _offered(category)[0]], limit=2)


def test_a_repeated_context_is_refused(registry, category_key):
    category = registry.category(category_key)
    first = _offered(category)[0]
    with pytest.raises(HTTPException, match="repeated"):
        checked_contexts(category, [first, first], limit=2)


def test_more_contexts_than_training_combined_are_refused(registry, category_key):
    category = registry.category(category_key)
    with pytest.raises(HTTPException, match="at most 1"):
        checked_contexts(category, _offered(category)[:2], limit=1)


def test_every_offered_context_is_presented_once(registry, category_key):
    category = registry.category(category_key)
    assert set(category.context_presentation) == set(category.available_contexts)


def test_a_category_without_presentation_shows_each_context_under_its_own_name(registry):
    for category in registry.categories.values():
        for key, shown in category.context_presentation.items():
            if shown.theme is None:
                assert shown.display_name == registry.contexts[key].display_name


def test_the_limit_is_the_most_contexts_a_training_set_combined():
    prepared = SimpleNamespace(
        set_fold=("train", "train", "test", "train"),
        category_keys=("a", "b"),
        set_category=np.array([0, 0, 0, 1]),
        set_combo=np.array([0, 1, 2, 0]),
        combo_context_slots=((1,), (1, 2), (1, 2, 3)),
    )
    assert trained_context_limits(prepared) == {"a": 2, "b": 1}


def test_the_stakeholder_limit_is_the_most_stakeholders_a_training_set_combined():
    prepared = SimpleNamespace(
        set_fold=("train", "train", "test", "train"),
        category_keys=("a", "b"),
        set_category=np.array([0, 0, 0, 1]),
        set_stakeholder_slots=((1,), (1, 2, 3), (1, 2, 3, 4), (5,)),
    )
    assert trained_stakeholder_limits(prepared) == {"a": 3, "b": 1}


def test_more_stakeholders_than_trained_are_refused():
    checked_stakeholder_count("a", ["x", "y", "z"], limit=3)
    with pytest.raises(HTTPException):
        checked_stakeholder_count("a", ["w", "x", "y", "z"], limit=3)
    with pytest.raises(HTTPException):
        checked_stakeholder_count("a", ["x", "x"], limit=3)
