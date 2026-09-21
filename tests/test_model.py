"""The model's structural promises.

These are not accuracy tests. They check the properties the architecture is
supposed to have regardless of how well it happens to be trained -- because
those are the properties that make adding a category cheap, and they are easy to
break without noticing.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from core.dataset import collate
from core.encoding import AlternativeInput, encode_set
from model import checkpoint as checkpoint_module
from model.comparator import Comparator
from model.recommender import ModelConfig, Recommender


def _item(registry, category_key, alternatives, context_keys, stakeholder_keys):
    encoded = encode_set(
        registry, category_key, alternatives, context_keys, stakeholder_keys
    )
    n = len(alternatives)
    return {
        "index": 0,
        "channels": encoded.channels,
        "level_slots": encoded.level_slots,
        "indicator_slots": encoded.indicator_slots[0],
        "family_slots": encoded.family_slots[0],
        "category_slot": encoded.category_slot,
        "category_key": category_key,
        "stakeholder_slots": encoded.stakeholder_slots,
        "context_slots": encoded.context_slots,
        "pref": np.full(n, np.nan, dtype=np.float32),
        "conf": np.full(n, np.nan, dtype=np.float32),
        "provenance": "control",
    }


@pytest.fixture(scope="module")
def model(registry):
    torch.manual_seed(0)
    return Recommender(ModelConfig.for_registry(registry, dim=32, encoder_blocks=1,
                                                comparator_blocks=1)).eval()


def test_no_category_is_named_in_the_model_or_core_or_train():
    """The design has leaked if a category key appears in the shared code.

    Checked against the registry's own category keys rather than a hardcoded
    word, so this keeps working when a second category is added.
    """
    root = Path(__file__).resolve().parents[1]
    import yaml

    seeds = yaml.safe_load(
        (root / "db" / "seeds" / "40_category_concrete.yaml").read_text(encoding="utf-8")
    )
    names = {row["key"] for row in seeds.get("category", [])}
    names |= {row["display_name"].lower() for row in seeds.get("category", [])}

    offenders = []
    for directory in ("model", "core", "train"):
        for path in (root / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for name in names:
                if name in text:
                    offenders.append(f"{path.relative_to(root)} mentions {name!r}")
    assert not offenders, offenders


def test_the_comparator_cannot_see_an_indicator():
    """The transferable stage takes alternative embeddings and nothing else.

    This boundary is what makes comparing alternatives in one category and in
    another literally the same operation.
    """
    signature = inspect.signature(Comparator.forward)
    assert set(signature.parameters) == {
        "self",
        "alternatives",
        "mask",
        "conditioning",
    }
    source = inspect.getsource(Comparator).lower()
    for word in ("indicator", "token", "level", "family"):
        assert word not in source.split("\"\"\"")[2].lower(), (
            f"comparator body references {word!r}"
        )


def test_set_size_is_not_capped(model, registry, category_key):
    """Two alternatives and many go through the same code."""
    from ingest.generators.parametric import ideal_alternative

    context = registry.category(category_key).default_context_key
    base = ideal_alternative(registry, category_key, [context])

    for size in (2, 5, 40):
        alternatives = [
            AlternativeInput(f"a{i}", dict(base.values), dict(base.levels))
            for i in range(size)
        ]
        batch = collate([_item(registry, category_key, alternatives, [context], [])])
        with torch.no_grad():
            scores = model(batch)
        assert scores.shape == (1, size)
        assert torch.isfinite(scores).all()


def test_padding_does_not_change_a_score(model, registry, category_key):
    """A set must score the same whatever it shares a batch with.

    If padding leaked, every reported number would depend on batch composition.
    """
    from ingest.generators.parametric import ideal_alternative

    context = registry.category(category_key).default_context_key
    base = ideal_alternative(registry, category_key, [context])
    small = [
        AlternativeInput(f"a{i}", dict(base.values), dict(base.levels)) for i in range(2)
    ]
    large = [
        AlternativeInput(f"b{i}", dict(base.values), dict(base.levels)) for i in range(5)
    ]

    alone = collate([_item(registry, category_key, small, [context], [])])
    together = collate(
        [
            _item(registry, category_key, small, [context], []),
            _item(registry, category_key, large, [context], []),
        ]
    )
    with torch.no_grad():
        alone_scores = model(alone)[0, :2]
        together_scores = model(together)[0, :2]
    assert torch.allclose(alone_scores, together_scores, atol=1e-5)


def test_alternative_order_does_not_change_the_scores(model, registry, category_key):
    """The set is a set. Permuting it must permute the scores, nothing more."""
    from ingest.generators.parametric import ideal_alternative, sweep_values

    category = registry.category(category_key)
    context = category.default_context_key
    varied = next(
        key
        for key in category.token_order
        if registry.indicator(key).value_type == "continuous"
        and not registry.indicator(key).is_derived
    )
    base = ideal_alternative(registry, category_key, [context], exclude={varied})
    values = sweep_values(registry, category_key, varied, 4)

    alternatives = [
        AlternativeInput(f"a{i}", {**base.values, varied: v}, dict(base.levels))
        for i, v in enumerate(values)
    ]
    permutation = [2, 0, 3, 1]
    shuffled = [alternatives[i] for i in permutation]

    with torch.no_grad():
        original = model(collate([_item(registry, category_key, alternatives, [context], [])]))[0, :4]
        permuted = model(collate([_item(registry, category_key, shuffled, [context], [])]))[0, :4]

    assert torch.allclose(original[permutation], permuted, atol=1e-5)


def test_a_new_indicator_starts_at_its_family_prior(registry):
    """The identity of an unseen indicator is its family, not noise.

    This is what makes the family embedding worth having: a category that
    introduces an indicator nothing has seen does not start from scratch.
    """
    config = ModelConfig.for_registry(registry, dim=16)
    model = Recommender(config)
    assert torch.allclose(
        model.tokens.indicator.weight, torch.zeros_like(model.tokens.indicator.weight)
    )
    assert not torch.allclose(
        model.tokens.family.weight, torch.zeros_like(model.tokens.family.weight)
    )


def test_a_checkpoint_survives_the_registry_growing(registry, tmp_path):
    """The promise that adding a category needs no model change.

    A checkpoint trained under a smaller registry must load into a model built
    for a larger one, with every previously learned row untouched.
    """
    torch.manual_seed(0)
    old_config = ModelConfig.for_registry(registry, dim=16, encoder_blocks=1,
                                          comparator_blocks=1)
    model = Recommender(old_config)
    with torch.no_grad():
        model.tokens.indicator.weight.normal_()
        model.conditioning.context.weight.normal_()
    original_indicators = model.tokens.indicator.weight.detach().clone()
    original_contexts = model.conditioning.context.weight.detach().clone()

    path = tmp_path / "model.pt"
    checkpoint_module.save(
        path,
        model,
        registry,
        registry_blob="indicator: []\n",
        meta=checkpoint_module.CheckpointMeta(
            registry_version="test",
            registry_content_hash=registry.content_hash,
            snapshot_hash=None,
            split_key=None,
            created_at=checkpoint_module.now(),
            metrics={},
        ),
    )

    # A registry that has grown: three more indicators, one more context, one
    # more category -- exactly what adding a category looks like.
    grown = ModelConfig(**{**old_config.to_dict()})
    grown.n_indicators += 3
    grown.n_contexts += 1
    grown.n_categories += 1
    grown.n_levels += 7

    state, report = checkpoint_module.grow_state_dict(
        model.state_dict(), old_config, grown
    )
    assert report

    bigger = Recommender(grown)
    bigger.load_state_dict(state)

    kept = bigger.tokens.indicator.weight[: old_config.n_indicators]
    assert torch.allclose(kept, original_indicators)
    added = bigger.tokens.indicator.weight[old_config.n_indicators :]
    assert torch.allclose(added, torch.zeros_like(added))

    kept_contexts = bigger.conditioning.context.weight[: old_config.n_contexts]
    assert torch.allclose(kept_contexts, original_contexts)


def test_a_checkpoint_refuses_to_shrink(registry):
    config = ModelConfig.for_registry(registry, dim=16)
    model = Recommender(config)
    smaller = ModelConfig(**config.to_dict())
    smaller.n_indicators -= 1
    with pytest.raises(ValueError, match="append-only"):
        checkpoint_module.grow_state_dict(model.state_dict(), config, smaller)
