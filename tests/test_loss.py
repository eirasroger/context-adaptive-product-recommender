"""The loss weights one decision as one decision, and one provenance as one
provenance."""

from __future__ import annotations

import pytest
import torch

from model.loss import (
    LossWeights,
    combine_by_group,
    composite_loss,
    pairwise_gap_loss,
    pairwise_per_set,
    pointwise_loss,
    pointwise_per_set,
)

FLAT = LossWeights()


def _mask(sizes, width):
    mask = torch.zeros(len(sizes), width, dtype=torch.bool)
    for row, size in enumerate(sizes):
        mask[row, :size] = True
    return mask


def _uniform(n, category="c", source="control"):
    return (source,) * n, (category,) * n


# ---------------------------------------------------------------------------
# Per-set behaviour
# ---------------------------------------------------------------------------


def test_pairwise_term_weights_sets_equally_regardless_of_size():
    """A five-alternative decision has ten pairs and a two-alternative one has
    one. Averaging within a set before combining across sets is what stops the
    larger decision counting ten times as much.
    """
    width = 5
    mask = _mask([2, 5], width)

    pref = torch.full((2, width), float("nan"))
    pref[0, :2] = torch.tensor([0.9, 0.1])
    pref[1, :5] = torch.linspace(0.9, 0.1, 5)

    scores = torch.where(torch.isnan(pref), torch.zeros_like(pref), pref + 0.1)
    scores[0, 0] -= 0.2
    scores[1, 0] -= 0.2
    conf = torch.ones_like(pref)

    per_set, counted = pairwise_per_set(scores, pref, conf, mask)
    assert counted.all()

    provenance, categories = _uniform(2)
    both = pairwise_gap_loss(scores, pref, conf, mask, provenance, categories, FLAT)
    assert float(both) == pytest.approx(float(per_set.mean()), abs=1e-6)


def test_padding_never_forms_a_pair():
    width = 5
    mask = _mask([2], width)
    pref = torch.full((1, width), float("nan"))
    pref[0, :2] = torch.tensor([0.8, 0.2])
    conf = torch.ones_like(pref)

    scores = torch.zeros(1, width)
    scores[0, :2] = torch.tensor([0.8, 0.2])
    # Absurd values in the padded slots must not matter.
    scores[0, 2:] = 99.0

    provenance, categories = _uniform(1)
    loss = pairwise_gap_loss(scores, pref, conf, mask, provenance, categories, FLAT)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_confidence_scales_the_pointwise_term():
    width = 2
    mask = _mask([2], width)
    # Deliberately asymmetric: the two alternatives are wrong by different
    # amounts, so reweighting them has somewhere to move the loss to.
    pref = torch.tensor([[0.9, 0.4]])
    scores = torch.tensor([[0.5, 0.5]])
    provenance, categories = _uniform(1)

    confident = pointwise_loss(
        scores, pref, torch.ones(1, 2), mask, provenance, categories, FLAT
    )
    unsure = pointwise_loss(
        scores, pref, torch.full((1, 2), 0.1), mask, provenance, categories, FLAT
    )
    # Weights are renormalised within a set, so uniform confidence cannot change
    # the result; what matters is that a low-confidence label does not dominate
    # a confident one beside it.
    assert float(confident) == pytest.approx(float(unsure), abs=1e-6)

    mixed = pointwise_loss(
        scores, pref, torch.tensor([[1.0, 0.05]]), mask, provenance, categories, FLAT
    )
    assert float(mixed) != pytest.approx(float(confident), abs=1e-6)


def test_unlabelled_alternatives_are_skipped():
    width = 3
    mask = _mask([3], width)
    pref = torch.tensor([[0.8, float("nan"), 0.2]])
    conf = torch.ones(1, width)
    scores = torch.tensor([[0.8, 0.999, 0.2]])
    provenance, categories = _uniform(1)

    loss = pointwise_loss(scores, pref, conf, mask, provenance, categories, FLAT)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Provenance weighting
# ---------------------------------------------------------------------------


def _degrade(scores, row):
    """Make one set's prediction clearly wrong, leaving the rest alone."""
    worse = scores.clone()
    worse[row] = torch.tensor([0.1, 0.9])
    return worse


def test_provenance_weights_apply_to_group_means_not_to_sets():
    """A handful of expert cases must outweigh a corpus of generated ones.

    Weighting each *set* by 0.2 would make an expert judgement count for less
    than a control case, which is backwards. Weighting the *group* by 0.2 gives
    the expert cases a fifth of the gradient however few of them there are --
    which is the entire reason for collecting them.
    """
    n_control = 60
    n = n_control + 1
    weights = LossWeights(
        provenance={"c": {"control": 0.4, "llm": 0.4, "expert": 0.2}}
    )

    pref = torch.tensor([[0.9, 0.1]]).repeat(n, 1)
    scores = torch.tensor([[0.5, 0.5]]).repeat(n, 1)
    conf = torch.ones(n, 2)
    mask = torch.ones(n, 2, dtype=torch.bool)
    provenance = ("control",) * n_control + ("expert",)
    categories = ("c",) * n

    base = float(pointwise_loss(scores, pref, conf, mask, provenance, categories, weights))
    expert = float(
        pointwise_loss(
            _degrade(scores, n_control), pref, conf, mask, provenance, categories, weights
        )
    )
    control = float(
        pointwise_loss(
            _degrade(scores, 0), pref, conf, mask, provenance, categories, weights
        )
    )

    # One expert set should move the loss far more than one control set, in
    # proportion to how outnumbered it is.
    assert expert - base > (control - base) * 10


def test_group_weighting_renormalises_over_what_the_batch_contains():
    """A batch with no expert cases must not produce a smaller gradient.

    It should simply be decided by the provenances it does have, rather than
    silently losing the missing group's share of the weight.
    """
    weights = LossWeights(
        provenance={"c": {"control": 0.4, "llm": 0.4, "expert": 0.2}}
    )
    per_set = torch.tensor([0.5, 0.5, 0.5])
    counted = torch.ones(3, dtype=torch.bool)

    only_control = combine_by_group(
        per_set, counted, ("control",) * 3, ("c",) * 3, weights
    )
    mixed = combine_by_group(
        per_set, counted, ("control", "llm", "expert"), ("c",) * 3, weights
    )
    assert float(only_control) == pytest.approx(0.5, abs=1e-6)
    assert float(mixed) == pytest.approx(0.5, abs=1e-6)


def test_categories_are_averaged_not_summed():
    """Adding a category must not inflate the loss."""
    weights = LossWeights(
        provenance={
            "a": {"control": 1.0},
            "b": {"control": 1.0},
        }
    )
    per_set = torch.tensor([0.4, 0.4])
    counted = torch.ones(2, dtype=torch.bool)

    one = combine_by_group(per_set[:1], counted[:1], ("control",), ("a",), weights)
    two = combine_by_group(per_set, counted, ("control",) * 2, ("a", "b"), weights)
    assert float(one) == pytest.approx(float(two), abs=1e-6)


def test_provenance_weights_come_from_the_registry(registry, category_key):
    weights = LossWeights(
        provenance={
            key: dict(category.provenance_weights)
            for key, category in registry.categories.items()
        }
    )
    declared = registry.category(category_key).provenance_weights
    assert declared
    for provenance, value in declared.items():
        assert weights.weight_for(category_key, provenance) == pytest.approx(value)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def test_composite_reports_its_terms_separately():
    width = 3
    mask = _mask([3], width)
    pref = torch.tensor([[0.9, 0.5, 0.1]])
    conf = torch.ones(1, width)
    scores = torch.tensor([[0.6, 0.5, 0.4]], requires_grad=True)

    breakdown = composite_loss(
        scores, pref, conf, mask, ("control",), ("c",), FLAT
    )
    assert float(breakdown.pointwise) > 0
    assert float(breakdown.pairwise) > 0
    assert float(breakdown.listwise) == 0.0
    breakdown.total.backward()
    assert scores.grad is not None


def test_reported_provenance_losses_are_unweighted():
    """The per-provenance numbers say how well each source is fitted.

    Mixing the weighting into them would make the report answer a different
    question from the one it appears to answer.
    """
    weights = LossWeights(
        provenance={"c": {"control": 0.4, "expert": 0.2}}
    )
    pref = torch.tensor([[0.9, 0.1], [0.9, 0.1]])
    scores = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    conf = torch.ones(2, 2)
    mask = torch.ones(2, 2, dtype=torch.bool)

    breakdown = composite_loss(
        scores, pref, conf, mask, ("control", "expert"), ("c", "c"), weights
    )
    per_set, _ = pointwise_per_set(scores, pref, conf, mask)
    assert breakdown.by_provenance["control"] == pytest.approx(float(per_set[0]), abs=1e-6)
    assert breakdown.by_provenance["expert"] == pytest.approx(float(per_set[1]), abs=1e-6)


def test_permutation_loss_is_off_by_default():
    """It separates near-ties, which is the opposite of the objective here.

    Implemented so the ablation can be run, never mixed in unless asked for.
    """
    assert LossWeights().listwise == 0.0
