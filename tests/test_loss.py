"""The loss weights one decision as one decision."""

from __future__ import annotations

import pytest
import torch

from model.loss import (
    LossWeights,
    composite_loss,
    pairwise_gap_loss,
    pointwise_loss,
)


def _mask(sizes, width):
    mask = torch.zeros(len(sizes), width, dtype=torch.bool)
    for row, size in enumerate(sizes):
        mask[row, :size] = True
    return mask


def test_pairwise_term_weights_sets_equally_regardless_of_size():
    """A five-alternative decision has ten pairs and a two-alternative one has
    one. Averaging within a set before averaging across sets is what stops the
    larger decision counting ten times as much.
    """
    width = 5
    mask = _mask([2, 5], width)

    pref = torch.full((2, width), float("nan"))
    pref[0, :2] = torch.tensor([0.9, 0.1])
    pref[1, :5] = torch.linspace(0.9, 0.1, 5)

    # Each set is wrong by the same amount per pair.
    scores = torch.where(torch.isnan(pref), torch.zeros_like(pref), pref + 0.1)
    scores[0, 0] -= 0.2
    scores[1, 0] -= 0.2

    conf = torch.ones_like(pref)
    ones = torch.ones(2)

    small = pairwise_gap_loss(scores[:1], pref[:1], conf[:1], mask[:1], ones[:1])
    large = pairwise_gap_loss(scores[1:], pref[1:], conf[1:], mask[1:], ones[1:])
    both = pairwise_gap_loss(scores, pref, conf, mask, ones)

    assert both == pytest.approx(float((small + large) / 2), abs=1e-6)


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

    loss = pairwise_gap_loss(scores, pref, conf, mask, torch.ones(1))
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_confidence_scales_the_pointwise_term():
    width = 2
    mask = _mask([2], width)
    # Deliberately asymmetric: the two alternatives are wrong by different
    # amounts, so reweighting them has somewhere to move the loss to.
    pref = torch.tensor([[0.9, 0.4]])
    scores = torch.tensor([[0.5, 0.5]])

    confident = pointwise_loss(scores, pref, torch.ones(1, 2), mask, torch.ones(1))
    unsure = pointwise_loss(
        scores, pref, torch.full((1, 2), 0.1), mask, torch.ones(1)
    )
    # Weights are renormalised, so uniform confidence cannot change the result;
    # what matters is that a low-confidence label does not dominate a confident
    # one within the same set.
    assert float(confident) == pytest.approx(float(unsure), abs=1e-6)

    mixed = pointwise_loss(
        scores, pref, torch.tensor([[1.0, 0.05]]), mask, torch.ones(1)
    )
    assert float(mixed) != pytest.approx(float(confident), abs=1e-6)


def test_unlabelled_alternatives_are_skipped():
    width = 3
    mask = _mask([3], width)
    pref = torch.tensor([[0.8, float("nan"), 0.2]])
    conf = torch.ones(1, width)
    scores = torch.tensor([[0.8, 0.999, 0.2]])

    loss = pointwise_loss(scores, pref, conf, mask, torch.ones(1))
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


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


def test_composite_reports_its_terms_separately():
    width = 3
    mask = _mask([3], width)
    pref = torch.tensor([[0.9, 0.5, 0.1]])
    conf = torch.ones(1, width)
    scores = torch.tensor([[0.6, 0.5, 0.4]], requires_grad=True)

    breakdown = composite_loss(
        scores, pref, conf, mask, ("control",), ("cat",), LossWeights()
    )
    assert float(breakdown.pointwise) > 0
    assert float(breakdown.pairwise) > 0
    assert float(breakdown.listwise) == 0.0
    breakdown.total.backward()
    assert scores.grad is not None


def test_permutation_loss_is_off_by_default():
    """It separates near-ties, which is the opposite of the objective here.

    Implemented so the ablation can be run, never mixed in unless asked for.
    """
    assert LossWeights().listwise == 0.0
