"""Training objective.

Two terms:

* a **pointwise** term, so the score lands in the right band -- weighted by the
  labeller's stated confidence and by how much the category's registry says to
  trust that provenance;
* a **pairwise-difference** term over every pair within a set, so the *gaps*
  between alternatives are right and not merely their order.

Pairs are averaged within a set and then across sets. Sets here hold between two
and five alternatives, which is one to ten pairs; summing instead would weight a
five-alternative decision ten times a two-alternative one for no reason. One
decision, one unit of weight.

Ranking-by-permutation losses are deliberately not used. Maximising the
likelihood of the correct ordering applies gradient pressure to *separate*
near-ties, which is the opposite of what is wanted here: if two products are
genuinely at 0.87 and 0.88 the model should say so, and a swap between them is
not an error. One is implemented below for the ablation that reports what
happens when you do use it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

MIN_CONFIDENCE = 0.05


@dataclass
class LossWeights:
    """How the two terms are mixed, and how provenance is weighted.

    ``provenance`` is keyed by category, because how far to trust a control case
    against an expert judgement is a property of the category, declared in the
    registry rather than hardcoded here.
    """

    pointwise: float = 1.0
    pairwise: float = 1.0
    listwise: float = 0.0  # ablation only
    provenance: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def weight_for(self, category_key: str, provenance_key: str) -> float:
        return self.provenance.get(category_key, {}).get(provenance_key, 1.0)


@dataclass
class LossBreakdown:
    total: torch.Tensor
    pointwise: torch.Tensor
    pairwise: torch.Tensor
    listwise: torch.Tensor
    by_provenance: dict[str, float]


def _valid(pref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return mask & ~torch.isnan(pref)


def pointwise_loss(
    scores: torch.Tensor,
    pref: torch.Tensor,
    conf: torch.Tensor,
    mask: torch.Tensor,
    set_weights: torch.Tensor,
) -> torch.Tensor:
    """Confidence-weighted smooth L1 against the label, averaged per set."""
    valid = _valid(pref, mask)
    if not valid.any():
        return scores.sum() * 0.0

    target = torch.nan_to_num(pref, nan=0.0)
    element = F.smooth_l1_loss(scores, target, reduction="none", beta=0.05)

    confidence = torch.nan_to_num(conf, nan=1.0).clamp(MIN_CONFIDENCE, 1.0)
    weights = confidence * valid.to(scores.dtype)

    per_set = (element * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-8)
    counted = valid.any(dim=1).to(scores.dtype)
    return (per_set * set_weights * counted).sum() / (
        (set_weights * counted).sum().clamp(min=1e-8)
    )


def pairwise_gap_loss(
    scores: torch.Tensor,
    pref: torch.Tensor,
    conf: torch.Tensor,
    mask: torch.Tensor,
    set_weights: torch.Tensor,
) -> torch.Tensor:
    """Penalise getting the gap between two alternatives wrong.

    Enumerated over the upper triangle and masked, so padded positions never
    form a pair. Averaged over the pairs in a set before averaging over sets.
    """
    valid = _valid(pref, mask)
    batch, alts = scores.shape
    if alts < 2:
        return scores.sum() * 0.0

    upper = torch.triu(
        torch.ones(alts, alts, dtype=torch.bool, device=scores.device), diagonal=1
    )
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1) & upper

    target = torch.nan_to_num(pref, nan=0.0)
    predicted_gap = scores.unsqueeze(2) - scores.unsqueeze(1)
    true_gap = target.unsqueeze(2) - target.unsqueeze(1)

    confidence = torch.nan_to_num(conf, nan=1.0).clamp(MIN_CONFIDENCE, 1.0)
    pair_confidence = torch.minimum(confidence.unsqueeze(2), confidence.unsqueeze(1))

    element = F.smooth_l1_loss(
        predicted_gap, true_gap, reduction="none", beta=0.05
    ) * pair_confidence * pair_mask.to(scores.dtype)

    pair_count = pair_mask.sum(dim=(1, 2)).to(scores.dtype)
    per_set = element.sum(dim=(1, 2)) / pair_count.clamp(min=1e-8)
    counted = (pair_count > 0).to(scores.dtype)
    return (per_set * set_weights * counted).sum() / (
        (set_weights * counted).sum().clamp(min=1e-8)
    )


def listwise_permutation_loss(
    scores: torch.Tensor,
    pref: torch.Tensor,
    mask: torch.Tensor,
    set_weights: torch.Tensor,
) -> torch.Tensor:
    """Likelihood of the correct permutation. Ablation only.

    Kept so the experiment that shows why it is the wrong objective can actually
    be run. It is never part of the default mix.
    """
    valid = _valid(pref, mask)
    target = torch.nan_to_num(pref, nan=-1e9)
    logits = torch.where(valid, scores, torch.full_like(scores, -1e9))

    order = torch.argsort(target, dim=1, descending=True)
    ordered = torch.gather(logits, 1, order)
    ordered_valid = torch.gather(valid, 1, order)

    # log-cumulative-sum-exp from the tail: at each position, the remaining
    # candidates are the ones ranked at or below it.
    flipped = torch.flip(ordered.masked_fill(~ordered_valid, -1e9), dims=[1])
    tail = torch.flip(torch.logcumsumexp(flipped, dim=1), dims=[1])

    element = (tail - ordered) * ordered_valid.to(scores.dtype)
    counts = ordered_valid.sum(dim=1).to(scores.dtype)
    per_set = element.sum(dim=1) / counts.clamp(min=1e-8)
    counted = (counts > 0).to(scores.dtype)
    return (per_set * set_weights * counted).sum() / (
        (set_weights * counted).sum().clamp(min=1e-8)
    )


def composite_loss(
    scores: torch.Tensor,
    pref: torch.Tensor,
    conf: torch.Tensor,
    mask: torch.Tensor,
    provenance: Sequence[str],
    category_keys: Sequence[str],
    weights: LossWeights,
) -> LossBreakdown:
    """The training objective, with its terms reported separately."""
    set_weights = torch.tensor(
        [
            weights.weight_for(category, source)
            for category, source in zip(category_keys, provenance)
        ],
        dtype=scores.dtype,
        device=scores.device,
    )

    point = pointwise_loss(scores, pref, conf, mask, set_weights)
    pair = pairwise_gap_loss(scores, pref, conf, mask, set_weights)
    listwise = (
        listwise_permutation_loss(scores, pref, mask, set_weights)
        if weights.listwise
        else scores.sum() * 0.0
    )

    total = (
        weights.pointwise * point
        + weights.pairwise * pair
        + weights.listwise * listwise
    )

    by_provenance: dict[str, float] = {}
    for source in sorted(set(provenance)):
        rows = torch.tensor(
            [key == source for key in provenance], device=scores.device
        )
        if rows.any():
            ones = torch.ones(int(rows.sum()), dtype=scores.dtype, device=scores.device)
            by_provenance[source] = float(
                pointwise_loss(
                    scores[rows], pref[rows], conf[rows], mask[rows], ones
                ).detach()
            )

    return LossBreakdown(
        total=total,
        pointwise=point.detach(),
        pairwise=pair.detach(),
        listwise=listwise.detach(),
        by_provenance=by_provenance,
    )
