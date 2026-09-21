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

**Provenance weights apply to group means, not to individual sets.** The loss
is ``sum over provenance of weight * mean(loss over that provenance's sets)``,
not a weighted average across sets. The difference is the whole point of having
expert data: there are a few hundred expert cases against tens of thousands of
control ones, so weighting each *set* by 0.2 would make an expert judgement
count for *less* than a generated one. Weighting the *group* by 0.2 gives the
expert cases a fifth of the total gradient signal, which is an enormous
per-example upweight and is what "expert cases refine the decision boundary"
actually means.

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


def combine_by_group(
    per_set: torch.Tensor,
    counted: torch.Tensor,
    provenance: Sequence[str],
    category_keys: Sequence[str],
    weights: LossWeights,
) -> torch.Tensor:
    """Weight provenance *group means*, then average over categories.

    Within a group, every set counts the same. Across groups, the registry's
    declared weights decide how much each provenance is allowed to move the
    model -- which is what lets a few hundred expert cases matter as much as a
    whole corpus of generated ones.

    Weights are renormalised over the groups actually present in the batch. A
    batch that happens to contain no expert cases should not therefore produce a
    smaller gradient; it should just be decided by the provenances it has.

    Categories are averaged rather than summed, so adding a category does not
    inflate the loss.
    """
    if per_set.numel() == 0:
        return per_set.sum() * 0.0

    buckets: dict[tuple[str, str], list[int]] = {}
    for index, (category, source) in enumerate(zip(category_keys, provenance)):
        if bool(counted[index]):
            buckets.setdefault((category, source), []).append(index)

    if not buckets:
        return per_set.sum() * 0.0

    categories = sorted({category for category, _ in buckets})
    total = per_set.sum() * 0.0

    for category in categories:
        subtotal = per_set.sum() * 0.0
        weight_sum = 0.0
        for (bucket_category, source), rows in buckets.items():
            if bucket_category != category:
                continue
            weight = weights.weight_for(category, source)
            if weight <= 0.0:
                continue
            index = torch.tensor(rows, device=per_set.device, dtype=torch.long)
            subtotal = subtotal + weight * per_set[index].mean()
            weight_sum += weight
        if weight_sum > 0.0:
            total = total + subtotal / weight_sum

    return total / len(categories)


def pointwise_per_set(
    scores: torch.Tensor,
    pref: torch.Tensor,
    conf: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Confidence-weighted smooth L1 against the label, one value per set.

    Returns the per-set loss and which sets carried a label at all.
    """
    valid = _valid(pref, mask)
    target = torch.nan_to_num(pref, nan=0.0)
    element = F.smooth_l1_loss(scores, target, reduction="none", beta=0.05)

    confidence = torch.nan_to_num(conf, nan=1.0).clamp(MIN_CONFIDENCE, 1.0)
    weights = confidence * valid.to(scores.dtype)

    per_set = (element * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-8)
    return per_set, valid.any(dim=1)


def pairwise_per_set(
    scores: torch.Tensor,
    pref: torch.Tensor,
    conf: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Error in the gap between two alternatives, averaged over a set's pairs.

    Enumerated over the upper triangle and masked, so padded positions never
    form a pair.
    """
    valid = _valid(pref, mask)
    batch, alts = scores.shape
    if alts < 2:
        zero = scores.sum(dim=1) * 0.0
        return zero, torch.zeros(batch, dtype=torch.bool, device=scores.device)

    upper = torch.triu(
        torch.ones(alts, alts, dtype=torch.bool, device=scores.device), diagonal=1
    )
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1) & upper

    target = torch.nan_to_num(pref, nan=0.0)
    predicted_gap = scores.unsqueeze(2) - scores.unsqueeze(1)
    true_gap = target.unsqueeze(2) - target.unsqueeze(1)

    confidence = torch.nan_to_num(conf, nan=1.0).clamp(MIN_CONFIDENCE, 1.0)
    pair_confidence = torch.minimum(confidence.unsqueeze(2), confidence.unsqueeze(1))

    element = (
        F.smooth_l1_loss(predicted_gap, true_gap, reduction="none", beta=0.05)
        * pair_confidence
        * pair_mask.to(scores.dtype)
    )

    pair_count = pair_mask.sum(dim=(1, 2)).to(scores.dtype)
    per_set = element.sum(dim=(1, 2)) / pair_count.clamp(min=1e-8)
    return per_set, pair_count > 0


def pointwise_loss(
    scores: torch.Tensor,
    pref: torch.Tensor,
    conf: torch.Tensor,
    mask: torch.Tensor,
    provenance: Sequence[str],
    category_keys: Sequence[str],
    weights: LossWeights,
) -> torch.Tensor:
    per_set, counted = pointwise_per_set(scores, pref, conf, mask)
    return combine_by_group(per_set, counted, provenance, category_keys, weights)


def pairwise_gap_loss(
    scores: torch.Tensor,
    pref: torch.Tensor,
    conf: torch.Tensor,
    mask: torch.Tensor,
    provenance: Sequence[str],
    category_keys: Sequence[str],
    weights: LossWeights,
) -> torch.Tensor:
    per_set, counted = pairwise_per_set(scores, pref, conf, mask)
    return combine_by_group(per_set, counted, provenance, category_keys, weights)


def listwise_permutation_loss(
    scores: torch.Tensor,
    pref: torch.Tensor,
    mask: torch.Tensor,
    provenance: Sequence[str],
    category_keys: Sequence[str],
    weights: LossWeights,
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
    return combine_by_group(
        per_set, counts > 0, provenance, category_keys, weights
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
    point = pointwise_loss(
        scores, pref, conf, mask, provenance, category_keys, weights
    )
    pair = pairwise_gap_loss(
        scores, pref, conf, mask, provenance, category_keys, weights
    )
    listwise = (
        listwise_permutation_loss(
            scores, pref, mask, provenance, category_keys, weights
        )
        if weights.listwise
        else scores.sum() * 0.0
    )

    total = (
        weights.pointwise * point
        + weights.pairwise * pair
        + weights.listwise * listwise
    )

    # Reported unweighted, so the numbers say how well each provenance is
    # actually fitted rather than how much it was allowed to matter.
    per_set, counted = pointwise_per_set(scores, pref, conf, mask)
    by_provenance: dict[str, float] = {}
    for source in sorted(set(provenance)):
        rows = [
            index
            for index, key in enumerate(provenance)
            if key == source and bool(counted[index])
        ]
        if rows:
            index = torch.tensor(rows, device=scores.device, dtype=torch.long)
            by_provenance[source] = float(per_set[index].mean().detach())

    return LossBreakdown(
        total=total,
        pointwise=point.detach(),
        pairwise=pair.detach(),
        listwise=listwise.detach(),
        by_provenance=by_provenance,
    )
