"""Evaluation metrics. Gap fidelity and band placement gate a release; the rest are reported."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.prepare import Prepared

#: Score difference below which two alternatives count as tied.
TIE_EPSILON = 0.03


@dataclass
class Predictions:
    """Model output gathered alongside the labels, one row per alternative."""

    set_index: np.ndarray
    position: np.ndarray
    predicted: np.ndarray
    actual: np.ndarray
    confidence: np.ndarray

    def __len__(self) -> int:
        return len(self.predicted)


@torch.no_grad()
def predict(model, loader: DataLoader, device: str | torch.device = "cpu") -> Predictions:
    model.eval()
    set_index, position, predicted, actual, confidence = [], [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        scores = model(batch).cpu().numpy()
        pref = batch.pref.cpu().numpy()
        conf = batch.conf.cpu().numpy()
        mask = batch.alternative_mask.cpu().numpy()
        indices = batch.set_indices.cpu().numpy()

        valid = mask & ~np.isnan(pref)
        rows, cols = np.nonzero(valid)
        set_index.append(indices[rows])
        position.append(cols)
        predicted.append(scores[rows, cols])
        actual.append(pref[rows, cols])
        confidence.append(conf[rows, cols])

    if not predicted:
        empty = np.array([], dtype=np.float64)
        return Predictions(empty, empty, empty, empty, empty)

    return Predictions(
        set_index=np.concatenate(set_index),
        position=np.concatenate(position),
        predicted=np.concatenate(predicted),
        actual=np.concatenate(actual),
        confidence=np.concatenate(confidence),
    )


def _grouped(predictions: Predictions) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for row, index in enumerate(predictions.set_index):
        groups[int(index)].append(row)
    return {
        index: (predictions.predicted[rows], predictions.actual[rows])
        for index, rows in groups.items()
    }


def gap_fidelity(predictions: Predictions) -> float:
    """Mean absolute error of within-set score differences. Lower is better."""
    errors = []
    for predicted, actual in _grouped(predictions).values():
        if len(predicted) < 2:
            continue
        upper = np.triu_indices(len(predicted), k=1)
        predicted_gaps = (predicted[:, None] - predicted[None, :])[upper]
        actual_gaps = (actual[:, None] - actual[None, :])[upper]
        errors.append(float(np.mean(np.abs(predicted_gaps - actual_gaps))))
    return float(np.mean(errors)) if errors else float("nan")


def band_placement(predictions: Predictions) -> float:
    """Mean absolute error of the set's mean score. Lower is better."""
    errors = [
        abs(float(predicted.mean()) - float(actual.mean()))
        for predicted, actual in _grouped(predictions).values()
    ]
    return float(np.mean(errors)) if errors else float("nan")


def pointwise_mae(predictions: Predictions) -> float:
    if not len(predictions):
        return float("nan")
    return float(np.mean(np.abs(predictions.predicted - predictions.actual)))


def top1_agreement(predictions: Predictions) -> float:
    """How often the model's first choice is the label's first choice."""
    hits, total = 0, 0
    for predicted, actual in _grouped(predictions).values():
        if len(predicted) < 2:
            continue
        total += 1
        # Picking any of several tied best alternatives counts as agreement.
        best = actual.max()
        if actual[int(np.argmax(predicted))] >= best - 1e-9:
            hits += 1
    return hits / total if total else float("nan")


def tie_tolerant_tau(predictions: Predictions, epsilon: float = TIE_EPSILON) -> float:
    """Rank correlation that skips pairs whose labels are within ``epsilon``."""
    concordant = discordant = 0
    for predicted, actual in _grouped(predictions).values():
        if len(predicted) < 2:
            continue
        upper = np.triu_indices(len(predicted), k=1)
        actual_gaps = (actual[:, None] - actual[None, :])[upper]
        predicted_gaps = (predicted[:, None] - predicted[None, :])[upper]

        decisive = np.abs(actual_gaps) >= epsilon
        agree = np.sign(actual_gaps[decisive]) == np.sign(predicted_gaps[decisive])
        concordant += int(agree.sum())
        discordant += int((~agree).sum())

    total = concordant + discordant
    return (concordant - discordant) / total if total else float("nan")


@dataclass
class MetricSet:
    n_sets: int
    n_alternatives: int
    gap_fidelity: float
    band_placement: float
    pointwise_mae: float
    top1_agreement: float
    tie_tolerant_tau: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n_sets": self.n_sets,
            "n_alternatives": self.n_alternatives,
            "gap_fidelity": self.gap_fidelity,
            "band_placement": self.band_placement,
            "pointwise_mae": self.pointwise_mae,
            "top1_agreement": self.top1_agreement,
            "tie_tolerant_tau": self.tie_tolerant_tau,
        }


def compute(predictions: Predictions) -> MetricSet:
    return MetricSet(
        n_sets=len(set(predictions.set_index.tolist())),
        n_alternatives=len(predictions),
        gap_fidelity=gap_fidelity(predictions),
        band_placement=band_placement(predictions),
        pointwise_mae=pointwise_mae(predictions),
        top1_agreement=top1_agreement(predictions),
        tie_tolerant_tau=tie_tolerant_tau(predictions),
    )


STRATA = ("category", "provenance", "context", "stakeholder", "set_size")


def strata_of(prepared: Prepared, set_index: int, kind: str) -> tuple[str, ...]:
    """The stratum labels of a comparison set; a set with two contexts counts under both."""
    if kind == "category":
        return (prepared.category_keys[int(prepared.set_category[set_index])],)
    if kind == "provenance":
        return (prepared.set_provenance[set_index],)
    if kind == "set_size":
        return (f"n={int(prepared.set_count[set_index])}",)
    if kind == "context":
        slots = prepared.combo_context_slots[int(prepared.set_combo[set_index])]
        return tuple(f"context_slot={slot}" for slot in slots) or ("none",)
    if kind == "stakeholder":
        slots = prepared.set_stakeholder_slots[set_index]
        return tuple(f"stakeholder_slot={slot}" for slot in slots) or ("none",)
    raise ValueError(f"unknown stratum {kind!r}")


def stratified(
    predictions: Predictions,
    prepared: Prepared,
    kinds: Iterable[str] = STRATA,
    registry=None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Metrics per stratum, so a regression in one cell cannot hide in the mean."""
    rows_by_set: dict[int, list[int]] = defaultdict(list)
    for row, index in enumerate(predictions.set_index):
        rows_by_set[int(index)].append(row)

    names = _slot_names(registry) if registry is not None else {}
    report: dict[str, dict[str, dict[str, float]]] = {}

    for kind in kinds:
        buckets: dict[str, list[int]] = defaultdict(list)
        for set_index, rows in rows_by_set.items():
            for label in strata_of(prepared, set_index, kind):
                buckets[names.get(label, label)].extend(rows)

        report[kind] = {}
        for label, rows in sorted(buckets.items()):
            subset = Predictions(
                set_index=predictions.set_index[rows],
                position=predictions.position[rows],
                predicted=predictions.predicted[rows],
                actual=predictions.actual[rows],
                confidence=predictions.confidence[rows],
            )
            report[kind][label] = compute(subset).as_dict()

    return report


def _slot_names(registry) -> dict[str, str]:
    names = {}
    for key, context in registry.contexts.items():
        names[f"context_slot={context.slot}"] = key
    for key, stakeholder in registry.stakeholders.items():
        names[f"stakeholder_slot={stakeholder.slot}"] = key
    return names


def family_breakdown(
    predictions: Predictions, prepared: Prepared, registry
) -> dict[str, dict[str, float]]:
    """Metrics per indicator family, over control cases whose varied indicator is known."""
    rows_by_family: dict[str, list[int]] = defaultdict(list)
    for row, index in enumerate(predictions.set_index):
        generator = prepared.set_generator[int(index)]
        if not generator:
            continue
        indicator = registry.indicators.get(generator)
        if indicator is None:
            continue
        rows_by_family[indicator.family_key].append(row)

    out = {}
    for family, rows in sorted(rows_by_family.items()):
        subset = Predictions(
            set_index=predictions.set_index[rows],
            position=predictions.position[rows],
            predicted=predictions.predicted[rows],
            actual=predictions.actual[rows],
            confidence=predictions.confidence[rows],
        )
        out[family] = compute(subset).as_dict()
    return out
