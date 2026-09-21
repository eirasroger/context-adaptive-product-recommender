"""Analyses behind the explorer.

Every analysis is generated from the registry, so a new category becomes
explorable as soon as its rows exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from core.dataset import collate
from core.encoding import AlternativeInput, encode_set
from core.registry import Registry
from ingest.generators import parametric

DEFAULT_STEPS = 21


@torch.no_grad()
def score(
    model,
    registry: Registry,
    category_key: str,
    alternatives: Sequence[AlternativeInput],
    context_keys: Sequence[str],
    stakeholder_keys: Sequence[str],
    device: str | torch.device = "cpu",
) -> np.ndarray:
    encoded = encode_set(
        registry, category_key, alternatives, context_keys, stakeholder_keys
    )
    n = len(alternatives)
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
        "pref": np.full(n, np.nan, dtype=np.float32),
        "conf": np.full(n, np.nan, dtype=np.float32),
        "provenance": "control",
    }
    model.eval()
    batch = collate([item]).to(device)
    return model(batch)[0, :n].cpu().numpy()


@torch.no_grad()
def score_many(
    model,
    registry: Registry,
    category_key: str,
    shortlists: Sequence[Sequence[AlternativeInput]],
    context_keys: Sequence[str],
    stakeholder_keys: Sequence[str],
    target: int,
    device: str | torch.device = "cpu",
    chunk: int = 256,
) -> np.ndarray:
    """Score many shortlists at once, returning one alternative's score from each.

    SHAP asks for hundreds of variants of the same comparison. Encoding them
    one at a time and running one forward pass each is the difference between
    seconds and tens of seconds, and nothing about the model requires it.
    """
    out = np.zeros(len(shortlists), dtype=np.float64)
    model.eval()

    for start in range(0, len(shortlists), chunk):
        window = shortlists[start : start + chunk]
        items = []
        for shortlist in window:
            encoded = encode_set(
                registry, category_key, shortlist, context_keys, stakeholder_keys
            )
            n = len(shortlist)
            items.append(
                {
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
            )
        batch = collate(items).to(device)
        scores = model(batch)[:, target].cpu().numpy()
        out[start : start + len(window)] = scores

    return out


@dataclass(frozen=True)
class Series:
    label: str
    points: list[float]


def response_curve(
    model,
    registry: Registry,
    category_key: str,
    indicator_key: str,
    context_keys: Sequence[str],
    stakeholder_keys: Sequence[str],
    steps: int = DEFAULT_STEPS,
    device: str | torch.device = "cpu",
) -> dict:
    """How the score moves as one indicator moves, everything else held ideal.

    One curve per stakeholder, so the spread between curves shows how much the
    archetype changes the model's sensitivity to that indicator.
    """
    category = registry.category(category_key)
    indicator = registry.indicator(indicator_key)
    values = parametric.sweep_values(registry, category_key, indicator_key, steps)

    series: list[dict] = []
    expected: list[float] = []
    for stakeholder_key in stakeholder_keys:
        case = parametric.make_case(
            registry,
            category_key,
            indicator_key,
            values,
            list(context_keys),
            stakeholder_key,
        )
        scores = score(
            model,
            registry,
            category_key,
            case.alternatives,
            context_keys,
            [stakeholder_key],
            device,
        )
        series.append(
            {"label": stakeholder_key, "points": [round(float(v), 4) for v in scores]}
        )
        if not expected:
            expected = [round(float(v), 4) for v in case.prefs]

    direction, priority = registry.resolve_direction(
        category_key, indicator_key, context_keys
    )
    spec = category.members[indicator_key].reference_range

    return {
        "category": category_key,
        "indicator": indicator_key,
        "display_name": indicator.display_name,
        "family": indicator.family_key,
        "unit": indicator.unit,
        "value_type": indicator.value_type,
        "definition": indicator.definition_text,
        "contexts": list(context_keys),
        "direction": direction,
        "priority": priority,
        "sweepable": category.members[indicator_key].is_sweepable,
        "control_note": category.members[indicator_key].control_note,
        "reference_range": None
        if spec is None
        else {"low": spec.ref_low, "high": spec.ref_high, "scale": spec.scale},
        "values": [v if isinstance(v, str) else round(float(v), 6) for v in values],
        "expected": expected,
        "series": series,
    }


def _matrix(
    model,
    registry: Registry,
    category_key: str,
    alternatives: Sequence[AlternativeInput],
    settings: Sequence[tuple[str, list[str], list[str]]],
    device: str | torch.device,
) -> list[dict]:
    rows = []
    for label, context_keys, stakeholder_keys in settings:
        scores = score(
            model,
            registry,
            category_key,
            alternatives,
            context_keys,
            stakeholder_keys,
            device,
        )
        order = np.argsort(-scores)
        rows.append(
            {
                "label": label,
                "scores": [round(float(v), 4) for v in scores],
                "winner": int(order[0]),
            }
        )
    return rows


def context_sensitivity(
    model,
    registry: Registry,
    category_key: str,
    alternatives: Sequence[AlternativeInput],
    stakeholder_keys: Sequence[str],
    device: str | torch.device = "cpu",
) -> dict:
    """The same shortlist scored under every context the category allows."""
    category = registry.category(category_key)
    contexts = sorted(category.available_contexts)
    rows = _matrix(
        model,
        registry,
        category_key,
        alternatives,
        [(key, [key], list(stakeholder_keys)) for key in contexts],
        device,
    )
    return {
        "axis": "context",
        "alternatives": [a.key for a in alternatives],
        "stakeholders": list(stakeholder_keys),
        "rows": rows,
        "changes_winner": len({row["winner"] for row in rows}) > 1,
    }


def stakeholder_sensitivity(
    model,
    registry: Registry,
    category_key: str,
    alternatives: Sequence[AlternativeInput],
    context_keys: Sequence[str],
    device: str | torch.device = "cpu",
) -> dict:
    """The same shortlist scored for every stakeholder archetype."""
    stakeholders = sorted(registry.stakeholders)
    rows = _matrix(
        model,
        registry,
        category_key,
        alternatives,
        [(key, list(context_keys), [key]) for key in stakeholders],
        device,
    )
    return {
        "axis": "stakeholder",
        "alternatives": [a.key for a in alternatives],
        "contexts": list(context_keys),
        "rows": rows,
        "changes_winner": len({row["winner"] for row in rows}) > 1,
    }


def _without(alternative: AlternativeInput, indicator_key: str) -> AlternativeInput:
    values = {k: v for k, v in alternative.values.items() if k != indicator_key}
    levels = {k: v for k, v in alternative.levels.items() if k != indicator_key}
    return AlternativeInput(key=alternative.key, values=values, levels=levels)


def attribution(
    model,
    registry: Registry,
    category_key: str,
    alternatives: Sequence[AlternativeInput],
    context_keys: Sequence[str],
    stakeholder_keys: Sequence[str],
    device: str | torch.device = "cpu",
) -> dict:
    """What each indicator contributes to each alternative's score.

    Measured by occlusion: withhold one indicator from every alternative, score
    again, and record the shift. The token representation makes this exact,
    because withholding an indicator is a state the model already understands
    rather than an out-of-distribution substitute value.

    A positive number means the indicator raised that alternative's score.
    """
    category = registry.category(category_key)
    baseline = score(
        model, registry, category_key, alternatives, context_keys, stakeholder_keys, device
    )

    supplied = [
        key
        for key in category.token_order
        if not registry.indicator(key).is_derived
        and any(
            key in alt.values or key in alt.levels for alt in alternatives
        )
    ]

    per_indicator = []
    for indicator_key in supplied:
        occluded = [_without(alt, indicator_key) for alt in alternatives]
        shifted = score(
            model, registry, category_key, occluded, context_keys, stakeholder_keys, device
        )
        delta = baseline - shifted
        per_indicator.append(
            {
                "indicator": indicator_key,
                "display_name": registry.indicator(indicator_key).display_name,
                "family": registry.indicator(indicator_key).family_key,
                "contribution": [round(float(v), 4) for v in delta],
                "magnitude": round(float(np.abs(delta).mean()), 4),
            }
        )

    per_indicator.sort(key=lambda row: row["magnitude"], reverse=True)

    families: dict[str, float] = {}
    for row in per_indicator:
        families[row["family"]] = families.get(row["family"], 0.0) + row["magnitude"]

    return {
        "alternatives": [a.key for a in alternatives],
        "baseline": [round(float(v), 4) for v in baseline],
        "contexts": list(context_keys),
        "stakeholders": list(stakeholder_keys),
        "indicators": per_indicator,
        "families": [
            {"family": key, "magnitude": round(value, 4)}
            for key, value in sorted(families.items(), key=lambda kv: -kv[1])
        ],
        "method": "occlusion",
    }


def example_shortlist(
    registry: Registry,
    category_key: str,
    context_keys: Sequence[str],
    size: int = 4,
) -> list[AlternativeInput]:
    """A spread of alternatives to explore when the caller supplies none.

    Built by sweeping the indicator the active context weighs most heavily, so
    the default view shows the model doing something rather than scoring four
    identical products.
    """
    candidates = parametric.varied_indicators(
        registry, category_key, list(context_keys)
    )
    if not candidates:
        candidates = list(registry.sweepable(category_key))

    ranked = sorted(
        candidates,
        key=lambda key: registry.resolve_direction(category_key, key, context_keys)[1],
        reverse=True,
    )
    indicator_key = ranked[0]
    values = parametric.sweep_values(registry, category_key, indicator_key, size)
    case = parametric.make_case(
        registry,
        category_key,
        indicator_key,
        values,
        list(context_keys),
        sorted(registry.stakeholders)[0],
    )
    return [
        AlternativeInput(
            key=f"Option {chr(65 + index)}",
            values=dict(alt.values),
            levels=dict(alt.levels),
        )
        for index, alt in enumerate(case.alternatives)
    ]
