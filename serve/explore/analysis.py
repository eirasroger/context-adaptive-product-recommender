"""Analyses behind the explorer.

Every analysis is generated from the registry, so a new category becomes
explorable as soon as its rows exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.encoding import AlternativeInput, encode_set
from core.scoring import Scorer, item, pad
from core.registry import Registry
from ingest.generators import parametric

DEFAULT_STEPS = 21


def score(
    scorer: Scorer,
    registry: Registry,
    category_key: str,
    alternatives: Sequence[AlternativeInput],
    context_keys: Sequence[str],
    stakeholder_keys: Sequence[str],
) -> np.ndarray:
    encoded = encode_set(
        registry, category_key, alternatives, context_keys, stakeholder_keys
    )
    return scorer(pad([item(encoded, category_key)]))[0, : len(alternatives)]


def score_many(
    scorer: Scorer,
    registry: Registry,
    category_key: str,
    shortlists: Sequence[Sequence[AlternativeInput]],
    context_keys: Sequence[str],
    stakeholder_keys: Sequence[str],
    target: int,
    chunk: int = 256,
) -> np.ndarray:
    """Score many shortlists at once, returning one alternative's score from each.

    SHAP asks for hundreds of variants of the same comparison. Encoding them
    one at a time and running one forward pass each is the difference between
    seconds and tens of seconds, and nothing about the model requires it.
    """
    out = np.zeros(len(shortlists), dtype=np.float64)
    for start in range(0, len(shortlists), chunk):
        window = shortlists[start : start + chunk]
        items = [
            item(
                encode_set(registry, category_key, shortlist, context_keys, stakeholder_keys),
                category_key,
            )
            for shortlist in window
        ]
        out[start : start + len(window)] = scorer(pad(items))[:, target]
    return out


@dataclass(frozen=True)
class Series:
    label: str
    points: list[float]


def response_curve(
    scorer: Scorer,
    registry: Registry,
    category_key: str,
    indicator_key: str,
    context_keys: Sequence[str],
    stakeholder_keys: Sequence[str],
    steps: int = DEFAULT_STEPS,
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
            scorer,
            registry,
            category_key,
            case.alternatives,
            context_keys,
            [stakeholder_key],
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
    scorer: Scorer,
    registry: Registry,
    category_key: str,
    alternatives: Sequence[AlternativeInput],
    settings: Sequence[tuple[str, list[str], list[str]]],
) -> list[dict]:
    rows = []
    for label, context_keys, stakeholder_keys in settings:
        scores = score(
            scorer,
            registry,
            category_key,
            alternatives,
            context_keys,
            stakeholder_keys,
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
    scorer: Scorer,
    registry: Registry,
    category_key: str,
    alternatives: Sequence[AlternativeInput],
    stakeholder_keys: Sequence[str],
) -> dict:
    """The same shortlist scored under every context the category allows."""
    category = registry.category(category_key)
    contexts = sorted(category.available_contexts)
    rows = _matrix(
        scorer,
        registry,
        category_key,
        alternatives,
        [(key, [key], list(stakeholder_keys)) for key in contexts],
    )
    return {
        "axis": "context",
        "alternatives": [a.key for a in alternatives],
        "stakeholders": list(stakeholder_keys),
        "rows": rows,
        "changes_winner": len({row["winner"] for row in rows}) > 1,
    }


def stakeholder_sensitivity(
    scorer: Scorer,
    registry: Registry,
    category_key: str,
    alternatives: Sequence[AlternativeInput],
    context_keys: Sequence[str],
) -> dict:
    """The same shortlist scored for every stakeholder archetype."""
    stakeholders = sorted(registry.stakeholders)
    rows = _matrix(
        scorer,
        registry,
        category_key,
        alternatives,
        [(key, list(context_keys), [key]) for key in stakeholders],
    )
    return {
        "axis": "stakeholder",
        "alternatives": [a.key for a in alternatives],
        "contexts": list(context_keys),
        "rows": rows,
        "changes_winner": len({row["winner"] for row in rows}) > 1,
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
