"""One control generator, driven entirely by the registry.

A control case holds every indicator at its ideal value, varies exactly one, and
maps that one to a preference score through a declared relationship. The
predecessor needed a near-duplicate script per indicator; here the indicator,
its direction, its range, its value type and its context-dependence all come
from registry rows, so a new category needs data and rows, never a new
generator.

Two things are fixed here that the per-indicator scripts got wrong:

* **The preference band is stakeholder-conditional.** Holding everything else
  ideal and equal, the varied indicator is the only signal, so the *direction*
  of the label is sound -- but the old generators gave every archetype the same
  spread. Someone who prioritises environmental impact should show a wider
  response to an environmental indicator than someone driven by cost. The
  registry's stakeholder priorities set the width.
* **The band floor moves with the set.** A shortlist where every option is poor
  lands low as a whole, rather than the best of a bad lot scoring near the top.

The output of this module doubles as the behavioural test suite, because the
mapping it applies is known exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from core.encoding import AlternativeInput
from core.registry import Registry

#: How far the preference band can spread, from the least to the most
#: responsive archetype. A stakeholder with no stated interest in an indicator
#: still responds to it -- just far less.
MIN_BAND = 0.12
MAX_BAND = 0.55

#: Where the band sits when the whole set is excellent, and when it is poor.
TOP_ANCHOR = 1.0


@dataclass(frozen=True)
class ProbeCase:
    """One generated comparison set with its known-correct labels."""

    category_key: str
    indicator_key: str
    context_keys: tuple[str, ...]
    stakeholder_key: str
    alternatives: tuple[AlternativeInput, ...]
    #: The varied indicator's raw value per alternative, in the same order.
    varied: tuple[float | str, ...]
    #: Quality of each alternative on the varied indicator, 0 worst to 1 best.
    quality: tuple[float, ...]
    prefs: tuple[float, ...]


# ---------------------------------------------------------------------------
# Ideal values
# ---------------------------------------------------------------------------


def ideal_for(
    registry: Registry,
    category_key: str,
    indicator_key: str,
    context_keys: Sequence[str],
) -> float | str | None:
    """The best value an indicator can take, according to the registry alone.

    Derived from the declared direction and range: where more is better the
    ideal is the top of the range, where less is better it is the bottom, and
    where the indicator is non-monotone it is the declared ideal. An indicator
    the registry gives no direction for and no context pulls on has no ideal,
    and is held at the middle of its range.
    """
    indicator = registry.indicator(indicator_key)
    member = registry.category(category_key).members.get(indicator_key)
    if member is None:
        return None

    if indicator.is_scale:
        positioned = [
            level for level in indicator.levels if level.normalised_position is not None
        ]
        if not positioned:
            return indicator.levels[0].key if indicator.levels else None
        direction, _ = registry.resolve_direction(
            category_key, indicator_key, context_keys
        )
        best = max if direction >= 0 else min
        return best(positioned, key=lambda level: level.normalised_position).key

    spec = member.reference_range
    if spec is None:
        return None

    declared_ideal = registry.declared_ideal(category_key, indicator_key, context_keys)
    if declared_ideal is not None:
        return declared_ideal

    direction, _ = registry.resolve_direction(category_key, indicator_key, context_keys)
    if direction > 0:
        return spec.ref_high
    if direction < 0:
        return spec.ref_low
    return 0.5 * (spec.ref_low + spec.ref_high)


def ideal_alternative(
    registry: Registry,
    category_key: str,
    context_keys: Sequence[str],
    exclude: Iterable[str] = (),
    jitter: float = 0.0,
    rng: random.Random | None = None,
) -> AlternativeInput:
    """An alternative that is ideal on every indicator but the excluded ones."""
    rng = rng or random.Random(0)
    excluded = set(exclude)
    values: dict[str, float] = {}
    levels: dict[str, str] = {}

    category = registry.category(category_key)
    for indicator_key in category.token_order:
        if indicator_key in excluded:
            continue
        indicator = registry.indicator(indicator_key)
        if indicator.is_derived:
            continue  # recomputed from its sources at encode time
        best = ideal_for(registry, category_key, indicator_key, context_keys)
        if best is None:
            continue
        if indicator.is_scale:
            levels[indicator_key] = str(best)
            continue
        value = float(best)
        if jitter:
            spec = category.members[indicator_key].reference_range
            width = (spec.ref_high - spec.ref_low) * jitter
            value = min(spec.ref_high, max(spec.ref_low, value + rng.uniform(-width, width)))
        values[indicator_key] = value

    return AlternativeInput(key="ideal", values=values, levels=levels)


# ---------------------------------------------------------------------------
# Quality and labels
# ---------------------------------------------------------------------------


def quality_of(
    registry: Registry,
    category_key: str,
    indicator_key: str,
    value: float | str,
    context_keys: Sequence[str],
) -> float:
    """How good a value is on this indicator, from 0 (worst) to 1 (best)."""
    indicator = registry.indicator(indicator_key)
    direction, _ = registry.resolve_direction(category_key, indicator_key, context_keys)

    if indicator.is_scale:
        position = indicator.level(str(value)).normalised_position
        if position is None:
            return 0.5
        return position if direction >= 0 else 1.0 - position

    spec = registry.category(category_key).members[indicator_key].reference_range
    if spec is None:
        return 0.5

    from core.encoding import normalise_reference

    normalised, _ = normalise_reference(float(value), spec)

    if spec.shape != "monotone":
        from core.encoding import _ideal_distance

        distance = _ideal_distance(normalised, spec)
        return 1.0 if distance is None else max(0.0, 1.0 - 2.0 * distance)

    if direction > 0:
        return normalised
    if direction < 0:
        return 1.0 - normalised
    return 1.0 - abs(normalised - 0.5) * 2.0


def band_width(
    registry: Registry, stakeholder_key: str, indicator_key: str
) -> float:
    """How wide a preference band this archetype spreads over this indicator.

    A stakeholder's declared priority for the indicator itself wins; failing
    that, its priority for the indicator's family. This is the whole fix for a
    generator that previously gave every archetype the same response to
    everything.
    """
    stakeholder = registry.stakeholders[stakeholder_key]
    indicator = registry.indicator(indicator_key)

    priority = stakeholder.indicator_priorities.get(indicator_key)
    if priority is None:
        priority = stakeholder.family_priorities.get(indicator.family_key, 0.5)

    return MIN_BAND + (MAX_BAND - MIN_BAND) * float(priority)


def labels_for(
    qualities: Sequence[float], width: float
) -> tuple[float, ...]:
    """Map qualities to preference scores inside a set-level band.

    The band's ceiling is pulled down by how good the *best* option in the set
    is, so a shortlist of uniformly poor options lands low as a whole. Within
    the band, spacing follows quality.
    """
    if not qualities:
        return ()
    best = max(qualities)
    ceiling = TOP_ANCHOR - (1.0 - best) * (1.0 - width)
    floor = max(0.0, ceiling - width)
    span = ceiling - floor
    best_quality = best if best > 0 else 1.0
    return tuple(
        round(min(1.0, max(0.0, floor + span * (q / best_quality))), 4)
        for q in qualities
    )


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------


def sweep_values(
    registry: Registry, category_key: str, indicator_key: str, steps: int
) -> list[float | str]:
    """Values spanning the indicator's declared range or its declared levels."""
    indicator = registry.indicator(indicator_key)
    if indicator.is_scale:
        return [level.key for level in indicator.levels]

    spec = registry.category(category_key).members[indicator_key].reference_range
    if spec is None:
        raise ValueError(f"{indicator_key} has no reference range in {category_key}")
    if steps < 2:
        raise ValueError("a sweep needs at least two steps")

    step = (spec.ref_high - spec.ref_low) / (steps - 1)
    return [spec.ref_low + step * index for index in range(steps)]


def make_case(
    registry: Registry,
    category_key: str,
    indicator_key: str,
    values: Sequence[float | str],
    context_keys: Sequence[str],
    stakeholder_key: str,
    jitter: float = 0.0,
    rng: random.Random | None = None,
) -> ProbeCase:
    """Build one control case varying a single indicator."""
    rng = rng or random.Random(0)
    base = ideal_alternative(
        registry, category_key, context_keys, exclude={indicator_key}, jitter=jitter, rng=rng
    )
    indicator = registry.indicator(indicator_key)

    alternatives = []
    qualities = []
    for position, value in enumerate(values):
        values_copy = dict(base.values)
        levels_copy = dict(base.levels)
        if indicator.is_scale:
            levels_copy[indicator_key] = str(value)
        else:
            values_copy[indicator_key] = float(value)
        alternatives.append(
            AlternativeInput(
                key=f"alt_{position + 1}", values=values_copy, levels=levels_copy
            )
        )
        qualities.append(
            quality_of(registry, category_key, indicator_key, value, context_keys)
        )

    width = band_width(registry, stakeholder_key, indicator_key)
    return ProbeCase(
        category_key=category_key,
        indicator_key=indicator_key,
        context_keys=tuple(context_keys),
        stakeholder_key=stakeholder_key,
        alternatives=tuple(alternatives),
        varied=tuple(values),
        quality=tuple(qualities),
        prefs=labels_for(qualities, width),
    )


def varied_indicators(
    registry: Registry, category_key: str, context_keys: Sequence[str]
) -> list[str]:
    """Indicators worth varying: those the registry gives a direction to here
    **and** declares safe to sweep.

    Two separate filters, for two separate reasons. An indicator with no
    direction under the active context carries no declared expectation, so
    sweeping it would assert nothing. An indicator marked ``control_mode:
    exclude`` has a direction that does not hold across its whole range -- it
    turns over, the sources disagree, or the question is open -- so sweeping it
    would assert something the registry does not actually claim, and would put
    that claim into the training data as if it were ground truth.
    """
    category = registry.category(category_key)
    out = []
    for indicator_key in category.token_order:
        if registry.indicator(indicator_key).is_derived:
            continue
        if not category.members[indicator_key].is_sweepable:
            continue
        if not registry.is_relevant(category_key, indicator_key, context_keys):
            continue
        direction, _ = registry.resolve_direction(
            category_key, indicator_key, context_keys
        )
        if direction != 0:
            out.append(indicator_key)
    return out
