"""Turn a comparison set into indicator tokens.

One token per indicator the category holds. A token carries who it is (an
identity embedding composed from its family and itself) and what it says about
this alternative (its value against the declared reference range, its value
against the other alternatives in the set, whether it is present, whether it is
relevant here, and which way the active context pulls it).

Two properties fall out of this representation and both matter:

* Adding an indicator adds a token, never a column, so no existing weight
  changes meaning. There is no feature vector to reorder.
* "Not applicable to this category" (no token at all) is distinguishable from
  "applicable but missing for this product" (a token with present = 0). A
  fixed-width vector cannot express that difference, and it is the difference
  that makes an open-ended set of categories workable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from core.registry import RangeSpec, Registry

#: Numeric channels on every token, in order. Named so that an ablation can
#: switch one off by name instead of by index.
CHANNELS: tuple[str, ...] = (
    "v_ref",           # value against the declared reference range
    "v_ref_saturated", # 1 when the raw value fell outside that range
    "v_withinset",     # value against the other alternatives in this set
    "v_withinset_defined",
    "present",         # the product has a value for this indicator
    "relevant",        # this indicator counts under the active context
    "direction",       # signed pull from the registry, -1 to +1
    "priority",        # how hard the active context pulls
    "ideal_distance",  # distance to the declared ideal, where non-monotone
    "has_ideal",
)

N_CHANNELS = len(CHANNELS)
CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNELS)}

#: Sentinel for a token whose indicator has no level (continuous or boolean).
NO_LEVEL = -1


@dataclass(frozen=True, slots=True)
class AlternativeInput:
    """One alternative's raw values, keyed by indicator.

    ``values`` holds numbers for continuous and boolean indicators; ``levels``
    holds level keys for ordinal and nominal ones. An indicator absent from both
    is missing, which is a different statement from a value of zero.
    """

    key: str
    values: Mapping[str, float] = None
    levels: Mapping[str, str] = None

    def __post_init__(self):
        object.__setattr__(self, "values", dict(self.values or {}))
        object.__setattr__(self, "levels", dict(self.levels or {}))


@dataclass(frozen=True, slots=True)
class EncodedSet:
    """A comparison set in the form the model consumes."""

    category_key: str
    category_slot: int
    #: (n_alternatives, n_tokens)
    indicator_slots: np.ndarray
    family_slots: np.ndarray
    level_slots: np.ndarray
    #: (n_alternatives, n_tokens, N_CHANNELS)
    channels: np.ndarray
    stakeholder_slots: tuple[int, ...]
    context_slots: tuple[int, ...]
    token_keys: tuple[str, ...]
    alternative_keys: tuple[str, ...]

    @property
    def n_alternatives(self) -> int:
        return self.channels.shape[0]

    @property
    def n_tokens(self) -> int:
        return self.channels.shape[1]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalise_reference(value: float, spec: RangeSpec) -> tuple[float, bool]:
    """Map a raw value into the unit interval using the declared range.

    Returns the normalised value and whether the raw value fell outside the
    range. Out-of-range values are clipped rather than allowed to rescale
    everything else, and the fact that clipping happened is reported to the
    model on its own channel instead of being hidden.
    """
    low, high = spec.ref_low, spec.ref_high
    if spec.scale == "log":
        if value <= 0.0:
            return 0.0, True
        raw = (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    else:
        raw = (value - low) / (high - low)
    clipped = min(1.0, max(0.0, raw))
    return clipped, not (0.0 <= raw <= 1.0)


def _ideal_distance(normalised: float, spec: RangeSpec) -> float | None:
    """Normalised distance to the declared ideal, for non-monotone indicators.

    Some indicators are bad in both directions -- too little and too much are
    each a problem -- and a signed direction cannot express that. An ideal point
    or band can.
    """
    if spec.shape == "monotone":
        return None
    if spec.shape == "ideal_point":
        if spec.ideal_value is None:
            return None
        ideal, _ = normalise_reference(spec.ideal_value, spec)
        return abs(normalised - ideal)
    if spec.ideal_low is None or spec.ideal_high is None:
        return None
    low, _ = normalise_reference(spec.ideal_low, spec)
    high, _ = normalise_reference(spec.ideal_high, spec)
    if normalised < low:
        return low - normalised
    if normalised > high:
        return normalised - high
    return 0.0


def _scalar_value(
    registry: Registry, indicator_key: str, alternative: AlternativeInput
) -> tuple[float | None, str | None]:
    """The comparable scalar for an alternative, plus its level key if any."""
    indicator = registry.indicator(indicator_key)
    if indicator.is_scale:
        level_key = alternative.levels.get(indicator_key)
        if level_key is None:
            return None, None
        level = indicator.level(level_key)
        if indicator.value_type == "nominal":
            # Nominal levels have no order, so there is no scalar to compare;
            # the identity of the level is carried by its embedding instead.
            return None, level_key
        return level.normalised_position, level_key
    value = alternative.values.get(indicator_key)
    return (None, None) if value is None else (float(value), None)


def apply_derivations(
    registry: Registry, category_key: str, alternative: AlternativeInput
) -> AlternativeInput:
    """Fill in derived indicators from their declared sources.

    Derived values are recomputed here rather than stored, so a derivation can
    be corrected in the registry without rewriting the corpus. A derived value
    is present only when every source it needs is present.
    """
    category = registry.category(category_key)
    values = dict(alternative.values)
    for indicator_key in category.token_order:
        indicator = registry.indicator(indicator_key)
        if not indicator.is_derived or not indicator.sources:
            continue
        total = 0.0
        complete = True
        for source_key, coefficient in indicator.sources:
            source_value = alternative.values.get(source_key)
            if source_value is None:
                complete = False
                break
            total += coefficient * float(source_value)
        if complete:
            values[indicator_key] = total
        else:
            values.pop(indicator_key, None)
    return AlternativeInput(key=alternative.key, values=values, levels=alternative.levels)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_set(
    registry: Registry,
    category_key: str,
    alternatives: Sequence[AlternativeInput],
    context_keys: Sequence[str],
    stakeholder_keys: Sequence[str],
    derive: bool = True,
) -> EncodedSet:
    """Encode one comparison set into tokens."""
    category = registry.category(category_key)
    token_keys = category.token_order
    n_alts = len(alternatives)
    n_tokens = len(token_keys)

    if n_alts == 0:
        raise ValueError("a comparison set needs at least one alternative")

    unknown = [key for key in context_keys if key not in category.available_contexts]
    if unknown:
        raise ValueError(
            f"context(s) {unknown} are not available for category {category_key!r}; "
            f"available: {sorted(category.available_contexts)}"
        )

    if derive:
        alternatives = [
            apply_derivations(registry, category_key, alternative)
            for alternative in alternatives
        ]

    indicator_slots = np.zeros((n_alts, n_tokens), dtype=np.int64)
    family_slots = np.zeros((n_alts, n_tokens), dtype=np.int64)
    level_slots = np.full((n_alts, n_tokens), NO_LEVEL, dtype=np.int64)
    channels = np.zeros((n_alts, n_tokens, N_CHANNELS), dtype=np.float32)

    for token_index, indicator_key in enumerate(token_keys):
        indicator = registry.indicator(indicator_key)
        member = category.members[indicator_key]
        spec = member.reference_range

        direction, priority = registry.resolve_direction(
            category_key, indicator_key, context_keys
        )
        relevant = registry.is_relevant(category_key, indicator_key, context_keys)

        scalars: list[float | None] = []
        for alt_index, alternative in enumerate(alternatives):
            scalar, level_key = _scalar_value(registry, indicator_key, alternative)
            scalars.append(scalar)

            indicator_slots[alt_index, token_index] = indicator.slot
            family_slots[alt_index, token_index] = indicator.family_slot
            if level_key is not None:
                level_slots[alt_index, token_index] = indicator.level(level_key).slot

            present = scalar is not None or level_key is not None
            row = channels[alt_index, token_index]
            row[CHANNEL_INDEX["present"]] = float(present)
            row[CHANNEL_INDEX["relevant"]] = float(relevant)
            row[CHANNEL_INDEX["direction"]] = direction
            row[CHANNEL_INDEX["priority"]] = priority

            if scalar is None:
                continue

            if indicator.is_scale:
                # An ordinal level's declared position is already normalised.
                normalised, saturated = float(scalar), False
            elif spec is not None:
                normalised, saturated = normalise_reference(scalar, spec)
            else:
                normalised, saturated = float(scalar), False

            row[CHANNEL_INDEX["v_ref"]] = normalised
            row[CHANNEL_INDEX["v_ref_saturated"]] = float(saturated)

            if spec is not None:
                distance = _ideal_distance(normalised, spec)
                if distance is not None:
                    row[CHANNEL_INDEX["ideal_distance"]] = distance
                    row[CHANNEL_INDEX["has_ideal"]] = 1.0

        _fill_within_set(channels, token_index, scalars)

    return EncodedSet(
        category_key=category_key,
        category_slot=category.slot,
        indicator_slots=indicator_slots,
        family_slots=family_slots,
        level_slots=level_slots,
        channels=channels,
        stakeholder_slots=tuple(
            registry.stakeholders[key].slot for key in stakeholder_keys
        ),
        context_slots=tuple(registry.contexts[key].slot for key in context_keys),
        token_keys=token_keys,
        alternative_keys=tuple(alternative.key for alternative in alternatives),
    )


def _fill_within_set(
    channels: np.ndarray, token_index: int, scalars: Sequence[float | None]
) -> None:
    """Normalise an indicator against the other alternatives in the set.

    Both value channels are needed. Within-set normalisation alone destroys the
    information that an entire shortlist is poor -- the best of a bad lot would
    look excellent. Reference normalisation alone loses resolution when the
    alternatives are clustered closely together.
    """
    present = [value for value in scalars if value is not None]
    if len(present) < 2:
        return
    low, high = min(present), max(present)
    if high <= low:
        return
    span = high - low
    for alt_index, value in enumerate(scalars):
        if value is None:
            continue
        row = channels[alt_index, token_index]
        row[CHANNEL_INDEX["v_withinset"]] = (value - low) / span
        row[CHANNEL_INDEX["v_withinset_defined"]] = 1.0
