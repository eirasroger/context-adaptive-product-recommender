from __future__ import annotations

import numpy as np
import pytest

from core.encoding import (
    CHANNEL_INDEX,
    NO_LEVEL,
    AlternativeInput,
    apply_derivations,
    encode_set,
    normalise_reference,
)


def _continuous(registry, category_key, predicate=lambda spec: True):
    category = registry.category(category_key)
    for key in category.token_order:
        indicator = registry.indicator(key)
        member = category.members[key]
        if (
            indicator.value_type == "continuous"
            and not indicator.is_derived
            and member.reference_range is not None
            and predicate(member.reference_range)
        ):
            return key, member.reference_range
    pytest.skip("no matching continuous indicator in the seeded registry")


def _ideal_values(registry, category_key, context_keys):
    from ingest.generators.parametric import ideal_alternative

    return ideal_alternative(registry, category_key, context_keys)


def test_reference_normalisation_uses_the_declared_range(registry, category_key):
    key, spec = _continuous(registry, category_key, lambda s: s.scale == "linear")
    low, saturated_low = normalise_reference(spec.ref_low, spec)
    high, _ = normalise_reference(spec.ref_high, spec)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(1.0)
    assert not saturated_low


def test_values_outside_the_declared_range_are_clipped_and_flagged(registry, category_key):
    key, spec = _continuous(registry, category_key, lambda s: s.scale == "linear")
    beyond = spec.ref_high + (spec.ref_high - spec.ref_low)
    value, saturated = normalise_reference(beyond, spec)
    assert value == pytest.approx(1.0)
    assert saturated


def test_missing_is_distinguishable_from_zero(registry, category_key):
    key, spec = _continuous(registry, category_key, lambda s: s.ref_low <= 0.0 <= s.ref_high)
    present = AlternativeInput("present", {key: 0.0})
    missing = AlternativeInput("missing", {})
    context = registry.category(category_key).default_context_key

    encoded = encode_set(registry, category_key, [present, missing], [context], [])
    token = encoded.token_keys.index(key)

    assert encoded.channels[0, token, CHANNEL_INDEX["present"]] == 1.0
    assert encoded.channels[1, token, CHANNEL_INDEX["present"]] == 0.0


def test_not_applicable_has_no_token_at_all(registry, category_key):
    category = registry.category(category_key)
    held = set(category.token_order)
    unheld = [key for key in registry.indicators if key not in held]

    encoded = encode_set(
        registry,
        category_key,
        [_ideal_values(registry, category_key, [category.default_context_key])],
        [category.default_context_key],
        [],
    )
    assert set(encoded.token_keys) == held
    for key in unheld:
        assert key not in encoded.token_keys


def test_within_set_channel_is_relative_to_the_other_alternatives(registry, category_key):
    key, spec = _continuous(registry, category_key)
    context = registry.category(category_key).default_context_key
    span = spec.ref_high - spec.ref_low

    alternatives = [
        AlternativeInput("low", {key: spec.ref_low}),
        AlternativeInput("mid", {key: spec.ref_low + span / 2}),
        AlternativeInput("high", {key: spec.ref_high}),
    ]
    encoded = encode_set(registry, category_key, alternatives, [context], [])
    token = encoded.token_keys.index(key)
    within = encoded.channels[:, token, CHANNEL_INDEX["v_withinset"]]
    assert within[0] == pytest.approx(0.0)
    assert within[2] == pytest.approx(1.0)
    assert 0.0 < within[1] < 1.0


def test_both_value_channels_are_needed_to_tell_two_sets_apart(registry, category_key):
    """Within-set normalisation alone makes a poor shortlist look like a good one."""
    key, spec = _continuous(registry, category_key)
    context = registry.category(category_key).default_context_key
    span = spec.ref_high - spec.ref_low

    good = [
        AlternativeInput("a", {key: spec.ref_low}),
        AlternativeInput("b", {key: spec.ref_low + span * 0.1}),
    ]
    poor = [
        AlternativeInput("a", {key: spec.ref_high - span * 0.1}),
        AlternativeInput("b", {key: spec.ref_high}),
    ]
    token_index = registry.category(category_key).token_order.index(key)

    good_encoded = encode_set(registry, category_key, good, [context], [])
    poor_encoded = encode_set(registry, category_key, poor, [context], [])

    within = CHANNEL_INDEX["v_withinset"]
    reference = CHANNEL_INDEX["v_ref"]
    assert np.allclose(
        good_encoded.channels[:, token_index, within],
        poor_encoded.channels[:, token_index, within],
    )
    assert not np.allclose(
        good_encoded.channels[:, token_index, reference],
        poor_encoded.channels[:, token_index, reference],
    )


def test_direction_comes_from_the_active_context(registry, category_key):
    category = registry.category(category_key)
    contexts = sorted(category.available_contexts)

    flipped = None
    for key in category.token_order:
        directions = {
            ctx: registry.resolve_direction(category_key, key, [ctx])[0]
            for ctx in contexts
        }
        values = [d for d in directions.values() if d != 0]
        if values and min(values) < 0 < max(values):
            flipped = (key, directions)
            break
    assert flipped is not None

    key, directions = flipped
    positive = next(ctx for ctx, d in directions.items() if d > 0)
    negative = next(ctx for ctx, d in directions.items() if d < 0)
    alternative = _ideal_values(registry, category_key, [positive])

    channel = CHANNEL_INDEX["direction"]
    for context, expected_sign in ((positive, 1), (negative, -1)):
        encoded = encode_set(registry, category_key, [alternative], [context], [])
        token = encoded.token_keys.index(key)
        assert np.sign(encoded.channels[0, token, channel]) == expected_sign


def test_ordinal_levels_use_their_declared_position(registry, category_key):
    category = registry.category(category_key)
    ordinal = [
        key
        for key in category.token_order
        if registry.indicator(key).value_type == "ordinal"
    ]
    if not ordinal:
        pytest.skip("no ordinal indicator in the seeded registry")

    key = ordinal[0]
    indicator = registry.indicator(key)
    context = category.default_context_key

    alternatives = [
        AlternativeInput(level.key, {}, {key: level.key}) for level in indicator.levels
    ]
    encoded = encode_set(registry, category_key, alternatives, [context], [])
    token = encoded.token_keys.index(key)

    observed = encoded.channels[:, token, CHANNEL_INDEX["v_ref"]]
    declared = [level.normalised_position for level in indicator.levels]
    assert np.allclose(observed, declared, atol=1e-6)
    assert (encoded.level_slots[:, token] != NO_LEVEL).all()


def test_derived_indicators_are_computed_from_their_sources(registry, category_key):
    category = registry.category(category_key)
    derived = [
        key for key in category.token_order if registry.indicator(key).is_derived
    ]
    if not derived:
        pytest.skip("no derived indicator in the seeded registry")

    key = derived[0]
    indicator = registry.indicator(key)
    values = {source: 0.02 * (index + 1) for index, (source, _) in enumerate(indicator.sources)}
    filled = apply_derivations(
        registry, category_key, AlternativeInput("a", values)
    )
    expected = sum(
        coefficient * values[source] for source, coefficient in indicator.sources
    )
    assert filled.values[key] == pytest.approx(expected)


def test_a_derived_indicator_is_absent_when_a_source_is_missing(registry, category_key):
    category = registry.category(category_key)
    derived = [
        key for key in category.token_order if registry.indicator(key).is_derived
    ]
    if not derived:
        pytest.skip("no derived indicator in the seeded registry")

    key = derived[0]
    sources = registry.indicator(key).sources
    partial = {source: 0.01 for source, _ in sources[:-1]}
    filled = apply_derivations(registry, category_key, AlternativeInput("a", partial))
    assert key not in filled.values


def test_encoding_refuses_a_context_the_category_cannot_use(registry, category_key):
    category = registry.category(category_key)
    unavailable = [
        key for key in registry.contexts if key not in category.available_contexts
    ]
    if not unavailable:
        pytest.skip("every context is available for this category")

    with pytest.raises(ValueError):
        encode_set(
            registry,
            category_key,
            [AlternativeInput("a", {})],
            [unavailable[0]],
            [],
        )
