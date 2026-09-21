"""The registry holds together, and the invariants it exists to enforce hold."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from db import slots, validate
from db.models import Indicator, Product


def test_seeded_registry_is_valid(seeded):
    assert validate.check(seeded) == []


def test_every_embedded_entity_has_a_contiguous_slot(seeded):
    sizes = slots.table_sizes(seeded)
    for table in slots.SLOTTED_TABLES:
        assert sizes.get(table, 0) > 0, f"{table} has no slots"
    assert validate._check_slots(seeded) == []


def test_slots_are_stable_across_reseeding(seeded):
    """Reapplying a seed must not renumber anything.

    If it did, every trained checkpoint would silently reinterpret its own
    weights the next time somebody ran the seeder.
    """
    from db.seed import seed

    before = {
        (row.table_name, row.entity_key): row.slot
        for row in seeded.query(slots.EmbeddingSlot).all()
    }
    seed(seeded)
    seeded.flush()
    after = {
        (row.table_name, row.entity_key): row.slot
        for row in seeded.query(slots.EmbeddingSlot).all()
    }
    assert before == after


def test_new_entity_appends_rather_than_inserts(seeded):
    """A new indicator takes the next slot and disturbs no existing one."""
    before = {
        row.entity_key: row.slot
        for row in seeded.query(slots.EmbeddingSlot).filter_by(table_name="indicator")
    }
    highest = max(before.values())

    new_slot = slots.allocate(seeded, "indicator", "a_brand_new_indicator")
    assert new_slot == highest + 1

    after = {
        row.entity_key: row.slot
        for row in seeded.query(slots.EmbeddingSlot).filter_by(table_name="indicator")
    }
    for key, slot in before.items():
        assert after[key] == slot

    seeded.rollback()


def test_context_availability_is_derived_from_indicator_membership(registry, category_key):
    """A context is live where its indicators are, and inert where they are not."""
    category = registry.category(category_key)
    assert category.available_contexts

    for context_key in category.available_contexts:
        context = registry.contexts[context_key]
        declarations = context.for_category(category_key)
        if context.is_baseline:
            continue
        assert any(key in category.members for key in declarations)


def test_required_declaration_makes_a_context_inert(registry, category_key):
    """A context requiring an indicator the category lacks must not be available.

    This is the mechanism a specialised requirement relies on to switch itself
    off for categories it does not apply to.
    """
    from core.registry import ContextSpec, DeclarationSpec, _derive_available

    category = registry.category(category_key)
    absent = ContextSpec(
        key="needs_something_absent",
        slot=999,
        display_name="Needs something absent",
        is_baseline=False,
        declarations={
            "*": {
                "an_indicator_this_category_lacks": DeclarationSpec(
                    indicator_key="an_indicator_this_category_lacks",
                    direction=1,
                    priority=0.9,
                    is_required=True,
                    ideal_value=None,
                )
            }
        },
    )
    available = _derive_available(
        category_key,
        set(category.members),
        {**registry.contexts, absent.key: absent},
        category.default_context_key,
        {},
    )
    assert absent.key not in available


def test_opposed_contexts_produce_opposite_directions(registry, category_key):
    """Somewhere in the registry, two contexts must disagree about an indicator.

    Not a property of any one category -- it is the thing the whole design is
    for. If no indicator inverts, there is nothing context-adaptive to learn.
    """
    category = registry.category(category_key)
    inverted = []
    for indicator_key in category.token_order:
        directions = {
            key: registry.resolve_direction(category_key, indicator_key, [key])[0]
            for key in category.available_contexts
        }
        values = [d for d in directions.values() if d != 0]
        if values and min(values) < 0 < max(values):
            inverted.append(indicator_key)
    assert inverted, "no indicator changes direction between contexts"


def test_ordinal_levels_are_ordered(registry):
    for indicator in registry.indicators.values():
        if indicator.value_type != "ordinal":
            continue
        positions = [level.normalised_position for level in indicator.levels]
        assert all(p is not None for p in positions)
        assert positions == sorted(positions)


def test_nominal_indicators_require_a_written_justification(seeded):
    """The registry refuses a nominal declaration nobody has argued for."""
    seeded.add(
        Indicator(
            key="unjustified_nominal",
            display_name="Unjustified",
            family_key=sorted(
                {i.family_key for i in seeded.query(Indicator).all()}
            )[0],
            value_type="nominal",
            definition_text="No order claimed, and no reason given.",
            default_direction=0,
        )
    )
    with pytest.raises(IntegrityError):
        seeded.flush()
    seeded.rollback()


def test_foreign_keys_are_enforced(seeded):
    """Without the pragma these constraints would silently not exist."""
    seeded.add(
        Product(
            id=10_000_001,
            category_key="no_such_category",
            source_key="no_such_source",
            created_at="now",
        )
    )
    with pytest.raises(IntegrityError):
        seeded.flush()
    seeded.rollback()


def test_excluded_indicators_are_never_swept(registry, category_key):
    """An indicator the registry refuses to sweep must not reach the generator.

    A control label asserts that the declared direction holds across the whole
    declared range. Where the registry says it does not, generating one would
    put a claim into the training data that the registry never made.
    """
    from ingest.generators import parametric

    category = registry.category(category_key)
    excluded = {
        key for key in category.token_order if not category.members[key].is_sweepable
    }
    assert excluded, "seed no longer exercises the exclusion path"

    for context_key in sorted(category.available_contexts):
        varied = set(
            parametric.varied_indicators(registry, category_key, [context_key])
        )
        assert not (varied & excluded), sorted(varied & excluded)


def test_every_exclusion_carries_a_reason(registry, category_key):
    """The reason is the point. A silent exclusion is an untested gap nobody sees."""
    category = registry.category(category_key)
    for key, member in category.members.items():
        if not member.is_sweepable:
            assert member.control_note and member.control_note.strip(), key


def test_a_sweepable_indicator_has_a_direction_to_sweep(registry, category_key, seeded):
    """Sweeping an indicator nothing gives a direction would assert nothing."""
    assert validate.check(seeded) == []

    category = registry.category(category_key)
    for key in registry.sweepable(category_key):
        directions = [
            registry.resolve_direction(category_key, key, [context])[0]
            for context in sorted(category.available_contexts)
        ]
        assert any(d != 0 for d in directions), key
