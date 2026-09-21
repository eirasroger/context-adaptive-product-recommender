"""Registry integrity checks.

The database enforces what a CHECK constraint can express. These are the rules
that span tables, which SQLite cannot enforce on its own: that every continuous
indicator a category holds has a range to normalise against, that an ordered
scale is actually ordered, that a category's own default context is available
for it, and so on.

Run before every release. A registry that fails these is not a registry that
produced a bad number somewhere -- it is one that will quietly change model
behaviour in a way nobody reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from db import slots
from db.models import (
    Category,
    CategoryContextOverride,
    CategoryIndicator,
    CategoryProvenanceWeight,
    Context,
    ContextDeclaration,
    Indicator,
    IndicatorDerivation,
    IndicatorFamily,
    IndicatorLevel,
    IndicatorReferenceRange,
    Provenance,
    Stakeholder,
    StakeholderPriority,
    WILDCARD,
)

WEIGHT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Problem:
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.detail}"


class RegistryInvalid(Exception):
    def __init__(self, problems: list[Problem]):
        self.problems = problems
        super().__init__(
            f"{len(problems)} registry problem(s):\n"
            + "\n".join(f"  {p}" for p in problems)
        )


def _all(session: Session, model) -> list:
    return list(session.execute(select(model)).scalars())


def check(session: Session) -> list[Problem]:
    """Return every rule violation found; empty means the registry is sound."""
    problems: list[Problem] = []

    indicators = {i.key: i for i in _all(session, Indicator)}
    families = {f.key for f in _all(session, IndicatorFamily)}
    categories = {c.key: c for c in _all(session, Category)}
    contexts = {c.key: c for c in _all(session, Context)}
    provenances = {p.key for p in _all(session, Provenance)}
    stakeholders = {s.key for s in _all(session, Stakeholder)}

    memberships: dict[str, dict[str, CategoryIndicator]] = {
        key: {} for key in categories
    }
    for row in _all(session, CategoryIndicator):
        memberships.setdefault(row.category_key, {})[row.indicator_key] = row

    ranges = {
        (r.category_key, r.indicator_key): r
        for r in _all(session, IndicatorReferenceRange)
    }

    # -- indicators ---------------------------------------------------------
    for key, indicator in indicators.items():
        if indicator.family_key not in families:
            problems.append(
                Problem("family_exists", f"indicator {key} names unknown family")
            )

        levels = sorted(
            (
                level
                for level in _all(session, IndicatorLevel)
                if level.indicator_key == key
            ),
            key=lambda level: level.level_index,
        )
        if indicator.value_type in ("ordinal", "nominal"):
            if len(levels) < 2:
                problems.append(
                    Problem(
                        "levels_declared",
                        f"{indicator.value_type} indicator {key} has "
                        f"{len(levels)} level(s); at least 2 are required",
                    )
                )
        elif levels:
            problems.append(
                Problem(
                    "levels_only_for_scales",
                    f"{indicator.value_type} indicator {key} declares levels",
                )
            )

        if indicator.value_type == "ordinal":
            positions = [level.normalised_position for level in levels]
            if any(position is None for position in positions):
                problems.append(
                    Problem(
                        "ordinal_positions",
                        f"ordinal indicator {key} has a level without a "
                        "declared position; an ordered scale must say where its "
                        "levels sit, since ordered scales are rarely evenly spaced",
                    )
                )
            elif any(a >= b for a, b in zip(positions, positions[1:])):
                problems.append(
                    Problem(
                        "ordinal_monotone",
                        f"ordinal indicator {key} has positions that do not "
                        "increase with level index",
                    )
                )

        if indicator.value_type == "nominal":
            for level in levels:
                if level.normalised_position is not None:
                    problems.append(
                        Problem(
                            "nominal_no_position",
                            f"nominal indicator {key} level {level.level_key} "
                            "declares a position; nominal levels have no order",
                        )
                    )

    # -- derivations --------------------------------------------------------
    derivations: dict[str, list[IndicatorDerivation]] = {}
    for row in _all(session, IndicatorDerivation):
        derivations.setdefault(row.derived_key, []).append(row)

    for key, indicator in indicators.items():
        if indicator.is_derived and not derivations.get(key):
            problems.append(
                Problem("derivation_declared", f"derived indicator {key} has no sources")
            )
        if not indicator.is_derived and derivations.get(key):
            problems.append(
                Problem(
                    "derivation_flagged",
                    f"indicator {key} has sources but is not marked derived",
                )
            )
        for row in derivations.get(key, []):
            if row.source_key not in indicators:
                problems.append(
                    Problem(
                        "derivation_source",
                        f"{key} derives from unknown indicator {row.source_key}",
                    )
                )
            elif indicators[row.source_key].is_derived:
                problems.append(
                    Problem(
                        "derivation_not_chained",
                        f"{key} derives from {row.source_key}, which is itself "
                        "derived; derivations are one level deep by design",
                    )
                )

    # -- categories ---------------------------------------------------------
    for key, category in categories.items():
        members = memberships.get(key, {})
        if not members:
            problems.append(Problem("category_nonempty", f"category {key} holds no indicators"))

        for indicator_key in members:
            if indicator_key not in indicators:
                problems.append(
                    Problem(
                        "membership_exists",
                        f"category {key} holds unknown indicator {indicator_key}",
                    )
                )
                continue
            indicator = indicators[indicator_key]
            if indicator.value_type == "continuous":
                if (key, indicator_key) not in ranges:
                    problems.append(
                        Problem(
                            "range_declared",
                            f"category {key} holds continuous indicator "
                            f"{indicator_key} with no declared reference range",
                        )
                    )
            if indicator.is_derived:
                for row in derivations.get(indicator_key, []):
                    if row.source_key not in members:
                        problems.append(
                            Problem(
                                "derivation_sources_held",
                                f"category {key} holds derived {indicator_key} "
                                f"but not its source {row.source_key}",
                            )
                        )

        weights = {
            row.provenance_key: row.weight
            for row in _all(session, CategoryProvenanceWeight)
            if row.category_key == key
        }
        missing = provenances - set(weights)
        if missing:
            problems.append(
                Problem(
                    "provenance_weights_complete",
                    f"category {key} has no loss weight for {sorted(missing)}",
                )
            )
        total = sum(weights.values())
        if weights and abs(total - 1.0) > WEIGHT_TOLERANCE:
            problems.append(
                Problem(
                    "provenance_weights_sum",
                    f"category {key} provenance weights sum to {total:.6f}, not 1",
                )
            )

        if category.default_context_key not in contexts:
            problems.append(
                Problem(
                    "default_context_exists",
                    f"category {key} names unknown default context "
                    f"{category.default_context_key}",
                )
            )
        elif category.default_context_key not in available_contexts(session, key):
            problems.append(
                Problem(
                    "default_context_available",
                    f"category {key} names default context "
                    f"{category.default_context_key}, which is not available for it",
                )
            )

    # -- contexts -----------------------------------------------------------
    declarations: dict[str, list[ContextDeclaration]] = {}
    for row in _all(session, ContextDeclaration):
        declarations.setdefault(row.context_key, []).append(row)

    for key, ctx in contexts.items():
        rows = declarations.get(key, [])
        if not rows and not ctx.is_baseline:
            problems.append(
                Problem(
                    "context_declares",
                    f"context {key} declares over nothing and is not a baseline, "
                    "so it would be silently available everywhere",
                )
            )
        for row in rows:
            if row.indicator_key not in indicators:
                problems.append(
                    Problem(
                        "declaration_indicator",
                        f"context {key} declares over unknown indicator "
                        f"{row.indicator_key}",
                    )
                )
            if row.category_key != WILDCARD and row.category_key not in categories:
                problems.append(
                    Problem(
                        "declaration_category",
                        f"context {key} has a declaration scoped to unknown "
                        f"category {row.category_key}",
                    )
                )

    for row in _all(session, CategoryContextOverride):
        if row.category_key not in categories:
            problems.append(
                Problem("override_category", f"override names unknown category {row.category_key}")
            )
        if row.context_key not in contexts:
            problems.append(
                Problem("override_context", f"override names unknown context {row.context_key}")
            )

    # -- stakeholders -------------------------------------------------------
    for row in _all(session, StakeholderPriority):
        if row.stakeholder_key not in stakeholders:
            problems.append(
                Problem(
                    "priority_stakeholder",
                    f"priority names unknown stakeholder {row.stakeholder_key}",
                )
            )
        known = families if row.target_kind == "family" else set(indicators)
        if row.target_key not in known:
            problems.append(
                Problem(
                    "priority_target",
                    f"stakeholder {row.stakeholder_key} has a priority for "
                    f"unknown {row.target_kind} {row.target_key}",
                )
            )

    # -- slots --------------------------------------------------------------
    problems.extend(_check_slots(session))

    return problems


def _check_slots(session: Session) -> list[Problem]:
    """Every embedded entity has a slot, and slots run contiguously from zero.

    Contiguity is not merely tidiness: the embedding tables are sized from the
    highest slot, so a gap would allocate a row of weights that nothing ever
    trains.
    """
    problems: list[Problem] = []
    expected = {
        "indicator_family": {f.key for f in _all(session, IndicatorFamily)},
        "indicator": {i.key for i in _all(session, Indicator)},
        "indicator_level": {
            f"{level.indicator_key}::{level.level_key}"
            for level in _all(session, IndicatorLevel)
        },
        "context": {c.key for c in _all(session, Context)},
        "stakeholder": {s.key for s in _all(session, Stakeholder)},
        "category": {c.key for c in _all(session, Category)},
    }

    from db.models import EmbeddingSlot

    for table, keys in expected.items():
        rows = [
            row for row in _all(session, EmbeddingSlot) if row.table_name == table
        ]
        allocated = {row.entity_key for row in rows}
        for missing in sorted(keys - allocated):
            problems.append(
                Problem("slot_allocated", f"{table}.{missing} has no embedding slot")
            )
        numbers = sorted(row.slot for row in rows)
        if numbers and numbers != list(range(len(numbers))):
            problems.append(
                Problem(
                    "slot_contiguous",
                    f"{table} slots are not contiguous from zero: {numbers}",
                )
            )
    return problems


def available_contexts(session: Session, category_key: str) -> set[str]:
    """Which contexts are live for a category.

    Derived, not maintained: a context is available when the category holds the
    indicators the context points at. A declaration marked required must be
    held, which is what makes a specialised requirement inert for categories it
    does not apply to. An explicit override wins over the derivation, for the
    case where a category happens to hold the right indicators but the context
    is meaningless for it anyway.
    """
    category = session.get(Category, category_key)
    if category is None:
        raise LookupError(f"unknown category {category_key!r}")

    held = {
        row.indicator_key
        for row in _all(session, CategoryIndicator)
        if row.category_key == category_key
    }

    overrides = {
        row.context_key: bool(row.is_applicable)
        for row in _all(session, CategoryContextOverride)
        if row.category_key == category_key
    }

    declarations: dict[str, list[ContextDeclaration]] = {}
    for row in _all(session, ContextDeclaration):
        if row.category_key in (WILDCARD, category_key):
            declarations.setdefault(row.context_key, []).append(row)

    available: set[str] = set()
    for ctx in _all(session, Context):
        if not ctx.is_active:
            continue
        if ctx.key in overrides:
            if overrides[ctx.key]:
                available.add(ctx.key)
            continue
        if ctx.is_baseline and category.default_context_key == ctx.key:
            available.add(ctx.key)
            continue
        rows = declarations.get(ctx.key, [])
        if not rows:
            continue
        required_held = all(
            row.indicator_key in held for row in rows if row.is_required
        )
        any_held = any(row.indicator_key in held for row in rows)
        if required_held and any_held:
            available.add(ctx.key)
    return available


def require_valid(session: Session) -> None:
    problems = check(session)
    if problems:
        raise RegistryInvalid(problems)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check registry integrity.")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    from db.session import create_db_engine, session_scope

    with session_scope(create_db_engine(args.db)) as session:
        problems = check(session)
        categories = [c.key for c in _all(session, Category)]
        availability = {key: sorted(available_contexts(session, key)) for key in categories}
        sizes = slots.table_sizes(session)

    for problem in problems:
        print(problem)
    if problems:
        raise SystemExit(1)

    print("registry ok")
    for key, contexts in availability.items():
        print(f"  {key}: contexts {contexts}")
    print(f"  embedding tables {sizes}")


if __name__ == "__main__":
    main()
