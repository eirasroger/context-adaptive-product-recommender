"""Immutable in-memory registry, built from the live database or a frozen release."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Mapping

import yaml

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

WILDCARD = "*"
SYNTHETIC_PROVENANCE = "synthetic"


@dataclass(frozen=True, slots=True)
class LevelSpec:
    key: str
    slot: int
    index: int
    normalised_position: float | None
    is_disqualifying: bool
    display_name: str


@dataclass(frozen=True, slots=True)
class IndicatorSpec:
    key: str
    slot: int
    display_name: str
    family_key: str
    family_slot: int
    value_type: str
    unit: str | None
    default_direction: int
    is_derived: bool
    definition_text: str
    levels: tuple[LevelSpec, ...] = ()
    sources: tuple[tuple[str, float], ...] = ()

    @property
    def is_scale(self) -> bool:
        return self.value_type in ("ordinal", "nominal")

    def level(self, level_key: str) -> LevelSpec:
        for level in self.levels:
            if level.key == level_key:
                return level
        raise LookupError(f"indicator {self.key} has no level {level_key!r}")


@dataclass(frozen=True, slots=True)
class RangeSpec:
    """The declared interval an indicator is normalised against."""

    ref_low: float
    ref_high: float
    scale: str
    shape: str
    ideal_value: float | None
    ideal_low: float | None
    ideal_high: float | None


@dataclass(frozen=True, slots=True)
class MembershipSpec:
    indicator_key: str
    relevance_mode: str
    direction_override: int | None
    reference_range: RangeSpec | None
    #: ``sweep`` lets control cases vary this indicator alone; ``exclude`` keeps it off the gate.
    control_mode: str = "sweep"
    control_note: str | None = None

    @property
    def is_sweepable(self) -> bool:
        return self.control_mode == "sweep"


@dataclass(frozen=True, slots=True)
class DeclarationSpec:
    indicator_key: str
    direction: int
    priority: float
    is_required: bool
    ideal_value: float | None


@dataclass(frozen=True, slots=True)
class ContextSpec:
    key: str
    slot: int
    display_name: str
    definition_text: str
    is_baseline: bool
    #: category key (or the wildcard) -> indicator key -> declaration
    declarations: Mapping[str, Mapping[str, DeclarationSpec]]

    def for_category(self, category_key: str) -> dict[str, DeclarationSpec]:
        """Declarations in force for a category: general rows, then overrides."""
        merged = dict(self.declarations.get(WILDCARD, {}))
        merged.update(self.declarations.get(category_key, {}))
        return merged


@dataclass(frozen=True, slots=True)
class StakeholderSpec:
    key: str
    slot: int
    display_name: str
    definition_text: str
    domain_scope_text: str | None
    #: read by the control-case generator only
    family_priorities: Mapping[str, float]
    indicator_priorities: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class CategorySpec:
    key: str
    slot: int
    display_name: str
    functional_unit_key: str
    functional_unit_display: str
    eligibility_precondition_text: str
    default_context_key: str
    members: Mapping[str, MembershipSpec]
    provenance_weights: Mapping[str, float]
    available_contexts: frozenset[str]
    #: indicator keys in slot order, stable because slots are append-only
    token_order: tuple[str, ...]

    @property
    def is_preview(self) -> bool:
        """Trained partly on synthetic labels, so its scores restate the registry's own rules."""
        return self.provenance_weights.get(SYNTHETIC_PROVENANCE, 0.0) > 0.0


@dataclass(frozen=True)
class Registry:
    version: str | None
    content_hash: str | None
    families: Mapping[str, int]
    indicators: Mapping[str, IndicatorSpec]
    contexts: Mapping[str, ContextSpec]
    stakeholders: Mapping[str, StakeholderSpec]
    categories: Mapping[str, CategorySpec]
    table_sizes: Mapping[str, int]
    provenances: tuple[str, ...] = ()

    def category(self, key: str) -> CategorySpec:
        try:
            return self.categories[key]
        except KeyError:
            raise LookupError(f"unknown category {key!r}") from None

    def indicator(self, key: str) -> IndicatorSpec:
        try:
            return self.indicators[key]
        except KeyError:
            raise LookupError(f"unknown indicator {key!r}") from None

    @cached_property
    def level_slots(self) -> Mapping[str, int]:
        return {
            f"{indicator.key}::{level.key}": level.slot
            for indicator in self.indicators.values()
            for level in indicator.levels
        }

    def resolve_direction(
        self,
        category_key: str,
        indicator_key: str,
        active_contexts: Iterable[str],
    ) -> tuple[float, float]:
        """Signed direction and priority; opposing contexts net out by priority."""
        category = self.category(category_key)
        weighted = 0.0
        total_priority = 0.0
        peak_priority = 0.0

        for context_key in active_contexts:
            context = self.contexts.get(context_key)
            if context is None:
                continue
            declaration = context.for_category(category_key).get(indicator_key)
            if declaration is None:
                continue
            weighted += declaration.direction * declaration.priority
            total_priority += declaration.priority
            peak_priority = max(peak_priority, declaration.priority)

        if total_priority > 0.0:
            direction = max(-1.0, min(1.0, weighted / total_priority))
            return direction, peak_priority

        member = category.members.get(indicator_key)
        if member is not None and member.direction_override is not None:
            return float(member.direction_override), 0.0
        return float(self.indicator(indicator_key).default_direction), 0.0

    def is_relevant(
        self,
        category_key: str,
        indicator_key: str,
        active_contexts: Iterable[str],
    ) -> bool:
        member = self.category(category_key).members.get(indicator_key)
        if member is None:
            return False
        if member.relevance_mode == "always":
            return True
        return any(
            indicator_key in self.contexts[key].for_category(category_key)
            for key in active_contexts
            if key in self.contexts
        )

    def sweepable(self, category_key: str) -> tuple[str, ...]:
        """Indicators this category allows a control case to vary on its own."""
        category = self.category(category_key)
        return tuple(
            key for key in category.token_order if category.members[key].is_sweepable
        )

    def declared_ideal(
        self,
        category_key: str,
        indicator_key: str,
        active_contexts: Iterable[str],
    ) -> float | None:
        """The ideal value for a non-monotone indicator, context first."""
        for context_key in active_contexts:
            context = self.contexts.get(context_key)
            if context is None:
                continue
            declaration = context.for_category(category_key).get(indicator_key)
            if declaration is not None and declaration.ideal_value is not None:
                return declaration.ideal_value
        member = self.category(category_key).members.get(indicator_key)
        if member is None or member.reference_range is None:
            return None
        spec = member.reference_range
        if spec.shape == "ideal_point":
            return spec.ideal_value
        if spec.shape == "ideal_band" and spec.ideal_low is not None:
            return 0.5 * (spec.ideal_low + spec.ideal_high)
        return None


def _slot_map(rows: Iterable) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        out.setdefault(row["table_name"], {})[row["entity_key"]] = int(row["slot"])
    return out


def from_document(
    document: Mapping[str, list[dict]],
    version: str | None = None,
    content_hash: str | None = None,
) -> Registry:
    slot_tables = _slot_map(document.get("embedding_slot", []))
    families = {row["key"]: slot_tables["indicator_family"][row["key"]] for row in document["indicator_family"]}

    levels_by_indicator: dict[str, list[LevelSpec]] = {}
    for row in document.get("indicator_level", []):
        levels_by_indicator.setdefault(row["indicator_key"], []).append(
            LevelSpec(
                key=row["level_key"],
                slot=slot_tables["indicator_level"][
                    f"{row['indicator_key']}::{row['level_key']}"
                ],
                index=int(row["level_index"]),
                normalised_position=row["normalised_position"],
                is_disqualifying=bool(row["is_disqualifying"]),
                display_name=row["display_name"],
            )
        )

    sources_by_indicator: dict[str, list[tuple[str, float]]] = {}
    for row in document.get("indicator_derivation", []):
        sources_by_indicator.setdefault(row["derived_key"], []).append(
            (row["source_key"], float(row["coefficient"]))
        )

    indicators: dict[str, IndicatorSpec] = {}
    for row in document["indicator"]:
        key = row["key"]
        levels = sorted(levels_by_indicator.get(key, []), key=lambda level: level.index)
        indicators[key] = IndicatorSpec(
            key=key,
            slot=slot_tables["indicator"][key],
            display_name=row["display_name"],
            family_key=row["family_key"],
            family_slot=families[row["family_key"]],
            value_type=row["value_type"],
            unit=row["unit"],
            default_direction=int(row["default_direction"]),
            is_derived=bool(row["is_derived"]),
            definition_text=row["definition_text"],
            levels=tuple(levels),
            sources=tuple(sorted(sources_by_indicator.get(key, []))),
        )

    declarations: dict[str, dict[str, dict[str, DeclarationSpec]]] = {}
    for row in document.get("context_declaration", []):
        declarations.setdefault(row["context_key"], {}).setdefault(
            row["category_key"], {}
        )[row["indicator_key"]] = DeclarationSpec(
            indicator_key=row["indicator_key"],
            direction=int(row["direction"]),
            priority=float(row["priority"]),
            is_required=bool(row["is_required"]),
            ideal_value=row["ideal_value"],
        )

    contexts = {
        row["key"]: ContextSpec(
            key=row["key"],
            slot=slot_tables["context"][row["key"]],
            display_name=row["display_name"],
            definition_text=row["definition_text"],
            is_baseline=bool(row["is_baseline"]),
            declarations=declarations.get(row["key"], {}),
        )
        for row in document["context"]
    }

    family_priorities: dict[str, dict[str, float]] = {}
    indicator_priorities: dict[str, dict[str, float]] = {}
    for row in document.get("stakeholder_priority", []):
        target = (
            family_priorities if row["target_kind"] == "family" else indicator_priorities
        )
        target.setdefault(row["stakeholder_key"], {})[row["target_key"]] = float(
            row["priority"]
        )

    stakeholders = {
        row["key"]: StakeholderSpec(
            key=row["key"],
            slot=slot_tables["stakeholder"][row["key"]],
            display_name=row["display_name"],
            definition_text=row["definition_text"],
            domain_scope_text=row["domain_scope_text"],
            family_priorities=family_priorities.get(row["key"], {}),
            indicator_priorities=indicator_priorities.get(row["key"], {}),
        )
        for row in document["stakeholder"]
    }

    units = {row["key"]: row["display_name"] for row in document["functional_unit"]}

    ranges: dict[tuple[str, str], RangeSpec] = {}
    for row in document.get("indicator_reference_range", []):
        ranges[(row["category_key"], row["indicator_key"])] = RangeSpec(
            ref_low=float(row["ref_low"]),
            ref_high=float(row["ref_high"]),
            scale=row["scale"],
            shape=row["shape"],
            ideal_value=row["ideal_value"],
            ideal_low=row["ideal_low"],
            ideal_high=row["ideal_high"],
        )

    members_by_category: dict[str, dict[str, MembershipSpec]] = {}
    for row in document.get("category_indicator", []):
        members_by_category.setdefault(row["category_key"], {})[
            row["indicator_key"]
        ] = MembershipSpec(
            indicator_key=row["indicator_key"],
            relevance_mode=row["relevance_mode"],
            direction_override=row["direction_override"],
            reference_range=ranges.get((row["category_key"], row["indicator_key"])),
            control_mode=row.get("control_mode", "sweep"),
            control_note=row.get("control_note"),
        )

    weights_by_category: dict[str, dict[str, float]] = {}
    for row in document.get("category_provenance_weight", []):
        weights_by_category.setdefault(row["category_key"], {})[
            row["provenance_key"]
        ] = float(row["weight"])

    overrides: dict[str, dict[str, bool]] = {}
    for row in document.get("category_context_override", []):
        overrides.setdefault(row["category_key"], {})[row["context_key"]] = bool(
            row["is_applicable"]
        )

    categories: dict[str, CategorySpec] = {}
    for row in document["category"]:
        key = row["key"]
        members = members_by_category.get(key, {})
        order = tuple(
            sorted(members, key=lambda indicator_key: indicators[indicator_key].slot)
        )
        categories[key] = CategorySpec(
            key=key,
            slot=slot_tables["category"][key],
            display_name=row["display_name"],
            functional_unit_key=row["functional_unit_key"],
            functional_unit_display=units[row["functional_unit_key"]],
            eligibility_precondition_text=row["eligibility_precondition_text"],
            default_context_key=row["default_context_key"],
            members=members,
            provenance_weights=weights_by_category.get(key, {}),
            available_contexts=frozenset(
                _derive_available(
                    key,
                    set(members),
                    contexts,
                    row["default_context_key"],
                    overrides.get(key, {}),
                )
            ),
            token_order=order,
        )

    table_sizes = {
        table: max(mapping.values()) + 1 for table, mapping in slot_tables.items()
    }

    return Registry(
        version=version,
        content_hash=content_hash,
        families=families,
        indicators=indicators,
        contexts=contexts,
        stakeholders=stakeholders,
        categories=categories,
        table_sizes=table_sizes,
        provenances=tuple(sorted(row["key"] for row in document.get("provenance", []))),
    )


def _derive_available(
    category_key: str,
    held: set[str],
    contexts: Mapping[str, ContextSpec],
    default_context_key: str,
    overrides: Mapping[str, bool],
) -> set[str]:
    """Contexts a category holds the indicators for, plus its baseline and overrides."""
    available: set[str] = set()
    for key, context in contexts.items():
        if key in overrides:
            if overrides[key]:
                available.add(key)
            continue
        if context.is_baseline and key == default_context_key:
            available.add(key)
            continue
        rows = context.for_category(category_key)
        if not rows:
            continue
        required_held = all(
            declaration.indicator_key in held
            for declaration in rows.values()
            if declaration.is_required
        )
        if required_held and any(indicator_key in held for indicator_key in rows):
            available.add(key)
    return available


def from_session(session: Session, version: str | None = None) -> Registry:
    """Build from the live database after validating it."""
    from db import release as release_module
    from db import validate

    validate.require_valid(session)
    document = release_module.export(session)
    return from_document(
        document, version=version, content_hash=release_module.content_hash(document)
    )


def from_release(session: Session, version: str) -> Registry:
    """Rehydrate the exact registry a given release froze."""
    from db.models import RegistryRelease

    row = session.get(RegistryRelease, version)
    if row is None:
        raise LookupError(f"no registry release {version!r}")
    document = yaml.safe_load(row.yaml_blob)
    return from_document(document, version=row.version, content_hash=row.content_hash)


def from_blob(blob: str, version: str | None = None, content_hash: str | None = None) -> Registry:
    """Rehydrate from a serialisation carried alongside a checkpoint."""
    return from_document(yaml.safe_load(blob), version=version, content_hash=content_hash)
