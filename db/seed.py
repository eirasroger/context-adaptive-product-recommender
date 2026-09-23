"""Load the YAML seed files into the registry tables, in filename order, idempotently."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from db import slots
from db.models import (
    Category,
    CategoryContextOverride,
    CategoryIndicator,
    CategoryProvenanceWeight,
    Context,
    ContextDeclaration,
    FunctionalUnit,
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
from db.session import create_db_engine, session_scope

SEED_DIR = Path(__file__).parent / "seeds"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _upsert(session: Session, model, pk: tuple | Any, **fields):
    existing = session.get(model, pk)
    if existing is None:
        row = model(**fields)
        session.add(row)
        session.flush()
        return row
    for name, value in fields.items():
        setattr(existing, name, value)
    session.flush()
    return existing


def _bool(value: Any) -> int:
    return 1 if value else 0


def _load_families(session: Session, rows: list[dict]) -> None:
    for row in rows:
        _upsert(
            session,
            IndicatorFamily,
            row["key"],
            key=row["key"],
            display_name=row["display_name"],
            definition_text=row["definition_text"].strip(),
            sort_order=row["sort_order"],
            is_active=_bool(row.get("is_active", True)),
        )
        slots.allocate(session, "indicator_family", row["key"])


def _load_provenance(session: Session, rows: list[dict]) -> None:
    for row in rows:
        _upsert(
            session,
            Provenance,
            row["key"],
            key=row["key"],
            display_name=row["display_name"],
            definition_text=row["definition_text"].strip(),
        )


def _load_indicators(session: Session, rows: list[dict]) -> None:
    for row in rows:
        key = row["key"]
        _upsert(
            session,
            Indicator,
            key,
            key=key,
            display_name=row["display_name"],
            family_key=row["family_key"],
            value_type=row["value_type"],
            unit=row.get("unit"),
            definition_text=row["definition_text"].strip(),
            default_direction=row.get("default_direction", 0),
            nominal_justification=row.get("nominal_justification"),
            is_derived=_bool(row.get("is_derived", False)),
            is_active=_bool(row.get("is_active", True)),
        )
        slots.allocate(session, "indicator", key)

        for level in row.get("levels", []):
            _upsert(
                session,
                IndicatorLevel,
                (key, level["level_key"]),
                indicator_key=key,
                level_key=level["level_key"],
                level_index=level["level_index"],
                display_name=level["display_name"],
                definition_text=level["definition_text"].strip(),
                normalised_position=level.get("normalised_position"),
                is_disqualifying=_bool(level.get("is_disqualifying", False)),
            )
            slots.allocate(
                session, "indicator_level", f"{key}::{level['level_key']}"
            )

        for source_key, coefficient in (row.get("derivation") or {}).items():
            _upsert(
                session,
                IndicatorDerivation,
                (key, source_key),
                derived_key=key,
                source_key=source_key,
                coefficient=float(coefficient),
            )


def _load_stakeholders(session: Session, rows: list[dict]) -> None:
    for row in rows:
        key = row["key"]
        _upsert(
            session,
            Stakeholder,
            key,
            key=key,
            display_name=row["display_name"],
            definition_text=row["definition_text"].strip(),
            domain_scope_text=(row.get("domain_scope_text") or "").strip() or None,
            is_active=_bool(row.get("is_active", True)),
        )
        slots.allocate(session, "stakeholder", key)

        priorities = row.get("priorities") or {}
        for kind in ("family", "indicator"):
            for target_key, priority in (priorities.get(kind) or {}).items():
                _upsert(
                    session,
                    StakeholderPriority,
                    (key, kind, target_key),
                    stakeholder_key=key,
                    target_kind=kind,
                    target_key=target_key,
                    priority=float(priority),
                )


def _load_contexts(session: Session, rows: list[dict]) -> None:
    for row in rows:
        key = row["key"]
        _upsert(
            session,
            Context,
            key,
            key=key,
            display_name=row["display_name"],
            definition_text=row["definition_text"].strip(),
            is_baseline=_bool(row.get("is_baseline", False)),
            is_active=_bool(row.get("is_active", True)),
        )
        slots.allocate(session, "context", key)

        for decl in row.get("declarations", []):
            category_key = decl.get("category_key", WILDCARD)
            _upsert(
                session,
                ContextDeclaration,
                (key, decl["indicator_key"], category_key),
                context_key=key,
                indicator_key=decl["indicator_key"],
                category_key=category_key,
                direction=decl["direction"],
                priority=float(decl["priority"]),
                is_required=_bool(decl.get("is_required", False)),
                ideal_value=decl.get("ideal_value"),
                note=decl.get("note"),
            )


def _load_categories(session: Session, doc: dict) -> None:
    for row in doc.get("functional_unit", []):
        _upsert(
            session,
            FunctionalUnit,
            row["key"],
            key=row["key"],
            display_name=row["display_name"],
            quantity_kind=row["quantity_kind"],
            unit=row["unit"],
            reference_service_life_years=row.get("reference_service_life_years"),
            definition_text=row["definition_text"].strip(),
        )

    default_note = (doc.get("range_source_note") or "Declared.").strip()

    for row in doc.get("category", []):
        key = row["key"]
        _upsert(
            session,
            Category,
            key,
            key=key,
            display_name=row["display_name"],
            definition_text=row["definition_text"].strip(),
            functional_unit_key=row["functional_unit_key"],
            eligibility_precondition_text=row[
                "eligibility_precondition_text"
            ].strip(),
            default_context_key=row["default_context_key"],
            is_active=_bool(row.get("is_active", True)),
        )
        slots.allocate(session, "category", key)

        for provenance_key, weight in (row.get("provenance_weights") or {}).items():
            _upsert(
                session,
                CategoryProvenanceWeight,
                (key, provenance_key),
                category_key=key,
                provenance_key=provenance_key,
                weight=float(weight),
            )

        for member in row.get("indicators", []):
            indicator_key = member["indicator_key"]
            _upsert(
                session,
                CategoryIndicator,
                (key, indicator_key),
                category_key=key,
                indicator_key=indicator_key,
                is_required=_bool(member.get("is_required", False)),
                relevance_mode=member.get("relevance_mode", "always"),
                direction_override=member.get("direction_override"),
                control_mode=member.get("control_mode", "sweep"),
                control_note=(member.get("control_note") or "").strip() or None,
                note=member.get("note"),
            )

            if "ref_low" in member:
                _upsert(
                    session,
                    IndicatorReferenceRange,
                    (key, indicator_key),
                    category_key=key,
                    indicator_key=indicator_key,
                    ref_low=float(member["ref_low"]),
                    ref_high=float(member["ref_high"]),
                    scale=member.get("scale", "linear"),
                    shape=member.get("shape", "monotone"),
                    ideal_value=member.get("ideal_value"),
                    ideal_low=member.get("ideal_low"),
                    ideal_high=member.get("ideal_high"),
                    source_note=(member.get("source_note") or default_note).strip(),
                )

        for override in row.get("context_overrides", []):
            _upsert(
                session,
                CategoryContextOverride,
                (key, override["context_key"]),
                category_key=key,
                context_key=override["context_key"],
                is_applicable=_bool(override["is_applicable"]),
                display_name=override.get("display_name"),
                justification=override["justification"].strip(),
            )


SECTION_LOADERS = (
    ("indicator_family", _load_families),
    ("provenance", _load_provenance),
    ("indicator", _load_indicators),
    ("stakeholder", _load_stakeholders),
    ("context", _load_contexts),
)


def apply_seed_document(session: Session, doc: dict) -> None:
    for section, loader in SECTION_LOADERS:
        if section in doc:
            loader(session, doc[section])
    if "category" in doc or "functional_unit" in doc:
        _load_categories(session, doc)


def seed(session: Session, seed_dir: Path = SEED_DIR) -> list[Path]:
    applied = []
    for path in sorted(seed_dir.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not doc:
            continue
        apply_seed_document(session, doc)
        applied.append(path)
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the registry from YAML.")
    parser.add_argument("--db", default=None, help="database file")
    parser.add_argument("--seed-dir", default=str(SEED_DIR))
    parser.add_argument(
        "--create",
        action="store_true",
        help="create the schema first, instead of relying on migrations",
    )
    args = parser.parse_args()

    engine = create_db_engine(args.db)
    if args.create:
        from db.session import create_all

        create_all(engine)

    with session_scope(engine) as session:
        applied = seed(session, Path(args.seed_dir))
        sizes = slots.table_sizes(session)

    for path in applied:
        print(f"applied {path.name}")
    print("embedding table sizes:", sizes)


if __name__ == "__main__":
    main()
