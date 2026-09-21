"""Freeze the registry into a hashed, diffable release.

Two things come out of a release:

* a **content hash** over a canonical serialisation, recorded with every
  checkpoint so a training run can be traced back to the exact semantics it was
  trained under;
* a **YAML mirror** written into the repository, so that a change to a direction
  sign or a reference range shows up as a reviewable diff rather than as an
  invisible row update.

The database remains the system of record. The mirror is an export, never an
input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from db import slots, validate
from db.models import (
    Category,
    CategoryContextOverride,
    CategoryIndicator,
    CategoryProvenanceWeight,
    Context,
    ContextDeclaration,
    EmbeddingSlot,
    FunctionalUnit,
    Indicator,
    IndicatorDerivation,
    IndicatorFamily,
    IndicatorLevel,
    IndicatorReferenceRange,
    Provenance,
    Stakeholder,
    StakeholderPriority,
)
from db.session import create_db_engine, session_scope

MIRROR_DIR = Path("registry")

#: Every registry table, with the columns and sort order that make the export
#: canonical. Two databases with the same semantics must serialise identically,
#: or the content hash means nothing.
EXPORT_SPEC: tuple[tuple[str, Any, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "indicator_family",
        IndicatorFamily,
        ("key", "display_name", "definition_text", "sort_order", "is_active"),
        ("key",),
    ),
    (
        "indicator",
        Indicator,
        (
            "key",
            "display_name",
            "family_key",
            "value_type",
            "unit",
            "definition_text",
            "default_direction",
            "nominal_justification",
            "is_derived",
            "is_active",
        ),
        ("key",),
    ),
    (
        "indicator_level",
        IndicatorLevel,
        (
            "indicator_key",
            "level_key",
            "level_index",
            "display_name",
            "definition_text",
            "normalised_position",
            "is_disqualifying",
        ),
        ("indicator_key", "level_index"),
    ),
    (
        "indicator_derivation",
        IndicatorDerivation,
        ("derived_key", "source_key", "coefficient"),
        ("derived_key", "source_key"),
    ),
    (
        "functional_unit",
        FunctionalUnit,
        (
            "key",
            "display_name",
            "quantity_kind",
            "unit",
            "reference_service_life_years",
            "definition_text",
        ),
        ("key",),
    ),
    (
        "category",
        Category,
        (
            "key",
            "display_name",
            "definition_text",
            "functional_unit_key",
            "eligibility_precondition_text",
            "default_context_key",
            "is_active",
        ),
        ("key",),
    ),
    (
        "category_indicator",
        CategoryIndicator,
        (
            "category_key",
            "indicator_key",
            "is_required",
            "relevance_mode",
            "direction_override",
            "control_mode",
            "control_note",
            "note",
        ),
        ("category_key", "indicator_key"),
    ),
    (
        "indicator_reference_range",
        IndicatorReferenceRange,
        (
            "category_key",
            "indicator_key",
            "ref_low",
            "ref_high",
            "scale",
            "shape",
            "ideal_value",
            "ideal_low",
            "ideal_high",
            "source_note",
        ),
        ("category_key", "indicator_key"),
    ),
    (
        "context",
        Context,
        ("key", "display_name", "definition_text", "is_baseline", "is_active"),
        ("key",),
    ),
    (
        "context_declaration",
        ContextDeclaration,
        (
            "context_key",
            "indicator_key",
            "category_key",
            "direction",
            "priority",
            "is_required",
            "ideal_value",
            "note",
        ),
        ("context_key", "category_key", "indicator_key"),
    ),
    (
        "category_context_override",
        CategoryContextOverride,
        ("category_key", "context_key", "is_applicable", "display_name", "justification"),
        ("category_key", "context_key"),
    ),
    (
        "stakeholder",
        Stakeholder,
        ("key", "display_name", "definition_text", "domain_scope_text", "is_active"),
        ("key",),
    ),
    (
        "stakeholder_priority",
        StakeholderPriority,
        ("stakeholder_key", "target_kind", "target_key", "priority"),
        ("stakeholder_key", "target_kind", "target_key"),
    ),
    (
        "provenance",
        Provenance,
        ("key", "display_name", "definition_text"),
        ("key",),
    ),
    (
        "category_provenance_weight",
        CategoryProvenanceWeight,
        ("category_key", "provenance_key", "weight"),
        ("category_key", "provenance_key"),
    ),
    (
        "embedding_slot",
        EmbeddingSlot,
        ("table_name", "entity_key", "slot"),
        ("table_name", "slot"),
    ),
)


def export(session: Session) -> dict[str, list[dict]]:
    """Serialise the registry into a canonical, order-stable structure."""
    document: dict[str, list[dict]] = {}
    for name, model, columns, order in EXPORT_SPEC:
        rows = list(session.execute(select(model)).scalars())
        rows.sort(key=lambda row: tuple(str(getattr(row, col)) for col in order))
        document[name] = [
            {col: getattr(row, col) for col in columns} for row in rows
        ]
    return document


def canonical_yaml(document: dict[str, list[dict]]) -> str:
    """YAML with fixed key order and no line wrapping, so diffs stay readable."""
    return yaml.safe_dump(
        document, sort_keys=False, allow_unicode=True, default_flow_style=False, width=10_000
    )


def content_hash(document: dict[str, list[dict]]) -> str:
    """Hash over a JSON rendering, which is stricter about types than YAML."""
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_mirror(document: dict[str, list[dict]], mirror_dir: Path = MIRROR_DIR) -> list[Path]:
    """Write one YAML file per registry table, for review."""
    mirror_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, rows in document.items():
        path = mirror_dir / f"{name}.yaml"
        path.write_text(canonical_yaml({name: rows}), encoding="utf-8")
        written.append(path)
    return written


def create_release(
    session: Session,
    version: str,
    notes: str,
    mirror_dir: Path = MIRROR_DIR,
) -> tuple[str, list[Path]]:
    """Validate, freeze and record a release. Returns its hash and mirror files."""
    from db.models import RegistryRelease

    validate.require_valid(session)

    document = export(session)
    blob = canonical_yaml(document)
    digest = content_hash(document)

    existing = session.get(RegistryRelease, version)
    if existing is not None and existing.content_hash != digest:
        raise ValueError(
            f"release {version} already exists with a different content hash; "
            "a released version is immutable, so bump the version instead"
        )

    if existing is None:
        session.add(
            RegistryRelease(
                version=version,
                created_at=datetime.now(timezone.utc).isoformat(),
                content_hash=digest,
                max_slots=json.dumps(slots.max_slots(session), sort_keys=True),
                yaml_blob=blob,
                notes=notes,
            )
        )
        session.flush()

    return digest, write_mirror(document, mirror_dir)


def latest_release(session: Session):
    from db.models import RegistryRelease

    return session.execute(
        select(RegistryRelease).order_by(RegistryRelease.created_at.desc()).limit(1)
    ).scalar_one_or_none()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut a registry release.")
    parser.add_argument("version", help="semantic version, e.g. 0.1.0")
    parser.add_argument("--db", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--mirror-dir", default=str(MIRROR_DIR))
    args = parser.parse_args()

    with session_scope(create_db_engine(args.db)) as session:
        digest, written = create_release(
            session, args.version, args.notes, Path(args.mirror_dir)
        )

    print(f"registry {args.version}  {digest}")
    print(f"mirror: {len(written)} files in {args.mirror_dir}/")


if __name__ == "__main__":
    main()
