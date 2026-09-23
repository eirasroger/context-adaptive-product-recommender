"""Append-only embedding slots: allocated once, never reused, burned on retirement."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import EmbeddingSlot

SLOTTED_TABLES = (
    "indicator_family",
    "indicator",
    "indicator_level",
    "context",
    "stakeholder",
    "category",
)


def allocate(session: Session, table_name: str, entity_key: str) -> int:
    """Return this entity's slot, allocating the next free one if it has none."""
    if table_name not in SLOTTED_TABLES:
        raise ValueError(f"{table_name!r} is not a slotted table")

    existing = session.get(EmbeddingSlot, (table_name, entity_key))
    if existing is not None:
        return existing.slot

    highest = session.execute(
        select(func.max(EmbeddingSlot.slot)).where(
            EmbeddingSlot.table_name == table_name
        )
    ).scalar()
    slot = 0 if highest is None else highest + 1

    session.add(
        EmbeddingSlot(
            table_name=table_name,
            entity_key=entity_key,
            slot=slot,
            allocated_at=datetime.now(timezone.utc).isoformat(),
        )
    )
    session.flush()
    return slot


def slot_of(session: Session, table_name: str, entity_key: str) -> int:
    row = session.get(EmbeddingSlot, (table_name, entity_key))
    if row is None:
        raise LookupError(f"no slot allocated for {table_name}.{entity_key}")
    return row.slot


def max_slots(session: Session) -> dict[str, int]:
    """Highest allocated slot per table."""
    rows = session.execute(
        select(EmbeddingSlot.table_name, func.max(EmbeddingSlot.slot)).group_by(
            EmbeddingSlot.table_name
        )
    ).all()
    return {table: int(highest) for table, highest in rows}


def table_sizes(session: Session) -> dict[str, int]:
    return {table: highest + 1 for table, highest in max_slots(session).items()}
