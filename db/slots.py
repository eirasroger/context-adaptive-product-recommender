"""Append-only slot allocation.

Every registry entity the model embeds owns an integer slot.  Slots are handed
out in insertion order and never reused, so adding an indicator, a context or a
stakeholder appends a row to an embedding table instead of shifting the meaning
of the existing ones.  This is what makes "a new category is registry rows plus
data" true at the level of the weights, not just at the level of the schema.

Retiring an entity sets ``is_active = 0`` on its own table and leaves the slot
burned.  Reusing a slot would silently hand one entity's learned behaviour to a
different one.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import EmbeddingSlot

#: The registry tables whose rows the model embeds.
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
    """Look up an existing slot, refusing to invent one."""
    row = session.get(EmbeddingSlot, (table_name, entity_key))
    if row is None:
        raise LookupError(f"no slot allocated for {table_name}.{entity_key}")
    return row.slot


def max_slots(session: Session) -> dict[str, int]:
    """Highest allocated slot per table, recorded with every registry release.

    A checkpoint uses this to know how wide its embedding tables were, so a
    later, larger registry can be loaded into it without guesswork.
    """
    rows = session.execute(
        select(EmbeddingSlot.table_name, func.max(EmbeddingSlot.slot)).group_by(
            EmbeddingSlot.table_name
        )
    ).all()
    return {table: int(highest) for table, highest in rows}


def table_sizes(session: Session) -> dict[str, int]:
    """Number of embedding rows required per table (highest slot plus one)."""
    return {table: highest + 1 for table, highest in max_slots(session).items()}
