"""Assign comparison sets to train, validation and test folds.

Assignment is a deterministic function of the set's own identifier and the name
of the split, so it is stable: adding data next month leaves every existing
assignment exactly where it was, and a set never drifts across the boundary
between one run and the next. That matters more than it sounds -- reshuffling on
every run makes two results incomparable without anyone noticing.

Stratified by category and provenance, so each fold keeps the same mixture of
control, model-labelled and expert-labelled cases as the corpus as a whole.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict

from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from db.models import ComparisonSet, DatasetSplit
from db.session import create_db_engine, session_scope

DEFAULT_SPLIT = "default"
FOLDS = ("train", "val", "test")


def _fraction(split_key: str, external_id: str) -> float:
    """A stable number in [0, 1) for this set under this split name."""
    digest = hashlib.sha256(f"{split_key}::{external_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign(
    session: Session,
    split_key: str = DEFAULT_SPLIT,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    replace: bool = False,
) -> dict[str, int]:
    """Write fold assignments for every comparison set."""
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must leave room for training")

    existing = session.execute(
        select(DatasetSplit).where(DatasetSplit.split_key == split_key).limit(1)
    ).first()
    if existing and not replace:
        raise ValueError(
            f"split {split_key!r} already exists; pass replace=True to rewrite it, "
            "bearing in mind that doing so invalidates comparisons with any result "
            "produced under the old assignment"
        )
    if existing:
        session.execute(delete(DatasetSplit).where(DatasetSplit.split_key == split_key))

    rows = session.execute(
        select(
            ComparisonSet.id, ComparisonSet.external_id,
            ComparisonSet.category_key, ComparisonSet.provenance_key,
        )
    ).all()

    # Stratify by ranking within each stratum rather than thresholding the hash
    # directly: for a small stratum -- expert cases, say -- raw thresholding can
    # easily leave a fold empty.
    strata: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for set_id, external_id, category_key, provenance_key in rows:
        strata[(category_key, provenance_key)].append(
            (_fraction(split_key, external_id), set_id)
        )

    assignments: list[dict] = []
    counts = dict.fromkeys(FOLDS, 0)
    for members in strata.values():
        members.sort()
        total = len(members)
        n_val = round(total * val_fraction)
        n_test = round(total * test_fraction)
        for index, (_, set_id) in enumerate(members):
            if index < n_val:
                fold = "val"
            elif index < n_val + n_test:
                fold = "test"
            else:
                fold = "train"
            counts[fold] += 1
            assignments.append(
                {"comparison_set_id": set_id, "split_key": split_key, "fold": fold}
            )

    for start in range(0, len(assignments), 5_000):
        session.execute(insert(DatasetSplit.__table__), assignments[start : start + 5_000])

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign train/val/test folds.")
    parser.add_argument("--db", default=None)
    parser.add_argument("--split-key", default=DEFAULT_SPLIT)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    with session_scope(create_db_engine(args.db)) as session:
        counts = assign(
            session,
            args.split_key,
            args.val_fraction,
            args.test_fraction,
            replace=args.replace,
        )

    print(f"split {args.split_key}:", counts)


if __name__ == "__main__":
    main()
