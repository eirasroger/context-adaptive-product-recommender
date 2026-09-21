"""Materialise the corpus into an immutable, content-hashed snapshot.

Training never reads the live database. It reads one of these, and records the
snapshot's hash and the registry version alongside the checkpoint, so a result
can always be traced back to the exact data and the exact semantics that
produced it. A snapshot is also fast to read repeatedly, which the database is
not.

The hash is computed over the logical rows before anything is written, so it
depends on the data rather than on compression settings or write order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import (
    ComparisonSet,
    ComparisonSetContext,
    ComparisonSetMember,
    ComparisonSetStakeholder,
    DatasetSplit,
    IndicatorValue,
    Label,
    LabelSet,
    Snapshot,
)
from db.session import create_db_engine, session_scope

SNAPSHOT_ROOT = Path("data/snapshots")

SETS_FILE = "sets.parquet"
MEMBERS_FILE = "members.parquet"
VALUES_FILE = "values.parquet"
MANIFEST_FILE = "manifest.json"

#: Multi-valued attachments are stored joined by this separator rather than as a
#: nested column: a comparison set has at most a handful of stakeholders, and a
#: flat string survives every parquet reader without ceremony.
JOIN = "|"


def _frame(session: Session, statement, columns: list[str]) -> pd.DataFrame:
    rows = session.execute(statement).all()
    return pd.DataFrame(rows, columns=columns)


def _hash_frames(frames: dict[str, pd.DataFrame]) -> str:
    """A hash over logical content, stable across writers and machines."""
    digest = hashlib.sha256()
    for name in sorted(frames):
        frame = frames[name]
        digest.update(name.encode("utf-8"))
        digest.update(JOIN.join(map(str, frame.columns)).encode("utf-8"))
        row_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy()
        digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def collect(
    session: Session,
    split_key: str,
    categories: Iterable[str] | None = None,
    provenances: Iterable[str] | None = None,
    folds: Iterable[str] | None = None,
    sources: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Pull the corpus into three flat frames: sets, members, values.

    ``sources`` exists so a run can be restricted to one ingested dataset. A
    reproduction of a published result has to be able to exclude data generated
    afterwards, or it is not a reproduction.
    """
    category_filter = sorted(categories) if categories else None
    provenance_filter = sorted(provenances) if provenances else None
    fold_filter = sorted(folds) if folds else None
    source_filter = sorted(sources) if sources else None

    statement = (
        select(
            ComparisonSet.id,
            ComparisonSet.external_id,
            ComparisonSet.category_key,
            ComparisonSet.provenance_key,
            ComparisonSet.generator_key,
            DatasetSplit.fold,
        )
        .join(
            DatasetSplit,
            (DatasetSplit.comparison_set_id == ComparisonSet.id)
            & (DatasetSplit.split_key == split_key),
        )
        .order_by(ComparisonSet.id)
    )
    if category_filter:
        statement = statement.where(ComparisonSet.category_key.in_(category_filter))
    if provenance_filter:
        statement = statement.where(ComparisonSet.provenance_key.in_(provenance_filter))
    if fold_filter:
        statement = statement.where(DatasetSplit.fold.in_(fold_filter))
    if source_filter:
        statement = statement.where(ComparisonSet.source_key.in_(source_filter))

    sets = _frame(
        session,
        statement,
        ["set_id", "external_id", "category_key", "provenance_key", "generator_key", "fold"],
    )
    if sets.empty:
        raise ValueError(
            f"no comparison sets matched split {split_key!r} with the given filters"
        )
    set_ids = set(sets["set_id"].tolist())

    stakeholders = _frame(
        session,
        select(
            ComparisonSetStakeholder.set_id, ComparisonSetStakeholder.stakeholder_key
        ).order_by(
            ComparisonSetStakeholder.set_id, ComparisonSetStakeholder.stakeholder_key
        ),
        ["set_id", "key"],
    )
    contexts = _frame(
        session,
        select(ComparisonSetContext.set_id, ComparisonSetContext.context_key).order_by(
            ComparisonSetContext.set_id, ComparisonSetContext.context_key
        ),
        ["set_id", "key"],
    )
    sets = sets.merge(
        _join_keys(stakeholders, "stakeholder_keys"), on="set_id", how="left"
    ).merge(_join_keys(contexts, "context_keys"), on="set_id", how="left")
    sets[["stakeholder_keys", "context_keys"]] = sets[
        ["stakeholder_keys", "context_keys"]
    ].fillna("")

    members = _frame(
        session,
        select(
            ComparisonSetMember.set_id,
            ComparisonSetMember.position,
            ComparisonSetMember.product_id,
            ComparisonSetMember.local_key,
            Label.pref,
            Label.conf,
            LabelSet.scale_semantics,
        )
        .join(LabelSet, LabelSet.comparison_set_id == ComparisonSetMember.set_id, isouter=True)
        .join(
            Label,
            (Label.label_set_id == LabelSet.id)
            & (Label.product_id == ComparisonSetMember.product_id),
            isouter=True,
        )
        .order_by(ComparisonSetMember.set_id, ComparisonSetMember.position),
        ["set_id", "position", "product_id", "local_key", "pref", "conf", "scale_semantics"],
    )
    members = members[members["set_id"].isin(set_ids)].reset_index(drop=True)
    members, multi_labelled = _aggregate_labellers(members)

    product_ids = set(members["product_id"].tolist())
    values = _frame(
        session,
        select(
            IndicatorValue.product_id,
            IndicatorValue.indicator_key,
            IndicatorValue.present,
            IndicatorValue.value_num,
            IndicatorValue.level_key,
        ).order_by(IndicatorValue.product_id, IndicatorValue.indicator_key),
        ["product_id", "indicator_key", "present", "value_num", "level_key"],
    )
    values = values[values["product_id"].isin(product_ids)].reset_index(drop=True)

    return {"sets": sets, "members": members, "values": values}


def _aggregate_labellers(members: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse several independent labellings of one set into one row each.

    The schema stores one label set per labeller on purpose, so expert scores
    are not averaged away at ingest time. A snapshot, though, needs exactly one
    target per alternative -- without this, two annotators would turn one
    alternative into two, and the model would be trained on a set that never
    existed.

    Preference is averaged; confidence is the mean confidence scaled down by how
    much the labellers disagreed, so a contested alternative carries less weight
    than a unanimous one. Where only one labeller scored a set, nothing changes.
    """
    keys = ["set_id", "position", "product_id", "local_key"]
    counts = members.groupby(keys, dropna=False).size()
    multi = int((counts > 1).sum())
    if multi == 0:
        return members, 0

    grouped = members.groupby(keys, dropna=False)
    aggregated = grouped.agg(
        pref=("pref", "mean"),
        spread=("pref", "std"),
        conf=("conf", "mean"),
        scale_semantics=("scale_semantics", "first"),
    ).reset_index()

    spread = aggregated["spread"].fillna(0.0).clip(0.0, 1.0)
    aggregated["conf"] = (aggregated["conf"].fillna(1.0) * (1.0 - spread)).clip(0.0, 1.0)
    aggregated = aggregated.drop(columns=["spread"])

    ordered = aggregated.sort_values(["set_id", "position"], kind="stable")
    return ordered.reset_index(drop=True)[
        ["set_id", "position", "product_id", "local_key", "pref", "conf", "scale_semantics"]
    ], multi


def _join_keys(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame({"set_id": [], column: []})
    return (
        frame.groupby("set_id")["key"]
        .apply(lambda keys: JOIN.join(sorted(keys)))
        .rename(column)
        .reset_index()
    )


def build(
    session: Session,
    split_key: str,
    registry_version: str,
    root: Path = SNAPSHOT_ROOT,
    categories: Iterable[str] | None = None,
    provenances: Iterable[str] | None = None,
    folds: Iterable[str] | None = None,
    sources: Iterable[str] | None = None,
) -> tuple[str, Path]:
    """Build the snapshot and register it. Returns its hash and directory."""
    from core import registry as registry_module

    registry = registry_module.from_release(session, registry_version)

    frames = collect(session, split_key, categories, provenances, folds, sources)
    digest = _hash_frames({**frames, "registry": pd.DataFrame({"h": [registry.content_hash]})})


    directory = root / digest
    directory.mkdir(parents=True, exist_ok=True)

    frames["sets"].to_parquet(directory / SETS_FILE, index=False)
    frames["members"].to_parquet(directory / MEMBERS_FILE, index=False)
    frames["values"].to_parquet(directory / VALUES_FILE, index=False)

    row_counts = {name: int(len(frame)) for name, frame in frames.items()}
    filter_spec = {
        "split_key": split_key,
        "categories": sorted(categories) if categories else None,
        "provenances": sorted(provenances) if provenances else None,
        "folds": sorted(folds) if folds else None,
        "sources": sorted(sources) if sources else None,
    }
    manifest = {
        "content_hash": digest,
        "registry_version": registry_version,
        "registry_content_hash": registry.content_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filter_spec": filter_spec,
        "row_counts": row_counts,
    }
    (directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    if session.get(Snapshot, digest) is None:
        session.add(
            Snapshot(
                content_hash=digest,
                registry_version=registry_version,
                split_key=split_key,
                created_at=manifest["created_at"],
                filter_spec=json.dumps(filter_spec, sort_keys=True),
                row_counts=json.dumps(row_counts, sort_keys=True),
                parquet_path=str(directory),
            )
        )

    return digest, directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a training snapshot.")
    parser.add_argument("registry_version")
    parser.add_argument("--db", default=None)
    parser.add_argument("--split-key", default="default")
    parser.add_argument("--root", default=str(SNAPSHOT_ROOT))
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--provenance", action="append", dest="provenances")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="restrict to one ingested dataset; repeatable",
    )
    args = parser.parse_args()

    with session_scope(create_db_engine(args.db)) as session:
        digest, directory = build(
            session,
            args.split_key,
            args.registry_version,
            root=Path(args.root),
            categories=args.categories,
            provenances=args.provenances,
            sources=args.sources,
        )

    print(f"snapshot {digest}")
    print(f"  {directory}")


if __name__ == "__main__":
    main()
