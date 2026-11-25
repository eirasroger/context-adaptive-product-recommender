"""Build the joint corpus from scratch: the main corpus, every category's dataset, and control cases for each category."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core import registry as registry_module
from core.registry import Registry
from db import release
from db.models import ComparisonSet
from db.seed import seed
from db.session import create_db_engine, db_path, session_scope
from ingest import split
from ingest.facade import ingest as facade
from ingest.generators import run as control
from ingest.generators.writer import write_cases

JOINT_DB = Path("data/joint.db")


def copy_corpus(source: Path, target: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(f"{target}{suffix}").unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_connection, target_connection = sqlite3.connect(source), sqlite3.connect(target)
    try:
        source_connection.backup(target_connection)
    finally:
        source_connection.close()
        target_connection.close()


def set_counts(session: Session, provenance: str | None = None) -> dict[str, int]:
    query = select(ComparisonSet.category_key, func.count()).group_by(ComparisonSet.category_key)
    if provenance is not None:
        query = query.where(ComparisonSet.provenance_key == provenance)
    return {key: int(count) for key, count in session.execute(query).all()}


def control_prefix(category_key: str) -> str:
    """The id prefix and source name of a category's generated control cases: facade_system gives facade."""
    return category_key.split("_")[0]


def add_controls(session: Session, registry: Registry, count: int, seed_value: int) -> None:
    """Control cases for every category that has none yet; concrete's come with the main corpus."""
    present = set_counts(session, "control")
    for category_key in sorted(registry.categories):
        if present.get(category_key):
            continue
        prefix = control_prefix(category_key)
        cases = control.generate(
            registry, category_key, sorted(registry.sweepable(category_key)), count=count, seed=seed_value
        )
        written = write_cases(session, registry, cases, source_key=f"{prefix}_control", prefix=prefix)
        print(f"control cases for {category_key}: {written['comparison_sets']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("release", help="registry version to cut, e.g. 0.7.0")
    parser.add_argument("note", help="what this release changes, recorded with it")
    parser.add_argument("--sets", type=int, default=16_000, help="synthetic facade shortlists, when resynthesising")
    parser.add_argument("--control-count", type=int, default=800, help="control cases per indicator")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resynthesise", action="store_true", help="rewrite the working facade dataset")
    parser.add_argument("--labels", choices=("llm", "rule"), default="llm", help="which facade labels to ingest")
    args = parser.parse_args()

    copy_corpus(db_path(), JOINT_DB)
    print(f"copied {db_path()} to {JOINT_DB}")

    with session_scope(create_db_engine(JOINT_DB)) as session:
        for path in seed(session):
            print(f"applied {path.name}")
        registry = registry_module.from_session(session)

        for table, count in facade.add_dataset(
            session, registry, args.labels, args.sets, args.seed, args.resynthesise
        ).items():
            print(f"  {table:28s} {count:>9,}")
        add_controls(session, registry, args.control_count, args.seed)

        empty = sorted(set(registry.categories) - set(set_counts(session)))
        if empty:
            raise SystemExit(f"the registry declares categories with no data in the corpus: {empty}")

        print(f"folds: {split.assign(session, replace=True)}")
        digest, _ = release.create_release(session, args.release, args.note)
        print(f"registry {args.release}  {digest[:12]}")


if __name__ == "__main__":
    main()
