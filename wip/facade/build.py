"""Build the working corpus: the main corpus, the facade overlay, its dataset and its control cases."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from core import registry as registry_module
from db import release
from db.seed import seed
from db.session import create_db_engine, db_path, session_scope
from ingest import split
from ingest.generators import run as control
from ingest.generators.writer import write_cases
from wip.facade import ingest, synthesise

WORK_DIR = Path("data/wip")
WORK_DB = WORK_DIR / "corpus.db"
MIRROR_DIR = WORK_DIR / "registry"
OVERLAY_DIR = Path(__file__).parent / "seeds"
RELEASE = "0.3.0-wip"
CONTROL_SOURCE = "facade_wip_control"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sets", type=int, default=16_000, help="synthetic shortlists")
    parser.add_argument("--control-count", type=int, default=800, help="control cases per indicator")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resynthesise", action="store_true", help="rewrite the working dataset")
    args = parser.parse_args()

    copy_corpus(db_path(), WORK_DB)
    print(f"copied {db_path()} to {WORK_DB}")

    with session_scope(create_db_engine(WORK_DB)) as session:
        for path in seed(session, OVERLAY_DIR):
            print(f"applied {path.name}")
        registry = registry_module.from_session(session)

        scenarios = synthesise.DATASET_DIR / synthesise.SCENARIOS_FILE
        labels = synthesise.DATASET_DIR / synthesise.LABELS_FILE
        if args.resynthesise or not scenarios.exists():
            synthesise.write(registry, synthesise.DATASET_DIR, args.sets, args.seed)
            print(f"wrote the working dataset to {synthesise.DATASET_DIR}")
        for table, count in ingest.ingest(session, registry, scenarios, labels).items():
            print(f"  {table:28s} {count:>9,}")

        category = synthesise.CATEGORY_KEY
        cases = control.generate(
            registry, category, sorted(registry.sweepable(category)), count=args.control_count, seed=args.seed
        )
        written = write_cases(session, registry, cases, source_key=CONTROL_SOURCE, prefix="wipfacade")
        print(f"control cases: {written['comparison_sets']:,}")

        folds = split.assign(session, replace=True)
        print(f"folds: {folds}")
        digest, _ = release.create_release(
            session, RELEASE, "Work in progress: facade systems on synthetic data.", MIRROR_DIR
        )
        print(f"registry {RELEASE}  {digest[:12]}")


if __name__ == "__main__":
    main()
