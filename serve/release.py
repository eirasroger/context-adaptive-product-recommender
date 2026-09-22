"""Promote a training run to the artefacts the service ships.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from serve.background import BACKGROUND_FILE, DEFAULT_ROWS, RELEASE_DIR, sample_from_database

MODEL_FILE = "model.pt"
MANIFEST_FILE = "manifest.json"
METRICS_FILE = "metrics.json"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "snapshot"
MAX_STRATUM_REGRESSION = 0.001

#: A checkpoint belongs in the repository. A corpus does not. If a promotion
#: produces artefacts past this, something is being shipped that should not be.
SIZE_WARNING_MB = 25


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


class GateFailed(Exception):
    pass


def refresh_fixture(snapshot_dir: Path, fixture_dir: Path = FIXTURE_DIR) -> dict:
    """Rewrite the evaluation fixture from the snapshot the model was scored on.

    The fixture holds data, so it goes stale the moment the corpus grows, which
    makes rebuilding it by hand a step that gets forgotten exactly once.
    """
    from snapshot.build import subset

    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists():
        raise GateFailed(
            f"no snapshot at {snapshot_dir}, so the evaluation fixture cannot "
            "be refreshed. Promote on the machine that trained the run, or "
            "pass force=True and rebuild the fixture separately."
        )

    fixture_dir = Path(fixture_dir)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    # The prepared cache is keyed on the snapshot's content hash, so a stale one
    # is inert rather than wrong. Removing it keeps the directory honest.
    for stale in fixture_dir.glob("*-prepared.pkl"):
        stale.unlink()
    return subset(snapshot_dir, fixture_dir, folds=["test"])


def regression_against_release(run_dir: Path, release_dir: Path):
    """Strata that got worse than the release this run would replace.

    The behavioural gate says the model learned the right shape. This says it
    did not get worse at anything on the way, which an overall average hides.
    """
    from eval import report as report_module

    baseline_path = Path(release_dir) / METRICS_FILE
    incoming_path = Path(run_dir) / METRICS_FILE
    if not baseline_path.exists() or not incoming_path.exists():
        return None

    return report_module.gate(
        json.loads(incoming_path.read_text(encoding="utf-8")),
        report_module.Thresholds(
            max_stratum_regression=MAX_STRATUM_REGRESSION,
            require_behavioural=False,
        ),
        json.loads(baseline_path.read_text(encoding="utf-8")),
    )


def promote(
    run_dir: Path,
    database: str | None = None,
    release_dir: Path = RELEASE_DIR,
    rows: int = DEFAULT_ROWS,
    notes: str = "",
    force: bool = False,
    fixture_dir: Path = FIXTURE_DIR,
    snapshot_root: Path | None = None,
) -> dict:
    run_dir = Path(run_dir)
    checkpoint = run_dir / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}")

    gate = _gate_result(run_dir)
    if gate is False and not force:
        raise GateFailed(
            f"{run_dir.name} did not pass the behavioural gate. Re-evaluate it with "
            "eval.run, or pass force=True and say in the notes why a failing model "
            "is being shipped."
        )
    if gate is None and not force:
        raise GateFailed(
            f"{run_dir.name} has no evaluation on record. Run eval.run against it "
            "before promoting, or pass force=True."
        )

    verdict = regression_against_release(run_dir, release_dir)
    if verdict is not None and not verdict.passed and not force:
        raise GateFailed(
            f"{run_dir.name} is worse than the release it would replace:\n"
            + "\n".join(f"  - {reason}" for reason in verdict.reasons)
            + "\nPass force=True and say in the notes why it ships anyway."
        )

    release_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, release_dir / MODEL_FILE)
    if (run_dir / METRICS_FILE).exists():
        shutil.copy2(run_dir / METRICS_FILE, release_dir / METRICS_FILE)

    from model import checkpoint as checkpoint_module

    described = checkpoint_module.describe(checkpoint)
    meta = described["meta"]

    try:
        from snapshot.build import SNAPSHOT_ROOT

        root = Path(snapshot_root) if snapshot_root else SNAPSHOT_ROOT
        refresh_fixture(root / (meta.get("snapshot_hash") or "unrecorded"), fixture_dir)
    except GateFailed:
        if not force:
            raise

    background_rows = 0
    if database:
        sampled = sample_from_database(database, rows)
        (release_dir / BACKGROUND_FILE).write_text(
            json.dumps({"origin": "corpus", "categories": sampled}, separators=(",", ":")),
            encoding="utf-8",
        )
        background_rows = sum(len(v) for v in sampled.values())

    evaluation = _evaluation(run_dir)

    manifest = {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "run": run_dir.name,
        "registry_version": meta.get("registry_version"),
        "registry_content_hash": meta.get("registry_content_hash"),
        "snapshot_hash": meta.get("snapshot_hash"),
        "split_key": meta.get("split_key"),
        "model_sha256": digest(release_dir / MODEL_FILE),
        "model_bytes": (release_dir / MODEL_FILE).stat().st_size,
        "background_rows": background_rows,
        "background_bytes": (release_dir / BACKGROUND_FILE).stat().st_size
        if (release_dir / BACKGROUND_FILE).exists()
        else 0,
        "config": described["config"],
        "evaluation": evaluation,
        "notes": notes,
    }
    (release_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _evaluation(run_dir: Path) -> dict | None:
    metrics = run_dir / "metrics.json"
    if not metrics.exists():
        return None
    results = json.loads(metrics.read_text(encoding="utf-8"))
    behavioural = results.get("behavioural") or {}
    return {
        "overall": results.get("overall"),
        "behavioural_passed": behavioural.get("passed"),
        "behavioural_summary": behavioural.get("summary"),
    }


def _gate_result(run_dir: Path) -> bool | None:
    evaluation = _evaluation(run_dir)
    if evaluation is None:
        return None
    return evaluation.get("behavioural_passed")


def read_manifest(release_dir: Path = RELEASE_DIR) -> dict | None:
    path = Path(release_dir) / MANIFEST_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a run to the served release.")
    parser.add_argument("run", help="a directory under runs/")
    parser.add_argument("--db", default="data/corpus.db", help="source for the SHAP background")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--release-dir", default=str(RELEASE_DIR))
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--force",
        action="store_true",
        help="promote despite a failing or missing evaluation",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="skip the background export and leave any existing file alone",
    )
    args = parser.parse_args()

    manifest = promote(
        Path(args.run),
        database=None if args.no_background else args.db,
        release_dir=Path(args.release_dir),
        rows=args.rows,
        notes=args.notes,
        force=args.force,
    )

    total = (manifest["model_bytes"] + manifest["background_bytes"]) / 1e6
    print(f"promoted {manifest['run']}")
    print(f"  registry   {manifest['registry_version']}")
    print(f"  snapshot   {(manifest['snapshot_hash'] or '')[:12]}")
    print(f"  model      {manifest['model_bytes'] / 1e6:.2f} MB  sha256 {manifest['model_sha256'][:12]}")
    print(f"  background {manifest['background_rows']} rows, "
          f"{manifest['background_bytes'] / 1e6:.2f} MB")
    evaluation = manifest.get("evaluation") or {}
    passed = evaluation.get("behavioural_passed")
    if passed is not None:
        print(f"  behavioural gate: {'PASS' if passed else 'FAIL (forced)'}")
    if total > SIZE_WARNING_MB:
        print(f"\nwarning: release is {total:.1f} MB, above the {SIZE_WARNING_MB} MB "
              "expected for a repository artefact")


if __name__ == "__main__":
    main()
