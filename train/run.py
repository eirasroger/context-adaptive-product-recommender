"""Train, evaluate and gate one run, writing everything into its run directory."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from core import registry as registry_module
from core.prepare import load_or_prepare
from db import release as release_module
from db.models import RegistryRelease
from db.session import create_db_engine, db_path, session_scope
from eval import behavioural, report as report_module
from model import checkpoint as checkpoint_module
from snapshot.build import SNAPSHOT_ROOT, build
from train import machine
from train.config import TrainConfig
from train.trainer import EpochRecord, train, write_history


def resolve_registry(config: TrainConfig, session):
    """The release the config names, or the newest one, written back into the config."""
    if config.registry_version is None:
        release = release_module.latest_release(session)
        if release is None:
            raise SystemExit(
                "the database holds no registry release; cut one with db.release"
            )
        config.registry_version = release.version
    else:
        release = session.get(RegistryRelease, config.registry_version)
        if release is None:
            raise SystemExit(f"no registry release {config.registry_version}")

    registry = registry_module.from_release(session, config.registry_version)
    return registry, release.yaml_blob


def resolve_snapshot(config: TrainConfig, session) -> tuple[str, Path]:
    if config.snapshot:
        directory = SNAPSHOT_ROOT / config.snapshot
        if not directory.exists():
            raise FileNotFoundError(f"no snapshot at {directory}")
        return config.snapshot, directory
    return build(
        session,
        config.split_key,
        config.registry_version,
        categories=config.categories or None,
        provenances=config.provenances or None,
    )


def timing_line(seconds: dict[str, float], epochs: int) -> str:
    per_epoch = seconds["train"] / max(1, epochs)
    parts = [f"{name} {value / 60:.1f} min" for name, value in seconds.items()]
    return f"{', '.join(parts)}; {epochs} epochs at {per_epoch:.1f} s each; total {sum(seconds.values()) / 60:.1f} min"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the recommender.")
    parser.add_argument("--config", default=None, help="training config YAML")
    parser.add_argument("--db", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--skip-behavioural",
        action="store_true",
        help="skip the behavioural gate (for a quick smoke run only)",
    )
    args = parser.parse_args()

    config = TrainConfig.load(args.config) if args.config else TrainConfig()
    if args.db:
        config.db = args.db
    if args.name:
        config.name = args.name
    if args.epochs is not None:
        config.optim.epochs = args.epochs
    if args.dim is not None:
        config.arch.dim = args.dim
    if args.batch_size is not None:
        config.optim.batch_size = args.batch_size

    config.db = str(config.db or db_path())
    engine = create_db_engine(config.db)
    with session_scope(engine) as session:
        registry, registry_blob = resolve_registry(config, session)
        snapshot_hash, snapshot_dir = resolve_snapshot(config, session)

    started = time.perf_counter()
    prepared = load_or_prepare(registry, snapshot_dir)
    seconds = {"prepare": time.perf_counter() - started}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config.output_dir) / f"{config.name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config.dump(run_dir / "config.yaml")

    print(f"registry {config.registry_version}  snapshot {snapshot_hash[:12]}")
    print(f"run {run_dir}")
    hardware = machine.describe(config.resolved_device())
    print(f"machine {machine.summary(hardware)}")

    def log(record: EpochRecord) -> None:
        print(record.line())

    started = time.perf_counter()
    model, history = train(config, registry, prepared, on_epoch=log)
    seconds["train"] = time.perf_counter() - started
    write_history(run_dir / "history.json", history)

    # Saved before evaluation, so a fault there cannot lose the trained weights.
    checkpoint_path = checkpoint_module.save(
        run_dir / "model.pt",
        model,
        registry,
        registry_blob,
        checkpoint_module.CheckpointMeta(
            registry_version=config.registry_version,
            registry_content_hash=registry.content_hash,
            snapshot_hash=snapshot_hash,
            split_key=config.split_key,
            created_at=checkpoint_module.now(),
            metrics={},
            notes=config.notes,
        ),
    )
    print(f"checkpoint {checkpoint_path}")

    device = config.resolved_device()
    started = time.perf_counter()
    results = report_module.evaluate_all(
        model, registry, prepared, device=device, fold="test"
    )
    seconds["evaluate"] = time.perf_counter() - started

    suite = None
    if not args.skip_behavioural:
        started = time.perf_counter()
        suite = behavioural.run(model, registry, device=device)
        seconds["behavioural"] = time.perf_counter() - started
        results["behavioural"] = {
            "passed": suite.passed,
            "summary": suite.summary(),
            "failures": [str(a) for a in suite.failures],
            "no_response": [str(a) for a in suite.flat],
        }

    (run_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    timing = timing_line(seconds, len(history))
    (run_dir / "run.json").write_text(
        json.dumps({"machine": hardware, "seconds": seconds, "epochs": len(history)}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        report_module.render(results, config.name)
        + f"\n## Run\n\n- Machine: {machine.summary(hardware)}\n- Time: {timing}\n",
        encoding="utf-8",
    )
    print(f"time {timing}")

    checkpoint_module.save(
        run_dir / "model.pt",
        model,
        registry,
        registry_blob,
        checkpoint_module.CheckpointMeta(
            registry_version=config.registry_version,
            registry_content_hash=registry.content_hash,
            snapshot_hash=snapshot_hash,
            split_key=config.split_key,
            created_at=checkpoint_module.now(),
            metrics=results.get("overall", {}),
            notes=config.notes,
        ),
    )

    print()
    print(report_module.render(results, config.name))
    if suite is not None:
        if suite.flat:
            print(
                f"\nnote: {len(suite.flat)} assertion(s) had no measurable "
                "response. Not a gate failure -- it means the data never "
                "isolates those indicators. See report.md."
            )
        if not suite.passed:
            print(f"\nbehavioural gate FAILED: {len(suite.failures)} assertion(s)")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
