"""The generic model against the 2025 concrete-only precursor, on the published concrete scenarios and the same folds."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from baselines.precursor_2025 import encoding as precursor
from core.dataset import ComparisonSetDataset, collate
from core.prepare import load_or_prepare
from db.session import create_db_engine, session_scope
from eval import metrics as M
from ingest.adapters.concrete_v1 import SOURCE_KEY as PUBLISHED_SOURCE, classify_provenance
from model import checkpoint as checkpoint_module
from snapshot.build import SNAPSHOT_ROOT, build
from train.concurrent import Job, run_jobs, sizing_line
from train.config import TrainConfig
from train.run import resolve_registry

ARMS = ("precursor", "generic_concrete", "generic_joint")
DEFAULT_CONFIG = Path("train/configs/default.yaml")
OUTPUT_ROOT = Path("runs")

#: Table 2 of the precursor paper: test-set metrics by number of alternatives.
PAPER_TABLE_2 = {
    2: {"sets": 3284, "mrr": 0.98, "ndcg": 0.97, "mae": 0.05, "mse": 0.007},
    3: {"sets": 3209, "mrr": 0.95, "ndcg": 0.95, "mae": 0.05, "mse": 0.007},
    4: {"sets": 3285, "mrr": 0.93, "ndcg": 0.92, "mae": 0.05, "mse": 0.007},
    5: {"sets": 3085, "mrr": 0.89, "ndcg": 0.89, "mae": 0.05, "mse": 0.008},
}

#: Reported per stratum, in this order.
METRICS = {
    "gap": M.gap_fidelity,
    "mae": M.pointwise_mae,
    "mse": lambda p: float(np.mean((p.predicted - p.actual) ** 2)) if len(p) else float("nan"),
    "regret": M.top1_regret,
    "top1": M.top1_agreement,
    "tau": M.tie_tolerant_tau,
    "mrr": M.mean_reciprocal_rank,
    "ndcg": M.ndcg,
}

BySet = dict[str, tuple[np.ndarray, np.ndarray]]


def by_set(predictions: M.Predictions, ids: list[str]) -> BySet:
    """Predicted and labelled scores per set id, alternatives in source order."""
    out: dict[str, tuple[list, list, list]] = {}
    for index, position, predicted, actual in zip(
        predictions.set_index, predictions.position, predictions.predicted, predictions.actual
    ):
        entry = out.setdefault(ids[int(index)], ([], [], []))
        entry[0].append(position)
        entry[1].append(predicted)
        entry[2].append(actual)
    return {
        sid: (np.asarray(p, dtype=float)[np.argsort(pos)], np.asarray(a, dtype=float)[np.argsort(pos)])
        for sid, (pos, p, a) in out.items()
    }


def assemble(sets: BySet, ids: list[str]) -> M.Predictions:
    sizes = [len(sets[sid][0]) for sid in ids]
    if not ids:
        empty = np.array([], dtype=float)
        return M.Predictions(empty, empty, empty, empty, empty)
    return M.Predictions(
        set_index=np.repeat(np.arange(len(ids)), sizes),
        position=np.concatenate([np.arange(n) for n in sizes]),
        predicted=np.concatenate([sets[sid][0] for sid in ids]),
        actual=np.concatenate([sets[sid][1] for sid in ids]),
        confidence=np.ones(sum(sizes)),
    )


def strata(ids: list[str], sets: BySet) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"all": list(ids)}
    for sid in ids:
        groups.setdefault(classify_provenance(sid)[0], []).append(sid)
        groups.setdefault(f"n={len(sets[sid][0])}", []).append(sid)
    return groups


def score(sets: BySet, ids: list[str]) -> dict[str, float]:
    predictions = assemble(sets, ids)
    return {"sets": len(ids), **{name: metric(predictions) for name, metric in METRICS.items()}}


def batch_grouped_retrieval(predictions: M.Predictions, batch_size: int) -> tuple[float, float]:
    """MRR and nDCG grouped as the 2025 evaluation script grouped them: relevance per set, queries keyed by position within a batch."""
    relevant = np.zeros(len(predictions), dtype=float)
    for index in np.unique(predictions.set_index):
        rows = predictions.set_index == index
        relevant[rows] = predictions.actual[rows] >= predictions.actual[rows].max() - 1e-9
    reciprocal, gains = [], []
    for key in np.unique(predictions.set_index % batch_size):
        rows = predictions.set_index % batch_size == key
        ranked = relevant[rows][np.argsort(-predictions.predicted[rows], kind="stable")]
        discounts = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
        reciprocal.append(1.0 / (1 + int(np.argmax(ranked))))
        gains.append(float((ranked * discounts).sum() / (np.sort(ranked)[::-1] * discounts).sum()))
    return float(np.mean(reciprocal)), float(np.mean(gains))


def reproduce(frozen_path: Path, labelled_path: Path, output: Path) -> None:
    """Score the published checkpoint on the published split: a faithful copy recovers Table 2."""
    frozen, labels = precursor.read_published(frozen_path, labelled_path)
    scenarios = precursor.encode(frozen, labels, [s["id"] for s in frozen])
    _, test = precursor.paper_split(scenarios)
    model = precursor.published_model(test[0]["features"].shape[1])

    rows = {}
    for size, paper in PAPER_TABLE_2.items():
        predictions, ids = precursor.predict(model, [s for s in test if len(s["features"]) == size])
        mrr_grouped, ndcg_grouped = batch_grouped_retrieval(predictions, precursor.source_config.BATCH_SIZE)
        rows[size] = {
            "sets": len(ids),
            "mrr_per_set": M.mean_reciprocal_rank(predictions),
            "mrr_batch_grouped": mrr_grouped,
            "ndcg_per_set": M.ndcg(predictions),
            "ndcg_batch_grouped": ndcg_grouped,
            "mae": M.pointwise_mae(predictions),
            "mse": METRICS["mse"](predictions),
        }

    lines = [
        "# Precursor reproduction",
        "",
        f"Published checkpoint `{precursor.PUBLISHED_CHECKPOINT.name}` on the published 70/30 split "
        f"(seed {precursor.PAPER_SPLIT_SEED}). Paper values in brackets. The grouped columns follow "
        f"the 2025 evaluation script, which keyed queries by position within a batch of "
        f"{precursor.source_config.BATCH_SIZE}; the per-set columns score each set as one query.",
        "",
        "| Alternatives | Sets | MRR grouped | MRR per set | nDCG grouped | nDCG per set | MAE | MSE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for size, ours in rows.items():
        paper = PAPER_TABLE_2[size]
        lines.append(
            f"| {size} | {ours['sets']:,} ({paper['sets']:,}) "
            f"| {ours['mrr_batch_grouped']:.3f} ({paper['mrr']:.2f}) | {ours['mrr_per_set']:.3f} "
            f"| {ours['ndcg_batch_grouped']:.3f} ({paper['ndcg']:.2f}) | {ours['ndcg_per_set']:.3f} "
            f"| {ours['mae']:.3f} ({paper['mae']:.2f}) | {ours['mse']:.4f} ({paper['mse']:.3f}) |"
        )
    _write(output, "reproduction", lines, {str(size): row for size, row in rows.items()})


def prepare_snapshots(config: TrainConfig) -> tuple[dict[str, str], dict[str, str]]:
    """Build each generic arm's snapshot and warm its prepared cache, so the training processes only read.

    Returns the snapshot hash per generic arm, and this repo's fold for every published concrete scenario.
    """
    with session_scope(create_db_engine(config.db)) as session:
        registry, _ = resolve_registry(config, session)
        published = build(
            session, config.split_key, config.registry_version,
            categories=["concrete"], sources=[PUBLISHED_SOURCE],
        )
        joint = build(
            session, config.split_key, config.registry_version,
            categories=config.categories or None, provenances=config.provenances or None,
        )
    prepared = load_or_prepare(registry, published[1])
    load_or_prepare(registry, joint[1])
    folds = dict(zip(prepared.set_external_ids, prepared.set_fold))
    return {"generic_concrete": published[0], "generic_joint": joint[0]}, folds


def precursor_seed(frozen_path: Path, labelled_path: Path, snapshot: str, seed: int, out: Path) -> None:
    """Train the precursor once on the published snapshot's folds; write its test-fold predictions."""
    config = TrainConfig.load(DEFAULT_CONFIG)
    with session_scope(create_db_engine(config.db)) as session:
        registry, _ = resolve_registry(config, session)
    prepared = load_or_prepare(registry, SNAPSHOT_ROOT / snapshot)
    folds = dict(zip(prepared.set_external_ids, prepared.set_fold))
    frozen, labels = precursor.read_published(frozen_path, labelled_path)

    fold_ids = {fold: {sid for sid, f in folds.items() if f == fold} for fold in ("train", "val", "test")}
    scenarios = precursor.encode(frozen, labels, fold_ids["train"])
    split = {fold: [s for s in scenarios if s["id"] in ids] for fold, ids in fold_ids.items()}
    model = precursor.fit(split["train"], split["val"], seed)
    sets = by_set(*precursor.predict(model, split["test"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({sid: [p.tolist(), a.tolist()] for sid, (p, a) in sets.items()}), encoding="utf-8")


def generic_predictions(run_dir: Path, device: str) -> tuple[BySet, dict[str, str]]:
    """A finished generic run's test-fold predictions, and the fold of every set in its snapshot."""
    meta = checkpoint_module.describe(run_dir / "model.pt")["meta"]
    registry = checkpoint_module.load_registry(run_dir / "model.pt")
    model, _, _ = checkpoint_module.load(run_dir / "model.pt", registry=registry, device=device)
    prepared = load_or_prepare(registry, SNAPSHOT_ROOT / str(meta["snapshot_hash"]))
    loader = DataLoader(ComparisonSetDataset(prepared, fold="test"), batch_size=256, shuffle=False, collate_fn=collate)
    predictions = M.predict(model, loader, device)
    return by_set(predictions, list(prepared.set_external_ids)), dict(zip(prepared.set_external_ids, prepared.set_fold))


def _jobs(
    base: TrainConfig, snapshots: dict[str, str], frozen_path: Path, labelled_path: Path,
    seeds: list[int], arms: list[str], output: Path,
) -> list[Job]:
    """One process per arm and seed, longest first; the first is also the heaviest, so its footprint sizes the rest."""
    jobs = []
    for arm in sorted(arms, key=lambda a: ("generic_joint", "precursor", "generic_concrete").index(a)):
        for seed in seeds:
            name = f"{arm}-seed{seed}"
            if arm == "precursor":
                command = [
                    sys.executable, "-u", "-m", "experiments.generic_vs_precursor", "precursor-seed",
                    str(frozen_path), str(labelled_path), "--seed", str(seed),
                    "--snapshot", snapshots["generic_concrete"], "--out", str(output / "precursor" / f"seed{seed}.json"),
                ]
            else:
                config = replace(
                    base, name=name, snapshot=snapshots[arm], optim=replace(base.optim, seed=seed),
                    notes=f"generic_vs_precursor, arm {arm}, seed {seed}",
                )
                config_file = config.dump(output / "jobs" / f"{name}.yaml")
                command = [sys.executable, "-u", "-m", "train.run", "--config", str(config_file), "--skip-behavioural"]
            jobs.append(Job(name=name, command=command, log=output / "logs" / f"{name}.log"))
    return jobs


def compare(
    frozen_path: Path, labelled_path: Path, seeds: list[int], arms: list[str], output: Path,
    epochs: int | None = None, parallel: int | None = None,
) -> None:
    base = TrainConfig.load(DEFAULT_CONFIG)
    base.promote = False
    base.output_dir = str(output)
    if epochs is not None:
        base.optim.epochs = epochs
    snapshots, folds = prepare_snapshots(base)
    test_ids = sorted(sid for sid, fold in folds.items() if fold == "test")

    jobs = _jobs(base, snapshots, frozen_path, labelled_path, seeds, arms, output)
    sizing, exits = run_jobs(jobs, base.resolved_device(), parallel)
    failed = sorted(name for name, code in exits.items() if code)
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}; logs in {output / 'logs'}")

    results: dict[str, dict[int, BySet]] = {arm: {} for arm in arms}
    for arm in arms:
        for seed in seeds:
            if arm == "precursor":
                raw = json.loads((output / "precursor" / f"seed{seed}.json").read_text(encoding="utf-8"))
                results[arm][seed] = {sid: (np.asarray(p), np.asarray(a)) for sid, (p, a) in raw.items()}
                continue
            (run_dir,) = [path.parent for path in output.glob(f"{arm}-seed{seed}-*/model.pt")]
            sets, run_folds = generic_predictions(run_dir, base.resolved_device())
            moved = [sid for sid in test_ids if run_folds.get(sid) != "test"]
            if moved:
                raise SystemExit(f"{arm}: {len(moved)} published test sets sit in another fold, e.g. {moved[:3]}")
            results[arm][seed] = sets

    scored = [sid for sid in test_ids if all(sid in runs[s] for runs in results.values() for s in seeds)]
    reference = results[arms[0]][seeds[0]]
    common = [
        sid for sid in scored
        if all(
            np.allclose(np.sort(results[arm][seed][sid][1]), np.sort(reference[sid][1]), atol=1e-6)
            for arm in arms for seed in seeds
        )
    ]

    groups = strata(common, reference)
    summary = {
        stratum: {arm: {seed: score(results[arm][seed], ids) for seed in seeds} for arm in arms}
        for stratum, ids in groups.items()
    }
    lines = _comparison_lines(summary, arms, seeds, len(test_ids), len(scored), len(common))
    _write(output, "comparison", lines + ["", sizing_line(sizing)], summary)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _comparison_lines(
    summary: dict, arms: list[str], seeds: list[int], n_test: int, n_scored: int, n_common: int
) -> list[str]:
    against = [arm for arm in arms if arm != "precursor"] if "precursor" in arms else []
    lines = [
        "# Generic model against the precursor",
        "",
        f"Published concrete test fold: {n_test:,} sets; {n_scored:,} scored by every arm; "
        f"{n_scored - n_common:,} dropped because their labels differ between arms. "
        f"Seeds {', '.join(map(str, seeds))}. Mean ± standard deviation over seeds; the difference "
        "to the precursor carries a 95 % interval from the seed spread.",
        "",
        "## Per seed, all sets",
        "",
        "A seed far from the others is a failed training; it widens its arm's spread below.",
        "",
        "| Arm | " + " | ".join(f"seed {seed}" for seed in seeds) + " |",
        "|---" * (1 + len(seeds)) + "|",
    ]
    for arm in arms:
        cells = [
            f"gap {summary['all'][arm][seed]['gap']:.4f}, tau {summary['all'][arm][seed]['tau']:.3f}"
            for seed in seeds
        ]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    for stratum, by_arm in summary.items():
        sets = next(iter(next(iter(by_arm.values())).values()))["sets"]
        columns = ["Metric", *arms, *(f"{arm} - precursor" for arm in against)]
        lines += ["", f"## {stratum} ({sets:,} sets)", "", "| " + " | ".join(columns) + " |", "|---" * len(columns) + "|"]
        for metric in METRICS:
            cells, stats = [], {}
            for arm in arms:
                mean, sd = _mean_sd([by_arm[arm][s][metric] for s in seeds])
                stats[arm] = (mean, sd)
                cells.append(f"{mean:.4f} ± {sd:.4f}")
            for arm in against:
                (m1, s1), (m0, s0) = stats[arm], stats["precursor"]
                half = 1.96 * np.sqrt((s1**2 + s0**2) / len(seeds))
                cells.append(f"{m1 - m0:+.4f} [{m1 - m0 - half:+.4f}, {m1 - m0 + half:+.4f}]")
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    return lines


def _write(output: Path, stem: str, lines: list[str], data: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / f"{stem}.json").write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("reproduce", "compare", "precursor-seed"):
        command = commands.add_parser(name)
        command.add_argument("scenarios", type=Path, help="frozen_dataset.json of the published corpus")
        command.add_argument("labels", type=Path, help="labelled_dataset.json of the published corpus")
    compare_command = commands.choices["compare"]
    compare_command.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    compare_command.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    compare_command.add_argument("--parallel", type=int, default=None,
                                 help="trainings at once; by default sized to this machine from the first one")
    compare_command.add_argument("--epochs", type=int, default=None,
                                 help="generic training epochs, for a quick check of the pipeline")
    seed_command = commands.choices["precursor-seed"]
    seed_command.description = "One precursor training, started by compare."
    seed_command.add_argument("--seed", type=int, required=True)
    seed_command.add_argument("--snapshot", required=True)
    seed_command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "precursor-seed":
        precursor_seed(args.scenarios, args.labels, args.snapshot, args.seed, args.out)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = OUTPUT_ROOT / f"generic-vs-precursor-{stamp}"
    if args.command == "reproduce":
        reproduce(args.scenarios, args.labels, output)
    else:
        compare(args.scenarios, args.labels, args.seeds, args.arms, output, args.epochs, args.parallel)


if __name__ == "__main__":
    main()
