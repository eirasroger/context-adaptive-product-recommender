"""Repeat probe: the labeller relabels held-out facade shortlists, to measure label noise and how close the model is to it."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from core.scoring import TIE_EPSILON
from ingest.facade.label import ROUND
from ingest.facade.synthesise import DATASET_DIR

PROBE_DIR = DATASET_DIR / "repeat_probe"
CHUNKS_DIR = PROBE_DIR / "chunks"
OUT_DIR = PROBE_DIR / "out"
PROBE_INDEX = PROBE_DIR / "index.json"
LABELS_FILE = DATASET_DIR / "llm_labelled_dataset.json"
BOOTSTRAP = 4000
FACADE = "facade_system"


def held_out_scenarios(snapshot_dir: Path) -> set[str]:
    import pandas as pd

    sets = pd.read_parquet(snapshot_dir / "sets.parquet")
    held = sets[(sets.category_key == FACADE) & (sets.provenance_key == "llm") & (sets.fold == "test")]
    return set(held.external_id)


def sent_records() -> dict[str, dict]:
    """Every scenario as the first labeller saw it, by id."""
    records = {}
    for path in sorted(ROUND.chunks_dir.glob("chunk_*.json")):
        for scenario in json.loads(path.read_text(encoding="utf-8"))["scenarios"]:
            records[scenario["scenario_id"]] = scenario
    return records


def write_chunks(snapshot_dir: Path, count: int, chunks: int, seed: int) -> None:
    if PROBE_INDEX.exists():
        raise SystemExit(f"{PROBE_INDEX} exists; the probe has been sent out already")
    records = sent_records()
    pool = sorted(held_out_scenarios(snapshot_dir) & set(records))
    chosen = random.Random(seed).sample(pool, count)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for number in range(chunks):
        batch = [records[scenario] for scenario in chosen[number::chunks]]
        (CHUNKS_DIR / f"chunk_{number + 1:02d}.json").write_text(
            json.dumps({"scenarios": batch}, indent=1), encoding="utf-8"
        )
    PROBE_INDEX.write_text(
        json.dumps({"seed": seed, "snapshot": snapshot_dir.name, "scenarios": chosen}, indent=1), encoding="utf-8"
    )
    print(f"{count} held-out scenarios in {chunks} chunks under {CHUNKS_DIR}")


def first_labels() -> dict[str, dict[str, float]]:
    """The labels training used, by scenario and product."""
    return {
        record["id"]: {alt["id_prod"]: float(alt["pref"]) for alt in record["labelled_alternatives"]}
        for record in json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    }


def repeat_labels() -> dict[str, dict[str, float]]:
    """The probe's labels, by scenario and product, through the letters the scenario was sent with."""
    letters = json.loads(ROUND.index_file.read_text(encoding="utf-8"))
    labels = {}
    for path in sorted(OUT_DIR.glob("chunk_*.json")):
        for scenario, rows in json.loads(path.read_text(encoding="utf-8"))["labels"].items():
            mapping = letters[scenario]["letters"]
            labels[scenario] = {mapping[row["id_prod"]]: float(row["pref"]) for row in rows}
    return labels


def model_scores(run_dir: Path, scenarios: set[str]) -> dict[str, dict[str, float]]:
    """The run's scores on its test fold, by scenario and product."""
    import torch
    from torch.utils.data import DataLoader

    from core.dataset import ComparisonSetDataset, collate
    from core.prepare import load_or_prepare
    from eval import metrics
    from model import checkpoint
    from snapshot.build import SNAPSHOT_ROOT

    model_path = run_dir / "model.pt"
    snapshot_dir = Path(SNAPSHOT_ROOT) / str(checkpoint.describe(model_path)["meta"]["snapshot_hash"])
    registry = checkpoint.load_registry(model_path)
    prepared = load_or_prepare(registry, snapshot_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = checkpoint.load(model_path, registry=registry, device=device)
    loader = DataLoader(ComparisonSetDataset(prepared, fold="test"), batch_size=256, shuffle=False, collate_fn=collate)
    predictions = metrics.predict(model, loader, device)

    scores: dict[str, dict[str, float]] = {}
    for set_index, position, predicted in zip(predictions.set_index, predictions.position, predictions.predicted):
        scenario = prepared.set_external_ids[int(set_index)]
        if scenario in scenarios:
            key = prepared.member_local_key[int(prepared.set_start[int(set_index)]) + int(position)]
            scores.setdefault(scenario, {})[key] = float(predicted)
    return scores


def agreement(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """One shortlist: gap and pointwise error, regret of the predicted first choice, and decisive pair counts."""
    upper = np.triu_indices(len(actual), k=1)
    predicted_gaps = (predicted[:, None] - predicted[None, :])[upper]
    actual_gaps = (actual[:, None] - actual[None, :])[upper]
    decisive = np.abs(actual_gaps) >= TIE_EPSILON
    agree = np.sign(actual_gaps[decisive]) == np.sign(predicted_gaps[decisive])
    first = int(np.argmax(predicted))
    return {
        "gap": float(np.mean(np.abs(predicted_gaps - actual_gaps))),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "regret": float(actual.max() - actual[first]),
        "top1": float(actual[first] >= actual.max() - 1e-9),
        "concordant": int(agree.sum()),
        "discordant": int((~agree).sum()),
    }


def compare(predicted: dict[str, dict[str, float]], actual: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for scenario in sorted(set(predicted) & set(actual)):
        keys = sorted(actual[scenario])
        if len(keys) >= 2 and set(keys) <= set(predicted[scenario]):
            out[scenario] = agreement(
                np.array([predicted[scenario][k] for k in keys]), np.array([actual[scenario][k] for k in keys])
            )
    return out


def mean_of(*labellings: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    shared = set.intersection(*(set(labels) for labels in labellings))
    return {
        scenario: {key: float(np.mean([labels[scenario][key] for labels in labellings])) for key in labellings[0][scenario]}
        for scenario in shared
    }


def interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.integers(0, len(values), (BOOTSTRAP, len(values)))
    return tuple(np.percentile(values[draws].mean(axis=1), [2.5, 97.5]))


def summary_row(name: str, rows: dict[str, dict[str, float]], rng: np.random.Generator) -> str:
    gaps = np.array([row["gap"] for row in rows.values()])
    low, high = interval(gaps, rng)
    concordant = sum(row["concordant"] for row in rows.values())
    discordant = sum(row["discordant"] for row in rows.values())
    tau = (concordant - discordant) / max(1, concordant + discordant)
    mae, regret, top1 = (np.mean([row[key] for row in rows.values()]) for key in ("mae", "regret", "top1"))
    return (
        f"| {name} | {len(rows)} | {gaps.mean():.4f} [{low:.4f}, {high:.4f}] | {mae:.4f} "
        f"| {regret:.4f} | {top1:.3f} | {tau:.3f} |"
    )


def analyse(run_dir: Path) -> str:
    first, second = first_labels(), repeat_labels()
    scenarios = set(second)
    model = model_scores(run_dir, scenarios)
    both = mean_of(first, second)
    rng = np.random.default_rng(0)

    between = compare(second, first)
    gaps_between = np.array([row["gap"] for row in between.values()])
    model_first = compare(model, first)
    lines = [
        f"# Repeat probe ({len(scenarios)} held-out facade shortlists, model `{run_dir.name}`)",
        "",
        "| Comparison | Sets | Gap [95 %] | MAE | Regret | Top-1 | Tau |",
        "|---|---|---|---|---|---|---|",
        summary_row("Second labels against first", between, rng),
        summary_row("Model against first labels", model_first, rng),
        summary_row("Model against second labels", compare(model, second), rng),
        summary_row("Model against the mean of both", compare(model, both), rng),
        "",
        "With independent noise of equal size in both labellings, a model that predicted the labeller's",
        "expected score exactly would sit at the labels' disagreement divided by the square root of two.",
        f"Floor on gap: {gaps_between.mean() / np.sqrt(2):.4f}; the model against the first labels: "
        f"{np.mean([row['gap'] for row in model_first.values()]):.4f}.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    chunks = commands.add_parser("chunks", help="sample held-out labelled shortlists and write probe chunks")
    chunks.add_argument("--snapshot", type=Path, required=True, help="snapshot directory whose test fold is sampled")
    chunks.add_argument("--sets", type=int, default=200)
    chunks.add_argument("--chunks", type=int, default=5)
    chunks.add_argument("--seed", type=int, default=0)
    analysis = commands.add_parser("analyse", help="compare the repeat labels with the first labels and the model")
    analysis.add_argument("--run", type=Path, required=True, help="a run trained on the probed snapshot")
    args = parser.parse_args()

    if args.command == "chunks":
        write_chunks(args.snapshot, args.sets, args.chunks, args.seed)
    else:
        print(analyse(args.run))


if __name__ == "__main__":
    main()
