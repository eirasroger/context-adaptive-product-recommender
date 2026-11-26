"""Compare two runs on the test sets neither trained on: by category and provenance, with paired bootstrap intervals on request."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from torch.utils.data import DataLoader

from core.dataset import ComparisonSetDataset, collate
from core.prepare import Prepared, load_or_prepare
from eval import metrics as metrics_module
from model import checkpoint as checkpoint_module
from snapshot.build import SNAPSHOT_ROOT

METRICS = ("gap_fidelity", "pointwise_mae", "top1_regret", "top1_agreement", "tie_tolerant_tau")
HEADERS = ("Gap", "MAE", "Regret", "Top-1", "Tau")


def load_run(run: Path):
    model, meta, _ = checkpoint_module.load(run / "model.pt")
    registry = checkpoint_module.load_registry(run / "model.pt")
    return model, load_or_prepare(registry, SNAPSHOT_ROOT / meta.snapshot_hash)


def untrained(external_ids: Sequence[str], test_sets: Iterable[int], old_fold_by_id: Mapping[str, str]) -> list[int]:
    """Test sets of the new run that the old run never trained on: in its test fold, or absent from its snapshot."""
    return [s for s in test_sets if old_fold_by_id.get(external_ids[s], "test") == "test"]


def rows_by_set(predictions: metrics_module.Predictions) -> dict[int, np.ndarray]:
    rows: dict[int, list[int]] = defaultdict(list)
    for row, index in enumerate(predictions.set_index):
        rows[int(index)].append(row)
    return {index: np.asarray(found) for index, found in rows.items()}


def scored(predictions: metrics_module.Predictions, rows: Mapping[int, np.ndarray], sets: Sequence[int]) -> np.ndarray:
    computed = metrics_module.compute(predictions.take(np.concatenate([rows[s] for s in sets]))).as_dict()
    return np.array([computed[name] for name in METRICS])


def groups_by_category(prepared: Prepared, sets: Sequence[int]) -> dict[str, list[int]]:
    """All, then each category followed by its provenances."""
    groups: dict[str, list[int]] = defaultdict(list)
    for s in sets:
        category = prepared.category_keys[int(prepared.set_category[s])]
        for name in ("All", category, f"{category}, {prepared.set_provenance[s]}"):
            groups[name].append(s)
    ordered = {"All": groups["All"]}
    for category in prepared.category_keys:
        if category in groups:
            ordered[category] = groups[category]
            for name in sorted(k for k in groups if k.startswith(f"{category}, ")):
                ordered[name] = groups[name]
    return ordered


def strata_within(prepared: Prepared, sets: Sequence[int]) -> dict[str, list[int]]:
    """One category's sets by number of contexts and by shortlist kind."""
    def contexts(s: int) -> int:
        return len(prepared.combo_context_slots[int(prepared.set_combo[s])])

    def kind(s: int) -> str:
        generator = prepared.set_generator[s]
        if generator == "across":
            return "across"
        return "within" if isinstance(generator, str) and generator.startswith("within") else "other"

    strata = {
        "All": list(sets),
        "Baseline or one context": [s for s in sets if contexts(s) <= 1],
        "Two contexts": [s for s in sets if contexts(s) == 2],
        "Three or four contexts": [s for s in sets if contexts(s) >= 3],
        "Within one typology": [s for s in sets if kind(s) == "within"],
        "Across typologies": [s for s in sets if kind(s) == "across"],
    }
    return {name: found for name, found in strata.items() if found}


def table(rows: list[tuple[str, int, list[str]]]) -> str:
    lines = ["| Test sets | Sets | " + " | ".join(HEADERS) + " |", "|---|---|" + "---|" * len(HEADERS)]
    lines += [f"| {name} | {count:,} | " + " | ".join(cells) + " |" for name, count, cells in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_run", type=Path)
    parser.add_argument("new_run", type=Path)
    parser.add_argument("--bootstrap", metavar="CATEGORY", help="also give this category's LLM sets by stratum, with intervals")
    parser.add_argument("--resamples", type=int, default=500)
    args = parser.parse_args()

    old_model, old_prepared = load_run(args.old_run)
    new_model, prepared = load_run(args.new_run)
    loader = DataLoader(ComparisonSetDataset(prepared, fold="test"), batch_size=256, shuffle=False, collate_fn=collate)
    old_predictions = metrics_module.predict(old_model, loader)
    new_predictions = metrics_module.predict(new_model, loader)
    old_rows, new_rows = rows_by_set(old_predictions), rows_by_set(new_predictions)

    old_fold_by_id = dict(zip(old_prepared.set_external_ids, old_prepared.set_fold))
    sets = untrained(prepared.set_external_ids, new_rows, old_fold_by_id)
    print(f"{len(sets):,} of {len(new_rows):,} test sets neither run trained on; old → new\n")

    rows = []
    for name, found in groups_by_category(prepared, sets).items():
        before, after = scored(old_predictions, old_rows, found), scored(new_predictions, new_rows, found)
        rows.append((name, len(found), [f"{b:.3f} → {a:.3f}" for b, a in zip(before, after)]))
    print(table(rows))

    if args.bootstrap:
        index = prepared.category_keys.index(args.bootstrap)
        llm = [s for s in sets if prepared.set_category[s] == index and prepared.set_provenance[s] == "llm"]
        rng = np.random.default_rng(0)
        rows = []
        for name, found in strata_within(prepared, llm).items():
            found = np.asarray(found)
            before, after = scored(old_predictions, old_rows, found), scored(new_predictions, new_rows, found)
            changes = []
            for _ in range(args.resamples):
                sample = rng.choice(found, size=len(found), replace=True)
                changes.append(scored(new_predictions, new_rows, sample) - scored(old_predictions, old_rows, sample))
            low, high = np.percentile(np.asarray(changes), [2.5, 97.5], axis=0)
            rows.append((name, len(found), [
                f"{b:.3f} → {a:.3f} [{lo:+.3f}, {hi:+.3f}]" for b, a, lo, hi in zip(before, after, low, high)
            ]))
        print(f"\n{args.bootstrap}, LLM sets; paired bootstrap over sets, {args.resamples} resamples, 95 % interval of the change\n")
        print(table(rows))


if __name__ == "__main__":
    main()
