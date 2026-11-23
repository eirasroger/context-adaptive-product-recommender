"""Epoch budget: the baseline trained at several epoch counts over several seeds on one split and snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from train.concurrent import Job, run_jobs, sizing_line

PREFIX = "epochs"
BUDGETS = (30, 45, 60)
SEEDS = (0, 1, 2)
#: Judged strata the budget is chosen on; the expert stratum is reported only, at 41 sets.
DECIDING_SCOPES = ("concrete/llm", "facade_system/llm")
REPORTED_SCOPES = ("overall", *DECIDING_SCOPES, "concrete/expert")
METRICS = ("gap_fidelity", "pointwise_mae", "top1_regret")
SECONDARY = ("top1_agreement", "tie_tolerant_tau")
SUMMARY = "summary.json"
REPORT = "report.md"
CURVES = "curves.png"


def arm_name(budget: int, seed: int) -> str:
    return f"{PREFIX}{budget}-seed{seed}"


def configure(config_path: Path, budget: int, seed: int, out: Path, snapshot: str):
    from train.config import TrainConfig

    config = TrainConfig.load(config_path)
    config.name = arm_name(budget, seed)
    config.output_dir = str(out)
    config.snapshot = snapshot
    config.optim.epochs = budget
    config.optim.seed = seed
    config.promote = False
    return config


def shared_snapshot(config_path: Path) -> str:
    """The config's split built once and its prepared cache warmed, so the runs only read; its hash."""
    from core.prepare import load_or_prepare
    from db.session import create_db_engine, db_path, session_scope
    from snapshot.build import build
    from train.config import TrainConfig
    from train.run import resolve_registry

    config = TrainConfig.load(config_path)
    with session_scope(create_db_engine(str(config.db or db_path()))) as session:
        registry, _ = resolve_registry(config, session)
        snapshot, directory = build(
            session, config.split_key, config.registry_version,
            categories=config.categories or None, provenances=config.provenances or None,
        )
    load_or_prepare(registry, directory)
    return snapshot


def scope_cell(results: dict, scope: str) -> dict | None:
    if scope == "overall":
        return results["overall"]
    return results.get("judged", {}).get("category_provenance", {}).get(scope)


def finished_runs(out: Path) -> list[dict]:
    """Budget, seed, epochs run, kept epoch, minutes and results of every run in ``out`` that reached its metrics."""
    import yaml

    runs = {}
    for metrics in sorted(out.glob("*/metrics.json")):
        directory = metrics.parent
        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        timing = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        kept = min(history, key=lambda record: record["val_loss"])
        budget, seed = config["optim"]["epochs"], config["optim"]["seed"]
        behavioural = json.loads(metrics.read_text(encoding="utf-8")).get("behavioural")
        runs[(budget, seed)] = {
            "budget": budget,
            "seed": seed,
            "run": directory.name,
            "epochs_run": len(history),
            "kept_epoch": kept["epoch"],
            "val_loss": kept["val_loss"],
            "minutes": timing["seconds"]["train"] / 60,
            "behavioural": "skipped" if behavioural is None
            else "pass" if behavioural["passed"] else f"{len(behavioural['failures'])} failed",
            "results": json.loads(metrics.read_text(encoding="utf-8")),
        }
    return [runs[key] for key in sorted(runs)]


def summarise(runs: Sequence[dict]) -> dict[int, dict[str, dict[str, float]]]:
    """Per budget and scope, the mean and seed standard deviation of each metric."""
    summary: dict[int, dict[str, dict[str, float]]] = {}
    for budget in sorted({run["budget"] for run in runs}):
        arm = [run for run in runs if run["budget"] == budget]
        summary[budget] = {}
        for scope in REPORTED_SCOPES:
            cells = [cell for run in arm if (cell := scope_cell(run["results"], scope)) is not None]
            if not cells:
                continue
            row = {"seeds": len(cells), "n_sets": cells[0]["n_sets"]}
            for metric in (*METRICS, *SECONDARY):
                values = np.array([cell[metric] for cell in cells], dtype=float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            summary[budget][scope] = row
    return summary


def within_noise(candidate: dict, best: dict, metric: str) -> bool:
    """A budget is level with the best when its mean is worse by no more than the larger seed spread of the two."""
    spread = max(candidate[f"{metric}_sd"], best[f"{metric}_sd"])
    return candidate[f"{metric}_mean"] - best[f"{metric}_mean"] <= spread


def shortest_level_budget(summary: dict[int, dict[str, dict[str, float]]]) -> int | None:
    """The shortest budget within noise of the best on every deciding scope and metric (all lower-is-better)."""
    for budget in sorted(summary):
        level = True
        for scope in DECIDING_SCOPES:
            for metric in METRICS:
                arms = {b: summary[b][scope] for b in summary if scope in summary[b]}
                if scope not in summary[budget] or len(arms) < 2:
                    return None
                best = min(arms.values(), key=lambda row: row[f"{metric}_mean"])
                level &= within_noise(summary[budget][scope], best, metric)
        if level:
            return budget
    return None


def render(runs: Sequence[dict], summary: dict, sizing: dict | None) -> str:
    budgets = sorted(summary)
    lines = [
        "# Epoch budget",
        "",
        f"Budgets {', '.join(map(str, budgets))} over seeds "
        f"{', '.join(map(str, sorted({run['seed'] for run in runs})))}, one split and snapshot. "
        "Mean ± seed standard deviation on the test fold; judged strata exclude control cases.",
        "",
        *([sizing_line(sizing), ""] if sizing else []),
    ]
    for scope in REPORTED_SCOPES:
        rows = [(budget, summary[budget][scope]) for budget in budgets if scope in summary[budget]]
        if not rows:
            continue
        lines += [
            f"## {scope} ({rows[0][1]['n_sets']:,} sets)",
            "",
            "| Epochs | Gap | MAE | Regret | Top-1 | Tau |",
            "|---|---|---|---|---|---|",
        ]
        for budget, row in rows:
            cells = " | ".join(
                f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_sd']:.4f}" for metric in (*METRICS, *SECONDARY)
            )
            lines.append(f"| {budget} | {cells} |")
        lines.append("")

    lines += [
        "## Every run",
        "",
        "| Run | Epochs run | Kept | Validation loss | Minutes | Behavioural | Facade LLM gap | Concrete LLM gap |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        facade = scope_cell(run["results"], "facade_system/llm")
        concrete = scope_cell(run["results"], "concrete/llm")
        lines.append(
            f"| {run['run']} | {run['epochs_run']} | {run['kept_epoch']} | {run['val_loss']:.4f} "
            f"| {run['minutes']:.1f} | {run['behavioural']} "
            f"| {facade['gap_fidelity']:.4f} | {concrete['gap_fidelity']:.4f} |"
        )
    chosen = shortest_level_budget(summary)
    lines += [
        "",
        "Shortest budget level with the best on gap, MAE and regret in "
        f"{' and '.join(DECIDING_SCOPES)}: " + (f"**{chosen} epochs**." if chosen else "none yet."),
    ]
    return "\n".join(lines) + "\n"


def write_summary(out: Path, sizing: dict | None) -> None:
    runs = finished_runs(out)
    if not runs:
        return
    summary = summarise(runs)
    listed = [{key: value for key, value in run.items() if key != "results"} for run in runs]
    (out / SUMMARY).write_text(
        json.dumps({"runs": listed, "sizing": sizing, "budgets": summary,
                    "shortest_level_budget": shortest_level_budget(summary)}, indent=2),
        encoding="utf-8",
    )
    (out / REPORT).write_text(render(runs, summary, sizing), encoding="utf-8")


def draw_curves(out: Path, seed: int) -> Path | None:
    from train import curves

    runs = {run["budget"]: out / run["run"] for run in finished_runs(out) if run["seed"] == seed}
    if len(runs) < 2:
        return None
    return curves.plot_runs(
        {f"{budget} epochs": curves.read_history(path) for budget, path in sorted(runs.items())}, out / CURVES
    )


def run(
    config_path: Path, budgets: Sequence[int], seeds: Sequence[int], behavioural_suite: bool,
    parallel: int | None = None, out: Path | None = None,
) -> Path:
    from train.config import TrainConfig

    base = TrainConfig.load(config_path)
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path(base.output_dir) / f"{PREFIX}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    snapshot = shared_snapshot(config_path)
    print(f"snapshot {snapshot[:12]}", flush=True)
    done = {(run["budget"], run["seed"]) for run in finished_runs(out)}
    jobs = []
    for budget in sorted(budgets, reverse=True):
        for seed in seeds:
            if (budget, seed) in done:
                continue
            config_file = configure(config_path, budget, seed, out, snapshot).dump(out / f"{arm_name(budget, seed)}.yaml")
            jobs.append(Job(
                name=config_file.stem,
                command=[sys.executable, "-u", "-m", "train.run", "--config", str(config_file)]
                + ([] if behavioural_suite else ["--skip-behavioural"]),
                log=config_file.with_suffix(".log"),
            ))
    sizing, _ = run_jobs(jobs, base.resolved_device(), parallel, on_finish=lambda job, code: write_summary(out, None))
    write_summary(out, sizing)
    draw_curves(out, seed=min(seeds))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("train/configs/default.yaml"))
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--skip-behavioural", action="store_true")
    parser.add_argument(
        "--parallel", type=int, default=None,
        help="runs trained at once; by default sized to this machine from the first run's footprint",
    )
    parser.add_argument("--out", type=Path, default=None, help="an earlier study's directory, to finish or extend it")
    args = parser.parse_args()

    out = run(args.config, args.budgets, args.seeds, not args.skip_behavioural, args.parallel, args.out)
    if (out / REPORT).exists():
        print((out / REPORT).read_text(encoding="utf-8"))
    print(f"runs in {out}")


if __name__ == "__main__":
    main()
