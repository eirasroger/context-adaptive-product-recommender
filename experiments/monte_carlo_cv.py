"""Monte Carlo cross-validation: the baseline retrained on fresh random splits, to measure how far its metrics move."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

SPLIT_PREFIX = "mccv"
METRICS = ("gap_fidelity", "pointwise_mae", "top1_agreement", "tie_tolerant_tau")
SUMMARY = "summary.json"
REPORT = "report.md"
#: One-sided 95 % bound on the relative gap difference between two runs of equal quality.
Z_95 = 1.645
#: Share of free cores, memory and GPU memory that concurrent runs may take.
HEADROOM = 0.8


@dataclass(frozen=True)
class Footprint:
    """What one training run uses once it is in its epoch loop."""

    cores: float
    ram_bytes: int
    gpu_bytes: int | None


@dataclass(frozen=True)
class Machine:
    physical_cores: int
    free_ram_bytes: int
    free_gpu_bytes: int | None


def capacity(machine: Machine, footprint: Footprint) -> tuple[int, dict[str, int]]:
    """How many runs fit at once, and the limit each resource sets.

    The machine is measured while the measured run is live, so it counts as one of the memory slots.
    """
    limits = {
        "cpu": int(machine.physical_cores * HEADROOM / max(footprint.cores, 1.0)),
        "ram": 1 + int(machine.free_ram_bytes * HEADROOM / max(footprint.ram_bytes, 1)),
    }
    if machine.free_gpu_bytes is not None and footprint.gpu_bytes:
        limits["gpu"] = 1 + int(machine.free_gpu_bytes * HEADROOM / footprint.gpu_bytes)
    return max(1, min(limits.values())), limits


def split_keys(repeats: int, first: int = 1) -> list[str]:
    return [f"{SPLIT_PREFIX}-{index}" for index in range(first, first + repeats)]


def configure(config_path: Path, split_key: str, output_dir: Path):
    from train.config import TrainConfig

    config = TrainConfig.load(config_path)
    config.split_key = split_key
    config.name = split_key
    config.output_dir = str(output_dir)
    config.snapshot = None
    config.promote = False
    return config


def scopes(results: dict) -> dict[str, dict[str, float]]:
    """Every scored scope of one run, flattened to ``kind/label``."""
    flat = {"overall": results["overall"]}
    for kind, cells in results.get("stratified", {}).items():
        for label, cell in cells.items():
            flat[f"{kind}/{label}"] = cell
    return flat


def gap_tolerance(coefficient_of_variation: float) -> float:
    """The relative gap increase that 95 % of runs of equal quality stay under."""
    return Z_95 * math.sqrt(2) * coefficient_of_variation


def summarise(runs: Sequence[dict]) -> dict[str, dict[str, float]]:
    """Mean, spread and range of each metric in each scope, over the runs that scored it."""
    by_scope: dict[str, list[dict[str, float]]] = {}
    for results in runs:
        for scope, cell in scopes(results).items():
            by_scope.setdefault(scope, []).append(cell)

    summary = {}
    for scope, cells in by_scope.items():
        row = {"runs": len(cells), "n_sets": float(np.mean([c["n_sets"] for c in cells]))}
        for metric in METRICS:
            values = np.array([c[metric] for c in cells], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        cv = row["gap_fidelity_sd"] / row["gap_fidelity_mean"]
        row["gap_cv"] = cv
        row["gap_tolerance"] = gap_tolerance(cv)
        summary[scope] = row
    return summary


def render(
    summary: dict[str, dict[str, float]], splits: Sequence[dict], min_sets: int, sizing: dict | None = None,
) -> str:
    lines = [
        "# Monte Carlo cross-validation",
        "",
        f"{len(splits)} random splits: " + ", ".join(f"`{s['split_key']}`" for s in splits) + ".",
        "",
        "Behavioural suite: " + ", ".join(
            f"{s['split_key']} {s['behavioural']}" for s in splits
        ) + ".",
        "",
        *([sizing_line(sizing), ""] if sizing else []),
        "Mean ± standard deviation across splits. Tolerance is the relative gap increase that",
        "95 % of equally good runs stay under; the regression gate removes test-fold",
        f"variation by rescoring on one fold, so read it as an upper bound. Scopes under {min_sets}",
        "sets are omitted.",
        "",
        "| Scope | Sets | Gap | Gap CV | Tolerance | Error | Top-1 | Tau |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scope, row in summary.items():
        if row["n_sets"] < min_sets:
            continue

        def cell(metric: str) -> str:
            return f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_sd']:.3f}"

        lines.append(
            f"| {scope} | {row['n_sets']:,.0f} | {cell('gap_fidelity')} | {row['gap_cv']:.1%} "
            f"| {row['gap_tolerance']:.0%} | {cell('pointwise_mae')} | {cell('top1_agreement')} "
            f"| {cell('tie_tolerant_tau')} |"
        )
    return "\n".join(lines) + "\n"


def sizing_line(sizing: dict) -> str:
    limits = sizing.get("limits")
    if not limits:
        return f"Trained {sizing['parallel']} at once ({sizing.get('reason', '')})."
    return f"Trained {sizing['parallel']} at once; limits " + ", ".join(
        f"{resource} {limit}" for resource, limit in limits.items()
    ) + "."


def prepare_split(config_path: Path, key: str, out: Path) -> Path:
    """Assign the split and build its snapshot here, so concurrent runs only read; the run's config file."""
    from db.session import create_db_engine, db_path, session_scope
    from ingest import split
    from snapshot.build import build
    from train.run import resolve_registry

    config = configure(config_path, key, out)
    config.db = str(config.db or db_path())
    with session_scope(create_db_engine(config.db)) as session:
        folds = split.assign(session, key, replace=True)
        resolve_registry(config, session)
        config.snapshot, _ = build(
            session, key, config.registry_version,
            categories=config.categories or None, provenances=config.provenances or None,
        )
    print(f"{key}: folds {folds}, snapshot {config.snapshot[:12]}", flush=True)
    return config.dump(out / f"{key}.yaml")


def free_gpu_bytes(device: str) -> int | None:
    """Device-wide free memory; the driver reports other processes' allocations, CUDA's own view on Windows does not."""
    if device != "cuda":
        return None
    try:
        listed = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return int(listed.split()[0]) << 20


def this_machine(device: str) -> Machine:
    import psutil

    return Machine(
        physical_cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        free_ram_bytes=psutil.virtual_memory().available,
        free_gpu_bytes=free_gpu_bytes(device),
    )


def measure(process: subprocess.Popen, log: Path, device: str, gpu_before: int | None) -> Footprint | None:
    """The footprint of a run once its first epoch is logged; None if it exits first."""
    import psutil

    while not any(line.startswith("epoch") for line in log.read_text(encoding="utf-8").splitlines()):
        if process.poll() is not None:
            return None
        time.sleep(2)
    probed = psutil.Process(process.pid)
    cores = probed.cpu_percent(interval=10) / 100
    gpu_after = free_gpu_bytes(device)
    return Footprint(
        cores=cores,
        ram_bytes=probed.memory_info().rss,
        gpu_bytes=None if gpu_before is None or gpu_after is None else max(0, gpu_before - gpu_after),
    )


def size_to_machine(process: subprocess.Popen, log: Path, device: str, gpu_before: int | None) -> dict:
    """Measure the first run, then decide how many can train alongside it."""
    footprint = measure(process, log, device, gpu_before)
    if footprint is None:
        return {"parallel": 1, "reason": "the measured run exited before its first epoch"}
    machine = this_machine(device)
    parallel, limits = capacity(machine, footprint)
    return {"parallel": parallel, "limits": limits, "footprint": asdict(footprint), "machine": asdict(machine)}


def launch(config_file: Path, behavioural_suite: bool) -> subprocess.Popen:
    command = [sys.executable, "-u", "-m", "train.run", "--config", str(config_file)]
    if not behavioural_suite:
        command.append("--skip-behavioural")
    log = config_file.with_suffix(".log").open("w", encoding="utf-8")
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)


def finished_runs(out: Path) -> list[dict]:
    """Split key, run name, behavioural verdict and results of every run in ``out`` that got as far as its metrics."""
    import yaml

    by_split = {}
    for metrics in sorted(out.glob("*/metrics.json")):
        config = yaml.safe_load((metrics.parent / "config.yaml").read_text(encoding="utf-8"))
        results = json.loads(metrics.read_text(encoding="utf-8"))
        behavioural = results.get("behavioural")
        by_split[config["split_key"]] = {
            "split_key": config["split_key"],
            "run": metrics.parent.name,
            "behavioural": "skipped" if behavioural is None
            else "pass" if behavioural["passed"] else f"{len(behavioural['failures'])} failed",
            "results": results,
        }
    return list(by_split.values())


def write_summary(out: Path, min_sets: int, sizing: dict) -> None:
    runs = finished_runs(out)
    if len(runs) < 2:
        return
    summary = summarise([r["results"] for r in runs])
    splits = [{k: v for k, v in r.items() if k != "results"} for r in runs]
    (out / SUMMARY).write_text(
        json.dumps({"splits": splits, "sizing": sizing, "scopes": summary}, indent=2), encoding="utf-8"
    )
    (out / REPORT).write_text(render(summary, splits, min_sets, sizing), encoding="utf-8")


def run(
    config_path: Path, repeats: int, first: int, behavioural_suite: bool, min_sets: int,
    parallel: int | None = None, out: Path | None = None,
) -> Path:
    from train.config import TrainConfig

    base = TrainConfig.load(config_path)
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path(base.output_dir) / f"{SPLIT_PREFIX}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    device = base.resolved_device()

    waiting = [prepare_split(config_path, key, out) for key in split_keys(repeats, first)]
    running: dict[Path, tuple[subprocess.Popen, float]] = {}

    def start(config_file: Path) -> subprocess.Popen:
        process = launch(config_file, behavioural_suite)
        running[config_file] = (process, time.perf_counter())
        print(f"{config_file.stem}: training", flush=True)
        return process

    if parallel is None:
        gpu_before = free_gpu_bytes(device)
        probe = waiting[0]
        sizing = size_to_machine(start(waiting.pop(0)), probe.with_suffix(".log"), device, gpu_before)
    else:
        sizing = {"parallel": parallel, "reason": "set by --parallel"}
    print(f"training {sizing['parallel']} at once: {json.dumps(sizing)}", flush=True)

    while waiting or running:
        while waiting and len(running) < sizing["parallel"]:
            start(waiting.pop(0))
        time.sleep(5)
        for config_file, (process, started) in list(running.items()):
            if process.poll() is None:
                continue
            del running[config_file]
            print(
                f"{config_file.stem}: finished, exit {process.returncode}, "
                f"{(time.perf_counter() - started) / 60:.1f} min",
                flush=True,
            )
            write_summary(out, min_sets, sizing)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("train/configs/default.yaml"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--first", type=int, default=1, help="index of the first split key, to extend an earlier run")
    parser.add_argument("--skip-behavioural", action="store_true")
    parser.add_argument("--min-sets", type=int, default=100, help="smallest scope shown in the report")
    parser.add_argument(
        "--parallel", type=int, default=None,
        help="splits trained at once; by default sized to this machine from the first run's footprint",
    )
    parser.add_argument("--out", type=Path, default=None, help="an earlier run's directory, to add splits to it")
    args = parser.parse_args()

    out = run(
        args.config, args.repeats, args.first, not args.skip_behavioural, args.min_sets,
        args.parallel, args.out,
    )
    if (out / REPORT).exists():
        print((out / REPORT).read_text(encoding="utf-8"))
    print(f"runs in {out}")


if __name__ == "__main__":
    main()
