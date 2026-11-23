"""Tuning studies: settings of the baseline config trained over seeds on one split and snapshot, read against a reference."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from train.concurrent import Job, run_jobs, sizing_line

STUDIES = Path(__file__).with_name("tuning.yaml")
#: Judged strata a setting is read on; the expert stratum is reported only, at 41 sets.
DECIDING_SCOPES = ("concrete/llm", "facade_system/llm")
REPORTED_SCOPES = ("overall", *DECIDING_SCOPES, "concrete/expert")
METRICS = ("gap_fidelity", "pointwise_mae", "top1_regret")
SECONDARY = ("top1_agreement", "tie_tolerant_tau")
SUMMARY = "summary.json"
REPORT = "report.md"
CURVES = "curves.png"


def load_study(name: str, path: Path = STUDIES) -> dict:
    studies = yaml.safe_load(path.read_text(encoding="utf-8"))
    if name not in studies:
        raise SystemExit(f"no study {name!r} in {path}; there are {', '.join(studies)}")
    study = with_trials(studies[name])
    if study["reference"] not in study["settings"]:
        raise SystemExit(f"study {name!r} names reference {study['reference']!r}, which is not one of its settings")
    return study


SEED_SEPARATOR = "-seed"


def sample(spec, rng: random.Random):
    """One draw from a search dimension: a list is a choice, ``{log: [low, high]}`` a log-uniform range."""
    if isinstance(spec, list):
        return rng.choice(spec)
    low, high = spec["log"]
    return float(f"{math.exp(rng.uniform(math.log(low), math.log(high))):.3g}")


def with_trials(study: dict) -> dict:
    """A search study's sampled trials added to its listed settings; the same seed draws the same trials."""
    search = study.get("search")
    if not search:
        return study
    rng = random.Random(search["seed"])
    trials = {
        f"trial-{number:02d}": {key: sample(spec, rng) for key, spec in search["space"].items()}
        for number in range(1, search["trials"] + 1)
    }
    return {**study, "settings": {**(study.get("settings") or {}), **trials}}


def run_name(setting: str, seed: int) -> str:
    return f"{setting}{SEED_SEPARATOR}{seed}"


def setting_of(name: str) -> str:
    return name.rsplit(SEED_SEPARATOR, 1)[0]


def apply_overrides(config, overrides: dict) -> None:
    for dotted, value in overrides.items():
        *path, field = dotted.split(".")
        target = config
        for part in path:
            target = getattr(target, part)
        if not hasattr(target, field):
            raise SystemExit(f"unknown config key {dotted!r}")
        setattr(target, field, value)


def configure(config_path: Path, setting: str, overrides: dict, seed: int, out: Path, snapshot: str):
    from train.config import TrainConfig

    config = TrainConfig.load(config_path)
    apply_overrides(config, overrides)
    config.name = run_name(setting, seed)
    config.output_dir = str(out)
    config.snapshot = snapshot
    config.optim.seed = seed
    config.promote = False
    return config


def cost(config) -> float:
    """A rough relative cost, so the heaviest run starts first and sets the concurrency."""
    arch = config.arch
    return arch.dim ** 2 * (arch.encoder_blocks + arch.comparator_blocks) * config.optim.epochs


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


def parameter_count(checkpoint: Path) -> int:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    return int(sum(tensor.numel() for tensor in state.values()))


def finished_runs(out: Path) -> list[dict]:
    """Setting, seed, size, epochs, timing and results of every run in ``out`` that reached its metrics."""
    runs = {}
    for metrics_path in sorted(out.glob("*/metrics.json")):
        directory = metrics_path.parent
        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        timing = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        results = json.loads(metrics_path.read_text(encoding="utf-8"))
        kept = min(history, key=lambda record: record["val_loss"])
        setting = setting_of(config["name"])
        behavioural = results.get("behavioural")
        runs[(setting, config["optim"]["seed"])] = {
            "setting": setting,
            "seed": config["optim"]["seed"],
            "run": directory.name,
            "parameters": parameter_count(directory / "model.pt"),
            "budget": config["optim"]["epochs"],
            "epochs_run": len(history),
            "kept_epoch": kept["epoch"],
            "val_loss": kept["val_loss"],
            "minutes": timing["seconds"]["train"] / 60,
            "behavioural": "skipped" if behavioural is None
            else "pass" if behavioural["passed"] else f"{len(behavioural['failures'])} failed",
            "results": results,
        }
    return list(runs.values())


def summarise(runs: Sequence[dict], order: Sequence[str]) -> dict[str, dict[str, dict[str, float]]]:
    """Per setting and scope, the mean and seed standard deviation of each metric."""
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for setting in order:
        arm = [run for run in runs if run["setting"] == setting]
        if not arm:
            continue
        summary[setting] = {}
        for scope in REPORTED_SCOPES:
            cells = [cell for run in arm if (cell := scope_cell(run["results"], scope)) is not None]
            if not cells:
                continue
            row = {"seeds": len(cells), "n_sets": cells[0]["n_sets"]}
            for metric in (*METRICS, *SECONDARY):
                values = np.array([cell[metric] for cell in cells], dtype=float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            summary[setting][scope] = row
    return summary


def verdict(candidate: dict, reference: dict, metric: str) -> str:
    """Better, worse or level against the reference, on a lower-is-better metric.

    A difference counts only when it exceeds the larger seed spread of the two settings.
    """
    spread = max(candidate[f"{metric}_sd"], reference[f"{metric}_sd"])
    difference = candidate[f"{metric}_mean"] - reference[f"{metric}_mean"]
    if difference < -spread:
        return "better"
    if difference > spread:
        return "worse"
    return "level"


def verdicts(summary: dict, reference: str) -> dict[str, dict[str, str]]:
    """Each setting's verdict against the reference on every deciding scope and metric."""
    out = {}
    for setting, scopes in summary.items():
        if setting == reference or reference not in summary:
            continue
        out[setting] = {
            f"{scope} {metric}": verdict(scopes[scope], summary[reference][scope], metric)
            for scope in DECIDING_SCOPES
            if scope in scopes and scope in summary[reference]
            for metric in METRICS
        }
    return out


SHORT = {"gap_fidelity": "gap", "pointwise_mae": "MAE", "top1_regret": "regret"}


def render(name: str, study: dict, runs: Sequence[dict], summary: dict, sizing: dict | None) -> str:
    reference = study["reference"]
    by_setting = {setting: [run for run in runs if run["setting"] == setting] for setting in summary}
    lines = [
        f"# Tuning study: {name}",
        "",
        f"{len(summary)} settings over seeds {', '.join(map(str, study['seeds']))}, one split and snapshot, "
        f"each read against `{reference}`. Mean ± seed standard deviation on the test fold; judged strata "
        "exclude control cases. A difference counts when it exceeds the larger seed spread of the two settings.",
        "",
        *([sizing_line(sizing), ""] if sizing else []),
        "## Settings",
        "",
        "| Setting | Overrides | Parameters | Minutes per run | Kept epoch | Behavioural |",
        "|---|---|---|---|---|---|",
    ]
    for setting, arm in by_setting.items():
        overrides = ", ".join(f"{key}={value}" for key, value in study["settings"][setting].items()) or "none"
        lines.append(
            f"| {setting} | {overrides} | {arm[0]['parameters']:,} "
            f"| {np.mean([run['minutes'] for run in arm]):.1f} "
            f"| {', '.join(str(run['kept_epoch']) for run in sorted(arm, key=lambda run: run['seed']))} "
            f"| {', '.join(run['behavioural'] for run in sorted(arm, key=lambda run: run['seed']))} |"
        )
    lines.append("")

    for scope in REPORTED_SCOPES:
        rows = [(setting, scopes[scope]) for setting, scopes in summary.items() if scope in scopes]
        if not rows:
            continue
        lines += [
            f"## {scope} ({rows[0][1]['n_sets']:,} sets)",
            "",
            "| Setting | Gap | MAE | Regret | Top-1 | Tau |",
            "|---|---|---|---|---|---|",
        ]
        for setting, row in rows:
            cells = " | ".join(
                f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_sd']:.4f}" for metric in (*METRICS, *SECONDARY)
            )
            lines.append(f"| {setting} | {cells} |")
        lines.append("")

    ranked = sorted(
        (setting for setting in summary if "facade_system/llm" in summary[setting]),
        key=lambda setting: summary[setting]["facade_system/llm"]["gap_fidelity_mean"],
    )
    if len(ranked) > 3:
        lines += [
            "## Ranked by facade LLM gap",
            "",
            "| Setting | Facade gap | Facade regret | Concrete gap | Concrete regret | Parameters | Overrides |",
            "|---|---|---|---|---|---|---|",
        ]
        for setting in ranked:
            facade, concrete = summary[setting]["facade_system/llm"], summary[setting]["concrete/llm"]
            overrides = ", ".join(f"{key}={value}" for key, value in study["settings"][setting].items()) or "none"
            lines.append(
                f"| {setting} | {facade['gap_fidelity_mean']:.4f} | {facade['top1_regret_mean']:.4f} "
                f"| {concrete['gap_fidelity_mean']:.4f} | {concrete['top1_regret_mean']:.4f} "
                f"| {by_setting[setting][0]['parameters']:,} | {overrides} |"
            )
        lines.append("")

    judged = verdicts(summary, reference)
    if judged:
        columns = list(next(iter(judged.values())))
        lines += [
            f"## Against {reference}",
            "",
            "| Setting | " + " | ".join(
                f"{scope.split('/')[0]} {SHORT[metric]}" for scope in DECIDING_SCOPES for metric in METRICS
            ) + " |",
            "|---|" + "---|" * len(columns),
        ]
        for setting, cells in judged.items():
            lines.append(f"| {setting} | " + " | ".join(cells[column] for column in columns) + " |")
        lines.append("")

    lines += [
        "## Every run",
        "",
        "| Run | Epochs run | Kept | Validation loss | Minutes | Behavioural | Facade LLM gap | Concrete LLM gap |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in sorted(runs, key=lambda run: (list(summary).index(run["setting"]), run["seed"])):
        facade = scope_cell(run["results"], "facade_system/llm")
        concrete = scope_cell(run["results"], "concrete/llm")
        lines.append(
            f"| {run['run']} | {run['epochs_run']} of {run['budget']} | {run['kept_epoch']} | {run['val_loss']:.4f} "
            f"| {run['minutes']:.1f} | {run['behavioural']} "
            f"| {facade['gap_fidelity']:.4f} | {concrete['gap_fidelity']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def write_summary(out: Path, name: str, study: dict, sizing: dict | None) -> None:
    runs = finished_runs(out)
    if not runs:
        return
    summary = summarise(runs, list(study["settings"]))
    listed = [{key: value for key, value in run.items() if key != "results"} for run in runs]
    (out / SUMMARY).write_text(
        json.dumps({"study": name, "definition": study, "runs": listed, "sizing": sizing, "settings": summary,
                    "verdicts": verdicts(summary, study["reference"])}, indent=2),
        encoding="utf-8",
    )
    (out / REPORT).write_text(render(name, study, runs, summary, sizing), encoding="utf-8")


def draw_curves(out: Path, study: dict) -> Path | None:
    """Validation and training curves of each setting's first seed, when they fit one figure."""
    from train import curves

    seed = min(study["seeds"])
    first = {run["setting"]: out / run["run"] for run in finished_runs(out) if run["seed"] == seed}
    ordered = {setting: first[setting] for setting in study["settings"] if setting in first}
    if not 2 <= len(ordered) <= len(curves.SERIES):
        return None
    return curves.plot_runs({setting: curves.read_history(path) for setting, path in ordered.items()}, out / CURVES)


def run(
    name: str, config_path: Path, behavioural_suite: bool, parallel: int | None = None, out: Path | None = None,
) -> Path:
    from train.config import TrainConfig

    study = load_study(name)
    base = TrainConfig.load(config_path)
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path(base.output_dir) / f"tuning-{name}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    snapshot = shared_snapshot(config_path)
    print(f"study {name}, snapshot {snapshot[:12]}, runs in {out}", flush=True)
    done = {(run["setting"], run["seed"]) for run in finished_runs(out)}
    configs = [
        configure(config_path, setting, overrides or {}, seed, out, snapshot)
        for setting, overrides in study["settings"].items()
        for seed in study["seeds"]
        if (setting, seed) not in done
    ]
    jobs = []
    for config in sorted(configs, key=cost, reverse=True):
        config_file = config.dump(out / f"{config.name}.yaml")
        jobs.append(Job(
            name=config.name,
            command=[sys.executable, "-u", "-m", "train.run", "--config", str(config_file)]
            + ([] if behavioural_suite else ["--skip-behavioural"]),
            log=config_file.with_suffix(".log"),
        ))
    sizing, _ = run_jobs(
        jobs, base.resolved_device(), parallel, on_finish=lambda job, code: write_summary(out, name, study, None)
    )
    write_summary(out, name, study, sizing)
    draw_curves(out, study)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", help=f"a study named in {STUDIES.name}")
    parser.add_argument("--config", type=Path, default=Path("train/configs/default.yaml"))
    parser.add_argument("--skip-behavioural", action="store_true")
    parser.add_argument(
        "--parallel", type=int, default=None,
        help="runs trained at once; by default sized to this machine from the first run's footprint",
    )
    parser.add_argument("--out", type=Path, default=None, help="an earlier run of the study, to finish or extend it")
    args = parser.parse_args()

    out = run(args.study, args.config, not args.skip_behavioural, args.parallel, args.out)
    if (out / REPORT).exists():
        print((out / REPORT).read_text(encoding="utf-8"))
    print(f"runs in {out}")


if __name__ == "__main__":
    main()
