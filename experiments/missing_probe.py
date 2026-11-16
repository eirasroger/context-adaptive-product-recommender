"""Missing-value probe: does a labeller mark a product down when its health status is hidden?"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ingest.facade.label import INDEX_FILE, open_registry, payload
from ingest.facade.synthesise import DATASET_DIR, SCENARIOS_FILE

PROBE_DIR = DATASET_DIR / "missing_probe"
CHUNKS_DIR = PROBE_DIR / "chunks"
OUT_DIR = PROBE_DIR / "out"
PROBE_INDEX = PROBE_DIR / "index.json"
FIGURE = Path(__file__).with_name("figures") / "missing-probe.json"

PROBED = "health"
VERSIONS = ("as_is_a", "as_is_b", "hidden")
BOOTSTRAP = 2000


def eligible(scenario: dict) -> bool:
    known = [a for a in scenario["alternatives"] if a.get(PROBED) is not None]
    return len(scenario["alternatives"]) >= 3 and len(known) >= 1


def hidden_version(scenario: dict, target: str) -> dict:
    return dict(
        scenario,
        alternatives=[
            {k: v for k, v in a.items() if not (a["id_prod"] == target and k == PROBED)}
            for a in scenario["alternatives"]
        ],
    )


def placement(shortlists: int, chunks: int) -> dict[tuple[int, int], int]:
    """Each version of a shortlist goes to a different chunk, so no labeller sees one shortlist twice."""
    return {(s, v): (s + v) % chunks for s in range(shortlists) for v in range(len(VERSIONS))}


def write_chunks(count: int, chunks: int, seed: int) -> None:
    if chunks < len(VERSIONS):
        raise ValueError(f"{len(VERSIONS)} versions need at least {len(VERSIONS)} chunks")
    registry = open_registry()
    sent = set(json.loads(INDEX_FILE.read_text(encoding="utf-8"))) if INDEX_FILE.exists() else set()
    scenarios = json.loads((DATASET_DIR / SCENARIOS_FILE).read_text(encoding="utf-8"))
    rng = random.Random(seed)
    chosen = rng.sample([s for s in scenarios if s["id"] not in sent and eligible(s)], count)

    work: list[list[dict]] = [[] for _ in range(chunks)]
    index: dict[str, dict] = {}
    for (s, v), chunk in placement(count, chunks).items():
        scenario = chosen[s]
        target = rng.choice([a["id_prod"] for a in scenario["alternatives"] if a.get(PROBED) is not None])
        shown = hidden_version(scenario, target) if VERSIONS[v] == "hidden" else scenario
        alternatives = list(shown["alternatives"])
        rng.shuffle(alternatives)
        sid = "m" + hashlib.sha1(f"{seed}:{s}:{v}".encode()).hexdigest()[:10]
        rendered, letters = payload(registry, dict(shown, id=sid, alternatives=alternatives))
        work[chunk].append(rendered)
        index[sid] = {"shortlist": s, "version": VERSIONS[v], "letters": letters, "target": target}

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for number, part in enumerate(work, start=1):
        rng.shuffle(part)
        (CHUNKS_DIR / f"chunk_{number:02d}.json").write_text(
            json.dumps({"scenarios": part}, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    PROBE_INDEX.write_text(json.dumps(index), encoding="utf-8")
    print(f"{len(index)} scenarios ({count} shortlists x {len(VERSIONS)} versions) in {chunks} chunks under {CHUNKS_DIR}")


def target_shifts(labels: Mapping[str, Mapping[str, float]], target: str) -> tuple[float, float]:
    """The hidden product's shift against the mean of the two as-is labels, and the as-is labels' own disagreement."""
    a, b, hidden = (labels[v] for v in VERSIONS)
    return hidden[target] - (a[target] + b[target]) / 2, abs(a[target] - b[target])


def summarise(shifts: Sequence[float], noise: Sequence[float], seed: int = 0) -> dict[str, float]:
    shifts_arr = np.asarray(shifts)
    draws = np.random.default_rng(seed).choice(shifts_arr, size=(BOOTSTRAP, len(shifts_arr))).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "shortlists": len(shifts_arr),
        "mean_shift": float(shifts_arr.mean()),
        "shift_low": float(low),
        "shift_high": float(high),
        "as_is_disagreement": float(np.mean(noise)),
    }


def analyse() -> None:
    index = json.loads(PROBE_INDEX.read_text(encoding="utf-8"))
    by_shortlist: dict[int, dict[str, dict[str, float]]] = {}
    targets: dict[int, str] = {}
    for path in sorted(OUT_DIR.glob("chunk_*.json")):
        for sid, rows in json.loads(path.read_text(encoding="utf-8"))["labels"].items():
            entry = index.get(sid)
            if entry is None:
                continue
            prefs = {entry["letters"][r["id_prod"]]: float(r["pref"]) for r in rows if r["id_prod"] in entry["letters"]}
            by_shortlist.setdefault(entry["shortlist"], {})[entry["version"]] = prefs
            targets[entry["shortlist"]] = entry["target"]
    complete = [s for s, labels in by_shortlist.items() if set(labels) == set(VERSIONS)]
    pairs = [target_shifts(by_shortlist[s], targets[s]) for s in complete]
    result = summarise([p[0] for p in pairs], [p[1] for p in pairs])
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    FIGURE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    penalised = result["shift_high"] < 0
    print("hiding health lowers the score" if penalised else "no penalty for a hidden health status beyond noise")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    chunk = commands.add_parser("chunks", help="sample shortlists and write the three versions in chunks")
    chunk.add_argument("--shortlists", type=int, default=40)
    chunk.add_argument("--chunks", type=int, default=3)
    chunk.add_argument("--seed", type=int, default=0)
    commands.add_parser("analyse", help="shift of the hidden product, written to experiments/figures")
    args = parser.parse_args()
    if args.command == "chunks":
        write_chunks(args.shortlists, args.chunks, args.seed)
    else:
        analyse()


if __name__ == "__main__":
    main()
