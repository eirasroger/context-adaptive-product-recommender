"""Context influence pilot: the same facade shortlists labelled under each candidate context, against a twice-labelled baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml
from scipy.stats import spearmanr

from core.scoring import TIE_EPSILON
from ingest.facade.label import open_registry, payload
from ingest.facade.synthesise import DATASET_DIR, SCENARIOS_FILE

SCREENING = Path(__file__).with_name("context_screening.yaml")
BRIEF_TEMPLATE = Path(__file__).with_name("context_pilot_brief.md")
PILOT_DIR = DATASET_DIR / "context_pilot"
CHUNKS_DIR = PILOT_DIR / "chunks"
OUT_DIR = PILOT_DIR / "out"
INDEX_FILE = PILOT_DIR / "index.json"
BRIEF_FILE = PILOT_DIR / "brief.md"
FIGURE = Path(__file__).with_name("figures") / "context-influence.csv"

BASELINES = ("baseline_a", "baseline_b")
STAKEHOLDER = "balanced_optimizer"
ACROSS_SHARE = 0.6
BOOTSTRAP = 2000


@dataclass(frozen=True)
class Candidate:
    key: str
    name: str
    description: str
    parent: str | None


def screening() -> dict:
    return yaml.safe_load(SCREENING.read_text(encoding="utf-8"))


def candidates(record: Mapping) -> list[Candidate]:
    return [
        Candidate(c["key"], c["name"], " ".join(c["description"].split()), c.get("parent"))
        for c in record["candidates"]
        if c["outcome"] == "pilot" or c.get("piloted")
    ]


def brief(record: Mapping) -> str:
    baseline = record["baseline"]
    lines = [f"- {baseline['name']}: {' '.join(baseline['description'].split())}"]
    lines += [
        f"- {c.name}: the baseline, except that {c.description[0].lower()}{c.description[1:]}"
        for c in candidates(record)
    ]
    return BRIEF_TEMPLATE.read_text(encoding="utf-8").replace("{situations}", "\n".join(lines))


def versions(record: Mapping) -> list[str]:
    return [*BASELINES, *(c.key for c in candidates(record))]


def situation_name(record: Mapping, version: str) -> str:
    if version in BASELINES:
        return record["baseline"]["name"]
    return next(c.name for c in candidates(record) if c.key == version)


def sample_shortlists(scenarios: Sequence[dict], count: int, seed: int) -> list[list[dict]]:
    """Shortlists of four or five products, mostly mixing typologies so that every context has something to act on."""
    rng = random.Random(seed)
    pool = [s for s in scenarios if len(s["alternatives"]) >= 4]
    across = [s for s in pool if s["shortlist"].startswith("across")]
    within = [s for s in pool if s["shortlist"].startswith("within")]
    n_across = round(count * ACROSS_SHARE)
    chosen = rng.sample(across, n_across) + rng.sample(within, count - n_across)
    return [s["alternatives"] for s in chosen]


def assign_chunks(shortlists: int, n_versions: int, chunks: int) -> dict[tuple[int, int], int]:
    """Every version of a shortlist lands in a different chunk, so no labeller sees one shortlist twice."""
    if n_versions > chunks:
        raise ValueError(f"{n_versions} versions need at least as many chunks, got {chunks}")
    return {(s, v): (s + v) % chunks for s in range(shortlists) for v in range(n_versions)}


def scenario_id(seed: int, shortlist: int, version: str) -> str:
    return "p" + hashlib.sha1(f"{seed}:{shortlist}:{version}".encode()).hexdigest()[:10]


def write_chunks(count: int, chunks: int, seed: int) -> None:
    record = screening()
    registry = open_registry()
    scenarios = json.loads((DATASET_DIR / SCENARIOS_FILE).read_text(encoding="utf-8"))
    shortlists = sample_shortlists(scenarios, count, seed)
    keys = versions(record)
    placement = assign_chunks(len(shortlists), len(keys), chunks)
    rng = random.Random(seed + 1)

    work: list[list[dict]] = [[] for _ in range(chunks)]
    index: dict[str, dict] = {}
    for (s, v), chunk in placement.items():
        version = keys[v]
        alternatives = list(shortlists[s])
        rng.shuffle(alternatives)
        sid = scenario_id(seed, s, version)
        shown, letters = payload(
            registry, {"id": sid, "contexts": [], "stakeholders": [STAKEHOLDER], "alternatives": alternatives}
        )
        shown["applications"] = [situation_name(record, version)]
        work[chunk].append(shown)
        index[sid] = {"shortlist": s, "version": version, "letters": letters}

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for number, part in enumerate(work, start=1):
        rng.shuffle(part)
        (CHUNKS_DIR / f"chunk_{number:02d}.json").write_text(
            json.dumps({"scenarios": part}, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    INDEX_FILE.write_text(json.dumps(index), encoding="utf-8")
    (PILOT_DIR / "shortlists.json").write_text(json.dumps(shortlists), encoding="utf-8")
    BRIEF_FILE.write_text(brief(record), encoding="utf-8")
    print(f"{len(index)} scenarios ({len(shortlists)} shortlists x {len(keys)} versions) in {chunks} chunks under {CHUNKS_DIR}")


def _sign(difference: float, eps: float) -> int:
    return 0 if abs(difference) < eps else (1 if difference > 0 else -1)


def pair_discordance(a: Mapping[str, float], b: Mapping[str, float], eps: float = TIE_EPSILON) -> float:
    """Share of product pairs ordered differently: 1 for a reversal, 0.5 where one side calls a tie."""
    scores = []
    for i, j in itertools.combinations(sorted(a), 2):
        sa, sb = _sign(a[i] - a[j], eps), _sign(b[i] - b[j], eps)
        scores.append(0.0 if sa == sb else (0.5 if 0 in (sa, sb) else 1.0))
    return float(np.mean(scores))


def winner_changed(a: Mapping[str, float], b: Mapping[str, float], eps: float = TIE_EPSILON) -> float:
    winners_a = {k for k, v in a.items() if v >= max(a.values()) - eps}
    winners_b = {k for k, v in b.items() if v >= max(b.values()) - eps}
    return 0.0 if winners_a & winners_b else 1.0


def mean_shift(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return float(np.mean([abs(a[k] - b[k]) for k in a]))


MEASURES = {"discordance": pair_discordance, "winner_changed": winner_changed, "mean_shift": mean_shift}


def collected() -> dict[tuple[int, str], dict[str, float]]:
    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    prefs: dict[tuple[int, str], dict[str, float]] = {}
    for path in sorted(OUT_DIR.glob("chunk_*.json")):
        for sid, rows in json.loads(path.read_text(encoding="utf-8"))["labels"].items():
            entry = index.get(sid)
            if entry is None:
                continue
            by_letter = {row["id_prod"]: float(row["pref"]) for row in rows}
            if set(by_letter) != set(entry["letters"]):
                continue
            prefs[(entry["shortlist"], entry["version"])] = {
                entry["letters"][letter]: pref for letter, pref in by_letter.items()
            }
    return prefs


def influence(
    prefs: Mapping[tuple[int, str], Mapping[str, float]], version: str, reference: Sequence[str], seed: int = 0
) -> dict[str, float]:
    """Distance from the reference labels, less the baseline's distance from itself, with a bootstrap 95 % interval."""
    shortlists = sorted(
        s for s in {s for s, _ in prefs} if all((s, v) in prefs for v in (version, *reference, *BASELINES))
    )
    result: dict[str, float] = {"shortlists": len(shortlists)}
    rng = np.random.default_rng(seed)
    for name, measure in MEASURES.items():
        excess = np.array([
            np.mean([measure(prefs[(s, version)], prefs[(s, r)]) for r in reference])
            - measure(prefs[(s, BASELINES[0])], prefs[(s, BASELINES[1])])
            for s in shortlists
        ])
        draws = rng.choice(excess, size=(BOOTSTRAP, len(excess))).mean(axis=1)
        result[name] = float(excess.mean())
        result[f"{name}_low"], result[f"{name}_high"] = (float(q) for q in np.quantile(draws, [0.025, 0.975]))
    return result


def numeric(registry, key: str, value) -> float | None:
    if value is None:
        return None
    indicator = registry.indicator(key)
    return indicator.level(value).normalised_position if indicator.is_scale else float(value)


def directions(
    prefs: Mapping[tuple[int, str], Mapping[str, float]], shortlists: Sequence[Sequence[dict]], version: str, registry
) -> dict[str, float]:
    """Rank correlation, across products, between an indicator's within-shortlist position and the context's preference shift."""
    keys = [k for k in registry.category("facade_system").token_order if not registry.indicator(k).is_derived]
    pairs: dict[str, list[tuple[float, float]]] = {k: [] for k in keys}
    for s, alternatives in enumerate(shortlists):
        if not all((s, v) in prefs for v in (version, *BASELINES)):
            continue
        for key in keys:
            values = {a["id_prod"]: numeric(registry, key, a.get(key)) for a in alternatives}
            known = [v for v in values.values() if v is not None]
            if len(known) < 2 or max(known) == min(known):
                continue
            low, high = min(known), max(known)
            for product, value in values.items():
                if value is None:
                    continue
                base = np.mean([prefs[(s, b)][product] for b in BASELINES])
                pairs[key].append(((value - low) / (high - low), prefs[(s, version)][product] - base))
    return {
        key: float(spearmanr(*zip(*rows)).statistic)
        for key, rows in pairs.items()
        if len(rows) >= 20 and len({r[1] for r in rows}) > 1
    }


def analyse() -> None:
    record = screening()
    registry = open_registry()
    prefs = collected()
    shortlists = json.loads((PILOT_DIR / "shortlists.json").read_text(encoding="utf-8"))
    rows = []
    for candidate in candidates(record):
        row = {"context": candidate.key, "reference": "baseline", **influence(prefs, candidate.key, BASELINES)}
        moved = sorted(directions(prefs, shortlists, candidate.key, registry).items(), key=lambda kv: -abs(kv[1]))
        row["drivers"] = "; ".join(f"{k} {v:+.2f}" for k, v in moved[:4])
        rows.append(row)
        if candidate.parent:
            rows.append({"context": candidate.key, "reference": candidate.parent,
                         **influence(prefs, candidate.key, (candidate.parent,)), "drivers": ""})
    noise = [pair_discordance(prefs[(s, BASELINES[0])], prefs[(s, BASELINES[1])])
             for s in {s for s, _ in prefs} if all((s, b) in prefs for b in BASELINES)]
    print(f"baseline against itself: discordance {np.mean(noise):.3f} over {len(noise)} shortlists")
    print("| context | against | n | discordance (95 % CI) | winner changed | mean shift | drivers |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['context']} | {r['reference']} | {r['shortlists']} "
              f"| {r['discordance']:+.3f} ({r['discordance_low']:+.3f} to {r['discordance_high']:+.3f}) "
              f"| {r['winner_changed']:+.2f} | {r['mean_shift']:+.3f} | {r['drivers']} |")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    with FIGURE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def status() -> None:
    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    done = set()
    for path in OUT_DIR.glob("chunk_*.json"):
        done |= set(json.loads(path.read_text(encoding="utf-8"))["labels"])
    print(f"{len(done & set(index))} of {len(index)} scenarios labelled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    chunk = commands.add_parser("chunks", help="sample shortlists and write the work chunks and brief")
    chunk.add_argument("--shortlists", type=int, default=50)
    chunk.add_argument("--chunks", type=int, default=18)
    chunk.add_argument("--seed", type=int, default=0)
    commands.add_parser("status", help="count labelled scenarios")
    commands.add_parser("analyse", help="influence per context, written to experiments/figures")
    args = parser.parse_args()
    if args.command == "chunks":
        write_chunks(args.shortlists, args.chunks, args.seed)
    elif args.command == "status":
        status()
    else:
        analyse()


if __name__ == "__main__":
    main()
