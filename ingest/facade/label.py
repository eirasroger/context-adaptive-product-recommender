"""Language-model labels for facade shortlists: the brief, the work chunks, and the collected labels."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import yaml

from core import registry as registry_module
from core.registry import Registry
from db.session import create_db_engine, session_scope
from ingest.facade.build import JOINT_DB
from ingest.facade.synthesise import CATEGORY_KEY, DATASET_DIR, LLM_LABELS_FILE, SCENARIOS_FILE

LABELLING_DIR = DATASET_DIR / "labelling"
CHUNKS_DIR = LABELLING_DIR / "chunks"
OUT_DIR = LABELLING_DIR / "out"
BRIEF_FILE = LABELLING_DIR / "brief.md"
INDEX_FILE = LABELLING_DIR / "index.json"
EXAMPLES = Path(__file__).with_name("labelling_examples.yaml")

LETTERS = "ABCDEFGHIJ"

BRIEF = Path(__file__).with_name("labelling_brief.md")


def open_registry() -> Registry:
    with session_scope(create_db_engine(JOINT_DB)) as session:
        return registry_module.from_session(session)


def _number(value):
    if value is None or isinstance(value, str):
        return value
    if value == 0:
        return 0.0
    digits = max(0, 3 - int(math.floor(math.log10(abs(value)))) - 1)
    return round(value, digits)


def brief() -> str:
    """The hand-written brief, followed by the worked examples."""
    lines = [BRIEF.read_text(encoding="utf-8").rstrip(), "", "# 7. Worked examples", ""]
    for number, example in enumerate(yaml.safe_load(EXAMPLES.read_text(encoding="utf-8"))["examples"], start=1):
        scenario = {k: example[k] for k in ("applications", "stakeholders", "alternatives")}
        lines += [
            f"Example {number}, input:", "", "```json", json.dumps(scenario, ensure_ascii=False), "```", "",
            "Output:", "", "```json", json.dumps(example["labels"], ensure_ascii=False, indent=1), "```", "",
        ]
    return "\n".join(lines)


def payload(registry: Registry, scenario: dict) -> tuple[dict, dict[str, str]]:
    """The scenario as the labeller sees it, with products lettered so no typology shows."""
    category = registry.category(CATEGORY_KEY)
    shown = [k for k in category.token_order if not registry.indicator(k).is_derived]
    letters: dict[str, str] = {}
    alternatives = []
    for letter, alternative in zip(LETTERS, scenario["alternatives"]):
        letters[letter] = alternative["id_prod"]
        row = {"id_prod": letter}
        for key in shown:
            value = alternative.get(key)
            indicator = registry.indicator(key)
            row[key] = indicator.level(value).display_name if indicator.is_scale and value else _number(value)
        alternatives.append(row)
    return {
        "scenario_id": scenario["id"],
        "applications": [registry.contexts[k].display_name for k in scenario["contexts"]],
        "stakeholders": [registry.stakeholders[k].display_name for k in scenario["stakeholders"]],
        "alternatives": alternatives,
    }, letters


def labelled_ids() -> set[str]:
    """Scenarios with labels returned in any output file, whatever round they came from."""
    done: set[str] = set()
    for path in OUT_DIR.glob("chunk_*.json"):
        done |= set(json.loads(path.read_text(encoding="utf-8"))["labels"])
    return done


def status() -> dict[str, int]:
    """Every scenario is labelled, pending (sent out, no labels back) or fresh (never sent)."""
    scenarios = json.loads((DATASET_DIR / SCENARIOS_FILE).read_text(encoding="utf-8"))
    index = json.loads(INDEX_FILE.read_text(encoding="utf-8")) if INDEX_FILE.exists() else {}
    done = labelled_ids()
    return {
        "scenarios": len(scenarios),
        "labelled": len(done),
        "pending": len(set(index) - done),
        "fresh": sum(1 for s in scenarios if s["id"] not in index),
    }


def write_chunks(registry: Registry, sets: int, chunks: int, seed: int, resend: bool = False) -> None:
    """A new round of chunks, numbered after the existing ones: fresh scenarios, or pending ones with resend."""
    scenarios = json.loads((DATASET_DIR / SCENARIOS_FILE).read_text(encoding="utf-8"))
    index: dict[str, dict[str, str]] = json.loads(INDEX_FILE.read_text(encoding="utf-8")) if INDEX_FILE.exists() else {}
    done = labelled_ids()
    pool = (
        [s for s in scenarios if s["id"] in index and s["id"] not in done]
        if resend
        else [s for s in scenarios if s["id"] not in index]
    )
    chosen = random.Random(seed).sample(pool, min(sets, len(pool)))
    sets = len(chosen)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    first = 1 + max((int(p.stem.split("_")[1][:2]) for p in CHUNKS_DIR.glob("chunk_*.json")), default=0)
    size = math.ceil(sets / chunks)
    for offset in range(chunks):
        part = []
        for scenario in chosen[offset * size:(offset + 1) * size]:
            shown, letters = payload(registry, scenario)
            part.append(shown)
            index[scenario["id"]] = letters
        path = CHUNKS_DIR / f"chunk_{first + offset:02d}.json"
        path.write_text(json.dumps({"scenarios": part}, ensure_ascii=False, indent=1), encoding="utf-8")
    INDEX_FILE.write_text(json.dumps(index), encoding="utf-8")
    print(f"{sets} scenarios in chunks {first:02d} to {first + chunks - 1:02d} under {CHUNKS_DIR}")


def collect() -> None:
    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    labelled, problems = [], []
    for path in sorted(OUT_DIR.glob("chunk_*.json")):
        for scenario_id, rows in json.loads(path.read_text(encoding="utf-8"))["labels"].items():
            letters = index.get(scenario_id)
            if letters is None:
                problems.append(f"{path.name}: unknown scenario {scenario_id}")
                continue
            by_letter = {row.get("id_prod"): row for row in rows}
            valid = set(by_letter) == set(letters) and all(
                0 <= float(row["pref"]) <= 1 and 0 <= float(row["conf"]) <= 1 for row in rows
            )
            if not valid:
                problems.append(f"{path.name}: {scenario_id} has missing, extra or out-of-range labels")
                continue
            labelled.append({
                "id": scenario_id,
                "labelled_alternatives": [
                    {"id_prod": letters[l], "pref": float(r["pref"]), "conf": float(r["conf"]), "reason": r["reason"]}
                    for l, r in sorted(by_letter.items())
                ],
            })
    (DATASET_DIR / LLM_LABELS_FILE).write_text(json.dumps(labelled, ensure_ascii=False), encoding="utf-8")
    print(f"{len(labelled)} of {len(index)} scenarios labelled; {len(problems)} rejected")
    for problem in problems[:20]:
        print("  " + problem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("brief", help="write the labelling brief")
    chunk = commands.add_parser("chunks", help="split a sample of scenarios into work chunks")
    chunk.add_argument("--sets", type=int, default=1000)
    chunk.add_argument("--chunks", type=int, default=20)
    chunk.add_argument("--seed", type=int, default=0)
    chunk.add_argument("--resend", action="store_true", help="chunk scenarios sent out earlier that came back unlabelled")
    commands.add_parser("collect", help="check the returned labels and write the labels file")
    commands.add_parser("status", help="count labelled, pending and fresh scenarios")
    args = parser.parse_args()

    if args.command == "collect":
        collect()
    elif args.command == "status":
        for name, count in status().items():
            print(f"{name:10s} {count:>6,}")
    elif args.command == "brief":
        LABELLING_DIR.mkdir(parents=True, exist_ok=True)
        BRIEF_FILE.write_text(brief(), encoding="utf-8")
        print(f"brief written to {BRIEF_FILE}")
    else:
        write_chunks(open_registry(), args.sets, args.chunks, args.seed, args.resend)


if __name__ == "__main__":
    main()
