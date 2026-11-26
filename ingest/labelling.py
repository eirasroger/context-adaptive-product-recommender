"""Language-model label rounds for any category: the brief, the work chunks, and the collected labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from core import registry as registry_module
from core.registry import Registry
from db.session import create_db_engine, session_scope
from ingest.build import JOINT_DB


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


def fingerprint(scenario: dict) -> str:
    """What a labeller judged: the situation, the stakeholders and every product's values."""
    judged = {k: scenario[k] for k in ("contexts", "stakeholders", "alternatives")}
    return hashlib.sha256(json.dumps(judged, sort_keys=True).encode()).hexdigest()[:16]


def next_chunk_number(paths: list[Path]) -> int:
    """One past the highest chunk number, reading only the leading digits, so chunk_13b counts as 13."""
    return 1 + max((int(re.match(r"chunk_(\d+)", p.stem).group(1)) for p in paths), default=0)


def of_kind(scenarios: list[dict], kind: str) -> list[dict]:
    """The scenarios whose shortlist is of this kind, or of a narrower kind under it."""
    return [s for s in scenarios if s["shortlist"] == kind or s["shortlist"].startswith(f"{kind}:")]


def read_labels(path: Path) -> dict | None:
    """A returned file's labels, or None when it is not UTF-8 JSON with a labels object."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))["labels"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return None


@dataclass(frozen=True)
class LabelRound:
    """Where one category's scenarios, brief and label files live."""

    category_key: str
    dataset_dir: Path
    scenarios_file: str
    labels_file: str
    brief_source: Path
    examples_source: Path
    letters: str

    @property
    def labelling_dir(self) -> Path:
        return self.dataset_dir / "labelling"

    @property
    def chunks_dir(self) -> Path:
        return self.labelling_dir / "chunks"

    @property
    def out_dir(self) -> Path:
        return self.labelling_dir / "out"

    @property
    def brief_file(self) -> Path:
        return self.labelling_dir / "brief.md"

    @property
    def index_file(self) -> Path:
        return self.labelling_dir / "index.json"

    def scenarios(self) -> list[dict]:
        return json.loads((self.dataset_dir / self.scenarios_file).read_text(encoding="utf-8"))

    def index(self) -> dict[str, dict]:
        return json.loads(self.index_file.read_text(encoding="utf-8")) if self.index_file.exists() else {}

    def brief(self) -> str:
        """The hand-written brief, followed by the worked examples."""
        lines = [self.brief_source.read_text(encoding="utf-8").rstrip(), "", "# 7. Worked examples", ""]
        examples = yaml.safe_load(self.examples_source.read_text(encoding="utf-8"))["examples"]
        for number, example in enumerate(examples, start=1):
            scenario = {k: example[k] for k in ("applications", "stakeholders", "alternatives")}
            lines += [
                f"Example {number}, input:", "", "```json", json.dumps(scenario, ensure_ascii=False), "```", "",
                "Output:", "", "```json", json.dumps(example["labels"], ensure_ascii=False, indent=1), "```", "",
            ]
        return "\n".join(lines)

    def payload(self, registry: Registry, scenario: dict) -> tuple[dict, dict[str, str]]:
        """The scenario as the labeller sees it, with products lettered so no typology shows."""
        category = registry.category(self.category_key)
        shown = [k for k in category.token_order if not registry.indicator(k).is_derived]
        letters: dict[str, str] = {}
        alternatives = []
        for letter, alternative in zip(self.letters, scenario["alternatives"]):
            letters[letter] = alternative["id_prod"]
            row = {"id_prod": letter}
            for key in shown:
                value = alternative.get(key)
                indicator = registry.indicator(key)
                row[key] = indicator.level(value).display_name if indicator.is_scale and value else _number(value)
            alternatives.append(row)
        return {
            "scenario_id": scenario["id"],
            "applications": [category.context_presentation[k].display_name for k in scenario["contexts"]],
            "stakeholders": [registry.stakeholders[k].display_name for k in scenario["stakeholders"]],
            "alternatives": alternatives,
        }, letters

    def labelled_ids(self) -> set[str]:
        """Scenarios with labels in any readable output file, whatever round they came from."""
        done: set[str] = set()
        for path in self.out_dir.glob("chunk_*.json"):
            done |= set(read_labels(path) or {})
        return done

    def status(self) -> dict[str, int]:
        """Every scenario is labelled, pending (sent out, no labels back) or fresh (never sent)."""
        scenarios = self.scenarios()
        index = self.index()
        done = self.labelled_ids()
        return {
            "scenarios": len(scenarios),
            "labelled": len(done),
            "pending": len(set(index) - done),
            "fresh": sum(1 for s in scenarios if s["id"] not in index),
        }

    def write_chunks(
        self, registry: Registry, sets: int, chunks: int, seed: int, resend: bool = False, shortlist: str | None = None
    ) -> None:
        """A new round of chunks, numbered after the existing ones: fresh scenarios, or pending ones with resend."""
        scenarios = self.scenarios()
        index = self.index()
        done = self.labelled_ids()
        pool = (
            [s for s in scenarios if s["id"] in index and s["id"] not in done]
            if resend
            else [s for s in scenarios if s["id"] not in index]
        )
        if shortlist is not None:
            pool = of_kind(pool, shortlist)
        chosen = random.Random(seed).sample(pool, min(sets, len(pool)))
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        first = next_chunk_number(list(self.chunks_dir.glob("chunk_*.json")))
        size = math.ceil(len(chosen) / chunks)
        for offset in range(chunks):
            part = []
            for scenario in chosen[offset * size:(offset + 1) * size]:
                shown, letters = self.payload(registry, scenario)
                part.append(shown)
                index[scenario["id"]] = {"letters": letters, "fingerprint": fingerprint(scenario)}
            path = self.chunks_dir / f"chunk_{first + offset:02d}.json"
            path.write_text(json.dumps({"scenarios": part}, ensure_ascii=False, indent=1), encoding="utf-8")
        self.index_file.write_text(json.dumps(index), encoding="utf-8")
        print(f"{len(chosen)} scenarios in chunks {first:02d} to {first + chunks - 1:02d} under {self.chunks_dir}")

    def collect(self) -> list[str]:
        """Write every valid label to the labels file; return what was refused, unreadable or skipped."""
        index = self.index()
        scenarios = {s["id"]: s for s in self.scenarios()}
        labelled, problems = [], []
        for path in sorted(self.out_dir.glob("chunk_*.json")):
            returned = read_labels(path)
            if returned is None:
                problems.append(f"{path.name}: unreadable; it must be UTF-8 JSON with a labels object")
                continue
            sent_path = self.chunks_dir / path.name
            if sent_path.exists():
                sent = {s["scenario_id"] for s in json.loads(sent_path.read_text(encoding="utf-8"))["scenarios"]}
                problems += [f"{path.name}: {sid} sent but not labelled" for sid in sorted(sent - set(returned))]
            for scenario_id, rows in returned.items():
                sent_as = index.get(scenario_id)
                if sent_as is None:
                    problems.append(f"{path.name}: unknown scenario {scenario_id}")
                    continue
                if scenario_id not in scenarios or fingerprint(scenarios[scenario_id]) != sent_as["fingerprint"]:
                    problems.append(f"{path.name}: {scenario_id} changed since it was labelled")
                    continue
                letters = sent_as["letters"]
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
        (self.dataset_dir / self.labels_file).write_text(json.dumps(labelled, ensure_ascii=False), encoding="utf-8")
        print(f"{len(labelled)} of {len(index)} scenarios labelled; {len(problems)} problems")
        for problem in problems[:20]:
            print("  " + problem)
        return problems


def main(label_round: LabelRound) -> None:
    parser = argparse.ArgumentParser(description=f"Language-model label rounds for {label_round.category_key}.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("brief", help="write the labelling brief")
    chunk = commands.add_parser("chunks", help="split a sample of scenarios into work chunks")
    chunk.add_argument("--sets", type=int, required=True)
    chunk.add_argument("--chunks", type=int, required=True)
    chunk.add_argument("--seed", type=int, required=True)
    chunk.add_argument("--resend", action="store_true", help="chunk scenarios sent out earlier that came back unlabelled")
    chunk.add_argument("--shortlist", help="draw only shortlists of this kind, such as unusual or within:etics")
    commands.add_parser("collect", help="check the returned labels and write the labels file")
    commands.add_parser("status", help="count labelled, pending and fresh scenarios")
    args = parser.parse_args()

    if args.command == "collect":
        label_round.collect()
    elif args.command == "status":
        for name, count in label_round.status().items():
            print(f"{name:10s} {count:>6,}")
    elif args.command == "brief":
        label_round.labelling_dir.mkdir(parents=True, exist_ok=True)
        label_round.brief_file.write_text(label_round.brief(), encoding="utf-8")
        print(f"brief written to {label_round.brief_file}")
    else:
        label_round.write_chunks(open_registry(), args.sets, args.chunks, args.seed, args.resend, args.shortlist)
