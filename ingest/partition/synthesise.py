"""Shortlists of composed partitions under contexts and stakeholders: the scenarios sent out for labelling."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from core.registry import Registry
from eval.metrics import SHORTLIST_ACROSS, SHORTLIST_WITHIN
from ingest.partition.compose import PLAUSIBLE, PRODUCTS

CATEGORY_KEY = "indoor_partition"
DATASET_DIR = Path("data/partition")
SCENARIOS_FILE = "scenarios.json"
LLM_LABELS_FILE = "llm_labelled_dataset.json"

#: Partitions of different typologies compete for the same wall in practice, so half the shortlists mix them.
WITHIN_TYPOLOGY_SHARE = 0.5
#: How many contexts beside the baseline a set carries; zero means the baseline alone.
CONTEXT_COUNT_WEIGHTS = {0: 0.2, 1: 0.55, 2: 0.25}
STAKEHOLDER_COUNT_WEIGHTS = {1: 0.6, 2: 0.3, 3: 0.1}
MIN_ALTS, MAX_ALTS = 2, 5
#: Share of appended sets that carry stretched products.
STRETCHED_SET_SHARE = 0.5
#: How far a stretch pushes a value, low or high side drawn evenly; each value moves alone, clipped to PLAUSIBLE.
FACTORS = {
    "gwp": ((0.3, 0.6), (1.6, 2.5)),
    "wdp": ((0.3, 0.6), (1.6, 3.0)),
    "fwu": ((0.3, 0.6), (1.6, 3.0)),
    "surface_mass": ((0.6, 0.8), (1.3, 1.6)),
    "u_value": ((0.4, 0.7), (1.4, 2.0)),
    "heat_capacity": ((0.5, 0.7), (1.4, 1.8)),
    "thickness": ((0.75, 0.9), (1.2, 1.4)),
    "cost_product": ((0.5, 0.7), (1.5, 2.5)),
    "cost_labour": ((0.5, 0.7), (1.5, 2.5)),
    "cost_maintenance": ((0.4, 0.7), (1.5, 2.5)),
}
SOUND_SHIFT_DB = (5.0, 10.0)
HIGH_SHARE = {"circ_orig": (40.0, 85.0), "fu_recyc": (50.0, 90.0)}
STRETCHABLE = (*FACTORS, "acoustic_reduction_pink", *HIGH_SHARE)
END_OF_LIFE = ("fu_recyc", "fu_incin", "fu_haz", "fu_inert")


def _pick(rng: random.Random, weights: dict):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def situation(rng: random.Random, baseline: str, others: list[str]) -> list[str]:
    """The baseline alone, or other contexts drawn uniformly; the baseline never joins another."""
    count = _pick(rng, CONTEXT_COUNT_WEIGHTS)
    return [baseline] if count == 0 else rng.sample(others, count)


def shortlist(rng: random.Random, pools: dict[str, list[dict]]) -> tuple[str, list[dict]]:
    size = rng.randint(MIN_ALTS, MAX_ALTS)
    if rng.random() < WITHIN_TYPOLOGY_SHARE:
        typology = rng.choice(sorted(pools))
        return f"{SHORTLIST_WITHIN}:{typology}", rng.sample(pools[typology], size)
    typologies = [rng.choice(sorted(pools)) for _ in range(size)]
    while len(set(typologies)) < 2:
        typologies[rng.randrange(size)] = rng.choice(sorted(pools))
    chosen: list[dict] = []
    for typology in typologies:
        candidate = rng.choice(pools[typology])
        while candidate in chosen:
            candidate = rng.choice(pools[typology])
        chosen.append(candidate)
    return SHORTLIST_ACROSS, chosen


def _clip(key: str, value: float) -> float:
    low, high = PLAUSIBLE[key]
    return round(min(high, max(low, value)), 4)


def stretched(product: dict, rng: random.Random, key: str) -> dict:
    """The product with two or three values pushed towards the edges of what is plausible."""
    variant = {**product, "id_prod": key}
    levers = rng.sample(STRETCHABLE, rng.randint(2, 3))
    for lever in levers:
        if lever in FACTORS:
            variant[lever] = _clip(lever, product[lever] * rng.uniform(*rng.choice(FACTORS[lever])))
        elif lever == "acoustic_reduction_pink":
            variant[lever] = _clip(lever, product[lever] + rng.choice((-1, 1)) * rng.uniform(*SOUND_SHIFT_DB))
        else:
            variant[lever] = round(rng.uniform(*HIGH_SHARE[lever]), 4)
    if "fu_recyc" in levers:
        rest = [k for k in END_OF_LIFE if k != "fu_recyc"]
        scale = (100.0 - variant["fu_recyc"]) / sum(product[k] for k in rest)
        for k in rest:
            variant[k] = round(product[k] * scale, 4)
    return variant


def stretched_shortlist(rng: random.Random, pools: dict[str, list[dict]], external_id: str) -> tuple[str, list[dict]]:
    """A shortlist with one alternative, two in sets of four or five, stretched."""
    kind, alternatives = shortlist(rng, pools)
    alternatives = list(alternatives)
    slots = rng.sample(range(len(alternatives)), 1 if len(alternatives) <= 3 else rng.randint(1, 2))
    for number, slot in enumerate(slots, start=1):
        alternatives[slot] = stretched(alternatives[slot], rng, f"{external_id}_stretch_{number}")
    return kind, alternatives


def synthesise(registry: Registry, products: list[dict], sets: int, seed: int, first: int = 1,
               stretched_share: float = 0.0) -> list[dict]:
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = {}
    for product in products:
        pools.setdefault(product["typology"], []).append(product)
    category = registry.category(CATEGORY_KEY)
    baseline = category.default_context_key
    others = sorted(category.available_contexts - {baseline})
    stakeholders = sorted(registry.stakeholders)
    scenarios = []
    for index in range(first, first + sets):
        external_id = f"partition_{index:06d}"
        contexts = situation(rng, baseline, others)
        chosen = rng.sample(stakeholders, _pick(rng, STAKEHOLDER_COUNT_WEIGHTS))
        if stretched_share and rng.random() < stretched_share:
            kind, alternatives = stretched_shortlist(rng, pools, external_id)
        else:
            kind, alternatives = shortlist(rng, pools)
        scenarios.append({"id": external_id, "shortlist": kind, "contexts": contexts,
                          "stakeholders": chosen, "alternatives": alternatives})
    return scenarios


def extend(registry: Registry, path: Path, products: list[dict], sets: int, seed: int) -> None:
    """Append sets after the last one, leaving every existing set as it was."""
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    first = 1 + max(int(s["id"].rsplit("_", 1)[1]) for s in scenarios)
    scenarios += synthesise(registry, products, sets, seed, first, STRETCHED_SET_SHARE)
    path.write_text(json.dumps(scenarios), encoding="utf-8")


def main() -> None:
    from ingest.labelling import open_registry

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command")
    more = commands.add_parser("extend", help="append sets, half of them holding stretched products")
    more.add_argument("--sets", type=int, required=True)
    more.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sets", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = DATASET_DIR / SCENARIOS_FILE
    products = json.loads(PRODUCTS.read_text(encoding="utf-8"))
    if args.command == "extend":
        extend(open_registry(), path, products, args.sets, args.seed)
        print(f"appended {args.sets} sets to {path}")
        return
    if path.exists():
        raise SystemExit(f"{path} exists; labels refer to its scenarios, so it is never rewritten; append with extend")
    scenarios = synthesise(open_registry(), products, args.sets, args.seed)
    path.write_text(json.dumps(scenarios), encoding="utf-8")
    print(f"wrote {len(scenarios)} scenarios to {path}")


if __name__ == "__main__":
    main()
