"""Shortlists of composed partitions under contexts and stakeholders: the scenarios sent out for labelling."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from core.registry import Registry
from eval.metrics import SHORTLIST_ACROSS, SHORTLIST_WITHIN
from ingest.partition.compose import PRODUCTS

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


def synthesise(registry: Registry, products: list[dict], sets: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = {}
    for product in products:
        pools.setdefault(product["typology"], []).append(product)
    category = registry.category(CATEGORY_KEY)
    baseline = category.default_context_key
    others = sorted(category.available_contexts - {baseline})
    stakeholders = sorted(registry.stakeholders)
    scenarios = []
    for index in range(1, sets + 1):
        contexts = situation(rng, baseline, others)
        chosen = rng.sample(stakeholders, _pick(rng, STAKEHOLDER_COUNT_WEIGHTS))
        kind, alternatives = shortlist(rng, pools)
        scenarios.append({"id": f"partition_{index:06d}", "shortlist": kind, "contexts": contexts,
                          "stakeholders": chosen, "alternatives": alternatives})
    return scenarios


def main() -> None:
    from ingest.labelling import open_registry

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sets", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = DATASET_DIR / SCENARIOS_FILE
    if path.exists():
        raise SystemExit(f"{path} exists; labels refer to its scenarios, so it is never rewritten")
    products = json.loads(PRODUCTS.read_text(encoding="utf-8"))
    scenarios = synthesise(open_registry(), products, args.sets, args.seed)
    path.write_text(json.dumps(scenarios), encoding="utf-8")
    print(f"wrote {len(scenarios)} scenarios to {path}")


if __name__ == "__main__":
    main()
