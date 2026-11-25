"""Write the working facade dataset: EPD-composed products, shortlists and rule labels."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Sequence

from core.registry import Registry
from eval.metrics import SHORTLIST_ACROSS, SHORTLIST_UNUSUAL, SHORTLIST_WITHIN
from ingest.generators import parametric
from ingest.facade import compose, variants

CATEGORY_KEY = "facade_system"
DATASET_DIR = Path("data/facade")
SCENARIOS_FILE = "frozen_dataset.json"
LABELS_FILE = "labelled_dataset.json"
LLM_LABELS_FILE = "llm_labelled_dataset.json"

PRODUCTS_PER_TYPOLOGY = 2_000
WITHIN_TYPOLOGY_SHARE = 0.75
#: How many contexts beside the baseline a set carries; zero means the baseline alone.
CONTEXT_COUNT_WEIGHTS = {0: 0.2, 1: 0.55, 2: 0.25}
#: Sets appended for situations where three or four contexts genuinely combine.
COMBINED_CONTEXT_WEIGHTS = {3: 2 / 3, 4: 1 / 3}
STAKEHOLDER_COUNT_WEIGHTS = {1: 0.6, 2: 0.3, 3: 0.1}
MIN_ALTS, MAX_ALTS = 2, 5
#: Share of variants that sit beside the product they vary, so the levers are the only difference.
PAIRED_VARIANT_SHARE = 0.5

MISSING_QUALITY = 0.3
LABEL_CONFIDENCE = 0.8


def _pick(rng: random.Random, weights: dict):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def stakeholder_priority(registry: Registry, stakeholder_key: str, indicator_key: str) -> float:
    stakeholder = registry.stakeholders[stakeholder_key]
    priority = stakeholder.indicator_priorities.get(indicator_key)
    if priority is None:
        family = registry.indicator(indicator_key).family_key
        priority = stakeholder.family_priorities.get(family, 0.5)
    return float(priority)


def utility(
    registry: Registry,
    alternative: dict,
    context_keys: Sequence[str],
    stakeholder_keys: Sequence[str],
) -> float:
    """Weighted mean quality; each family's weight is shared among its directed indicators."""
    category = registry.category(CATEGORY_KEY)
    directed = []
    for key in category.token_order:
        indicator = registry.indicator(key)
        if indicator.is_derived:
            continue
        direction, context_priority = registry.resolve_direction(CATEGORY_KEY, key, context_keys)
        if direction != 0:
            directed.append((key, indicator.family_key, abs(direction), context_priority))

    family_size: dict[str, int] = {}
    for _, family, _, _ in directed:
        family_size[family] = family_size.get(family, 0) + 1

    total = weight_sum = 0.0
    for key, family, strength, context_priority in directed:
        stakeholder = sum(stakeholder_priority(registry, s, key) for s in stakeholder_keys) / len(stakeholder_keys)
        weight = stakeholder * strength * (0.5 + 1.5 * context_priority) / family_size[family]
        value = alternative.get(key)
        quality = (
            MISSING_QUALITY
            if value is None
            else parametric.quality_of(registry, CATEGORY_KEY, key, value, context_keys)
        )
        total += weight * quality
        weight_sum += weight
    return total / weight_sum


def rule_labels(utilities: Sequence[float], rng: random.Random) -> list[float]:
    """Order within the shortlist, placed on a band that sits lower when every option is poor."""
    mean = sum(utilities) / len(utilities)
    spread = max(max(utilities) - min(utilities), 0.1)
    labels = []
    for value in utilities:
        relative = min(1.0, max(0.0, 0.5 + (value - mean) / spread))
        absolute = min(1.0, max(0.0, (value - 0.3) / 0.4))
        pref = 0.1 + 0.8 * (0.6 * relative + 0.4 * absolute) + rng.gauss(0, 0.015)
        labels.append(round(min(0.99, max(0.01, pref)), 3))
    return labels


def situation(
    rng: random.Random, baseline: str, others: Sequence[str], counts: dict[int, float] = CONTEXT_COUNT_WEIGHTS
) -> list[str]:
    """The baseline alone, or other contexts drawn uniformly; the baseline never joins another."""
    count = _pick(rng, counts)
    return [baseline] if count == 0 else rng.sample(list(others), count)


def shortlist(rng: random.Random, pools: dict[str, list[dict]]) -> tuple[str, list[dict]]:
    size = rng.randint(MIN_ALTS, MAX_ALTS)
    if rng.random() < WITHIN_TYPOLOGY_SHARE:
        typology = rng.choice(list(pools))
        return f"{SHORTLIST_WITHIN}:{typology}", rng.sample(pools[typology], size)
    typologies = [rng.choice(list(pools)) for _ in range(size)]
    while len(set(typologies)) < 2:
        typologies[rng.randrange(size)] = rng.choice(list(pools))
    chosen: list[dict] = []
    for typology in typologies:
        candidate = rng.choice(pools[typology])
        while candidate in chosen:
            candidate = rng.choice(pools[typology])
        chosen.append(candidate)
    return SHORTLIST_ACROSS, chosen


def synthesise(
    registry: Registry, library: compose.Library, sets: int, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    pools = {
        typology: [
            compose.product(library, rng, typology, f"{typology}_{index:04d}")
            for index in range(PRODUCTS_PER_TYPOLOGY)
        ]
        for typology in compose.TYPOLOGIES
    }
    category = registry.category(CATEGORY_KEY)
    baseline = category.default_context_key
    others = sorted(category.available_contexts - {baseline})
    stakeholders = sorted(registry.stakeholders)

    scenarios, labels = [], []
    for index in range(sets):
        chosen_contexts = situation(rng, baseline, others)
        chosen_stakeholders = rng.sample(stakeholders, _pick(rng, STAKEHOLDER_COUNT_WEIGHTS))
        kind, alternatives = shortlist(rng, pools)
        scenario, label = _scenario(
            registry, rng, f"synthetic_{index + 1:06d}", kind, alternatives, chosen_contexts, chosen_stakeholders
        )
        scenarios.append(scenario)
        labels.append(label)
    return scenarios, labels


def _scenario(
    registry: Registry, rng: random.Random, external_id: str, kind: str, alternatives: list[dict],
    contexts: list[str], stakeholders: list[str],
) -> tuple[dict, dict]:
    utilities = [utility(registry, alt, contexts, stakeholders) for alt in alternatives]
    return (
        {"id": external_id, "shortlist": kind, "contexts": contexts, "stakeholders": stakeholders,
         "alternatives": alternatives},
        {
            "id": external_id,
            "labelled_alternatives": [
                {"id_prod": alt["id_prod"], "pref": pref, "conf": LABEL_CONFIDENCE, "reason": "Synthetic rule label."}
                for alt, pref in zip(alternatives, rule_labels(utilities, rng))
            ],
        },
    )


def unusual_shortlist(rng: random.Random, pools: dict[str, list[dict]], external_id: str) -> tuple[str, list[dict]]:
    """A shortlist with one or two alternatives replaced by variants, half of them beside the product they vary."""
    kind, alternatives = shortlist(rng, pools)
    alternatives = list(alternatives)
    replaced = rng.sample(range(len(alternatives)), 1 if len(alternatives) <= 3 else rng.randint(1, 2))
    for number, slot in enumerate(replaced, start=1):
        others = [a for i, a in enumerate(alternatives) if i not in replaced]
        parent = rng.choice(others) if others and rng.random() < PAIRED_VARIANT_SHARE else alternatives[slot]
        alternatives[slot] = variants.variant(parent, rng, f"{external_id}_variant_{number}")
    return f"{SHORTLIST_UNUSUAL}:{kind}", alternatives


def extend(registry: Registry, directory: Path, sets: int, seed: int, counts: dict[int, float]) -> int:
    """Append sets drawn from the products already in the dataset, leaving every existing set as it was."""
    return _append(registry, directory, sets, seed, counts, lambda rng, pools, _: shortlist(rng, pools))


def extend_unusual(registry: Registry, directory: Path, sets: int, seed: int) -> int:
    """Append sets that each hold an unusual variant of a product already in the dataset."""
    return _append(registry, directory, sets, seed, CONTEXT_COUNT_WEIGHTS, unusual_shortlist)


def _append(
    registry: Registry, directory: Path, sets: int, seed: int, counts: dict[int, float],
    draw: Callable[[random.Random, dict[str, list[dict]], str], tuple[str, list[dict]]],
) -> int:
    scenarios_path, labels_path = directory / SCENARIOS_FILE, directory / LABELS_FILE
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    pools: dict[str, dict[str, dict]] = {}
    for scenario in scenarios:
        for alternative in scenario["alternatives"]:
            if "derived_from" not in alternative:
                pools.setdefault(alternative["typology"], {})[alternative["id_prod"]] = alternative
    products = {typology: list(found.values()) for typology, found in sorted(pools.items())}

    category = registry.category(CATEGORY_KEY)
    others = sorted(category.available_contexts - {category.default_context_key})
    stakeholders = sorted(registry.stakeholders)
    rng = random.Random(seed)
    first = 1 + max(int(s["id"].rsplit("_", 1)[1]) for s in scenarios)
    for index in range(first, first + sets):
        external_id = f"synthetic_{index:06d}"
        contexts = situation(rng, category.default_context_key, others, counts)
        chosen = rng.sample(stakeholders, _pick(rng, STAKEHOLDER_COUNT_WEIGHTS))
        kind, alternatives = draw(rng, products, external_id)
        scenario, label = _scenario(registry, rng, external_id, kind, alternatives, contexts, chosen)
        scenarios.append(scenario)
        labels.append(label)
    scenarios_path.write_text(json.dumps(scenarios), encoding="utf-8")
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    return sets


def write(
    registry: Registry, library: compose.Library, directory: Path, sets: int, seed: int = 0
) -> tuple[Path, Path]:
    scenarios, labels = synthesise(registry, library, sets, seed)
    directory.mkdir(parents=True, exist_ok=True)
    scenarios_path, labels_path = directory / SCENARIOS_FILE, directory / LABELS_FILE
    scenarios_path.write_text(json.dumps(scenarios), encoding="utf-8")
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    return scenarios_path, labels_path


def main() -> None:
    import argparse

    from core import registry as registry_module
    from db.session import create_db_engine, session_scope
    from ingest.build import JOINT_DB

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    more = commands.add_parser("extend", help="append sets where three or four contexts combine")
    more.add_argument("--sets", type=int, required=True)
    more.add_argument("--seed", type=int, required=True)
    unusual = commands.add_parser("unusual", help="append sets that each hold an unusual variant of a product")
    unusual.add_argument("--sets", type=int, required=True)
    unusual.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    with session_scope(create_db_engine(JOINT_DB)) as session:
        registry = registry_module.from_session(session)
    if args.command == "unusual":
        added = extend_unusual(registry, DATASET_DIR, args.sets, args.seed)
    else:
        added = extend(registry, DATASET_DIR, args.sets, args.seed, COMBINED_CONTEXT_WEIGHTS)
    print(f"appended {added} sets to {DATASET_DIR / SCENARIOS_FILE}")


if __name__ == "__main__":
    main()
