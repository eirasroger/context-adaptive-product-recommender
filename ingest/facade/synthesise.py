"""Write the working facade dataset: EPD-composed products, shortlists and rule labels."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

from core.registry import Registry
from ingest.generators import parametric
from ingest.facade import compose

CATEGORY_KEY = "facade_system"
DATASET_DIR = Path("data/facade")
SCENARIOS_FILE = "frozen_dataset.json"
LABELS_FILE = "labelled_dataset.json"
LLM_LABELS_FILE = "llm_labelled_dataset.json"

PRODUCTS_PER_TYPOLOGY = 2_000
WITHIN_TYPOLOGY_SHARE = 0.75
SECOND_CONTEXT_SHARE = 0.25
STAKEHOLDER_COUNT_WEIGHTS = {1: 0.6, 2: 0.3, 3: 0.1}
MIN_ALTS, MAX_ALTS = 2, 5

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


def shortlist(rng: random.Random, pools: dict[str, list[dict]]) -> tuple[str, list[dict]]:
    size = rng.randint(MIN_ALTS, MAX_ALTS)
    if rng.random() < WITHIN_TYPOLOGY_SHARE:
        typology = rng.choice(list(pools))
        return f"within:{typology}", rng.sample(pools[typology], size)
    typologies = [rng.choice(list(pools)) for _ in range(size)]
    while len(set(typologies)) < 2:
        typologies[rng.randrange(size)] = rng.choice(list(pools))
    chosen: list[dict] = []
    for typology in typologies:
        candidate = rng.choice(pools[typology])
        while candidate in chosen:
            candidate = rng.choice(pools[typology])
        chosen.append(candidate)
    return "across", chosen


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
    contexts = sorted(registry.category(CATEGORY_KEY).available_contexts)
    stakeholders = sorted(registry.stakeholders)

    scenarios, labels = [], []
    for index in range(sets):
        chosen_contexts = [rng.choice(contexts)]
        if rng.random() < SECOND_CONTEXT_SHARE:
            chosen_contexts.append(rng.choice([c for c in contexts if c != chosen_contexts[0]]))
        chosen_stakeholders = rng.sample(stakeholders, _pick(rng, STAKEHOLDER_COUNT_WEIGHTS))
        kind, alternatives = shortlist(rng, pools)

        external_id = f"synthetic_{index + 1:06d}"
        scenarios.append(
            {
                "id": external_id,
                "shortlist": kind,
                "contexts": chosen_contexts,
                "stakeholders": chosen_stakeholders,
                "alternatives": alternatives,
            }
        )
        utilities = [utility(registry, alt, chosen_contexts, chosen_stakeholders) for alt in alternatives]
        labels.append(
            {
                "id": external_id,
                "labelled_alternatives": [
                    {"id_prod": alt["id_prod"], "pref": pref, "conf": LABEL_CONFIDENCE, "reason": "Synthetic rule label."}
                    for alt, pref in zip(alternatives, rule_labels(utilities, rng))
                ],
            }
        )
    return scenarios, labels


def write(
    registry: Registry, library: compose.Library, directory: Path, sets: int, seed: int = 0
) -> tuple[Path, Path]:
    scenarios, labels = synthesise(registry, library, sets, seed)
    directory.mkdir(parents=True, exist_ok=True)
    scenarios_path, labels_path = directory / SCENARIOS_FILE, directory / LABELS_FILE
    scenarios_path.write_text(json.dumps(scenarios), encoding="utf-8")
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    return scenarios_path, labels_path
