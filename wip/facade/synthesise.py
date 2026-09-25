"""Write the working facade dataset: typology-templated products, shortlists and rule labels."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable, Sequence

from core.registry import Registry
from ingest.generators import parametric

CATEGORY_KEY = "facade_system"
DATASET_DIR = Path("data/wip/facade")
SCENARIOS_FILE = "frozen_dataset.json"
LABELS_FILE = "labelled_dataset.json"

PRODUCTS_PER_TYPOLOGY = 2_500
WITHIN_TYPOLOGY_SHARE = 0.75
SECOND_CONTEXT_SHARE = 0.25
STAKEHOLDER_COUNT_WEIGHTS = {1: 0.6, 2: 0.3, 3: 0.1}
MIN_ALTS, MAX_ALTS = 2, 5

SURFACE_RESISTANCES = 0.17
MISSING_QUALITY = 0.3
LABEL_CONFIDENCE = 0.8
HEALTH_WEIGHTS = {"h0": 0.05, "h1": 0.35, "h2": 0.35, "h3": 0.10, "h4": 0.08, "h5": 0.05, "h6": 0.02}


def _pick(rng: random.Random, weights: dict):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def mass_law_reduction(surface_mass: float) -> float:
    """Airborne sound reduction of a single leaf by the CTE DB-HR mass law."""
    if surface_mass <= 150:
        return 16.6 * math.log10(surface_mass) + 5
    return 36.5 * math.log10(surface_mass) - 38.5


def transmittance(base_resistance: float, insulation_mm: float, conductivity: float) -> float:
    return 1.0 / (SURFACE_RESISTANCES + base_resistance + insulation_mm / 1000 / conductivity)


def end_of_life(rng: random.Random, recovered: tuple, incinerated: tuple, hazardous: tuple) -> dict:
    shares = {
        "fu_recyc": rng.uniform(*recovered),
        "fu_incin": rng.uniform(*incinerated),
        "fu_haz": rng.uniform(*hazardous),
    }
    shares["fu_inert"] = max(0.0, 100.0 - sum(shares.values()))
    return shares


INSULATION = {
    "mineral_wool": {"conductivity": (0.033, 0.040), "density": (40, 140), "gwp_per_kg": 1.2},
    "eps": {"conductivity": (0.031, 0.038), "density": (15, 25), "gwp_per_kg": 3.3},
    "pir": {"conductivity": (0.022, 0.025), "density": (38, 45), "gwp_per_kg": 3.0},
}


def _insulation(rng: random.Random, material: str, thickness_mm: float) -> dict:
    spec = INSULATION[material]
    mass = rng.uniform(*spec["density"]) * thickness_mm / 1000
    return {
        "conductivity": rng.uniform(*spec["conductivity"]),
        "mass": mass,
        "gwp": mass * spec["gwp_per_kg"],
    }


def _substrate(rng: random.Random) -> dict:
    """A heavy wall of brick, block or concrete that ETICS and ventilated facades sit on."""
    mass = rng.uniform(150, 350)
    return {
        "mass": mass,
        "thickness": rng.uniform(150, 250),
        "resistance": rng.uniform(0.2, 0.6),
        "gwp": mass * rng.uniform(0.18, 0.25),
        "heat_capacity": min(0.6 * mass, 250) * rng.uniform(0.9, 1.1),
    }


def sandwich_panel(rng: random.Random) -> dict:
    core = _pick(rng, {"mineral_wool": 0.6, "pir": 0.4})
    thickness = rng.uniform(80, 250) if core == "mineral_wool" else rng.uniform(60, 200)
    conductivity = rng.uniform(0.040, 0.044) if core == "mineral_wool" else rng.uniform(0.022, 0.025)
    core_mass = (rng.uniform(90, 120) if core == "mineral_wool" else rng.uniform(38, 45)) * thickness / 1000
    low_carbon_steel = rng.random() < 0.25
    steel_mass = rng.uniform(9, 12)
    gwp = steel_mass * (1.1 if low_carbon_steel else 2.6) + core_mass * (0.6 if core == "mineral_wool" else 3.0)
    return {
        "u_value": transmittance(0.0, thickness, conductivity),
        "acoustic_reduction": rng.uniform(29, 34) if core == "mineral_wool" else rng.uniform(24, 28),
        "surface_mass": steel_mass + core_mass,
        "heat_capacity": rng.uniform(5, 15),
        "fire_reaction": _pick(rng, {"a2": 0.7, "a1": 0.3} if core == "mineral_wool" else {"b": 0.6, "c": 0.3, "d": 0.1}),
        "thickness": thickness + 1,
        "gwp": gwp * rng.uniform(0.9, 1.1),
        "cost_product": rng.uniform(45, 90) + 0.1 * thickness,
        "cost_labour": rng.uniform(15, 30),
        "cost_maintenance": rng.uniform(10, 30),
        "circ_orig": rng.uniform(40, 75) if low_carbon_steel else rng.uniform(15, 40),
        **end_of_life(rng, (70, 90), (0, 3) if core == "mineral_wool" else (8, 20), (0, 1)),
    }


def masonry(rng: random.Random) -> dict:
    """Cavity wall: facing brick outside, insulation in the cavity, a light inner leaf."""
    material = _pick(rng, {"mineral_wool": 0.5, "eps": 0.5})
    insulation_mm = rng.uniform(40, 120)
    insulation = _insulation(rng, material, insulation_mm)
    leaves = rng.uniform(170, 230) + rng.uniform(60, 90) + 20
    mass = leaves + insulation["mass"]
    return {
        "u_value": transmittance(rng.uniform(0.45, 0.75), insulation_mm, insulation["conductivity"]),
        "acoustic_reduction": mass_law_reduction(leaves) + rng.uniform(-1, 3),
        "surface_mass": mass,
        "heat_capacity": rng.uniform(40, 100),
        "fire_reaction": _pick(rng, {"a1": 0.85, "a2": 0.15}),
        "thickness": 115 + 70 + 30 + insulation_mm,
        "gwp": leaves * rng.uniform(0.20, 0.26) + insulation["gwp"] + rng.uniform(4, 10),
        "cost_product": rng.uniform(40, 80),
        "cost_labour": rng.uniform(50, 90),
        "cost_maintenance": rng.uniform(5, 20),
        "circ_orig": rng.uniform(0, 25),
        **end_of_life(rng, (20, 60), (1, 4) if material == "eps" else (0, 1), (0, 0.5)),
    }


def etics(rng: random.Random) -> dict:
    """External thermal insulation composite system on a heavy substrate."""
    material = _pick(rng, {"eps": 0.65, "mineral_wool": 0.35})
    insulation_mm = rng.uniform(60, 200)
    insulation = _insulation(rng, material, insulation_mm)
    substrate = _substrate(rng)
    render = rng.uniform(15, 25)
    return {
        "u_value": transmittance(substrate["resistance"], insulation_mm, insulation["conductivity"]),
        "acoustic_reduction": mass_law_reduction(substrate["mass"])
        + (rng.uniform(-4, 0) if material == "eps" else rng.uniform(0, 3)),
        "surface_mass": substrate["mass"] + insulation["mass"] + render,
        "heat_capacity": substrate["heat_capacity"],
        "fire_reaction": _pick(rng, {"b": 0.7, "c": 0.3} if material == "eps" else {"a2": 0.75, "a1": 0.25}),
        "thickness": substrate["thickness"] + insulation_mm + 10,
        "gwp": substrate["gwp"] + insulation["gwp"] + rng.uniform(8, 15),
        "cost_product": rng.uniform(45, 85) + rng.uniform(25, 40),
        "cost_labour": rng.uniform(45, 80),
        "cost_maintenance": rng.uniform(25, 60),
        "circ_orig": rng.uniform(0, 15),
        **end_of_life(rng, (5, 30), (3, 10) if material == "eps" else (0, 2), (0, 2)),
    }


CLADDING = {
    "ceramic": {"fire": {"a1": 1.0}, "mass": (30, 40), "gwp": (15, 25), "cost": (60, 120), "combustible": False},
    "stone": {"fire": {"a1": 1.0}, "mass": (50, 80), "gwp": (8, 15), "cost": (90, 180), "combustible": False},
    "fibre_cement": {"fire": {"a2": 1.0}, "mass": (15, 25), "gwp": (10, 20), "cost": (40, 80), "combustible": False},
    "hpl": {"fire": {"b": 0.5, "c": 0.5}, "mass": (10, 15), "gwp": (10, 18), "cost": (50, 90), "combustible": True},
    "composite_panel": {"fire": {"b": 0.5, "d": 0.3, "e": 0.2}, "mass": (6, 10), "gwp": (25, 45), "cost": (45, 85), "combustible": True},
}


def ventilated(rng: random.Random) -> dict:
    """Rainscreen cladding on a subframe, over mineral wool on a heavy substrate."""
    cladding = CLADDING[rng.choice(list(CLADDING))]
    insulation_mm = rng.uniform(60, 160)
    insulation = _insulation(rng, "mineral_wool", insulation_mm)
    substrate = _substrate(rng)
    aluminium_subframe = rng.random() < 0.6
    return {
        "u_value": transmittance(substrate["resistance"], insulation_mm, insulation["conductivity"]),
        "acoustic_reduction": mass_law_reduction(substrate["mass"]) + rng.uniform(1, 5),
        "surface_mass": substrate["mass"] + insulation["mass"] + rng.uniform(*cladding["mass"]) + rng.uniform(3, 6),
        "heat_capacity": substrate["heat_capacity"],
        "fire_reaction": _pick(rng, cladding["fire"]),
        "thickness": substrate["thickness"] + insulation_mm + rng.uniform(30, 50) + rng.uniform(10, 40),
        "gwp": substrate["gwp"] + insulation["gwp"] + rng.uniform(*cladding["gwp"])
        + (rng.uniform(15, 40) if aluminium_subframe else rng.uniform(8, 15)),
        "cost_product": rng.uniform(*cladding["cost"]) + rng.uniform(30, 60),
        "cost_labour": rng.uniform(50, 100),
        "cost_maintenance": rng.uniform(8, 25),
        "circ_orig": rng.uniform(10, 45),
        **end_of_life(rng, (45, 85), (3, 12) if cladding["combustible"] else (0, 3), (0, 1)),
    }


TYPOLOGIES: dict[str, Callable[[random.Random], dict]] = {
    "sandwich_panel": sandwich_panel,
    "masonry": masonry,
    "etics": etics,
    "ventilated": ventilated,
}

#: Chance a value is left out, where it differs from the default.
MISSING_SHARE = {"wdp": 0.10, "fwu": 0.10, "b": 0.10}
MISSING_BY_TYPOLOGY = {("etics", "acoustic_reduction"): 0.25, ("sandwich_panel", "heat_capacity"): 0.30}
DEFAULT_MISSING_SHARE = 0.03


def product(rng: random.Random, typology: str, key: str) -> dict:
    values = TYPOLOGIES[typology](rng)
    gwp = values["gwp"]
    values["wdp"] = gwp * rng.uniform(0.05, 0.2)
    values["fwu"] = gwp * rng.uniform(0.002, 0.008)
    values["b"] = min(1.0, max(0.0, 0.05 + gwp / 250 + rng.uniform(-0.08, 0.08)))
    values["health"] = _pick(rng, HEALTH_WEIGHTS)

    kept = {}
    for name, value in values.items():
        share = MISSING_BY_TYPOLOGY.get((typology, name), MISSING_SHARE.get(name, DEFAULT_MISSING_SHARE))
        if rng.random() < share:
            continue
        kept[name] = value if isinstance(value, str) else round(value, 4)
    return {"id_prod": key, "typology": typology, **kept}


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


def synthesise(registry: Registry, sets: int, seed: int = 0) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    pools = {
        typology: [product(rng, typology, f"{typology}_{index:04d}") for index in range(PRODUCTS_PER_TYPOLOGY)]
        for typology in TYPOLOGIES
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


def write(registry: Registry, directory: Path, sets: int, seed: int = 0) -> tuple[Path, Path]:
    scenarios, labels = synthesise(registry, sets, seed)
    directory.mkdir(parents=True, exist_ok=True)
    scenarios_path, labels_path = directory / SCENARIOS_FILE, directory / LABELS_FILE
    scenarios_path.write_text(json.dumps(scenarios), encoding="utf-8")
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    return scenarios_path, labels_path
