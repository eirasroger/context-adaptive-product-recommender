"""Unusual but plausible facade products: a real product redesigned along a few levers, its physics kept."""

from __future__ import annotations

import math
import random
from typing import Callable

from ingest.facade import compose

#: Thermal resistance of the wall around the insulation, m2K/W, as compose builds each typology.
BASE_RESISTANCE = {"sandwich_panel": 0.0, "etics": 0.4, "ventilated": 0.4, "precast": 0.1, "masonry": 0.6}
#: Embodied carbon of installed insulation, kg CO2e per m3; wood fibre stores biogenic carbon.
INSULATION_GWP_PER_M3 = {"eps": 60.0, "xps": 100.0, "pir": 130.0, "wool": 120.0, "wood": -60.0}
#: The insulant a lever has chosen, carried to later levers and dropped from the variant.
INSULANT = "_insulant"
SANDWICH_CORES = ("wool", "pir", "eps")
ETICS_INSULANTS = ("eps", "wool", "wood", "pir")
SANDWICH_CONDUCTIVITY_ALLOWANCE = 0.004
MIN_INSULATION_MM = 20.0
U_RANGE = (0.10, 0.60)

#: Substrate behind ETICS or a ventilated cladding: aerated block at the light end, dense concrete at the heavy.
SUBSTRATE_DENSITY = (0.45, 2.4)
SUBSTRATE_THICKNESS_MM = (150.0, 300.0)
SUBSTRATE_MASS = (90.0, 650.0)
SUBSTRATE_GWP_PER_KG = (0.08, 0.30)
PRECAST_LEAF_MASS = (240.0, 600.0)
MASONRY_LEAVES_MASS = (150.0, 420.0)

GWP_FACTOR = (0.5, 1.8)
WATER_FACTOR = (0.4, 2.5)
#: Highest plausible share of secondary material by typology, set by its main material.
RECYCLED_CONTENT_CAP = {"sandwich_panel": 75.0, "precast": 30.0, "masonry": 20.0, "etics": 20.0, "ventilated": 40.0}
RECOVERED_SHARE = (5.0, 95.0)
COST_RANGE = {"cost_product": (40.0, 230.0), "cost_labour": (15.0, 100.0), "cost_maintenance": (5.0, 60.0)}
HEALTH_LEVELS = ("h1", "h2", "h3", "h4", "h5", "h6")

#: Hard limits on every value a variant can carry.
PLAUSIBLE = {
    "gwp": (-60.0, 400.0),
    "wdp": (-10.0, 120.0),
    "fwu": (0.0, 8.0),
    "circ_orig": (0.0, 100.0),
    "fu_recyc": (0.0, 100.0),
    "fu_incin": (0.0, 100.0),
    "fu_inert": (0.0, 100.0),
    "fu_haz": (0.0, 100.0),
    "u_value": (0.08, 0.70),
    "acoustic_reduction": (20.0, 65.0),
    "surface_mass": (8.0, 800.0),
    "heat_capacity": (3.0, 250.0),
    "thickness": (40.0, 550.0),
    **COST_RANGE,
}

PHYSICAL = ("mass", "insulant", "insulation", "cladding")
FREE = ("carbon", "content", "end_of_life", "cost", "health")
APPLICABLE = {
    "sandwich_panel": ("insulant", "insulation", *FREE),
    "etics": ("mass", "insulant", "insulation", *FREE),
    "ventilated": ("mass", "insulation", "cladding", *FREE),
    "precast": ("mass", "insulation", *FREE),
    "masonry": ("mass", "insulation", *FREE),
}
LEVERS_PER_VARIANT = (2, 4)


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def insulant_of(product: dict) -> str:
    """The insulation a composed product most likely holds, read from what its kind leaves visible."""
    if INSULANT in product:
        return product[INSULANT]
    typology = product["typology"]
    if typology == "sandwich_panel":
        return "wool" if product["acoustic_reduction"] >= 28.5 else "pir"
    if typology == "etics":
        return {"a1": "wool", "a2": "wool", "d": "wood", "e": "xps"}.get(product.get("fire_reaction"), "eps")
    if typology == "precast":
        return "eps"
    return "wool"


def conductivity(typology: str, insulant: str) -> float:
    allowance = SANDWICH_CONDUCTIVITY_ALLOWANCE if typology == "sandwich_panel" else 0.0
    return compose.INSULATION[insulant][0] + allowance


def insulation_mm(product: dict, insulant: str) -> float:
    resistance = 1 / product["u_value"] - compose.SURFACE_RESISTANCES - BASE_RESISTANCE[product["typology"]]
    return max(resistance, 0.1) * conductivity(product["typology"], insulant) * 1000


def substrate_mass(product: dict) -> float:
    typology = product["typology"]
    if typology == "precast":
        return product["heat_capacity"] / 0.36
    if typology == "masonry":
        return max(product["surface_mass"] - 10.0, 100.0)
    return product["heat_capacity"] / 0.6


def _adjust(values: dict, key: str, change: float) -> None:
    if values.get(key) is not None:
        values[key] += change


def _swap_insulation(values: dict, old: str, new: str, old_mm: float, new_mm: float) -> None:
    old_density, new_density = compose.INSULATION[old][1], compose.INSULATION[new][1]
    values["thickness"] += new_mm - old_mm
    values["surface_mass"] += (new_density * new_mm - old_density * old_mm) / 1000
    _adjust(values, "gwp", (INSULATION_GWP_PER_M3[new] * new_mm - INSULATION_GWP_PER_M3[old] * old_mm) / 1000)


def change_mass(values: dict, rng: random.Random) -> None:
    """A lighter or heavier substrate; sound follows the mass law, thermal mass and thickness follow the mass."""
    typology, old = values["typology"], substrate_mass(values)
    if typology == "precast":
        new = rng.uniform(*PRECAST_LEAF_MASS)
        values["thickness"] += (new - old) / 2.4
        sound_old, sound_new = 0.6 * old, 0.6 * new
        thermal_old, thermal_new = min(0.36 * old, 250), min(0.36 * new, 250)
        carbon_per_kg = (values.get("gwp") or 0.0) / values["surface_mass"]
    elif typology == "masonry":
        new = rng.uniform(*MASONRY_LEAVES_MASS)
        sound_old, sound_new = old, new
        thermal_old, thermal_new = old, new
        carbon_per_kg = (values.get("gwp") or 0.0) / values["surface_mass"]
    else:
        density, thickness = rng.uniform(*SUBSTRATE_DENSITY), rng.uniform(*SUBSTRATE_THICKNESS_MM)
        new = min(max(density * thickness, SUBSTRATE_MASS[0]), SUBSTRATE_MASS[1])
        values["thickness"] += new / density - old / 1.4
        sound_old, sound_new = old, new
        thermal_old, thermal_new = min(0.6 * old, 250), min(0.6 * new, 250)
        carbon_per_kg = rng.uniform(*SUBSTRATE_GWP_PER_KG)
    values["surface_mass"] += new - old
    values["acoustic_reduction"] += compose.mass_law_reduction(sound_new) - compose.mass_law_reduction(sound_old)
    values["heat_capacity"] *= thermal_new / thermal_old
    _adjust(values, "gwp", (new - old) * carbon_per_kg)


def change_insulant(values: dict, rng: random.Random) -> None:
    """Another insulation at the same U-value; thickness, mass, carbon and fire class follow the material."""
    typology, old = values["typology"], insulant_of(values)
    new = rng.choice([kind for kind in (SANDWICH_CORES if typology == "sandwich_panel" else ETICS_INSULANTS) if kind != old])
    old_mm = insulation_mm(values, old)
    new_mm = old_mm * conductivity(typology, new) / conductivity(typology, old)
    _swap_insulation(values, old, new, old_mm, new_mm)
    values[INSULANT] = new
    values["fire_reaction"] = compose.insulation_fire(new, rng)
    if typology == "sandwich_panel":
        values["acoustic_reduction"] = rng.uniform(29, 34) if new == "wool" else rng.uniform(24, 28)
    else:
        bonus = {"eps": -2.0}
        values["acoustic_reduction"] += bonus.get(new, 1.5) - bonus.get(old, 1.5)


def change_insulation(values: dict, rng: random.Random) -> None:
    """More or less insulation; a lower U-value costs thickness, mass, carbon and product cost."""
    typology, insulant = values["typology"], insulant_of(values)
    lam = conductivity(typology, insulant)
    old_mm = insulation_mm(values, insulant)
    target = _log_uniform(rng, *U_RANGE)
    new_mm = max(old_mm + (1 / target - 1 / values["u_value"]) * lam * 1000, MIN_INSULATION_MM)
    values["u_value"] = 1 / (1 / values["u_value"] + (new_mm - old_mm) / 1000 / lam)
    _swap_insulation(values, insulant, insulant, old_mm, new_mm)
    _adjust(values, "cost_product", 0.1 * (new_mm - old_mm))


def change_cladding(values: dict, rng: random.Random) -> None:
    """Another rainscreen; its material sets the reaction to fire."""
    old_mass = max(values["surface_mass"] - substrate_mass(values) - 15.0, 5.0)
    kind = rng.choice([k for k in compose.CLADDING if k != "precast"])
    fire, typical_mass, _ = compose.CLADDING[kind]
    values["fire_reaction"] = compose.weighted_choice(rng, fire)
    values["surface_mass"] += typical_mass - old_mass


def change_carbon(values: dict, rng: random.Random) -> None:
    """Another manufacturing route: cement, steel or electricity of lower or higher carbon, and its water."""
    _scale(values, "gwp", _log_uniform(rng, *GWP_FACTOR))
    _scale(values, "wdp", _log_uniform(rng, *WATER_FACTOR))
    _scale(values, "fwu", _log_uniform(rng, *WATER_FACTOR))


def change_content(values: dict, rng: random.Random) -> None:
    if values.get("circ_orig") is not None:
        values["circ_orig"] = rng.uniform(0.0, RECYCLED_CONTENT_CAP[values["typology"]])


def change_end_of_life(values: dict, rng: random.Random) -> None:
    """Another end-of-life route: dry-fixed and demountable, or bonded and landfilled."""
    if values.get("fu_recyc") is None:
        return
    room = 100.0 - (values.get("fu_haz") or 0.0) - (values.get("fu_incin") or 0.0)
    values["fu_recyc"] = min(rng.uniform(*RECOVERED_SHARE), room)
    if values.get("fu_inert") is not None:
        values["fu_inert"] = room - values["fu_recyc"]


def change_cost(values: dict, rng: random.Random) -> None:
    """Another market: product line, regional wage level and upkeep drawn apart from the typology."""
    for key, bounds in COST_RANGE.items():
        if values.get(key) is not None:
            values[key] = rng.uniform(*bounds)


def change_health(values: dict, rng: random.Random) -> None:
    if values.get("health") is not None:
        values["health"] = rng.choice(HEALTH_LEVELS)


def _scale(values: dict, key: str, factor: float) -> None:
    if values.get(key) is not None:
        values[key] *= factor


LEVERS: dict[str, Callable[[dict, random.Random], None]] = {
    "mass": change_mass,
    "insulant": change_insulant,
    "insulation": change_insulation,
    "cladding": change_cladding,
    "carbon": change_carbon,
    "content": change_content,
    "end_of_life": change_end_of_life,
    "cost": change_cost,
    "health": change_health,
}


def variant(parent: dict, rng: random.Random, key: str, levers: tuple[str, ...] | None = None) -> dict:
    """The parent redesigned along a few of its typology's levers, physical ones first."""
    applicable = APPLICABLE[parent["typology"]]
    chosen = levers or tuple(rng.sample(applicable, rng.randint(*LEVERS_PER_VARIANT)))
    ordered = [lever for lever in LEVERS if lever in chosen]
    values = dict(parent)
    for lever in ordered:
        LEVERS[lever](values, rng)
    values.pop(INSULANT, None)
    for name, (low, high) in PLAUSIBLE.items():
        if values.get(name) is not None:
            values[name] = round(min(max(values[name], low), high), 4)
    return {**values, "id_prod": key, "derived_from": parent["id_prod"], "levers": ordered}
