"""Facade systems composed from real EPD layers; physics supplies what the EPDs do not declare."""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

from ingest.epd.layers import (
    LIMITS_PER_KG,
    Build,
    Layer,
    health_status,
    mass_law_reduction,
    number,
    per_unit,
    weighted_choice,
)

CANDIDATES = Path("data/facade/epd/candidates.csv")

SURFACE_RESISTANCES = 0.17
#: Product rules for groups that match a facade term but are another product.
OFF_SCOPE_PCR = ("windows and doors", "technical - chemical", "technical-chemical", "werkmörtel", "dübel")
OFF_SCOPE_NAME = ("mortar", "mörtel", "adhesive", "kleber", "fastening", "dowel", "geosynth", "masonry cement",
                  "support system", "wall ties", "paver", "retaining")

#: Insulation classes: typical declared conductivity (W/mK) and density (kg/m3) when the EPD gives none.
INSULATION = {
    "wool": (0.035, 100.0),
    "eps": (0.033, 18.0),
    "xps": (0.034, 33.0),
    "pir": (0.023, 32.0),
    "wood": (0.042, 150.0),
}
INSULATION_PATTERNS = {
    "pir": r"polyisocyanurate|polyurethane|\bpir\b|\bpur\b|\bpu\b|phenolic",
    "xps": r"\bxps\b|extruded polystyrene",
    "eps": r"\beps\b|expanded polystyrene|polystyrol",
    "wood": r"wood ?fib|holzfaser",
    "wool": r"wool|wolle|rocksilk|paroc|isover|rockwool|mineral",
}

#: Cladding classes: reaction to fire, typical mass (kg/m2) when the EPD gives none, and whether it burns.
CLADDING = {
    "ceramic": ({"a1": 1.0}, 35.0, False),
    "precast": ({"a1": 1.0}, 380.0, False),
    "metal": ({"a1": 0.7, "a2": 0.3}, 8.0, False),
    "fibre_cement": ({"a2": 1.0}, 18.0, False),
    "stone_wool": ({"a2": 1.0}, 10.0, False),
    "hpl": ({"b": 0.5, "c": 0.5}, 12.0, True),
    "composite": ({"b": 0.4, "c": 0.3, "d": 0.3}, 8.0, True),
    "timber": ({"d": 1.0}, 12.0, True),
}
CLADDING_PATTERNS = {
    "precast": r"precast|concrete cladding",
    "ceramic": r"ceramic|porcelain|terracotta|tile|stoneo|stone-faced|granite",
    "fibre_cement": r"fib(re|er) ?cement|faserzement|hardie|duraflex|cemintel|nichiha|hicem|hiden|flat sheet",
    "stone_wool": r"rockpanel|stone wool",
    "hpl": r"\bhpl\b|laminate|fundermax|resopal|polyrey|trespa",
    "metal": r"alumin|steel|zinc|copper|kalzip|metal",
    "timber": r"wood|timber|spruce|larch|accoya|superwood|pine|cedar",
    "composite": r"composite|wpc|honeycomb",
}

def _classify(text: str, patterns: dict[str, str]) -> str | None:
    lowered = text.lower()
    return next((kind for kind, pattern in patterns.items() if re.search(pattern, lowered)), None)


def in_metres(thickness: float | None) -> float | None:
    """A declared layer thickness in metres; some EPDs enter millimetres, and a few enter nothing usable."""
    if thickness is None:
        return None
    if thickness > 1.0:
        thickness /= 1000
    return thickness if thickness >= 0.005 else None


def layer_from(row: dict) -> Layer | None:
    """Place one EPD in the layer library, or leave it out when it is off scope or cannot be scaled."""
    name, pcr = row["name"], (row["pcr"] or "").lower()
    lowered = name.lower()
    if any(p in pcr for p in OFF_SCOPE_PCR) or any(p in lowered for p in OFF_SCOPE_NAME):
        return None
    if number(row["gwp_total_a1a3"]) is None:
        return None

    unit, amount = row["declared_unit"], number(row["declared_amount"]) or 1.0
    mass, thickness = number(row["mass_kg"]), in_metres(number(row["thickness_m"]))
    density, grammage = number(row["density_kg_m3"]), number(row["grammage_kg_m2"])
    if thickness is None and grammage and density:
        thickness = grammage / density
    typology = row["typology"]
    common = {"uuid": row["uuid"], "name": name}

    def per_m2(role: str, kind: str) -> Layer | None:
        if unit != "m2":
            return None
        return Layer(role, kind, values=per_unit(row, amount), mass=None if mass is None else mass / amount,
                     thickness_m=thickness, **common)

    metal_faced = re.search(r"steel|metal|alumin|\bpir\b|\bpur\b|polyurethane|double skin|insulated panel", lowered)
    faced_panel = "panel" in lowered and re.search(r"metal face|double skin|steel|wall and roof", lowered)
    if "concrete" not in lowered and (
        typology == "sandwich_panel" or ("sandwich" in lowered and metal_faced) or faced_panel
    ):
        if "core insulation" in lowered or "sip" in lowered.split():
            return None
        return per_m2("sandwich_panel", _classify(name, INSULATION_PATTERNS) or "wool")

    if typology == "precast" or "precast" in lowered:
        if "cladding" in lowered:
            return per_m2("cladding", "precast")
        if re.search(r"insulated|sandwich|three-layer|two-layer", lowered) and unit == "kg":
            return Layer("precast", _classify(name, INSULATION_PATTERNS) or "eps",
                         values=per_unit(row, amount), mass=1.0, **common)
        if unit == "kg":
            return Layer("substrate", "concrete", values=per_unit(row, amount), mass=1.0, **common)
        return None

    insulation = _classify(name, INSULATION_PATTERNS)
    if typology == "etics":
        kit = per_m2("etics_kit", insulation or "kit")
        return kit if kit and (insulation is None or thickness) else None

    if typology == "insulation_layer" or (insulation and re.search(r"insulation|slab|roll|board|dämm", lowered)):
        if insulation is None:
            return None
        if unit == "m3":
            divisor = amount
        elif unit == "m2" and thickness:
            divisor = amount * thickness
        elif unit == "kg" and density:
            divisor = amount / density
        else:
            return None
        per_m3_mass = mass / divisor if mass else density
        return Layer("insulation", insulation, values=per_unit(row, divisor), mass=per_m3_mass, density=density, **common)

    if typology == "masonry" and unit in ("kg", "t") and re.search(r"brick|block|ziegel|clay", lowered):
        return Layer("substrate", "brick", values=per_unit(row, amount * (1000 if unit == "t" else 1)), mass=1.0, **common)

    if typology == "render" and unit == "kg":
        return Layer("render", "render", values=per_unit(row, amount), mass=1.0, **common)

    if typology == "ventilated":
        kind = _classify(name, CLADDING_PATTERNS)
        return per_m2("cladding", kind) if kind else None
    return None


#: Values set aside by the last load, as (layer name, flow), for the build to report.
set_aside: list[tuple[str, str]] = []


def without_unit_errors(layers: list[Layer]) -> list[Layer]:
    set_aside.clear()
    cleaned = []
    for layer in layers:
        values = dict(layer.values)
        if layer.mass:
            for key, (low, high) in LIMITS_PER_KG.items():
                value = values[key]
                if value is not None and not low <= value / layer.mass <= high:
                    values[key] = None
                    set_aside.append((layer.name, key))
        cleaned.append(Layer(layer.role, layer.kind, layer.uuid, layer.name, values, layer.mass, layer.thickness_m, layer.density))
    return cleaned


class Library:
    """The layer EPDs available to each role and kind."""

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    @classmethod
    def load(cls, path: Path = CANDIDATES) -> "Library":
        if not path.exists():
            raise FileNotFoundError(f"no EPD table at {path}; run python -m ingest.facade.epd_search first")
        with path.open(encoding="utf-8") as handle:
            layers = [layer for row in csv.DictReader(handle) if (layer := layer_from(row))]
        return cls(without_unit_errors(layers))

    def of(self, role: str, *kinds: str) -> list[Layer]:
        return [l for l in self.layers if l.role == role and (not kinds or l.kind in kinds)]

    def pick(self, rng: random.Random, role: str, *kinds: str) -> Layer:
        options = self.of(role, *kinds) or self.of(role)
        if not options:
            raise LookupError(f"the EPD table holds no {role} layer")
        return rng.choice(options)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for layer in self.layers:
            counts[f"{layer.role}:{layer.kind}"] = counts.get(f"{layer.role}:{layer.kind}", 0) + 1
        return dict(sorted(counts.items()))


def transmittance(base_resistance: float, insulation_mm: float, conductivity: float) -> float:
    return 1.0 / (SURFACE_RESISTANCES + base_resistance + insulation_mm / 1000 / conductivity)


def _add_insulation(build: Build, library: Library, rng: random.Random, kinds: dict[str, float], thickness_mm: float) -> tuple[str, float]:
    kind = weighted_choice(rng, kinds)
    layer = library.pick(rng, "insulation", kind)
    conductivity, typical_density = INSULATION[layer.kind]
    metres = thickness_mm / 1000
    build.add(layer, metres, (layer.mass or typical_density) * metres)
    return layer.kind, conductivity


def _add_substrate(build: Build, library: Library, rng: random.Random, mass: float) -> None:
    build.add(library.pick(rng, "substrate", weighted_choice(rng, {"brick": 0.6, "concrete": 0.4})), mass, mass)


def insulation_fire(kind: str, rng: random.Random) -> str:
    table = {"eps": {"b": 0.7, "c": 0.3}, "xps": {"b": 0.4, "c": 0.4, "e": 0.2}, "pir": {"b": 0.6, "c": 0.3, "d": 0.1},
             "wood": {"b": 0.3, "c": 0.4, "d": 0.3}, "wool": {"a2": 0.7, "a1": 0.3}}
    return weighted_choice(rng, table[kind])


def sandwich_panel(library: Library, rng: random.Random) -> tuple[Build, dict]:
    layer = library.pick(rng, "sandwich_panel")
    conductivity, _ = INSULATION[layer.kind]
    thickness_mm = layer.thickness_m * 1000 if layer.thickness_m else rng.uniform(80, 200)
    mass = layer.mass or (10.0 + (110 if layer.kind == "wool" else 40) * thickness_mm / 1000)
    build = Build()
    build.add(layer, 1.0, mass)
    return build, {
        "u_value": transmittance(0.0, thickness_mm, conductivity + 0.004),
        "acoustic_reduction": rng.uniform(29, 34) if layer.kind == "wool" else rng.uniform(24, 28),
        "heat_capacity": rng.uniform(5, 15),
        "fire_reaction": insulation_fire(layer.kind, rng),
        "thickness": thickness_mm,
        "cost_product": rng.uniform(45, 90) + 0.1 * thickness_mm,
        "cost_labour": rng.uniform(15, 30),
        "cost_maintenance": rng.uniform(10, 30),
    }


def precast(library: Library, rng: random.Random) -> tuple[Build, dict]:
    layer = library.pick(rng, "precast")
    insulation_mm = rng.uniform(100, 200)
    concrete_mass = rng.uniform(380, 520)
    conductivity, _ = INSULATION.get(layer.kind, INSULATION["eps"])
    build = Build()
    build.add(layer, concrete_mass, concrete_mass)
    return build, {
        "u_value": transmittance(0.1, insulation_mm, conductivity),
        "acoustic_reduction": mass_law_reduction(concrete_mass * 0.6) + rng.uniform(0, 3),
        "heat_capacity": min(0.6 * concrete_mass * 0.6, 250) * rng.uniform(0.9, 1.1),
        "fire_reaction": "a1",
        "thickness": concrete_mass / 2.4 + insulation_mm,
        "cost_product": rng.uniform(90, 160),
        "cost_labour": rng.uniform(20, 40),
        "cost_maintenance": rng.uniform(5, 15),
    }


def etics(library: Library, rng: random.Random) -> tuple[Build, dict]:
    build = Build()
    substrate_mass = rng.uniform(150, 350)
    _add_substrate(build, library, rng, substrate_mass)
    kit = library.pick(rng, "etics_kit")
    if kit.kind != "kit" and kit.thickness_m:
        kind, conductivity, insulation_mm = kit.kind, INSULATION[kit.kind][0], kit.thickness_m * 1000
        build.add(kit, 1.0, kit.mass or 20.0)
    else:
        insulation_mm = rng.uniform(60, 200)
        build.add(kit, 1.0, kit.mass or 15.0)
        kind, conductivity = _add_insulation(build, library, rng, {"eps": 0.6, "wool": 0.3, "wood": 0.1}, insulation_mm)
    return build, {
        "u_value": transmittance(rng.uniform(0.2, 0.6), insulation_mm, conductivity),
        "acoustic_reduction": mass_law_reduction(substrate_mass) + (rng.uniform(-4, 0) if kind == "eps" else rng.uniform(0, 3)),
        "heat_capacity": min(0.6 * substrate_mass, 250) * rng.uniform(0.9, 1.1),
        "fire_reaction": insulation_fire(kind, rng),
        "thickness": substrate_mass / 1.4 + insulation_mm + 10,
        "cost_product": rng.uniform(70, 125),
        "cost_labour": rng.uniform(45, 80),
        "cost_maintenance": rng.uniform(25, 60),
    }


def ventilated(library: Library, rng: random.Random) -> tuple[Build, dict]:
    build = Build()
    substrate_mass = rng.uniform(150, 350)
    _add_substrate(build, library, rng, substrate_mass)
    insulation_mm = rng.uniform(60, 160)
    _, conductivity = _add_insulation(build, library, rng, {"wool": 1.0}, insulation_mm)
    aluminium = rng.random() < 0.6
    build.add_estimate(rng.uniform(3, 6), gwp=rng.uniform(15, 40) if aluminium else rng.uniform(8, 15))
    cladding = library.pick(rng, "cladding")
    fire, typical_mass, _ = CLADDING[cladding.kind]
    build.add(cladding, 1.0, cladding.mass or typical_mass)
    return build, {
        "u_value": transmittance(rng.uniform(0.2, 0.6), insulation_mm, conductivity),
        "acoustic_reduction": mass_law_reduction(substrate_mass) + rng.uniform(1, 5),
        "heat_capacity": min(0.6 * substrate_mass, 250) * rng.uniform(0.9, 1.1),
        "fire_reaction": weighted_choice(rng, fire),
        "thickness": substrate_mass / 1.4 + insulation_mm + rng.uniform(40, 90),
        "cost_product": rng.uniform(90, 230),
        "cost_labour": rng.uniform(50, 100),
        "cost_maintenance": rng.uniform(8, 25),
    }


def masonry(library: Library, rng: random.Random) -> tuple[Build, dict]:
    """Cavity wall: facing brick outside, insulation in the cavity, a light inner leaf."""
    build = Build()
    outer, inner = rng.uniform(170, 230), rng.uniform(60, 90)
    build.add(library.pick(rng, "substrate", "brick"), outer, outer)
    insulation_mm = rng.uniform(40, 120)
    _, conductivity = _add_insulation(build, library, rng, {"wool": 0.5, "eps": 0.5}, insulation_mm)
    build.add(library.pick(rng, "substrate", "brick"), inner, inner)
    renders = library.of("render")
    if renders:
        build.add(rng.choice(renders), 5.0, 5.0)
    else:
        build.add_estimate(5.0)
    return build, {
        "u_value": transmittance(rng.uniform(0.45, 0.75), insulation_mm, conductivity),
        "acoustic_reduction": mass_law_reduction(outer + inner) + rng.uniform(-1, 3),
        "heat_capacity": rng.uniform(40, 100),
        "fire_reaction": weighted_choice(rng, {"a1": 0.85, "a2": 0.15}),
        "thickness": 115 + 70 + 30 + insulation_mm,
        "cost_product": rng.uniform(40, 80),
        "cost_labour": rng.uniform(50, 90),
        "cost_maintenance": rng.uniform(5, 20),
    }


TYPOLOGIES = {
    "sandwich_panel": sandwich_panel,
    "precast": precast,
    "etics": etics,
    "ventilated": ventilated,
    "masonry": masonry,
}


def product(library: Library, rng: random.Random, typology: str, key: str) -> dict:
    build, performance = TYPOLOGIES[typology](library, rng)
    values = {**build.indicators(), **performance, "health": health_status(rng)}
    kept = {
        name: value if isinstance(value, str) else round(value, 4)
        for name, value in values.items()
        if value is not None
    }
    return {"id_prod": key, "typology": typology, "sources": sorted(set(build.sources)), **kept}
