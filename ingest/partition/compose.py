"""Indoor partitions composed layer by layer from real EPDs; physics and published ranges supply the rest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass, replace
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

CURATED = Path("data/partition/epd/curated.csv")
PRODUCTS = Path("data/partition/products.json")
PRODUCTS_PER_TYPOLOGY = 2_500

INTERIOR_SURFACE_RESISTANCES = 0.26
UNFILLED_CAVITY_RESISTANCE = 0.17
BOARD_CONDUCTIVITY = 0.25
SPECIFIC_HEAT = 1.0
HEAT_DEPTH_MM = 85.0
REFERENCE_SERVICE_LIFE_DECADES = 5

#: Germany and the UK price the same wall at up to twice Spain, mostly through labour (docs/PARTITIONS.md 12).
MARKET_FACTOR = (1.0, 2.0)

#: CYPE's ten-year upkeep runs about 2 % of the price on bare masonry and 5 % on framed walls; one range for all
#: typologies keeps the cost mix from naming the typology (the facade lesson, docs/FINDINGS.md B9).
MAINTENANCE_SHARE_PER_DECADE = (0.02, 0.07)

#: Loose bounds that only rule out impossible walls; everything inside them is kept for the model to see.
PLAUSIBLE = {
    "gwp": (-20.0, 250.0),
    "wdp": (-10.0, 150.0),
    "fwu": (0.0, 8.0),
    "acoustic_reduction_pink": (25.0, 80.0),
    "u_value": (0.1, 4.0),
    "surface_mass": (10.0, 600.0),
    "heat_capacity": (3.0, 250.0),
    "thickness": (40.0, 450.0),
    "cost_product": (2.0, 400.0),
    "cost_labour": (2.0, 250.0),
    "cost_maintenance": (0.0, 150.0),
}


@dataclass(frozen=True)
class Board:
    epd_layer: str
    mass_12: tuple[float, float]
    mass_15: tuple[float, float]
    single_layer_grade_12: int
    single_layer_grade_15: int
    sound_bonus: tuple[float, float]
    cost: tuple[float, float]
    fire: str = "a2"


#: Grades follow British Gypsum's White Book and Placo's impact energies (docs/PARTITIONS.md 9.5): board choice
#: sets the grade, stud width only the height.
BOARDS = {
    "standard": Board("plasterboard", (8.0, 10.5), (10.0, 12.5), 2, 2, (0.0, 0.0), (2.5, 5.5)),
    "fire": Board("plasterboard", (9.5, 11.0), (11.5, 13.5), 2, 3, (0.0, 1.0), (3.5, 7.0)),
    "moisture": Board("plasterboard", (8.0, 10.5), (10.0, 12.5), 2, 2, (0.0, 0.0), (3.5, 14.0)),
    "acoustic": Board("plasterboard", (11.0, 13.5), (13.0, 15.0), 3, 3, (2.0, 3.5), (6.0, 17.0)),
    "impact": Board("plasterboard", (11.5, 13.5), (13.5, 15.5), 4, 4, (1.0, 2.0), (7.0, 18.0)),
    "fibre": Board("gypsum_fibreboard", (14.0, 16.0), (17.0, 19.0), 4, 4, (1.0, 2.0), (8.0, 15.0)),
}
BOARD_WEIGHTS = {"standard": 0.42, "fire": 0.18, "moisture": 0.12, "acoustic": 0.12, "impact": 0.08, "fibre": 0.08}
STUD_STEEL_KG = {50: (1.4, 2.2), 75: (1.8, 2.7), 100: (2.2, 3.2)}


@dataclass(frozen=True)
class Unit:
    epd_layer: str
    thickness: tuple[float, float]
    density: tuple[float, float]
    conductivity: tuple[float, float]
    bed_kg: tuple[float, float]
    sound_bonus: tuple[float, float]
    cost: tuple[float, float]
    labour: tuple[float, float]
    grade: int = 3
    plastered: bool = True


UNITS = {
    "clay_hollow": Unit("clay_unit", (50, 115), (600, 950), (0.30, 0.50), (12, 30), (0, 1), (6, 13), (10, 16)),
    "clay_perforated": Unit("clay_unit", (115, 240), (750, 1200), (0.25, 0.45), (20, 40), (0, 1), (10, 22), (12, 22)),
    "aerated_concrete": Unit("aerated_concrete", (75, 150), (350, 650), (0.09, 0.17), (2, 5), (0, 2), (14, 28), (8, 14)),
    "calcium_silicate": Unit("calcium_silicate", (70, 175), (1400, 2000), (0.7, 1.1), (3, 8), (0, 1), (14, 30), (10, 18)),
    "concrete_block": Unit("concrete_block", (70, 200), (900, 1500), (0.5, 1.0), (15, 35), (-1, 1), (9, 22), (12, 22)),
    "gypsum_block": Unit("gypsum_block", (60, 100), (800, 1100), (0.30, 0.45), (2, 4), (0, 2), (12, 20), (8, 14),
                         plastered=False),
}
UNIT_WEIGHTS = {"clay_hollow": 0.24, "clay_perforated": 0.16, "aerated_concrete": 0.17, "calcium_silicate": 0.14,
                "concrete_block": 0.14, "gypsum_block": 0.15}
#: Leaves thin enough to pair on elastic bands.
DOUBLE_LEAF_UNITS = {"clay_hollow": 0.35, "aerated_concrete": 0.2, "gypsum_block": 0.2, "calcium_silicate": 0.15,
                     "concrete_block": 0.1}

TYPOLOGIES = ("framed", "masonry", "lined_masonry", "double_leaf")


class Library:
    """Real EPD values per kg, by layer type, from the curated partition search."""

    def __init__(self, layers: dict[str, list[Layer]]):
        self.layers = layers

    @classmethod
    def load(cls, path: Path = CURATED) -> "Library":
        if not path.exists():
            raise FileNotFoundError(f"no EPD table at {path}; run python -m ingest.partition.epd_search first")
        layers: dict[str, list[Layer]] = {}
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (layer := per_kg_layer(row)) is not None:
                    layers.setdefault(layer.role, []).append(layer)
        return cls(layers)

    def pick(self, rng: random.Random, name: str) -> Layer:
        """One real EPD; a flow it leaves undeclared comes from another EPD of the same layer."""
        options = self.layers.get(name)
        if not options:
            raise LookupError(f"the EPD table holds no {name} layer")
        chosen = rng.choice(options)
        values = dict(chosen.values)
        for key, value in values.items():
            if value is None:
                donors = [o.values[key] for o in options if o.values[key] is not None]
                values[key] = rng.choice(donors) if donors else None
        return replace(chosen, values=values)

    def summary(self) -> dict[str, int]:
        return {name: len(layers) for name, layers in sorted(self.layers.items())}


def per_kg_layer(row: dict) -> Layer | None:
    unit, amount = row["declared_unit"], number(row["declared_amount"]) or 1.0
    mass = number(row["mass_kg"]) or (amount if unit == "kg" else None)
    if not mass or number(row["gwp_total_a1a3"]) is None:
        return None
    values = per_unit(row, mass)
    for key, (low, high) in LIMITS_PER_KG.items():
        if values[key] is not None and not low <= values[key] <= high:
            values[key] = None
    if values["gwp"] is None:
        return None
    return Layer(row["typology"], row["typology"], row["uuid"], row["name"], values, mass=1.0)


def _between(rng: random.Random, bounds: tuple[float, float]) -> float:
    return rng.uniform(*bounds)


def _add(build: Build, library: Library, rng: random.Random, layer_name: str, kg: float) -> None:
    build.add(library.pick(rng, layer_name), kg, kg)


@dataclass
class Face:
    """What one face of the wall contributes beyond its core."""

    thickness: float = 0.0
    resistance: float = 0.0
    heat_mass: float = 0.0
    sound_gain: float = 0.0
    grade: int | None = None
    product: float = 0.0
    labour: float = 0.0
    fire: str | None = None
    reaches_core: bool = True


def board_face(build: Build, library: Library, rng: random.Random, boards: int, kinds: list[str]) -> Face:
    """Boards screwed to a frame on one face, outermost last, with joint compound on the face."""
    face = Face()
    for kind in kinds[:boards]:
        board = BOARDS[kind]
        thick = 15.0 if rng.random() < 0.35 else 12.5
        kg = _between(rng, board.mass_15 if thick == 15.0 else board.mass_12)
        _add(build, library, rng, board.epd_layer, kg)
        face.thickness += thick
        face.resistance += thick / 1000 / BOARD_CONDUCTIVITY
        face.heat_mass += kg
        face.sound_gain += _between(rng, board.sound_bonus)
        face.product += _between(rng, board.cost)
        face.fire = board.fire
    outer = BOARDS[kinds[boards - 1]]
    face.grade = 4 if boards >= 2 else (outer.single_layer_grade_15 if face.thickness >= 15 else outer.single_layer_grade_12)
    _add(build, library, rng, "joint_compound", rng.uniform(0.4, 1.2))
    face.product += rng.uniform(0.6, 1.5)
    return face


def plaster_face(build: Build, library: Library, rng: random.Random) -> Face:
    thick = rng.uniform(8, 20)
    kg = thick * rng.uniform(1.0, 1.7)
    _add(build, library, rng, "gypsum_plaster", kg)
    return Face(thickness=thick, resistance=thick / 1000 / 0.5, heat_mass=kg, grade=3,
                product=rng.uniform(1.0, 4.0), labour=rng.uniform(7.0, 15.0))


def skim_face(build: Build, library: Library, rng: random.Random) -> Face:
    kg = rng.uniform(0.5, 1.5)
    _add(build, library, rng, "joint_compound", kg)
    return Face(thickness=2.0, heat_mass=kg, grade=3, product=rng.uniform(0.5, 1.5), labour=rng.uniform(2.0, 5.0))


def lining_face(build: Build, library: Library, rng: random.Random) -> Face:
    """A plasterboard lining on one face of masonry: bonded with adhesive dabs, or on its own studs with wool."""
    kind = weighted_choice(rng, BOARD_WEIGHTS)
    if rng.random() < 0.45:
        face = board_face(build, library, rng, 1, [kind])
        _add(build, library, rng, "gypsum_plaster", rng.uniform(2.5, 5.0))
        gap = rng.uniform(8, 15)
        face.thickness += gap
        face.sound_gain = rng.uniform(-2.0, 3.0)
        face.grade = max(face.grade, 3)
        face.labour = rng.uniform(7.0, 11.0)
    else:
        face = board_face(build, library, rng, 1 if rng.random() < 0.8 else 2, [kind, kind])
        stud = rng.choice((40, 50, 75))
        _add(build, library, rng, "steel_profile", rng.uniform(0.9, 1.8))
        wool_mm = 0.0 if rng.random() < 0.15 else stud * rng.uniform(0.6, 1.0)
        if wool_mm:
            _add(build, library, rng, "mineral_wool", wool_mm / 1000 * rng.uniform(12, 50))
        face.thickness += stud + rng.uniform(0, 10)
        face.reaches_core = False
        face.resistance += wool_mm / 1000 / rng.uniform(0.032, 0.040) if wool_mm else UNFILLED_CAVITY_RESISTANCE
        face.sound_gain = rng.uniform(8, 16) if wool_mm else rng.uniform(3, 8)
        face.product += rng.uniform(4.0, 8.0) + 0.06 * wool_mm
        face.labour = rng.uniform(8.0, 13.0)
    return face


@dataclass
class Core:
    thickness: float
    resistance: float
    heat_mass_per_face: float
    sound: float
    grade: int
    product: float
    labour: float
    fire: str


def masonry_core(build: Build, library: Library, rng: random.Random, unit_key: str,
                 thickness: tuple[float, float] | None = None) -> Core:
    unit = UNITS[unit_key]
    thick = _between(rng, thickness or unit.thickness)
    kg = thick / 1000 * _between(rng, unit.density)
    _add(build, library, rng, unit.epd_layer, kg)
    bed_layer = "gypsum_plaster" if unit_key == "gypsum_block" else "mortar"
    _add(build, library, rng, bed_layer, _between(rng, unit.bed_kg))
    reach = min(thick / 2, HEAT_DEPTH_MM)
    return Core(
        thickness=thick,
        resistance=thick / 1000 / _between(rng, unit.conductivity),
        heat_mass_per_face=kg * reach / thick,
        sound=_between(rng, unit.sound_bonus),
        grade=unit.grade if thick >= 70 else 2,
        product=_between(rng, unit.cost),
        labour=_between(rng, unit.labour),
        fire="a1",
    )


def _faced(build: Build, core: Core, faces: list[Face], sound: float, typology: str) -> dict:
    resistance = INTERIOR_SURFACE_RESISTANCES + core.resistance + sum(f.resistance for f in faces)
    heat = statistics.mean(f.heat_mass + (core.heat_mass_per_face if f.reaches_core else 0.0) for f in faces) * SPECIFIC_HEAT
    fires = [f.fire for f in faces if f.fire] + [core.fire]
    return {
        "typology": typology,
        "acoustic_reduction_pink": sound,
        "u_value": 1 / resistance,
        "heat_capacity": heat,
        "thickness": core.thickness + sum(f.thickness for f in faces),
        "fire_reaction": "a2" if "a2" in fires else "a1",
        "impact_resistance": min(f.grade if f.grade is not None else core.grade for f in faces),
        "cost_product": core.product + sum(f.product for f in faces),
        "cost_labour": core.labour + sum(f.labour for f in faces),
    }


def framed(build: Build, library: Library, rng: random.Random) -> dict:
    stud = weighted_choice(rng, {50: 0.35, 75: 0.45, 100: 0.2})
    frames = 2 if rng.random() < 0.15 else 1
    boards = weighted_choice(rng, {1: 0.5, 2: 0.42, 3: 0.08})
    kind = weighted_choice(rng, BOARD_WEIGHTS)
    inner = "standard" if boards > 1 and rng.random() < 0.25 else kind
    kinds = [inner] * (boards - 1) + [kind]
    faces = [board_face(build, library, rng, boards, kinds) for _ in range(2)]

    for _ in range(frames):
        _add(build, library, rng, "steel_profile", _between(rng, STUD_STEEL_KG[stud]))
    fill = 0.0 if rng.random() < 0.12 else rng.uniform(0.5, 1.0)
    cavity = stud * frames + (rng.uniform(5, 40) if frames == 2 else 0.0)
    wool_mm = fill * stud * frames
    if wool_mm:
        _add(build, library, rng, "mineral_wool", wool_mm / 1000 * rng.uniform(12, 60))
    wool_resistance = wool_mm / 1000 / rng.uniform(0.032, 0.040)
    air = UNFILLED_CAVITY_RESISTANCE * frames if fill < 0.95 else 0.0

    weighted_rw = (35 + 0.05 * (stud - 50) + (4 + 5 * fill if fill else 0)
                   + sum(rng.uniform(5, 8) for _ in range(boards - 1))
                   + (rng.uniform(9, 15) if frames == 2 else 0)
                   + sum(f.sound_gain for f in faces) / 2 + rng.uniform(-1.5, 1.5))
    spectrum = rng.uniform(-6, -2) if frames == 2 else rng.uniform(-5, -1)
    core = Core(thickness=cavity, resistance=wool_resistance + air, heat_mass_per_face=0.0, sound=0.0, grade=4,
                product=frames * rng.uniform(3.0, 6.0) + 0.06 * wool_mm,
                labour=rng.uniform(10.0, 16.0) + 2.5 * (boards - 1) * 2 + rng.uniform(6, 10) * (frames - 1),
                fire="a2")
    return _faced(build, core, faces, weighted_rw + spectrum, "framed")


def masonry(build: Build, library: Library, rng: random.Random) -> dict:
    unit_key = weighted_choice(rng, UNIT_WEIGHTS)
    core = masonry_core(build, library, rng, unit_key)
    finish = plaster_face if UNITS[unit_key].plastered else skim_face
    faces = [finish(build, library, rng) for _ in range(2)]
    sound = mass_law_reduction(build.mass) + core.sound + rng.uniform(-2, 2)
    return _faced(build, core, faces, sound, "masonry")


def lined_masonry(build: Build, library: Library, rng: random.Random) -> dict:
    unit_key = weighted_choice(rng, UNIT_WEIGHTS)
    core = masonry_core(build, library, rng, unit_key)
    lined = 2 if rng.random() < 0.4 else 1
    finish = plaster_face if UNITS[unit_key].plastered else skim_face
    faces = [lining_face(build, library, rng) for _ in range(lined)] + [finish(build, library, rng) for _ in range(2 - lined)]
    gains = sorted((f.sound_gain for f in faces), reverse=True)
    sound = mass_law_reduction(build.mass) + core.sound + gains[0] + 0.5 * gains[1] + rng.uniform(-2, 2)
    return _faced(build, core, faces, sound, "lined_masonry")


def double_leaf(build: Build, library: Library, rng: random.Random) -> dict:
    leaves = [masonry_core(build, library, rng, weighted_choice(rng, DOUBLE_LEAF_UNITS), (50, 115)) for _ in range(2)]
    wool_mm = rng.uniform(30, 60)
    _add(build, library, rng, "mineral_wool", wool_mm / 1000 * rng.uniform(12, 60))
    build.add_estimate(rng.uniform(0.3, 0.8), gwp=rng.uniform(1.0, 3.0))
    faces = [plaster_face(build, library, rng) for _ in range(2)]
    core = Core(
        thickness=sum(l.thickness for l in leaves) + wool_mm + rng.uniform(0, 10),
        resistance=sum(l.resistance for l in leaves) + wool_mm / 1000 / rng.uniform(0.032, 0.040),
        heat_mass_per_face=statistics.mean(l.heat_mass_per_face for l in leaves),
        sound=0.0,
        grade=min(l.grade for l in leaves),
        product=sum(l.product for l in leaves) + 0.06 * wool_mm + rng.uniform(2, 5),
        labour=sum(l.labour for l in leaves) + rng.uniform(3, 6),
        fire="a1",
    )
    sound = mass_law_reduction(build.mass) + rng.uniform(7, 14)
    return _faced(build, core, faces, sound, "double_leaf")


BUILDERS = {"framed": framed, "masonry": masonry, "lined_masonry": lined_masonry, "double_leaf": double_leaf}


def priced(values: dict, rng: random.Random) -> dict:
    """Spread costs over European markets: labour scales with the market, materials less so; no country is kept."""
    market = _between(rng, MARKET_FACTOR)
    direct = values["cost_product"] + values["cost_labour"]
    product = values["cost_product"] * math.sqrt(market) * rng.uniform(0.7, 1.4)
    labour = values["cost_labour"] * market * rng.uniform(0.8, 1.25)
    maintenance = direct * _between(rng, MAINTENANCE_SHARE_PER_DECADE) * REFERENCE_SERVICE_LIFE_DECADES * market ** 0.75
    return {**values, "cost_product": product, "cost_labour": labour, "cost_maintenance": maintenance}


IMPACT_LEVELS = {1: "category_i", 2: "category_ii", 3: "category_iii", 4: "category_iv"}


def plausible(values: dict) -> bool:
    return all(values.get(key) is None or low <= values[key] <= high for key, (low, high) in PLAUSIBLE.items())


def product(library: Library, rng: random.Random, typology: str, key: str, attempts: int = 20) -> dict:
    """One partition; a draw outside the loose physical bounds is drawn again."""
    for _ in range(attempts):
        build = Build()
        values = priced(BUILDERS[typology](build, library, rng), rng)
        values = {**build.indicators(), **values}
        if plausible(values):
            break
    grade = values.pop("impact_resistance")
    if rng.random() < 0.1:
        grade = max(1, grade - 1)
    values["impact_resistance"] = IMPACT_LEVELS[grade]
    values["health"] = health_status(rng)
    kept = {name: v if isinstance(v, str) else round(v, 4) for name, v in values.items() if v is not None}
    return {"id_prod": key, "sources": sorted(set(build.sources)), **kept}


def compose_all(library: Library, per_typology: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [
        product(library, rng, typology, f"partition_{typology}_{index:04d}")
        for typology in TYPOLOGIES
        for index in range(per_typology)
    ]


def describe(products: list[dict]) -> str:
    keys = ["gwp", "acoustic_reduction_pink", "surface_mass", "u_value", "thickness", "heat_capacity",
            "cost_product", "cost_labour", "cost_maintenance", "fu_recyc", "circ_orig"]
    lines = []
    for typology in TYPOLOGIES:
        group = [p for p in products if p["typology"] == typology]
        cells = []
        for key in keys:
            values = sorted(p[key] for p in group if key in p)
            if values:
                cells.append(f"{key} {values[len(values) // 10]:.1f}-{values[len(values) * 9 // 10]:.1f}")
        grades = {g: sum(p["impact_resistance"] == g for p in group) for g in IMPACT_LEVELS.values()}
        lines.append(f"{typology} ({len(group)}): " + "; ".join(cells) + f"; impact {grades}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-typology", type=int, default=PRODUCTS_PER_TYPOLOGY)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    library = Library.load()
    print(f"EPD layers: {library.summary()}")
    products = compose_all(library, args.per_typology, args.seed)
    PRODUCTS.parent.mkdir(parents=True, exist_ok=True)
    PRODUCTS.write_text(json.dumps(products), encoding="utf-8")
    print(f"wrote {len(products)} products to {PRODUCTS}")
    print("10th to 90th percentiles:")
    print(describe(products))


if __name__ == "__main__":
    main()
