"""Context influence pilot for indoor partitions: plausible products composed from CEC tables and reviewed manufacturer data, labelled under each candidate context."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml
from scipy.stats import spearmanr

from experiments.context_influence import (
    BASELINES,
    Candidate,
    assign_chunks,
    influence,
    pair_discordance,
    scenario_id,
)

SCREENING = Path(__file__).with_name("partition_context_screening.yaml")
BRIEF_TEMPLATE = Path(__file__).with_name("partition_pilot_brief.md")
PILOT_DIR = Path("data/partition/context_pilot")
FIGURES = Path(__file__).with_name("figures")

ACROSS_SHARE = 0.6
LETTERS = "ABCDE"
SURFACE_RESISTANCES = 0.26
PLASTER_FACE_MASS = 17.0
SPECIFIC_HEAT = 1.0
HEAT_DEPTH_MM = 85.0

SHOWN = (
    "gwp", "circ_orig", "fu_recyc", "fu_inert", "fu_haz",
    "cost_product", "cost_labour", "cost_maintenance", "health", "fire_reaction",
    "impact_resistance", "sound_reduction", "u_value", "surface_mass", "heat_capacity", "thickness",
)
IMPACT_LEVELS = ("Category I", "Category II", "Category III", "Category IV")
IMPACT_POSITIONS = dict(zip(IMPACT_LEVELS, (0.0, 1 / 3, 2 / 3, 1.0)))
MASONRY_IMPACT = 3
NARROW_STUD_IMPACT_CAP = 2
HEALTH_LEVELS = (
    "Hazardous substances present", "Undocumented", "No listed substances, uncertified",
    "Material Health Bronze", "Silver", "Gold", "Platinum",
)
HEALTH_POSITIONS = dict(zip(HEALTH_LEVELS, (0.0, 0.4, 0.9, 0.925, 0.95, 0.975, 1.0)))
HEALTH_WEIGHTS = (0.02, 0.33, 0.30, 0.15, 0.10, 0.07, 0.03)
FIRE_POSITIONS = {"Class A2": 0.9, "Class A1": 1.0}


@dataclass(frozen=True)
class Board:
    thickness: float
    mass: float
    gwp: float
    cost: float
    single_layer_impact: int
    sound_bonus: float = 0.0


BOARDS = {
    "standard_12": Board(12.5, 8.7, 1.0, 2.5, 2),
    "standard_15": Board(15.0, 10.8, 1.3, 3.2, 2),
    "fire_12": Board(12.5, 10.3, 1.4, 3.5, 2),
    "fire_15": Board(15.0, 12.5, 1.6, 4.3, 3),
    "moisture_12": Board(12.5, 8.7, 1.3, 3.4, 2),
    "hard_15": Board(15.0, 13.5, 1.7, 6.0, 4),
    "acoustic_12": Board(12.5, 12.4, 1.5, 5.5, 3, sound_bonus=3.0),
}


def framed_impact(board: Board, boards_per_face: int, stud_mm: int) -> int:
    """EAD use category from the UK duty-grade rules: two boards per face meet Severe, a single board sets the grade, 48 mm studs cap it at Medium."""
    grade = 4 if boards_per_face >= 2 else board.single_layer_impact
    return min(grade, NARROW_STUD_IMPACT_CAP) if stud_mm < 70 else grade


def impact_label(grade: int) -> str:
    return IMPACT_LEVELS[grade - 1]


@dataclass(frozen=True)
class Wool:
    conductivity: float
    density: float
    gwp_per_mm: float
    cost_per_mm: float


WOOLS = {
    "glass": Wool(0.035, 18.0, 0.020, 0.06),
    "rock": Wool(0.036, 40.0, 0.026, 0.09),
}


@dataclass(frozen=True)
class FramedSystem:
    key: str
    stud_mm: int
    frames: int
    boards_per_face: int
    board: str
    wool_mm: float
    sound: float
    thickness: float


FRAMED = (
    FramedSystem("t3_48_single", 48, 1, 1, "standard_12", 45, 39.5, 73),
    FramedSystem("t3_48_double_board", 48, 1, 2, "standard_12", 45, 52.5, 98),
    FramedSystem("t3_70_single", 70, 1, 1, "standard_15", 65, 46.9, 100),
    FramedSystem("t3_70_double_board", 70, 1, 2, "standard_12", 65, 53.5, 120),
    FramedSystem("t3_70_double_15", 70, 1, 2, "standard_15", 65, 54.0, 130),
    FramedSystem("t3_90_double_board", 90, 1, 2, "standard_12", 80, 55.0, 140),
    FramedSystem("t3_twin_frame", 48, 2, 2, "standard_12", 90, 57.0, 176),
    FramedSystem("t3_twin_frame_wide", 70, 2, 2, "standard_12", 130, 64.0, 220),
)
FRAME_STEEL_KG = {48: 1.8, 70: 2.2, 90: 2.6}
FRAME_COST = {48: 3.5, 70: 4.0, 90: 4.8}
STEEL_GWP_PER_KG = 2.5
JOINTS_MASS, JOINTS_GWP, JOINTS_COST = 0.7, 0.2, 2.0


@dataclass(frozen=True)
class Leaf:
    """A CEC single-leaf partition plastered on both faces (CEC v05.0 4.4.1.1)."""

    sound: tuple[float, float]
    mass: tuple[float, float]
    resistance: float
    leaf_mm: float
    gwp_per_kg: float
    cost_product: float
    cost_labour: float


LEAVES = {
    "lh70": Leaf((36, 37), (89, 97), 0.21, 70, 0.21, 8.0, 16.0),
    "lgf70": Leaf((33, 34), (70, 80), 0.38, 70, 0.21, 7.0, 12.0),
    "lh110": Leaf((40, 42), (127, 160), 0.28, 110, 0.21, 10.0, 18.0),
    "lp115": Leaf((42, 44), (150, 161), 0.23, 115, 0.21, 12.0, 20.0),
    "lp240": Leaf((49, 50), (284, 313), 0.40, 240, 0.21, 22.0, 30.0),
    "bc140": Leaf((43, 45), (136, 160), 0.37, 140, 0.21, 14.0, 17.0),
    "bh100": Leaf((41, 41), (151, 151), 0.21, 100, 0.12, 11.0, 17.0),
}
PLASTER_MM, PLASTER_GWP_PER_FACE, PLASTER_COST, PLASTER_LABOUR = 15.0, 2.6, 4.0, 9.0


@dataclass(frozen=True)
class Lining:
    """A CEC lining on one face (CEC v05.0 4.4.1.3): TR1 on studs, TR2 bonded."""

    resistance_without_wool: float
    wool_mm: float
    thickness: float
    mass: float
    gwp: float
    cost_product: float
    cost_labour: float
    board_face_mass: float
    impact: int


LININGS = {
    "tr1": Lining(0.21, 50, 73, 13.9, 6.9, 10.0, 10.0, 10.8, 2),
    "tr2": Lining(0.04, 30, 40, 8.0, 2.2, 7.0, 6.0, 7.5, 3),
}
LINING_SOUND_GAIN = {("tr1", "lh70"): 16, ("tr1", "lp115"): 14, ("tr2", "lgf70"): 10}
LINED = (("lh70", "tr1", 1), ("lh70", "tr1", 2), ("lp115", "tr1", 1), ("lgf70", "tr2", 1))


@dataclass(frozen=True)
class DoubleLeaf:
    """Two CEC leaves on elastic bands with 40 mm of wool between them (CEC v05.0 4.4.2)."""

    key: str
    faces: tuple[str, str]
    resistance_without_wool: float
    sound: tuple[float, float]
    mass: tuple[float, float]
    thickness: float


DOUBLES = (
    DoubleLeaf("t2_lh_lh", ("lh70", "lh70"), 0.63, (53, 55), (148, 170), 210),
    DoubleLeaf("t2_lgf_lgf", ("lgf70", "lgf70"), 0.97, (53, 55), (110, 130), 210),
    DoubleLeaf("t2_lp_lh", ("lp115", "lh70"), 0.58, (58, 61), (184, 241), 255),
    DoubleLeaf("t2_bc_lgf", ("bc140", "lgf70"), 0.81, (58, 61), (168, 209), 280),
)
DOUBLE_WOOL_MM, BANDS_GWP, BANDS_COST = 40.0, 0.5, 3.0


def face_heat_capacity(leaf: Leaf, leaf_mass: float) -> float:
    """Plaster plus the leaf within the periodic penetration depth, never past the midplane."""
    reach = min(leaf.leaf_mm / 2, HEAT_DEPTH_MM)
    return (PLASTER_FACE_MASS + leaf_mass * reach / leaf.leaf_mm) * SPECIFIC_HEAT


def _shares(rng: random.Random, family: str) -> dict[str, float]:
    if family == "t3":
        circ, recyc, incin = rng.uniform(8, 25), rng.uniform(20, 45), rng.uniform(2, 6)
    elif family == "t1_lined":
        circ, recyc, incin = rng.uniform(3, 12), rng.uniform(30, 60), rng.uniform(0.5, 2)
    else:
        circ, recyc, incin = rng.uniform(0, 6), rng.uniform(30, 70), rng.uniform(0, 1)
    hazardous = 0.3 if rng.random() < 0.08 else 0.0
    return {
        "circ_orig": round(circ),
        "fu_recyc": round(recyc),
        "fu_inert": round(100 - recyc - incin - hazardous),
        "fu_haz": hazardous,
    }


def _health(rng: random.Random) -> str | None:
    return None if rng.random() < 0.2 else rng.choices(HEALTH_LEVELS, HEALTH_WEIGHTS)[0]


def _noisy(rng: random.Random, value: float, spread: float) -> float:
    return value * rng.uniform(1 - spread, 1 + spread)


def framed_product(rng: random.Random, system: FramedSystem, board_key: str, wool_key: str) -> dict:
    board, wool = BOARDS[board_key], WOOLS[wool_key]
    layers = 2 * system.boards_per_face
    steel = FRAME_STEEL_KG[system.stud_mm] * system.frames
    resistance = SURFACE_RESISTANCES + layers * board.thickness / 1000 / 0.25 + system.wool_mm / 1000 / wool.conductivity
    mass = layers * board.mass + wool.density * system.wool_mm / 1000 + steel + JOINTS_MASS
    gwp = layers * board.gwp + wool.gwp_per_mm * system.wool_mm + steel * STEEL_GWP_PER_KG + JOINTS_GWP
    thickness = system.thickness + layers * (board.thickness - BOARDS[system.board].thickness)
    product = layers * board.cost + FRAME_COST[system.stud_mm] * system.frames + wool.cost_per_mm * system.wool_mm + JOINTS_COST
    labour = 11 + 3 * layers + 9 * (system.frames - 1)
    return {
        "family": "t3",
        "sound_reduction": round(system.sound + board.sound_bonus + rng.uniform(-0.6, 0.6), 1),
        "u_value": round(1 / resistance, 2),
        "surface_mass": round(_noisy(rng, mass, 0.05), 1),
        "heat_capacity": round(system.boards_per_face * board.mass * SPECIFIC_HEAT, 1),
        "thickness": round(thickness),
        "fire_reaction": "Class A2",
        "impact_resistance": impact_label(framed_impact(board, system.boards_per_face, system.stud_mm)),
        "gwp": round(_noisy(rng, gwp, 0.2), 1),
        "cost_product": round(_noisy(rng, product, 0.15), 1),
        "cost_labour": round(_noisy(rng, labour, 0.15), 1),
        "cost_maintenance": round(_noisy(rng, 5 if board_key == "hard_15" else 8, 0.25), 1),
        **_shares(rng, "t3"),
        "health": _health(rng),
    }


def leaf_product(rng: random.Random, key: str) -> dict:
    leaf = LEAVES[key]
    mass = rng.uniform(*leaf.mass)
    leaf_mass = mass - 2 * PLASTER_FACE_MASS
    return {
        "family": "t1",
        "sound_reduction": round(rng.uniform(*leaf.sound), 1),
        "u_value": round(1 / (SURFACE_RESISTANCES + leaf.resistance), 2),
        "surface_mass": round(mass, 1),
        "heat_capacity": round(face_heat_capacity(leaf, leaf_mass), 1),
        "thickness": round(leaf.leaf_mm + 2 * PLASTER_MM),
        "fire_reaction": "Class A1",
        "impact_resistance": impact_label(MASONRY_IMPACT),
        "gwp": round(_noisy(rng, leaf_mass * leaf.gwp_per_kg + 2 * PLASTER_GWP_PER_FACE, 0.15), 1),
        "cost_product": round(_noisy(rng, leaf.cost_product + 2 * PLASTER_COST, 0.15), 1),
        "cost_labour": round(_noisy(rng, leaf.cost_labour + 2 * PLASTER_LABOUR, 0.15), 1),
        "cost_maintenance": round(_noisy(rng, 3.5, 0.3), 1),
        **_shares(rng, "t1"),
        "health": _health(rng),
    }


def lined_product(rng: random.Random, leaf_key: str, lining_key: str, faces: int) -> dict:
    base = leaf_product(rng, leaf_key)
    leaf, lining = LEAVES[leaf_key], LININGS[lining_key]
    wool_resistance = lining.wool_mm / 1000 / WOOLS["glass"].conductivity
    gain = LINING_SOUND_GAIN[(lining_key, leaf_key)]
    plastered_faces = 2 - faces
    leaf_mass = base["surface_mass"] - 2 * PLASTER_FACE_MASS
    face_values = [lining.board_face_mass] * faces + [face_heat_capacity(leaf, leaf_mass)] * plastered_faces
    return {
        **base,
        "family": "t1_lined",
        "sound_reduction": round(base["sound_reduction"] + gain * (1.0 if faces == 1 else 1.5), 1),
        "u_value": round(1 / (SURFACE_RESISTANCES + leaf.resistance + faces * (lining.resistance_without_wool + wool_resistance)), 2),
        "surface_mass": round(base["surface_mass"] + faces * lining.mass, 1),
        "heat_capacity": round(float(np.mean(face_values)), 1),
        "thickness": base["thickness"] + faces * lining.thickness,
        "fire_reaction": "Class A2",
        "impact_resistance": impact_label(min(MASONRY_IMPACT, lining.impact)),
        "gwp": round(base["gwp"] + faces * _noisy(rng, lining.gwp, 0.2), 1),
        "cost_product": round(base["cost_product"] + faces * _noisy(rng, lining.cost_product, 0.15), 1),
        "cost_labour": round(base["cost_labour"] + faces * _noisy(rng, lining.cost_labour, 0.15), 1),
        "cost_maintenance": round(_noisy(rng, 5.5, 0.3), 1),
        **_shares(rng, "t1_lined"),
    }


def double_leaf_product(rng: random.Random, system: DoubleLeaf) -> dict:
    mass = rng.uniform(*system.mass)
    faces = [LEAVES[k] for k in system.faces]
    leaf_mass = mass - 2 * PLASTER_FACE_MASS
    share = [f.leaf_mm / sum(x.leaf_mm for x in faces) for f in faces]
    wool_resistance = DOUBLE_WOOL_MM / 1000 / WOOLS["glass"].conductivity
    gwp = leaf_mass * 0.21 + 2 * PLASTER_GWP_PER_FACE + WOOLS["glass"].gwp_per_mm * DOUBLE_WOOL_MM + BANDS_GWP
    product = sum(f.cost_product for f in faces) + 2 * PLASTER_COST + WOOLS["glass"].cost_per_mm * DOUBLE_WOOL_MM + BANDS_COST
    labour = sum(f.cost_labour for f in faces) + 2 * PLASTER_LABOUR + 4
    return {
        "family": "t2",
        "sound_reduction": round(rng.uniform(*system.sound), 1),
        "u_value": round(1 / (system.resistance_without_wool + wool_resistance), 2),
        "surface_mass": round(mass, 1),
        "heat_capacity": round(float(np.mean([face_heat_capacity(f, leaf_mass * s) for f, s in zip(faces, share)])), 1),
        "thickness": round(system.thickness),
        "fire_reaction": "Class A1",
        "impact_resistance": impact_label(MASONRY_IMPACT),
        "gwp": round(_noisy(rng, gwp, 0.15), 1),
        "cost_product": round(_noisy(rng, product, 0.15), 1),
        "cost_labour": round(_noisy(rng, labour, 0.15), 1),
        "cost_maintenance": round(_noisy(rng, 3.0, 0.3), 1),
        **_shares(rng, "t2"),
        "health": _health(rng),
    }


FRAMED_VARIANTS = (("standard", "glass"), ("fire", "rock"), ("acoustic_12", "glass"))


def _board_for(system: FramedSystem, variant: str) -> str:
    size = system.board.split("_")[1]
    if variant == "standard":
        return system.board
    if variant == "fire":
        return f"fire_{size}"
    return "acoustic_12" if size == "12" else "hard_15"


def compose(seed: int) -> list[dict]:
    """About sixty products spanning the three typologies, each with every indicator the labeller sees."""
    rng = random.Random(seed)
    products = []
    for system in FRAMED:
        for variant, wool in FRAMED_VARIANTS:
            products.append({"source": f"{system.key}:{variant}", **framed_product(rng, system, _board_for(system, variant), wool)})
    for key in LEAVES:
        for copy in range(2):
            products.append({"source": f"t1_{key}:{copy}", **leaf_product(rng, key)})
    for leaf_key, lining_key, faces in LINED:
        for copy in range(2):
            products.append({"source": f"t1_{leaf_key}+{lining_key}x{faces}:{copy}", **lined_product(rng, leaf_key, lining_key, faces)})
    for system in DOUBLES:
        for copy in range(2):
            products.append({"source": f"{system.key}:{copy}", **double_leaf_product(rng, system)})
    for number, product in enumerate(products, start=1):
        product["id_prod"] = f"partition_{number:03d}"
    return products


def typology(product: Mapping) -> str:
    return "t1" if product["family"].startswith("t1") else product["family"]


def sample_shortlists(products: Sequence[dict], count: int, seed: int) -> list[list[dict]]:
    """Shortlists of four or five products, mostly mixing typologies so that every context has something to act on."""
    rng = random.Random(seed)
    by_family: dict[str, list[dict]] = {}
    for product in products:
        by_family.setdefault(typology(product), []).append(product)
    shortlists = []
    n_across = round(count * ACROSS_SHARE)
    for number in range(count):
        size = rng.choice((4, 5))
        if number < n_across:
            families = rng.sample(sorted(by_family), 2) if rng.random() < 0.5 else sorted(by_family)
            chosen: list[dict] = [rng.choice(by_family[f]) for f in families]
            pool = [p for f in families for p in by_family[f] if p not in chosen]
            chosen += rng.sample(pool, size - len(chosen))
        else:
            family = rng.choice([f for f in sorted(by_family) if len(by_family[f]) >= size])
            chosen = rng.sample(by_family[family], size)
        shortlists.append(chosen)
    rng.shuffle(shortlists)
    return shortlists


def screening() -> dict:
    return yaml.safe_load(SCREENING.read_text(encoding="utf-8"))


def candidates(record: Mapping, only: Sequence[str] | None = None) -> list[Candidate]:
    """Candidates sent to the pilot, or the named subset of them."""
    chosen = [
        Candidate(c["key"], c["name"], " ".join(c["description"].split()), None)
        for c in record["candidates"]
        if c["outcome"] == "pilot"
    ]
    if only is None:
        return chosen
    unknown = set(only) - {c.key for c in chosen}
    if unknown:
        raise ValueError(f"not piloted candidates: {sorted(unknown)}")
    return [c for c in chosen if c.key in only]


def brief(record: Mapping, only: Sequence[str] | None = None) -> str:
    baseline = record["baseline"]
    lines = [f"- {baseline['name']}: {' '.join(baseline['description'].split())}"]
    lines += [
        f"- {c.name}: the baseline, except that {c.description[0].lower()}{c.description[1:]}"
        for c in candidates(record, only)
    ]
    return BRIEF_TEMPLATE.read_text(encoding="utf-8").replace("{situations}", "\n".join(lines))


def versions(record: Mapping, only: Sequence[str] | None = None) -> list[str]:
    return [*BASELINES, *(c.key for c in candidates(record, only))]


def situation_name(record: Mapping, version: str) -> str:
    if version in BASELINES:
        return record["baseline"]["name"]
    return next(c.name for c in candidates(record) if c.key == version)


def shown(product: Mapping) -> dict:
    return {key: product.get(key) for key in SHOWN}


def pilot_candidates(directory: Path) -> list[str] | None:
    settings = directory / "pilot.json"
    return json.loads(settings.read_text(encoding="utf-8"))["candidates"] if settings.exists() else None


def write_chunks(directory: Path, count: int, chunks: int, seed: int, only: Sequence[str] | None) -> None:
    record = screening()
    products = compose(seed)
    shortlists = sample_shortlists(products, count, seed)
    keys = versions(record, only)
    placement = assign_chunks(len(shortlists), len(keys), chunks)
    rng = random.Random(seed + 1)

    work: list[list[dict]] = [[] for _ in range(chunks)]
    index: dict[str, dict] = {}
    for (s, v), chunk in placement.items():
        version = keys[v]
        alternatives = list(shortlists[s])
        rng.shuffle(alternatives)
        sid = scenario_id(seed, s, version)
        letters = {letter: a["id_prod"] for letter, a in zip(LETTERS, alternatives)}
        work[chunk].append({
            "scenario_id": sid,
            "applications": [situation_name(record, version)],
            "stakeholders": ["Balanced optimizer"],
            "alternatives": [{"id_prod": letter, **shown(a)} for letter, a in zip(LETTERS, alternatives)],
        })
        index[sid] = {"shortlist": s, "version": version, "letters": letters}

    (directory / "chunks").mkdir(parents=True, exist_ok=True)
    (directory / "out").mkdir(parents=True, exist_ok=True)
    for number, part in enumerate(work, start=1):
        rng.shuffle(part)
        (directory / "chunks" / f"chunk_{number:02d}.json").write_text(
            json.dumps({"scenarios": part}, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    (directory / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (directory / "products.json").write_text(json.dumps(products, indent=1), encoding="utf-8")
    (directory / "shortlists.json").write_text(json.dumps(shortlists), encoding="utf-8")
    (directory / "brief.md").write_text(brief(record, only), encoding="utf-8")
    (directory / "pilot.json").write_text(json.dumps({"seed": seed, "candidates": list(only) if only else None}), encoding="utf-8")
    print(f"{len(products)} products; {len(index)} scenarios ({len(shortlists)} shortlists x {len(keys)} versions) in {chunks} chunks under {directory / 'chunks'}")


def collected(directory: Path) -> dict[tuple[int, str], dict[str, float]]:
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    prefs: dict[tuple[int, str], dict[str, float]] = {}
    for path in sorted((directory / "out").glob("chunk_*.json")):
        for sid, rows in json.loads(path.read_text(encoding="utf-8"))["labels"].items():
            entry = index.get(sid)
            if entry is None:
                continue
            by_letter = {row["id_prod"]: float(row["pref"]) for row in rows}
            if set(by_letter) != set(entry["letters"]):
                continue
            prefs[(entry["shortlist"], entry["version"])] = {
                entry["letters"][letter]: pref for letter, pref in by_letter.items()
            }
    return prefs


def numeric(key: str, value) -> float | None:
    if value is None:
        return None
    if key == "health":
        return HEALTH_POSITIONS[value]
    if key == "fire_reaction":
        return FIRE_POSITIONS[value]
    if key == "impact_resistance":
        return IMPACT_POSITIONS[value]
    return float(value)


def directions(prefs: Mapping[tuple[int, str], Mapping[str, float]], shortlists: Sequence[Sequence[dict]], version: str) -> dict[str, float]:
    """Rank correlation, across products, between an indicator's within-shortlist position and the context's preference shift."""
    pairs: dict[str, list[tuple[float, float]]] = {k: [] for k in SHOWN}
    for s, alternatives in enumerate(shortlists):
        if not all((s, v) in prefs for v in (version, *BASELINES)):
            continue
        for key in SHOWN:
            values = {a["id_prod"]: numeric(key, a.get(key)) for a in alternatives}
            known = [v for v in values.values() if v is not None]
            if len(known) < 2 or max(known) == min(known):
                continue
            low, high = min(known), max(known)
            for product, value in values.items():
                if value is None:
                    continue
                base = np.mean([prefs[(s, b)][product] for b in BASELINES])
                pairs[key].append(((value - low) / (high - low), prefs[(s, version)][product] - base))
    return {
        key: float(spearmanr(*zip(*rows)).statistic)
        for key, rows in pairs.items()
        if len(rows) >= 20 and len({r[1] for r in rows}) > 1
    }


def figure_path(directory: Path) -> Path:
    suffix = "" if directory == PILOT_DIR else "-" + directory.name.removeprefix("context_pilot_")
    return FIGURES / f"partition-context-influence{suffix}.csv"


def analyse(directory: Path) -> None:
    record = screening()
    prefs = collected(directory)
    shortlists = json.loads((directory / "shortlists.json").read_text(encoding="utf-8"))
    rows = []
    for candidate in candidates(record, pilot_candidates(directory)):
        row = {"context": candidate.key, **influence(prefs, candidate.key, BASELINES)}
        moved = sorted(directions(prefs, shortlists, candidate.key).items(), key=lambda kv: -abs(kv[1]))
        row["drivers"] = "; ".join(f"{k} {v:+.2f}" for k, v in moved[:4])
        rows.append(row)
    noise = [pair_discordance(prefs[(s, BASELINES[0])], prefs[(s, BASELINES[1])])
             for s in {s for s, _ in prefs} if all((s, b) in prefs for b in BASELINES)]
    print(f"baseline against itself: discordance {np.mean(noise):.3f} over {len(noise)} shortlists")
    print("| context | n | discordance (95 % CI) | winner changed | mean shift | drivers |")
    print("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -r["discordance"]):
        print(f"| {r['context']} | {r['shortlists']} "
              f"| {r['discordance']:+.3f} ({r['discordance_low']:+.3f} to {r['discordance_high']:+.3f}) "
              f"| {r['winner_changed']:+.2f} | {r['mean_shift']:+.3f} | {r['drivers']} |")
    figure = figure_path(directory)
    figure.parent.mkdir(parents=True, exist_ok=True)
    with figure.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def status(directory: Path) -> None:
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    done = set()
    for path in (directory / "out").glob("chunk_*.json"):
        done |= set(json.loads(path.read_text(encoding="utf-8"))["labels"])
    print(f"{len(done & set(index))} of {len(index)} scenarios labelled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=PILOT_DIR, help="pilot directory under data/partition")
    commands = parser.add_subparsers(dest="command", required=True)
    chunk = commands.add_parser("chunks", help="compose products, sample shortlists, write the work chunks and brief")
    chunk.add_argument("--shortlists", type=int, default=50)
    chunk.add_argument("--chunks", type=int, default=18)
    chunk.add_argument("--seed", type=int, default=0)
    chunk.add_argument("--candidates", help="comma-separated subset of piloted candidates; all when omitted")
    commands.add_parser("status", help="count labelled scenarios")
    commands.add_parser("analyse", help="influence per context, written to experiments/figures")
    args = parser.parse_args()
    if args.command == "chunks":
        only = args.candidates.split(",") if args.candidates else None
        write_chunks(args.dir, args.shortlists, args.chunks, args.seed, only)
    elif args.command == "status":
        status(args.dir)
    else:
        analyse(args.dir)


if __name__ == "__main__":
    main()
