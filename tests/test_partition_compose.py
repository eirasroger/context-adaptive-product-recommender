import random

import pytest

from ingest.facade.compose import FLOWS, Build, Layer
from ingest.partition import compose

LAYERS = ("plasterboard", "gypsum_fibreboard", "steel_profile", "mineral_wool", "joint_compound", "gypsum_plaster",
          "mortar", "clay_unit", "aerated_concrete", "calcium_silicate", "concrete_block", "gypsum_block")


def _library() -> compose.Library:
    flows = {"gwp": 0.3, "wdp": 0.05, "fw": 0.002, "sm": 0.1, "recovered": 0.5, "mer": 0.01, "hwd": 0.0}
    return compose.Library({
        name: [Layer(name, name, f"{name}-uuid", name, {**dict.fromkeys(FLOWS), **flows}, 1.0)]
        for name in LAYERS
    })


@pytest.fixture(scope="module")
def products() -> list[dict]:
    return compose.compose_all(_library(), 150, seed=3)


def test_every_typology_is_composed_inside_the_loose_bounds(products):
    assert {p["typology"] for p in products} == set(compose.TYPOLOGIES)
    assert sum(compose.plausible(p) for p in products) >= 0.97 * len(products)


def test_composition_is_repeatable_from_its_seed():
    assert compose.compose_all(_library(), 5, seed=9) == compose.compose_all(_library(), 5, seed=9)


def test_products_carry_no_market_or_country(products):
    assert not any({"market", "country", "market_factor"} & set(p) for p in products)


def test_two_boards_per_face_grade_severe_whatever_the_stud():
    face = compose.board_face(Build(), _library(), random.Random(0), 2, ["standard", "standard"])
    assert face.grade == 4


def test_one_standard_board_grades_medium_and_one_impact_board_severe():
    rng = random.Random(0)
    assert compose.board_face(Build(), _library(), rng, 1, ["standard"]).grade == 2
    assert compose.board_face(Build(), _library(), rng, 1, ["impact"]).grade == 4


def test_a_lining_on_its_own_studs_hides_the_masonry_heat_capacity():
    rng = random.Random(0)
    linings = [compose.lining_face(Build(), _library(), rng) for _ in range(40)]
    assert any(not f.reaches_core for f in linings) and any(f.reaches_core for f in linings)
    assert all(f.reaches_core == (f.thickness < 40) for f in linings)


def test_an_undeclared_flow_is_borrowed_from_another_epd_of_the_same_layer():
    declared = Layer("mortar", "mortar", "a", "a", {**dict.fromkeys(FLOWS), "gwp": 0.2, "wdp": 0.07}, 1.0)
    missing = Layer("mortar", "mortar", "b", "b", {**dict.fromkeys(FLOWS), "gwp": 0.3}, 1.0)
    library = compose.Library({"mortar": [declared, missing]})
    assert all(library.pick(random.Random(seed), "mortar").values["wdp"] == 0.07 for seed in range(10))
