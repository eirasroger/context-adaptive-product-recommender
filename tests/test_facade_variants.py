import json
import random

import pytest

from ingest.facade import synthesise, variants
from ingest.facade.label import of_kind

PRODUCTS = {
    "sandwich_panel": {"gwp": 35.0, "wdp": 11.0, "fwu": 0.2, "circ_orig": 13.0, "fu_recyc": 44.0, "fu_incin": 0.0,
                       "fu_inert": 56.0, "fu_haz": 0.0, "cost_product": 80.0, "cost_labour": 22.0,
                       "cost_maintenance": 20.0, "u_value": 0.27, "acoustic_reduction": 26.0, "surface_mass": 17.0,
                       "heat_capacity": 10.0, "fire_reaction": "b", "thickness": 110.0, "health": "h2"},
    "etics": {"gwp": 50.0, "wdp": 12.0, "fwu": 0.3, "circ_orig": 0.2, "fu_recyc": 66.0, "fu_incin": 0.0,
              "fu_inert": 33.998, "fu_haz": 0.002, "cost_product": 98.0, "cost_labour": 63.0, "cost_maintenance": 43.0,
              "u_value": 0.22, "acoustic_reduction": 48.0, "surface_mass": 267.0, "heat_capacity": 148.0,
              "fire_reaction": "b", "thickness": 324.0, "health": "h1"},
    "ventilated": {"gwp": 83.0, "circ_orig": 1.0, "fu_recyc": 67.0, "fu_incin": 0.0, "fu_inert": 32.99, "fu_haz": 0.01,
                   "cost_product": 160.0, "cost_labour": 74.0, "cost_maintenance": 16.0, "u_value": 0.27,
                   "acoustic_reduction": 52.0, "surface_mass": 280.0, "heat_capacity": 150.0, "fire_reaction": "a2",
                   "thickness": 354.0},
    "precast": {"gwp": 77.0, "fwu": 1.0, "circ_orig": 4.5, "fu_recyc": 70.0, "fu_incin": 0.0, "fu_inert": 29.95,
                "fu_haz": 0.05, "cost_product": 126.0, "cost_labour": 30.0, "cost_maintenance": 10.0, "u_value": 0.19,
                "acoustic_reduction": 52.0, "surface_mass": 452.0, "heat_capacity": 162.0, "fire_reaction": "a1",
                "thickness": 337.0, "health": "h3"},
    "masonry": {"gwp": 54.0, "wdp": 12.0, "fwu": 0.18, "circ_orig": 0.03, "fu_recyc": 74.0, "fu_incin": 0.0,
                "fu_inert": 25.997, "fu_haz": 0.003, "cost_product": 60.0, "cost_labour": 71.0, "cost_maintenance": 12.5,
                "u_value": 0.32, "acoustic_reduction": 51.5, "surface_mass": 283.0, "heat_capacity": 69.0,
                "fire_reaction": "a1", "thickness": 294.0, "health": "h2"},
}


def _product(typology: str, number: int = 0) -> dict:
    return {"id_prod": f"{typology}_{number:04d}", "typology": typology, "sources": [], **PRODUCTS[typology]}


def _variants(typology: str, levers: tuple[str, ...] | None = None, n: int = 300) -> list[tuple[dict, dict]]:
    rng = random.Random(0)
    parent = _product(typology)
    return [(parent, variants.variant(parent, rng, f"v{i}", levers)) for i in range(n)]


@pytest.mark.parametrize("typology", ["etics", "ventilated", "precast", "masonry"])
def test_sound_reduction_and_thermal_mass_follow_a_changed_substrate_mass(typology):
    for parent, variant in _variants(typology, ("mass",)):
        heavier = variant["surface_mass"] > parent["surface_mass"]
        assert (variant["acoustic_reduction"] > parent["acoustic_reduction"]) == heavier
        assert variant["heat_capacity"] >= parent["heat_capacity"] or not heavier


@pytest.mark.parametrize("typology", list(PRODUCTS))
def test_a_lower_u_value_costs_thickness(typology):
    for parent, variant in _variants(typology, ("insulation",)):
        if variant["u_value"] < parent["u_value"]:
            assert variant["thickness"] > parent["thickness"]
        elif variant["u_value"] > parent["u_value"]:
            assert variant["thickness"] < parent["thickness"]


def test_a_sandwich_core_sets_both_fire_class_and_sound_reduction():
    for _, variant in _variants("sandwich_panel", ("insulant",)):
        mineral_wool = variant["fire_reaction"] in ("a1", "a2")
        assert (variant["acoustic_reduction"] >= 29) == mineral_wool


@pytest.mark.parametrize("typology", ["precast", "masonry"])
def test_a_heavy_mineral_wall_stays_non_combustible(typology):
    for _, variant in _variants(typology):
        assert variant["fire_reaction"] in ("a1", "a2")


@pytest.mark.parametrize("typology", list(PRODUCTS))
def test_end_of_life_shares_never_exceed_the_whole(typology):
    for _, variant in _variants(typology):
        shares = [variant.get(k) or 0.0 for k in ("fu_recyc", "fu_incin", "fu_inert", "fu_haz")]
        assert sum(shares) <= 100.0 + 1e-6


def test_a_lever_never_fills_a_missing_value():
    for _, variant in _variants("ventilated", n=500):
        assert "health" not in variant and "wdp" not in variant and "fwu" not in variant


@pytest.mark.parametrize("typology", list(PRODUCTS))
def test_a_variant_carries_only_indicators_and_its_record_of_origin(typology):
    for parent, variant in _variants(typology):
        assert set(variant) == set(parent) | {"derived_from", "levers"}
        assert variant["derived_from"] == parent["id_prod"]
        assert variant["levers"] and set(variant["levers"]) <= set(variants.APPLICABLE[typology])


@pytest.mark.parametrize("typology", list(PRODUCTS))
def test_every_value_stays_within_the_plausible_limits(typology):
    for _, variant in _variants(typology):
        for key, (low, high) in variants.PLAUSIBLE.items():
            if variant.get(key) is not None:
                assert low <= variant[key] <= high


def test_unusual_sets_are_appended_after_the_existing_ones_and_each_holds_a_variant(registry, tmp_path):
    products = [_product(typology, i) for typology in PRODUCTS for i in range(6)]
    existing = [{"id": "synthetic_000001", "shortlist": "across", "contexts": ["hot_summer"],
                 "stakeholders": ["balanced_optimizer"], "alternatives": products}]
    (tmp_path / synthesise.SCENARIOS_FILE).write_text(json.dumps(existing), encoding="utf-8")
    (tmp_path / synthesise.LABELS_FILE).write_text(json.dumps([{"id": "synthetic_000001"}]), encoding="utf-8")

    synthesise.extend_unusual(registry, tmp_path, sets=60, seed=0)

    scenarios = json.loads((tmp_path / synthesise.SCENARIOS_FILE).read_text(encoding="utf-8"))
    labels = json.loads((tmp_path / synthesise.LABELS_FILE).read_text(encoding="utf-8"))
    assert scenarios[0] == existing[0]
    assert [s["id"] for s in scenarios] == [entry["id"] for entry in labels]
    variant_ids = []
    for scenario in scenarios[1:]:
        assert scenario["shortlist"].startswith("unusual:")
        found = [a["id_prod"] for a in scenario["alternatives"] if "derived_from" in a]
        assert 1 <= len(found) <= 2
        variant_ids += found
    assert len(variant_ids) == len(set(variant_ids))


def test_chunks_of_a_kind_take_its_narrower_kinds_and_nothing_else():
    scenarios = [{"shortlist": k} for k in ("unusual:within:etics", "unusual:across", "within:etics", "across")]
    assert [s["shortlist"] for s in of_kind(scenarios, "unusual")] == ["unusual:within:etics", "unusual:across"]
    assert [s["shortlist"] for s in of_kind(scenarios, "within")] == ["within:etics"]
