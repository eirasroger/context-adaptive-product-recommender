from __future__ import annotations

from ingest.facade import compose
from ingest.facade.compose import FLOWS, Layer, in_metres, without_unit_errors


def _layer(mass: float, **values: float) -> Layer:
    return Layer("substrate", "brick", "uuid", "name", {**dict.fromkeys(FLOWS), **values}, mass)


def test_a_thickness_entered_in_millimetres_is_read_as_millimetres():
    assert in_metres(0.12) == 0.12
    assert in_metres(120.0) == 0.12
    assert in_metres(0.001) is None


def test_a_mass_flow_larger_than_the_product_is_set_aside():
    kept, = without_unit_errors([_layer(10.0, recovered=12.0, sm=4.0)])
    assert kept.values["recovered"] is None
    assert kept.values["sm"] == 4.0
    assert compose.set_aside == [("name", "recovered")]


def test_stored_biogenic_carbon_survives_the_limits():
    kept, = without_unit_errors([_layer(10.0, gwp=-15.0)])
    assert kept.values["gwp"] == -15.0
