from __future__ import annotations

from ingest.facade import compose
from ingest.epd import layers
from ingest.epd.layers import FLOWS, Layer
from ingest.facade.compose import in_metres, without_unit_errors


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


def test_a_share_of_products_has_an_unknown_health_status():
    import random

    rng = random.Random(0)
    statuses = [layers.health_status(rng) for _ in range(5000)]
    missing = statuses.count(None) / len(statuses)
    assert abs(missing - layers.HEALTH_MISSING_SHARE) < 0.02
    assert {s for s in statuses if s is not None} <= set(layers.HEALTH_WEIGHTS)


def test_stored_biogenic_carbon_survives_the_limits():
    kept, = without_unit_errors([_layer(10.0, gwp=-15.0)])
    assert kept.values["gwp"] == -15.0
