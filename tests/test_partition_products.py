from pathlib import Path

import yaml

from ingest.partition.epd_search import curated, layer_of

REFERENCE = Path(__file__).resolve().parents[1] / "ingest" / "partition" / "reference_systems.yaml"
TYPOLOGIES = {"t1", "t2", "t3", "gypsum_block"}
KINDS = {"lab", "calculated", "in_situ", "secondary"}


def _values(entry, key):
    value = entry.get(key)
    if value is None:
        return []
    return [v for v in (value if isinstance(value, list) else [value]) if v is not None]


def test_every_reference_system_names_its_market_typology_kind_and_source():
    for entry in yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))["systems"]:
        assert isinstance(entry["market"], str) and len(entry["market"]) == 2, entry["id"]
        assert entry["typology"] in TYPOLOGIES, entry["id"]
        assert entry["kind"] in KINDS, entry["id"]
        assert entry["source"].startswith("https://"), entry["id"]


def test_reference_sound_and_mass_values_are_physically_plausible():
    for entry in yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))["systems"]:
        assert all(25 <= v <= 85 for v in _values(entry, "rw") + _values(entry, "ra")), entry["id"]
        assert all(-15 <= v <= 1 for v in _values(entry, "c") + _values(entry, "ctr")), entry["id"]
        assert all(10 <= v <= 400 for v in _values(entry, "mass_kg_m2")), entry["id"]


def test_every_reference_cost_names_a_known_source_and_splits_within_its_total():
    reference = yaml.safe_load((REFERENCE.parent / "reference_costs.yaml").read_text(encoding="utf-8"))
    for entry in reference["costs"]:
        assert isinstance(entry["market"], str) and len(entry["market"]) == 2, entry
        assert entry["source"] in reference["sources"], entry
        assert entry["kind"] in {"itemised", "guide"}, entry
        if entry["kind"] == "itemised":
            assert entry["product"] + entry["labour"] <= entry["total"] * 1.001, entry


def test_layer_names_in_several_languages_find_their_layer():
    assert layer_of("Rigips Metallprofile CW") == "steel_profile"
    assert layer_of("Lana di roccia per pareti") == "mineral_wool"
    assert layer_of("Carreau de plâtre Placo") == "gypsum_block"
    assert layer_of("KS-ISO Kalksandstein") == "calcium_silicate"
    assert layer_of("Floor screed") is None


def test_curation_drops_products_a_partition_cannot_use():
    assert not curated("calcium_silicate", "StoPrim Silikat")
    assert not curated("steel_profile", "Hot rolled carbon steel profiles")
    assert not curated("partition_system", "T50/T50 LOFT all glass partition wall system")
    assert curated("steel_profile", "Gypframe® Metal Profiles")
    assert curated("plasterboard", "Gyproc WallBoard 12.5 mm")
