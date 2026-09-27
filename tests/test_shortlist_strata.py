import random
from types import SimpleNamespace

from eval.metrics import strata_of
from ingest.facade.synthesise import shortlist


def _shortlist_strata(generator):
    return strata_of(SimpleNamespace(set_generator=[generator]), 0, "shortlist")


def test_a_single_typology_shortlist_counts_as_within_and_under_its_typology():
    assert _shortlist_strata("within:etics") == ("within", "within:etics")


def test_a_mixed_shortlist_counts_as_across():
    assert _shortlist_strata("across") == ("across",)


def test_control_cases_and_unlabelled_generators_stay_out_of_the_stratum():
    assert _shortlist_strata("gwp") == ()
    assert _shortlist_strata(None) == ()
    assert _shortlist_strata(float("nan")) == ()


def test_every_synthesised_shortlist_lands_in_the_stratum():
    pools = {t: [{"id": f"{t}_{i}"} for i in range(10)] for t in ("etics", "masonry", "precast")}
    rng = random.Random(0)
    for _ in range(200):
        generator, _ = shortlist(rng, pools)
        assert _shortlist_strata(generator)
