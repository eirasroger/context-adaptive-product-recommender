from experiments.partition_context_pilot import (
    LEAVES,
    brief,
    candidates,
    compose,
    face_heat_capacity,
    sample_shortlists,
    screening,
    typology,
)


def test_framed_walls_are_lighter_and_better_insulated_than_single_brick_walls():
    products = compose(0)
    framed = [p for p in products if p["family"] == "t3"]
    brick = [p for p in products if p["family"] == "t1"]
    assert max(p["surface_mass"] for p in framed) < min(p["surface_mass"] for p in brick)
    assert max(p["u_value"] for p in framed) < min(p["u_value"] for p in brick)


def test_a_lining_on_both_faces_beats_one_face_on_sound_and_heat():
    products = compose(0)
    one = [p for p in products if "+tr1x1" in p["source"] and "lh70" in p["source"]]
    two = [p for p in products if "+tr1x2" in p["source"]]
    assert min(p["sound_reduction"] for p in two) > max(p["sound_reduction"] for p in one)
    assert max(p["u_value"] for p in two) < min(p["u_value"] for p in one)


def test_heat_capacity_never_reaches_past_the_midplane():
    leaf = LEAVES["lh70"]
    assert face_heat_capacity(leaf, 100.0) == 17.0 + 50.0


def test_end_of_life_shares_stay_within_the_whole():
    for product in compose(0):
        total = product["fu_recyc"] + product["fu_inert"] + product["fu_haz"]
        assert 90 <= total <= 101


def test_most_shortlists_mix_typologies():
    shortlists = sample_shortlists(compose(0), 50, 0)
    mixed = sum(len({typology(p) for p in s}) > 1 for s in shortlists)
    assert 0.5 <= mixed / len(shortlists) <= 0.8
    assert all(4 <= len(s) <= 5 for s in shortlists)
    assert all(len({p["id_prod"] for p in s}) == len(s) for s in shortlists)


def test_every_piloted_context_is_described_in_the_brief():
    record = screening()
    text = brief(record)
    assert "{situations}" not in text
    for candidate in candidates(record):
        assert f"- {candidate.name}:" in text


def test_every_screened_out_candidate_gives_a_reason():
    for candidate in screening()["candidates"]:
        assert candidate["outcome"] in {"pilot", "compliance", "value", "stakeholder", "scope"}
        assert candidate["outcome"] == "pilot" or candidate.get("reason")
