import pytest

from experiments.context_influence import (
    assign_chunks,
    brief,
    candidates,
    pair_discordance,
    screening,
    winner_changed,
)


def test_no_chunk_holds_one_shortlist_twice():
    placement = assign_chunks(shortlists=50, n_versions=15, chunks=18)
    seen = {}
    for (shortlist, _), chunk in placement.items():
        assert (shortlist, chunk) not in seen
        seen[(shortlist, chunk)] = True


def test_more_versions_than_chunks_is_refused():
    with pytest.raises(ValueError):
        assign_chunks(shortlists=2, n_versions=5, chunks=4)


def test_identical_labels_have_no_discordance():
    labels = {"a": 0.9, "b": 0.5, "c": 0.2}
    assert pair_discordance(labels, labels) == 0
    assert winner_changed(labels, labels) == 0


def test_a_reversal_is_fully_discordant():
    assert pair_discordance({"a": 0.9, "b": 0.5, "c": 0.2}, {"a": 0.2, "b": 0.5, "c": 0.9}) == 1


def test_a_tie_on_one_side_counts_half():
    assert pair_discordance({"a": 0.8, "b": 0.5}, {"a": 0.51, "b": 0.5}) == 0.5


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
