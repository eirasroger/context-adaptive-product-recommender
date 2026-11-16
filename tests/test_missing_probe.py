from experiments.missing_probe import PROBED, VERSIONS, hidden_version, placement, target_shifts


def test_no_chunk_holds_one_shortlist_twice():
    seen = set()
    for (shortlist, _), chunk in placement(shortlists=40, chunks=3).items():
        assert (shortlist, chunk) not in seen
        seen.add((shortlist, chunk))


def test_only_the_target_loses_its_health_status():
    scenario = {"alternatives": [{"id_prod": "a", PROBED: "h2", "gwp": 1.0}, {"id_prod": "b", PROBED: "h3"}]}
    hidden = hidden_version(scenario, "a")
    assert hidden["alternatives"] == [{"id_prod": "a", "gwp": 1.0}, {"id_prod": "b", PROBED: "h3"}]
    assert scenario["alternatives"][0][PROBED] == "h2"


def test_the_shift_is_measured_against_both_unchanged_labels():
    labels = dict(zip(VERSIONS, ({"a": 0.8}, {"a": 0.6}, {"a": 0.5})))
    shift, disagreement = target_shifts(labels, "a")
    assert abs(shift - (-0.2)) < 1e-9
    assert abs(disagreement - 0.2) < 1e-9
