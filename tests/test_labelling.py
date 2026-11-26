from pathlib import Path

from ingest.labelling import next_chunk_number


def test_chunk_numbers_past_99_continue_the_sequence():
    names = ["chunk_13b.json", "chunk_99.json", "chunk_100.json", "chunk_118.json"]
    assert next_chunk_number([Path(n) for n in names]) == 119


def test_a_suffixed_chunk_counts_by_its_number():
    assert next_chunk_number([Path("chunk_13b.json"), Path("chunk_02.json")]) == 14


def test_the_first_chunk_is_one():
    assert next_chunk_number([]) == 1


def _round_with_two_sent_scenarios(tmp_path):
    import json

    from ingest.labelling import LabelRound, fingerprint

    label_round = LabelRound(
        category_key="any",
        dataset_dir=tmp_path,
        scenarios_file="scenarios.json",
        labels_file="labels.json",
        brief_source=tmp_path / "brief.md",
        examples_source=tmp_path / "examples.yaml",
        letters="AB",
    )
    scenarios = [
        {"id": sid, "contexts": [], "stakeholders": [], "alternatives": [{"id_prod": "p1"}, {"id_prod": "p2"}]}
        for sid in ("s1", "s2")
    ]
    (tmp_path / "scenarios.json").write_text(json.dumps(scenarios), encoding="utf-8")
    label_round.chunks_dir.mkdir(parents=True)
    label_round.out_dir.mkdir(parents=True)
    label_round.index_file.write_text(json.dumps({
        s["id"]: {"letters": {"A": "p1", "B": "p2"}, "fingerprint": fingerprint(s)} for s in scenarios
    }), encoding="utf-8")
    (label_round.chunks_dir / "chunk_01.json").write_text(
        json.dumps({"scenarios": [{"scenario_id": "s1"}, {"scenario_id": "s2"}]}), encoding="utf-8"
    )
    return label_round


def test_collect_reports_a_scenario_the_labeller_skipped(tmp_path):
    import json

    label_round = _round_with_two_sent_scenarios(tmp_path)
    rows = [{"id_prod": "A", "pref": 0.7, "conf": 0.6, "reason": "r"}, {"id_prod": "B", "pref": 0.3, "conf": 0.6, "reason": "r"}]
    (label_round.out_dir / "chunk_01.json").write_text(json.dumps({"labels": {"s1": rows}}), encoding="utf-8")

    problems = label_round.collect()

    assert problems == ["chunk_01.json: s2 sent but not labelled"]
    assert [s["id"] for s in json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))] == ["s1"]


def test_collect_reports_a_file_saved_in_the_windows_code_page(tmp_path):
    label_round = _round_with_two_sent_scenarios(tmp_path)
    (label_round.out_dir / "chunk_01.json").write_bytes('{"labels": {"s1": [{"reason": "40 kg/m²"}]}}'.encode("cp1252"))

    problems = label_round.collect()

    assert problems == ["chunk_01.json: unreadable; it must be UTF-8 JSON with a labels object"]
    assert label_round.status()["pending"] == 2
