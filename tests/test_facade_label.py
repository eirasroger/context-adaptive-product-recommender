from pathlib import Path

from ingest.facade.label import next_chunk_number


def test_chunk_numbers_past_99_continue_the_sequence():
    names = ["chunk_13b.json", "chunk_99.json", "chunk_100.json", "chunk_118.json"]
    assert next_chunk_number([Path(n) for n in names]) == 119


def test_a_suffixed_chunk_counts_by_its_number():
    assert next_chunk_number([Path("chunk_13b.json"), Path("chunk_02.json")]) == 14


def test_the_first_chunk_is_one():
    assert next_chunk_number([]) == 1
