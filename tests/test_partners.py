from __future__ import annotations

import pytest

from serve import partners


def test_an_issued_entry_names_the_partner_its_key_belongs_to(capsys):
    partners.main(["someone"])
    lines = capsys.readouterr().out.splitlines()
    key = lines[0].split()[-1]
    entry = lines[1].split()[-1]

    assert partners.parse(entry) == {partners.digest(key): "someone"}


def test_several_entries_are_read_from_one_variable():
    raw = f"a:{partners.digest('x')}, b:{partners.digest('y')},"
    assert sorted(partners.parse(raw).values()) == ["a", "b"]


def test_an_unset_variable_admits_no_partner():
    assert partners.parse("") == {}


@pytest.mark.parametrize("entry", ["someone", "someone:abc", f":{'0' * 64}", f"Some One:{'0' * 64}"])
def test_a_malformed_entry_stops_the_service_from_starting(entry):
    with pytest.raises(ValueError):
        partners.parse(entry)


def test_the_key_itself_never_appears_in_the_entry(capsys):
    partners.main(["someone"])
    key, entry = (line.split()[-1] for line in capsys.readouterr().out.splitlines())
    assert key not in entry
