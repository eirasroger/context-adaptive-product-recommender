"""The frontend draws everything from the registry through the API."""

from __future__ import annotations

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
GENERATED = FRONTEND / "src" / "api" / "schema.d.ts"


def _sources() -> list[Path]:
    files = [FRONTEND / "index.html"]
    files += [
        path
        for path in (FRONTEND / "src").rglob("*")
        if path.suffix in {".ts", ".tsx", ".css", ".html"} and path != GENERATED
    ]
    return files


def test_the_frontend_names_no_registry_key(registry):
    """Keys are matched as quoted literals, which is how code would name one;
    several are ordinary words, such as `b` and `density`. Category names are
    distinctive enough to match anywhere."""
    keys = (
        set(registry.indicators)
        | set(registry.contexts)
        | set(registry.stakeholders)
        | set(registry.categories)
    )
    literal = re.compile(r"""["'`](%s)["'`]""" % "|".join(map(re.escape, sorted(keys))))
    categories = {key.lower() for key in registry.categories} | {
        category.display_name.lower() for category in registry.categories.values()
    }

    sources = _sources()
    assert len(sources) > 1, "found no frontend sources; the layout has changed"

    offenders = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        offenders += [f"{path.name} names {m.group(1)!r}" for m in literal.finditer(text)]
        offenders += [f"{path.name} mentions {name!r}" for name in categories if name in text.lower()]
    assert not offenders, offenders
