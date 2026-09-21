"""Background sample for SHAP.

SHAP perturbs a feature towards other plausible values, so it needs a reference
distribution. The right one is the distribution the model was fitted on.

Two sources, tried in order:

1. A file shipped with the release. Small, self-contained, and the only option
   that works where there is no database, which includes every serverless
   deployment.
2. The live database, when one is attached. Used in development, and the source
   the released file is exported from.

Falling back to neither is survivable. SHAP then perturbs over the declared
reference ranges and says so in its response.

The corpus holds training data. For a category seeded from generated cases that
data is synthetic, which makes it a sound reference distribution for an
explanation and an unsound source of products to compare.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

RELEASE_DIR = Path(__file__).parent / "release"
BACKGROUND_FILE = "background.json"

ENV_DB = "RECOMMENDER_DB"
ENV_BACKGROUND = "RECOMMENDER_BACKGROUND"

DEFAULT_ROWS = 256


class Background:
    """Where SHAP gets its reference values, and what to call it in a report."""

    def __init__(self, rows: dict[str, list[dict]], origin: str):
        self._rows = rows
        self.origin = origin

    @property
    def available(self) -> bool:
        return bool(self._rows)

    def size(self, category_key: str) -> int:
        return len(self._rows.get(category_key, []))

    def rows(
        self, category_key: str, indicator_keys: Sequence[str]
    ) -> list[dict[str, float | str | None]]:
        wanted = set(indicator_keys)
        return [
            {k: v for k, v in row.items() if k in wanted}
            for row in self._rows.get(category_key, [])
        ]

    # -- construction ------------------------------------------------------

    @classmethod
    def open(cls) -> "Background":
        path = Path(os.environ.get(ENV_BACKGROUND, RELEASE_DIR / BACKGROUND_FILE))
        if path.exists():
            return cls.from_file(path)

        database = os.environ.get(ENV_DB)
        if database and Path(database).exists():
            return cls.from_database(database)

        return cls({}, "reference ranges")

    @classmethod
    def from_file(cls, path: Path) -> "Background":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload.get("categories", {}), payload.get("origin", "released sample"))

    @classmethod
    def from_database(cls, database: str, rows: int = DEFAULT_ROWS) -> "Background":
        return cls(sample_from_database(database, rows), "corpus")


def sample_from_database(database: str, rows: int = DEFAULT_ROWS) -> dict[str, list[dict]]:
    """Draw a sample of products per category.

    Imported lazily so a deployment that ships the exported file does not need
    SQLAlchemy or the database package at all.
    """
    from sqlalchemy import func, select

    from db.models import Category, IndicatorValue, Product
    from db.session import create_db_engine, session_factory

    session = session_factory(create_db_engine(database))()
    try:
        categories = [key for (key,) in session.execute(select(Category.key)).all()]
        sampled: dict[str, list[dict]] = {}

        for category_key in categories:
            ids = [
                int(row[0])
                for row in session.execute(
                    select(Product.id)
                    .where(Product.category_key == category_key)
                    .order_by(func.random())
                    .limit(rows)
                ).all()
            ]
            if not ids:
                continue

            values = session.execute(
                select(
                    IndicatorValue.product_id,
                    IndicatorValue.indicator_key,
                    IndicatorValue.present,
                    IndicatorValue.value_num,
                    IndicatorValue.level_key,
                ).where(IndicatorValue.product_id.in_(ids))
            ).all()

            grouped: dict[int, dict] = {i: {} for i in ids}
            for product_id, key, present, value_num, level_key in values:
                if not present:
                    grouped[product_id][key] = None
                elif level_key is not None:
                    grouped[product_id][key] = level_key
                else:
                    grouped[product_id][key] = float(value_num)

            sampled[category_key] = [grouped[i] for i in ids if grouped[i]]

        return sampled
    finally:
        session.close()
