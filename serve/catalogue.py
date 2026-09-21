"""Background sample for SHAP, drawn from the training corpus.

SHAP needs values to perturb a feature towards, and the honest choice is the
distribution the model was fitted on. The corpus supplies that.

This is the only thing the serving layer reads from the database. The corpus
holds training data, which for the seeded category is synthetic, so it is
suitable as a reference distribution for an explanation and unsuitable as a
source of products to compare. Products to compare come from the person using
the tool.

Optional. Without a database, SHAP falls back to a background spanning the
declared reference ranges and says so in its response.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from db.models import IndicatorValue, Product
from db.session import create_db_engine, session_factory

ENV_DB = "RECOMMENDER_DB"


class Catalogue:
    def __init__(self, engine: Engine):
        self.engine = engine
        self._sessions = session_factory(engine)

    @classmethod
    def open(cls, path: str | Path | None = None) -> "Catalogue | None":
        target = path or os.environ.get(ENV_DB)
        if not target or not Path(target).exists():
            return None
        return cls(create_db_engine(target))

    def _session(self) -> Session:
        return self._sessions()

    def count(self, category_key: str) -> int:
        with self._session() as session:
            return int(
                session.execute(
                    select(func.count())
                    .select_from(Product)
                    .where(Product.category_key == category_key)
                ).scalar()
                or 0
            )

    def background(
        self, category_key: str, indicator_keys: Sequence[str], size: int = 120
    ) -> list[dict[str, float | str | None]]:
        with self._session() as session:
            ids = [
                int(row[0])
                for row in session.execute(
                    select(Product.id)
                    .where(Product.category_key == category_key)
                    .order_by(func.random())
                    .limit(size)
                ).all()
            ]
            if not ids:
                return []
            rows = session.execute(
                select(
                    IndicatorValue.product_id,
                    IndicatorValue.indicator_key,
                    IndicatorValue.present,
                    IndicatorValue.value_num,
                    IndicatorValue.level_key,
                ).where(IndicatorValue.product_id.in_(ids))
            ).all()

        wanted = set(indicator_keys)
        grouped: dict[int, dict[str, float | str | None]] = {i: {} for i in ids}
        for product_id, key, present, value_num, level_key in rows:
            if key not in wanted:
                continue
            if not present:
                grouped[product_id][key] = None
            elif level_key is not None:
                grouped[product_id][key] = level_key
            else:
                grouped[product_id][key] = float(value_num)

        return [grouped[i] for i in ids if grouped[i]]
