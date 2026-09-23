"""SHAP attribution, run offline.

Seconds per alternative, which suits aggregate analysis over many instances.
The serving path never imports it, and the deployed bundle carries none of its
dependencies. `experiments.attribution` compares it with the withholding
measure.

Features are indicators, one column each, which is the granularity a specifier
reads. Continuous indicators carry their value, ordered scales carry their level
index, and a sentinel stands for a value the product does not have, so that
"unknown" is something SHAP can move a feature into rather than a hole in the
matrix.

The rest of the shortlist is held fixed while the target alternative's features
are perturbed. The score is relative to the alternatives on the table, so an
explanation has to be made in the presence of those alternatives to mean
anything.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.encoding import AlternativeInput
from core.registry import Registry
from serve.explore.analysis import score_many

class ExplanationUnavailable(RuntimeError):
    """Raised when the deployment was built without the explanation stack."""


def available() -> bool:
    """Whether SHAP can run here.

    It carries numba, llvmlite, scipy, scikit-learn and pandas behind it, which
    is roughly 330 MB. A deployment under a size limit can leave all of that out
    and still score, so the import is checked rather than assumed.
    """
    return importlib.util.find_spec("shap") is not None


MISSING = -1.0

DEFAULT_BACKGROUND = 60
DEFAULT_SAMPLES = 400
MAX_SAMPLES = 4000


@dataclass(frozen=True)
class Column:
    indicator_key: str
    display_name: str
    family: str
    value_type: str
    levels: tuple[str, ...]

    def encode(self, values: dict, levels: dict) -> float:
        if self.value_type in ("ordinal", "nominal"):
            level = levels.get(self.indicator_key)
            return MISSING if level is None else float(self.levels.index(level))
        value = values.get(self.indicator_key)
        return MISSING if value is None else float(value)

    def decode(self, cell: float, values: dict, levels: dict) -> None:
        if cell <= MISSING:
            return
        if self.value_type in ("ordinal", "nominal"):
            index = int(round(cell))
            if 0 <= index < len(self.levels):
                levels[self.indicator_key] = self.levels[index]
            return
        values[self.indicator_key] = float(cell)


def columns_for(registry: Registry, category_key: str) -> list[Column]:
    category = registry.category(category_key)
    out = []
    for key in category.token_order:
        indicator = registry.indicator(key)
        if indicator.is_derived:
            continue
        out.append(
            Column(
                indicator_key=key,
                display_name=indicator.display_name,
                family=indicator.family_key,
                value_type=indicator.value_type,
                levels=tuple(level.key for level in indicator.levels),
            )
        )
    return out


def _row(columns: Sequence[Column], alternative: AlternativeInput) -> np.ndarray:
    return np.array(
        [c.encode(alternative.values, alternative.levels) for c in columns],
        dtype=np.float64,
    )


def sample_products(
    database: str, category_key: str, rows: int = DEFAULT_BACKGROUND
) -> list[dict]:
    """Random products from the corpus, in the form `explain` takes as its background."""
    from sqlalchemy import func, select

    from db.models import IndicatorValue, Product
    from db.session import create_db_engine, session_factory

    session = session_factory(create_db_engine(database))()
    try:
        ids = [
            int(product_id)
            for (product_id,) in session.execute(
                select(Product.id)
                .where(Product.category_key == category_key)
                .order_by(func.random())
                .limit(rows)
            ).all()
        ]
        grouped: dict[int, dict] = {product_id: {} for product_id in ids}
        for product_id, key, present, value_num, level_key in session.execute(
            select(
                IndicatorValue.product_id,
                IndicatorValue.indicator_key,
                IndicatorValue.present,
                IndicatorValue.value_num,
                IndicatorValue.level_key,
            ).where(IndicatorValue.product_id.in_(ids))
        ).all():
            if not present:
                grouped[product_id][key] = None
            elif level_key is not None:
                grouped[product_id][key] = level_key
            else:
                grouped[product_id][key] = float(value_num)
        return [grouped[product_id] for product_id in ids if grouped[product_id]]
    finally:
        session.close()


def _background(
    columns: Sequence[Column],
    registry: Registry,
    category_key: str,
    samples: Sequence[dict] | None,
    size: int,
) -> np.ndarray:
    """Values to perturb against.

    Real products when the database is available. Otherwise a spread over each
    indicator's declared reference range, which is a weaker baseline and is
    reported as such.
    """
    if samples:
        rows = []
        for sample in samples:
            values = {k: v for k, v in sample.items() if isinstance(v, (int, float))}
            levels = {k: v for k, v in sample.items() if isinstance(v, str)}
            rows.append(_row(columns, AlternativeInput("bg", values, levels)))
        if rows:
            return np.array(rows[:size], dtype=np.float64)

    category = registry.category(category_key)
    rng = np.random.default_rng(0)
    rows = np.zeros((size, len(columns)), dtype=np.float64)
    for index, column in enumerate(columns):
        if column.levels:
            rows[:, index] = rng.integers(0, len(column.levels), size=size)
            continue
        spec = category.members[column.indicator_key].reference_range
        if spec is None:
            rows[:, index] = 0.0
        else:
            rows[:, index] = rng.uniform(spec.ref_low, spec.ref_high, size=size)
    return rows


def explain(
    model,
    registry: Registry,
    category_key: str,
    alternatives: Sequence[AlternativeInput],
    target: int,
    context_keys: Sequence[str],
    stakeholder_keys: Sequence[str],
    background_samples: Sequence[dict] | None = None,
    background_origin: str = "reference ranges",
    background_size: int = DEFAULT_BACKGROUND,
    nsamples: int = DEFAULT_SAMPLES,
    device: str = "cpu",
) -> dict:
    try:
        import shap
    except ImportError as error:
        raise ExplanationUnavailable(
            "This deployment was built without the explanation stack. Install "
            "shap to enable it."
        ) from error

    columns = columns_for(registry, category_key)
    others = list(alternatives)
    subject = others[target]

    def predict(matrix: np.ndarray) -> np.ndarray:
        shortlists = []
        for row in matrix:
            values: dict[str, float] = {}
            levels: dict[str, str] = {}
            for column, cell in zip(columns, row):
                column.decode(float(cell), values, levels)
            shortlist = list(others)
            shortlist[target] = AlternativeInput(subject.key, values, levels)
            shortlists.append(shortlist)

        return score_many(
            model,
            registry,
            category_key,
            shortlists,
            context_keys,
            stakeholder_keys,
            target,
            device,
        )

    background = _background(
        columns, registry, category_key, background_samples, background_size
    )
    summary = shap.kmeans(background, min(10, len(background)))

    explainer = shap.KernelExplainer(predict, summary)
    row = _row(columns, subject).reshape(1, -1)
    raw = explainer.shap_values(
        row, nsamples=min(nsamples, MAX_SAMPLES), silent=True
    )
    if isinstance(raw, list):
        raw = raw[0]
    contributions = np.asarray(raw, dtype=np.float64).reshape(-1)

    baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    observed = float(predict(row)[0])

    per_indicator = [
        {
            "indicator": column.indicator_key,
            "display_name": column.display_name,
            "family": column.family,
            "value": _readable(column, row[0][index], subject),
            "contribution": round(float(contributions[index]), 5),
        }
        for index, column in enumerate(columns)
    ]
    per_indicator.sort(key=lambda entry: abs(entry["contribution"]), reverse=True)

    families: dict[str, float] = {}
    for entry in per_indicator:
        families[entry["family"]] = families.get(entry["family"], 0.0) + entry["contribution"]

    return {
        "method": "shap",
        "target": subject.key,
        "score": round(observed, 4),
        "baseline": round(baseline, 4),
        "indicators": per_indicator,
        "families": [
            {"family": key, "contribution": round(value, 5)}
            for key, value in sorted(
                families.items(), key=lambda kv: abs(kv[1]), reverse=True
            )
        ],
        "background": background_origin if background_samples else "reference ranges",
        "background_size": int(len(background)),
        "nsamples": int(min(nsamples, MAX_SAMPLES)),
    }


def _readable(column: Column, cell: float, alternative: AlternativeInput) -> str:
    if cell <= MISSING:
        return "unknown"
    if column.levels:
        index = int(round(cell))
        return column.levels[index] if 0 <= index < len(column.levels) else "unknown"
    value = alternative.values.get(column.indicator_key, cell)
    return f"{value:g}"
