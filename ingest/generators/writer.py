"""Persist generated control cases into the corpus.

Generated cases go through exactly the same tables as ingested ones. There is no
separate path for synthetic data, which is what makes provenance a genuine
column rather than a label attached to a parallel pipeline.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Iterable, Sequence

from sqlalchemy import insert
from sqlalchemy.orm import Session

from core.registry import Registry
from db.models import (
    ComparisonSet,
    ComparisonSetContext,
    ComparisonSetMember,
    ComparisonSetStakeholder,
    DataSource,
    IndicatorValue,
    Label,
    LabelSet,
    Product,
)
from ingest.generators.parametric import ProbeCase

GENERATED_SOURCE = "generated_control"


def ensure_source(session: Session, key: str = GENERATED_SOURCE, note: str = "") -> None:
    session.merge(
        DataSource(
            key=key,
            display_name="Registry-driven control cases",
            citation=note or "Generated from the registry by ingest.generators.parametric.",
            doi=None,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            adapter_version="parametric_v1",
            content_hash=hashlib.sha256(key.encode("utf-8")).hexdigest(),
        )
    )
    session.flush()


def _next_id(session: Session, model) -> int:
    from sqlalchemy import func, select

    highest = session.execute(select(func.max(model.id))).scalar()
    return 1 if highest is None else int(highest) + 1


def write_cases(
    session: Session,
    registry: Registry,
    cases: Sequence[ProbeCase],
    source_key: str = GENERATED_SOURCE,
    prefix: str = "gen",
) -> dict[str, int]:
    """Write generated cases and their labels. Returns row counts."""
    ensure_source(session, source_key)
    now = datetime.now(timezone.utc).isoformat()

    product_id = _next_id(session, Product)
    set_id = _next_id(session, ComparisonSet)
    label_set_id = _next_id(session, LabelSet)

    products, values, sets, members = [], [], [], []
    set_contexts, set_stakeholders, label_sets, labels = [], [], [], []

    for ordinal, case in enumerate(cases):
        category = registry.category(case.category_key)
        external_id = f"{prefix}_{case.indicator_key}_{ordinal}"

        sets.append(
            {
                "id": set_id,
                "category_key": case.category_key,
                "source_key": source_key,
                "provenance_key": "control",
                # The varied indicator is the generator identity: one parametric
                # generator, told which indicator to sweep.
                "generator_key": case.indicator_key,
                "external_id": external_id,
                "created_at": now,
            }
        )
        for context_key in case.context_keys:
            set_contexts.append({"set_id": set_id, "context_key": context_key})
        set_stakeholders.append(
            {"set_id": set_id, "stakeholder_key": case.stakeholder_key}
        )
        label_sets.append(
            {
                "id": label_set_id,
                "comparison_set_id": set_id,
                "provenance_key": "control",
                "labeller_key": f"parametric:{case.indicator_key}",
                # A control label is a declared function of a declared range, so
                # it means something on its own rather than only relative to the
                # set it appears in.
                "scale_semantics": "absolute_reference",
                "method_note": (
                    f"All indicators ideal except {case.indicator_key}; band width "
                    f"from the priorities of {case.stakeholder_key}."
                ),
                "created_at": now,
            }
        )

        for position, alternative in enumerate(case.alternatives):
            products.append(
                {
                    "id": product_id,
                    "category_key": case.category_key,
                    "source_key": source_key,
                    "external_ref": f"{external_id}/{alternative.key}",
                    "display_name": alternative.key,
                    "created_at": now,
                }
            )
            members.append(
                {
                    "set_id": set_id,
                    "product_id": product_id,
                    "category_key": case.category_key,
                    "position": position,
                    "local_key": alternative.key,
                }
            )
            for indicator_key in category.token_order:
                if registry.indicator(indicator_key).is_derived:
                    # Recomputed from its sources when encoding, never stored.
                    continue
                if indicator_key in alternative.values:
                    values.append(
                        {
                            "product_id": product_id,
                            "indicator_key": indicator_key,
                            "present": 1,
                            "value_num": float(alternative.values[indicator_key]),
                            "level_key": None,
                        }
                    )
                elif indicator_key in alternative.levels:
                    values.append(
                        {
                            "product_id": product_id,
                            "indicator_key": indicator_key,
                            "present": 1,
                            "value_num": None,
                            "level_key": alternative.levels[indicator_key],
                        }
                    )
                else:
                    values.append(
                        {
                            "product_id": product_id,
                            "indicator_key": indicator_key,
                            "present": 0,
                            "value_num": None,
                            "level_key": None,
                        }
                    )

            labels.append(
                {
                    "label_set_id": label_set_id,
                    "product_id": product_id,
                    "pref": float(case.prefs[position]),
                    "conf": 1.0,
                    "rationale": (
                        f"{case.indicator_key}={case.varied[position]!r}; "
                        f"quality {case.quality[position]:.3f}"
                    ),
                }
            )
            product_id += 1

        label_set_id += 1
        set_id += 1

    for model, rows in (
        (ComparisonSet, sets),
        (Product, products),
        (ComparisonSetMember, members),
        (ComparisonSetContext, set_contexts),
        (ComparisonSetStakeholder, set_stakeholders),
        (IndicatorValue, values),
        (LabelSet, label_sets),
        (Label, labels),
    ):
        for start in range(0, len(rows), 5_000):
            session.execute(insert(model.__table__), rows[start : start + 5_000])

    return {
        "comparison_sets": len(sets),
        "products": len(products),
        "indicator_values": len(values),
        "labels": len(labels),
    }


def generate_for_category(
    registry: Registry,
    category_key: str,
    stakeholders: Iterable[str] | None = None,
    steps: int = 5,
    jitter: float = 0.05,
    seed: int = 0,
) -> list[ProbeCase]:
    """Sweep every indicator the registry gives a direction to, per context."""
    import random

    from ingest.generators import parametric

    rng = random.Random(seed)
    category = registry.category(category_key)
    stakeholder_keys = list(stakeholders or registry.stakeholders)

    cases: list[ProbeCase] = []
    for context_key in sorted(category.available_contexts):
        for indicator_key in parametric.varied_indicators(
            registry, category_key, [context_key]
        ):
            values = parametric.sweep_values(
                registry, category_key, indicator_key, steps
            )
            for stakeholder_key in stakeholder_keys:
                cases.append(
                    parametric.make_case(
                        registry,
                        category_key,
                        indicator_key,
                        values,
                        [context_key],
                        stakeholder_key,
                        jitter=jitter,
                        rng=rng,
                    )
                )
    return cases
