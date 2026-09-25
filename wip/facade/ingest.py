"""Read the working facade dataset into the corpus, keyed by registry keys throughout."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, insert, select
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
from ingest.adapters.concrete_v1 import IngestError, file_hash
from wip.facade.synthesise import CATEGORY_KEY

ADAPTER_VERSION = "facade_wip_v0"
SOURCE_KEY = "facade_wip_synthetic"
PROVENANCE_KEY = "synthetic"
METADATA_FIELDS = {"id_prod", "typology"}
CHUNK = 5_000


def _next_id(session: Session, model) -> int:
    highest = session.execute(select(func.max(model.id))).scalar()
    return 1 if highest is None else int(highest) + 1


def _value_row(registry: Registry, product_id: int, indicator_key: str, raw) -> dict:
    row = {"product_id": product_id, "indicator_key": indicator_key, "present": 0, "value_num": None, "level_key": None}
    if raw is None:
        return row
    if registry.indicator(indicator_key).is_scale:
        return {**row, "present": 1, "level_key": str(raw)}
    return {**row, "present": 1, "value_num": float(raw)}


def ingest(session: Session, registry: Registry, scenarios_path: Path, labels_path: Path) -> dict[str, int]:
    if session.get(DataSource, SOURCE_KEY) is not None:
        raise IngestError(f"source {SOURCE_KEY!r} is already in this corpus; rebuild it from scratch")

    category = registry.category(CATEGORY_KEY)
    membership = sorted(category.members)
    now = datetime.now(timezone.utc).isoformat()
    session.add(
        DataSource(
            key=SOURCE_KEY,
            display_name="Facade systems, working dataset",
            citation="Synthetic working dataset from wip.facade.synthesise. For development only.",
            doi=None,
            ingested_at=now,
            adapter_version=ADAPTER_VERSION,
            content_hash=file_hash(scenarios_path, labels_path),
        )
    )
    session.flush()

    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    labels = {entry["id"]: entry for entry in json.loads(labels_path.read_text(encoding="utf-8"))}

    rows: dict[type, list[dict]] = {model: [] for model in (
        ComparisonSet, Product, ComparisonSetMember, ComparisonSetStakeholder,
        ComparisonSetContext, IndicatorValue, LabelSet, Label,
    )}
    product_id = _next_id(session, Product)
    set_id = _next_id(session, ComparisonSet)
    label_set_id = _next_id(session, LabelSet)
    unknown: set[str] = set()

    for scenario in scenarios:
        external_id = scenario["id"]
        rows[ComparisonSet].append({
            "id": set_id, "category_key": CATEGORY_KEY, "source_key": SOURCE_KEY,
            "provenance_key": PROVENANCE_KEY, "generator_key": scenario["shortlist"],
            "external_id": external_id, "created_at": now,
        })
        rows[ComparisonSetContext] += [{"set_id": set_id, "context_key": key} for key in scenario["contexts"]]
        rows[ComparisonSetStakeholder] += [{"set_id": set_id, "stakeholder_key": key} for key in scenario["stakeholders"]]
        rows[LabelSet].append({
            "id": label_set_id, "comparison_set_id": set_id, "provenance_key": PROVENANCE_KEY,
            "labeller_key": f"rule:{ADAPTER_VERSION}", "scale_semantics": "within_set_relative",
            "method_note": "Declared rule over registry directions and stakeholder priorities.",
            "created_at": now,
        })
        prefs = {row["id_prod"]: row for row in labels[external_id]["labelled_alternatives"]}

        for position, alternative in enumerate(scenario["alternatives"]):
            local_key = alternative["id_prod"]
            unknown.update(set(alternative) - METADATA_FIELDS - set(membership))
            rows[Product].append({
                "id": product_id, "category_key": CATEGORY_KEY, "source_key": SOURCE_KEY,
                "external_ref": f"{external_id}/{local_key}", "display_name": local_key, "created_at": now,
            })
            rows[ComparisonSetMember].append({
                "set_id": set_id, "product_id": product_id, "category_key": CATEGORY_KEY,
                "position": position, "local_key": local_key,
            })
            rows[IndicatorValue] += [
                _value_row(registry, product_id, key, alternative.get(key)) for key in membership
            ]
            label = prefs[local_key]
            rows[Label].append({
                "label_set_id": label_set_id, "product_id": product_id,
                "pref": float(label["pref"]), "conf": float(label["conf"]), "rationale": label.get("reason"),
            })
            product_id += 1
        set_id += 1
        label_set_id += 1

    if unknown:
        raise IngestError(f"fields with no registry indicator in {CATEGORY_KEY}: {sorted(unknown)}")

    for model, batch in rows.items():
        for start in range(0, len(batch), CHUNK):
            session.execute(insert(model.__table__), batch[start : start + CHUNK])
    return {model.__tablename__: len(batch) for model, batch in rows.items()}
