"""Adapter for the published concrete corpus.

The corpus is not regenerated natively. Reproducing the published results under
the general architecture is the experiment that establishes the architecture is
sound, and regenerating the data would forfeit it. So this adapter converts the
released files into the universal schema verbatim, and every judgement it makes
about the source format lives here rather than leaking into anything downstream.

Three translations happen at this boundary and nowhere else:

* provenance stops being a prefix on an identifier string and becomes a column;
* the health score stops being an integer on a scale treated as continuous and
  becomes an ordinal level;
* the cost object stops being summed into a single number and becomes three
  values, with the total derived from them on demand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import insert
from sqlalchemy.orm import Session

from db.models import (
    Category,
    ComparisonSet,
    ComparisonSetContext,
    ComparisonSetMember,
    ComparisonSetStakeholder,
    Context,
    DataSource,
    IndicatorValue,
    Label,
    LabelSet,
    Product,
    Stakeholder,
)
from db.session import create_db_engine, session_scope

ADAPTER_VERSION = "concrete_v1.1"
CATEGORY_KEY = "concrete"
SOURCE_KEY = "concrete_doi_data3164_v1"

#: Source field name -> registry indicator key. Only the renamed ones appear;
#: anything not listed keeps its name.
FIELD_RENAMES = {"SCM_content": "scm_content"}

#: The source nests costs under one object.
COST_FIELDS = {"c_p": "cost_product", "c_w": "cost_labour", "c_m": "cost_maintenance"}

#: Indicators the source carries as a plain number but the registry declares as
#: an ordered scale. The value is the prefix of the level key.
ORDINAL_FIELDS = {"health": "h"}

#: Fields on an alternative that are identifiers or containers, not indicators.
NON_INDICATOR_FIELDS = {"id_prod", "c"}

CHUNK = 5_000

#: What to do when the source labels the same alternative twice with different
#: scores. Refusing is the default because a duplicate means the labelling of
#: that set is not trustworthy, and quietly keeping whichever row happened to be
#: read last is how a corrupt label becomes a training target.
DUPLICATE_POLICIES = ("fail", "first", "drop_set")


class IngestError(Exception):
    pass


# ---------------------------------------------------------------------------
# Source reading
# ---------------------------------------------------------------------------


def file_hash(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def classify_provenance(external_id: str) -> tuple[str, str | None]:
    """Recover provenance and generator from the source's identifier scheme.

    This is the one place that reads meaning out of an identifier string. Past
    this function, provenance is a column.
    """
    text = str(external_id)
    if text.startswith("control_"):
        remainder = text[len("control_") :]
        generator = remainder.rsplit("_", 1)[0] if "_" in remainder else remainder
        return "control", generator
    if text.startswith("expert"):
        return "expert", None
    return "llm", None


def _normalise(text: str) -> str:
    """Fold the cosmetic differences between a source label and a display name.

    Ampersands, hyphens and stray punctuation differ between the source prose
    and the registry; the words do not.
    """
    folded = str(text).strip().lower().replace("&", "and").replace("-", " ")
    return " ".join(folded.strip(" .,;:").split())


def _display_name_map(session: Session, model) -> dict[str, str]:
    return {_normalise(row.display_name): row.key for row in session.query(model).all()}


def _match_prose(prose: str, lookup: dict[str, str], kind: str) -> str:
    """Match a source's prose label against a registry display name.

    The source records stakeholders as a full sentence whose opening clause is
    the archetype name; contexts are recorded as the bare name.
    """
    text = str(prose).strip()
    head = text.split(":", 1)[0]
    for candidate in (_normalise(head), _normalise(text)):
        if candidate in lookup:
            return lookup[candidate]
    raise IngestError(f"no registry {kind} matches {prose!r}")


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------


def extract_values(
    alternative: dict, known_indicators: set[str]
) -> tuple[dict[str, float], dict[str, str], list[str]]:
    """Split one source alternative into numeric values and ordinal levels.

    A null in the source means unknown, and stays unknown: it is never imputed.
    A zero is a real value and is kept.
    """
    values: dict[str, float] = {}
    levels: dict[str, str] = {}
    unknown: list[str] = []

    for field, raw in alternative.items():
        if field in NON_INDICATOR_FIELDS:
            continue
        key = FIELD_RENAMES.get(field, field)
        if key in ORDINAL_FIELDS:
            if raw is not None:
                levels[key] = f"{ORDINAL_FIELDS[key]}{int(raw)}"
            continue
        if key not in known_indicators:
            unknown.append(field)
            continue
        if raw is not None:
            values[key] = float(raw)

    for field, raw in (alternative.get("c") or {}).items():
        key = COST_FIELDS.get(field)
        if key is None:
            unknown.append(f"c.{field}")
        elif raw is not None:
            values[key] = float(raw)

    return values, levels, unknown


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def ingest(
    session: Session,
    scenarios_path: Path,
    labels_path: Path,
    limit: int | None = None,
    doi: str | None = None,
    citation: str = "",
    on_duplicate_label: str = "fail",
) -> dict[str, int]:
    if on_duplicate_label not in DUPLICATE_POLICIES:
        raise ValueError(f"on_duplicate_label must be one of {DUPLICATE_POLICIES}")
    category = session.get(Category, CATEGORY_KEY)
    if category is None:
        raise IngestError(
            f"category {CATEGORY_KEY!r} is not in the registry; seed it first"
        )

    from db.models import CategoryIndicator

    membership = {
        row.indicator_key
        for row in session.query(CategoryIndicator).filter_by(
            category_key=CATEGORY_KEY
        )
    }
    stakeholder_lookup = _display_name_map(session, Stakeholder)
    context_lookup = _display_name_map(session, Context)

    now = datetime.now(timezone.utc).isoformat()
    session.merge(
        DataSource(
            key=SOURCE_KEY,
            display_name="Concrete alternatives with stakeholder and context labels",
            citation=citation or "Published open-access dataset.",
            doi=doi,
            ingested_at=now,
            adapter_version=ADAPTER_VERSION,
            content_hash=file_hash(scenarios_path, labels_path),
        )
    )
    session.flush()

    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    raw_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    label_index = {str(entry["id"]): entry for entry in raw_labels}

    if limit is not None:
        scenarios = scenarios[:limit]
        label_index = {
            key: value
            for key, value in label_index.items()
            if key in {str(s["id"]) for s in scenarios}
        }

    scenario_ids = {str(s["id"]) for s in scenarios}
    orphans = sorted(set(label_index) - scenario_ids)
    if orphans:
        raise IngestError(
            f"{len(orphans)} label set(s) reference scenarios that do not exist, "
            f"first few: {orphans[:5]}. Refusing to drop them silently -- resolve "
            "the discrepancy in the source before ingesting."
        )

    products: list[dict] = []
    values: list[dict] = []
    sets: list[dict] = []
    members: list[dict] = []
    set_stakeholders: list[dict] = []
    set_contexts: list[dict] = []
    label_sets: list[dict] = []
    labels: list[dict] = []

    product_id = _next_id(session, Product)
    set_id = _next_id(session, ComparisonSet)
    label_set_id = _next_id(session, LabelSet)
    unknown_fields: set[str] = set()
    missing_labels = 0
    duplicate_sets = 0

    for scenario in scenarios:
        external_id = str(scenario["id"])
        provenance, generator = classify_provenance(external_id)

        sets.append(
            {
                "id": set_id,
                "category_key": CATEGORY_KEY,
                "source_key": SOURCE_KEY,
                "provenance_key": provenance,
                "generator_key": generator,
                "external_id": external_id,
                "created_at": now,
            }
        )

        for prose in scenario.get("stakeholder_preference", []):
            set_stakeholders.append(
                {
                    "set_id": set_id,
                    "stakeholder_key": _match_prose(
                        prose, stakeholder_lookup, "stakeholder"
                    ),
                }
            )
        for prose in scenario.get("situations", []):
            set_contexts.append(
                {
                    "set_id": set_id,
                    "context_key": _match_prose(prose, context_lookup, "context"),
                }
            )

        label_entry = label_index.get(external_id)
        labelled, duplicated = _index_labels(
            (label_entry or {}).get("labelled_alternatives", [])
        )
        if duplicated:
            if on_duplicate_label == "fail":
                raise IngestError(
                    f"set {external_id} labels {sorted(duplicated)} more than once "
                    "with differing scores. Pass --on-duplicate-label to say what "
                    "should happen; keeping one arbitrarily is not a decision this "
                    "adapter will make on its own."
                )
            duplicate_sets += 1
            if on_duplicate_label == "drop_set":
                label_entry = None

        if label_entry is None:
            missing_labels += 1
        else:
            label_sets.append(
                {
                    "id": label_set_id,
                    "comparison_set_id": set_id,
                    "provenance_key": provenance,
                    "labeller_key": _labeller(provenance, generator),
                    # Control labels are a declared function of a declared
                    # range; the assessment brief asked for suitability within
                    # the set. Those are different statements, so which one a
                    # score is gets recorded rather than assumed.
                    "scale_semantics": (
                        "absolute_reference"
                        if provenance == "control"
                        else "within_set_relative"
                    ),
                    "method_note": f"As published; ingested by {ADAPTER_VERSION}.",
                    "created_at": now,
                }
            )

        for position, alternative in enumerate(scenario.get("alternatives", [])):
            local_key = str(alternative["id_prod"])
            numeric, level_keys, unknown = extract_values(alternative, membership)
            unknown_fields.update(unknown)

            products.append(
                {
                    "id": product_id,
                    "category_key": CATEGORY_KEY,
                    "source_key": SOURCE_KEY,
                    "external_ref": f"{external_id}/{local_key}",
                    "display_name": local_key,
                    "created_at": now,
                }
            )
            members.append(
                {
                    "set_id": set_id,
                    "product_id": product_id,
                    "category_key": CATEGORY_KEY,
                    "position": position,
                    "local_key": local_key,
                }
            )

            # A row for every indicator the category holds, so that "applicable
            # but missing" is stored rather than inferred from an absent row.
            for indicator_key in sorted(membership):
                if indicator_key in numeric:
                    values.append(
                        {
                            "product_id": product_id,
                            "indicator_key": indicator_key,
                            "present": 1,
                            "value_num": numeric[indicator_key],
                            "level_key": None,
                        }
                    )
                elif indicator_key in level_keys:
                    values.append(
                        {
                            "product_id": product_id,
                            "indicator_key": indicator_key,
                            "present": 1,
                            "value_num": None,
                            "level_key": level_keys[indicator_key],
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

            row = labelled.get(local_key)
            if row is not None and row.get("pref") is not None:
                labels.append(
                    {
                        "label_set_id": label_set_id,
                        "product_id": product_id,
                        "pref": float(row["pref"]),
                        "conf": None if row.get("conf") is None else float(row["conf"]),
                        "rationale": row.get("reason"),
                    }
                )

            product_id += 1

        if label_entry is not None:
            label_set_id += 1
        set_id += 1

    if unknown_fields:
        raise IngestError(
            "source fields with no registry indicator: "
            f"{sorted(unknown_fields)}. Add them to the registry or add an "
            "explicit rename; dropping them silently would change the model's "
            "inputs without anyone reviewing it."
        )

    _bulk(session, ComparisonSet, sets)
    _bulk(session, Product, products)
    _bulk(session, ComparisonSetMember, members)
    _bulk(session, ComparisonSetStakeholder, set_stakeholders)
    _bulk(session, ComparisonSetContext, set_contexts)
    _bulk(session, IndicatorValue, values)
    _bulk(session, LabelSet, label_sets)
    _bulk(session, Label, labels)

    return {
        "comparison_sets": len(sets),
        "products": len(products),
        "indicator_values": len(values),
        "label_sets": len(label_sets),
        "labels": len(labels),
        "unlabelled_sets": missing_labels,
        "sets_with_duplicate_labels": duplicate_sets,
    }


def _index_labels(rows: list[dict]) -> tuple[dict[str, dict], set[str]]:
    """Index labels by alternative, reporting any alternative labelled twice."""
    indexed: dict[str, dict] = {}
    duplicated: set[str] = set()
    for row in rows:
        key = str(row["id_prod"])
        if key in indexed:
            if indexed[key].get("pref") != row.get("pref"):
                duplicated.add(key)
            continue  # first occurrence wins
        indexed[key] = row
    return indexed, duplicated


def _labeller(provenance: str, generator: str | None) -> str:
    if provenance == "control":
        return f"generator:{generator}"
    if provenance == "expert":
        # The published corpus is already aggregated across annotators. The
        # schema holds one label set per annotator, so per-annotator scores can
        # be ingested later without a migration -- but what is on offer today is
        # the aggregate, and it says so.
        return "published_aggregate"
    return "llm_published"


def _next_id(session: Session, model) -> int:
    from sqlalchemy import func, select

    highest = session.execute(select(func.max(model.id))).scalar()
    return 1 if highest is None else int(highest) + 1


def _bulk(session: Session, model, rows: list[dict]) -> None:
    for start in range(0, len(rows), CHUNK):
        session.execute(insert(model.__table__), rows[start : start + CHUNK])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenarios", help="frozen scenarios JSON from the release")
    parser.add_argument("labels", help="labels JSON from the release")
    parser.add_argument("--db", default=None)
    parser.add_argument("--limit", type=int, default=None, help="first N scenarios")
    parser.add_argument("--doi", default=None)
    parser.add_argument("--citation", default="")
    parser.add_argument(
        "--on-duplicate-label",
        choices=DUPLICATE_POLICIES,
        default="fail",
        help="what to do when the source labels one alternative twice",
    )
    args = parser.parse_args()

    with session_scope(create_db_engine(args.db)) as session:
        counts = ingest(
            session,
            Path(args.scenarios),
            Path(args.labels),
            limit=args.limit,
            doi=args.doi,
            citation=args.citation,
            on_duplicate_label=args.on_duplicate_label,
        )

    for name, count in counts.items():
        print(f"{name:20s} {count:>10,}")


if __name__ == "__main__":
    main()
