"""Generate control cases from the registry and write them to the corpus.

Used to fill coverage gaps. An indicator that nothing in the corpus ever varies
on its own gives the model no isolated signal for it, so the model learns to
ignore it -- which the behavioural suite reports as no measurable response.
Generating cases for it is the fix, and it needs no new code per indicator
because the sweep comes from the registry.

Only indicators the registry declares sweepable are eligible. One whose declared
direction does not hold across its whole range is refused rather than quietly
skipped: asking for it means the caller believes something the registry denies,
and that is worth an error rather than a silent omission.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.registry import Registry
from db.models import ComparisonSet
from db.session import create_db_engine, session_scope
from ingest.generators import parametric
from ingest.generators.writer import GENERATED_SOURCE, write_cases

#: Matches the shape of decisions in the published corpus: a shortlist is a
#: handful of options, not a catalogue.
DEFAULT_MIN_ALTS = 2
DEFAULT_MAX_ALTS = 5

#: Wobble applied to the indicators being held at their ideal, so the model
#: cannot learn "everything is exactly ideal" as a shortcut for the label.
DEFAULT_JITTER = 0.04


def existing_coverage(session: Session, category_key: str) -> Counter:
    """How many control sets exist per generator key for a category."""
    rows = session.execute(
        select(ComparisonSet.generator_key, func.count())
        .where(
            ComparisonSet.category_key == category_key,
            ComparisonSet.provenance_key == "control",
            ComparisonSet.generator_key.is_not(None),
        )
        .group_by(ComparisonSet.generator_key)
    ).all()
    return Counter({key: int(count) for key, count in rows})


def uncovered(registry: Registry, session: Session, category_key: str) -> list[str]:
    """Sweepable indicators no control set currently varies on its own.

    Matched on the generator key. A dataset ingested from outside may label its
    generators with names of its own that do not correspond to indicator keys,
    in which case an indicator can look uncovered when it is not -- so the
    resolved list is always printed before anything is generated.
    """
    covered = set(existing_coverage(session, category_key))
    return [key for key in registry.sweepable(category_key) if key not in covered]


def sample_values(
    registry: Registry,
    category_key: str,
    indicator_key: str,
    n_alternatives: int,
    rng: random.Random,
) -> list:
    """Values for one case: spread across the declared range, order shuffled.

    Anchoring the extremes guarantees the case actually spans the range rather
    than clustering by chance, which is what makes the relationship learnable
    from a handful of alternatives.
    """
    indicator = registry.indicator(indicator_key)
    if indicator.is_scale:
        levels = [level.key for level in indicator.levels]
        if n_alternatives >= len(levels):
            chosen = levels[:]
        else:
            chosen = rng.sample(levels, n_alternatives)
        rng.shuffle(chosen)
        return chosen

    spec = registry.category(category_key).members[indicator_key].reference_range
    low, high = spec.ref_low, spec.ref_high
    span = high - low

    values = [low, high][:n_alternatives]
    while len(values) < n_alternatives:
        values.append(low + span * rng.random())
    # Nudge the anchors inward a little so every case is not identical at the
    # endpoints.
    values = [
        min(high, max(low, value + rng.uniform(-0.02, 0.02) * span))
        for value in values
    ]
    rng.shuffle(values)
    return values


def generate(
    registry: Registry,
    category_key: str,
    indicators: list[str],
    count: int,
    min_alts: int = DEFAULT_MIN_ALTS,
    max_alts: int = DEFAULT_MAX_ALTS,
    jitter: float = DEFAULT_JITTER,
    seed: int = 0,
) -> list[parametric.ProbeCase]:
    """Build ``count`` cases per indicator, spread over contexts and stakeholders."""
    rng = random.Random(seed)
    category = registry.category(category_key)
    stakeholder_keys = sorted(registry.stakeholders)

    cases: list[parametric.ProbeCase] = []
    for indicator_key in indicators:
        # Only contexts under which this indicator is both relevant and
        # directed; elsewhere the sweep would assert nothing.
        usable = [
            context_key
            for context_key in sorted(category.available_contexts)
            if indicator_key
            in parametric.varied_indicators(registry, category_key, [context_key])
        ]
        if not usable:
            continue

        for index in range(count):
            context_key = usable[index % len(usable)]
            stakeholder_key = stakeholder_keys[index % len(stakeholder_keys)]
            n_alternatives = rng.randint(min_alts, max_alts)
            values = sample_values(
                registry, category_key, indicator_key, n_alternatives, rng
            )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("category")
    parser.add_argument("--db", default=None)
    parser.add_argument("--registry-version", default=None)
    parser.add_argument(
        "--indicator",
        action="append",
        dest="indicators",
        help="indicator to sweep; repeatable. Defaults to every sweepable one.",
    )
    parser.add_argument(
        "--only-uncovered",
        action="store_true",
        help="restrict to sweepable indicators no control set currently varies",
    )
    parser.add_argument("--count", type=int, default=1000, help="cases per indicator")
    parser.add_argument("--min-alts", type=int, default=DEFAULT_MIN_ALTS)
    parser.add_argument("--max-alts", type=int, default=DEFAULT_MAX_ALTS)
    parser.add_argument("--jitter", type=float, default=DEFAULT_JITTER)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-key", default=GENERATED_SOURCE)
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be generated"
    )
    args = parser.parse_args()

    from core import registry as registry_module

    with session_scope(create_db_engine(args.db)) as session:
        if args.registry_version:
            registry = registry_module.from_release(session, args.registry_version)
        else:
            registry = registry_module.from_session(session)

        sweepable = set(registry.sweepable(args.category))
        coverage = existing_coverage(session, args.category)

        if args.indicators:
            requested = list(args.indicators)
            refused = [key for key in requested if key not in sweepable]
            if refused:
                raise SystemExit(
                    f"the registry does not declare {refused} sweepable for "
                    f"{args.category!r}. Generating control cases for them would "
                    "assert a relationship the registry says does not hold across "
                    "the range. Change the declaration, or leave them out."
                )
        elif args.only_uncovered:
            requested = uncovered(registry, session, args.category)
        else:
            requested = sorted(sweepable)

        print(f"category {args.category}, registry {registry.version}")
        print(f"  sweepable      : {len(sweepable)}")
        print(f"  excluded       : {len(registry.category(args.category).token_order) - len(sweepable)}")
        print("  existing control coverage by generator key:")
        for key, total in sorted(coverage.items()):
            print(f"      {key:24s} {total:>7,}")
        print(f"  generating for : {requested}")

        if not requested:
            print("nothing to generate")
            return

        cases = generate(
            registry,
            args.category,
            requested,
            count=args.count,
            min_alts=args.min_alts,
            max_alts=args.max_alts,
            jitter=args.jitter,
            seed=args.seed,
        )
        print(f"  cases built    : {len(cases):,}")

        if args.dry_run:
            print("dry run; nothing written")
            return

        counts = write_cases(
            session, registry, cases, source_key=args.source_key, prefix="gen"
        )

    for name, total in counts.items():
        print(f"{name:20s} {total:>10,}")


if __name__ == "__main__":
    main()
