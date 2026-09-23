"""Why one alternative is ahead of another.

The question a person actually has is never "what is this alternative worth
against an average product". It is "why this one and not that one", which is a
comparison between two specific things on the table.

So for every indicator where the two genuinely differ, hand the leader the
rival's value for that one indicator, leave everything else alone, and re-score.
If the lead survives, that indicator was not decisive. If it collapses, it was.

Every row is a real score from the real model. Nothing is sampled or
approximated, and the whole set runs as one batched forward pass.

The limit worth knowing: one indicator changes at a time, so a lead that rests
on two indicators together shows no single culprit. That case is reported rather
than papered over.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from core.encoding import AlternativeInput
from core.registry import Registry
from core.scoring import Scorer
from serve.explore.analysis import score, score_many

#: A gap change below this is noise from the model rather than an effect.
NEGLIGIBLE = 0.002

#: How many contributing indicators to name when nothing is decisive on its own.
NAMED_CAUSES = 2


@dataclass(frozen=True)
class Difference:
    indicator: str
    display_name: str
    family: str
    unit: str | None
    leader_value: str
    rival_value: str
    new_gap: float
    closes: float
    flips: bool
    favours: str


def _display(registry: Registry, indicator_key: str, value) -> str:
    if value is None:
        return "unknown"
    indicator = registry.indicator(indicator_key)
    if indicator.is_scale:
        try:
            return indicator.level(str(value)).display_name
        except LookupError:
            return str(value)
    return f"{value:g}"


def _value_of(alternative: AlternativeInput, key: str):
    if key in alternative.values:
        return alternative.values[key]
    return alternative.levels.get(key)


def _swap(leader: AlternativeInput, rival: AlternativeInput, key: str) -> AlternativeInput:
    values, levels = dict(leader.values), dict(leader.levels)
    values.pop(key, None)
    levels.pop(key, None)
    if key in rival.values:
        values[key] = rival.values[key]
    if key in rival.levels:
        levels[key] = rival.levels[key]
    return AlternativeInput(leader.key, values, levels)


def _summarise(rival: str, leader: str, head: dict) -> str:
    decisive = head["decisive"]
    contributing = head["contributing"]
    wins = head["rival_wins"]

    if decisive:
        names = " and ".join(d["display_name"].lower() for d in decisive[:2])
        first = decisive[0]
        text = (
            f"{rival} is behind only because of {names}. "
            f"If {leader} had {rival}'s {first['display_name'].lower()} "
            f"of {first['rival_value']}, {rival} would win."
        )
    elif contributing:
        names = " and ".join(
            c["display_name"].lower() for c in contributing[:NAMED_CAUSES]
        )
        text = (
            f"{rival} loses on {names}, and no single change would put it ahead."
        )
    else:
        text = f"Nothing on its own accounts for the gap to {rival}."

    if wins:
        names = ", ".join(w["display_name"].lower() for w in wins[:3])
        text += f" It does beat {leader} on {names}."
    return text


def _comparable(registry: Registry, indicator_key: str, value):
    """A number where larger is better, so alternatives can be ranked on it."""
    if value is None:
        return None
    indicator = registry.indicator(indicator_key)
    if indicator.is_scale:
        try:
            return indicator.level(str(value)).normalised_position
        except LookupError:
            return None
    return float(value)


def _wins(
    registry: Registry,
    category_key: str,
    alternatives: Sequence[AlternativeInput],
    context_keys: Sequence[str],
    comparisons: Sequence[dict],
) -> dict[str, list[str]]:
    """Which indicators each alternative is genuinely best on.

    The counterfactuals say which indicators moved the result. The registry's
    declared direction says who is actually best at each one. Using the
    counterfactuals for both would report a pairwise result as though it were an
    outright claim, and an alternative would be credited with winning on cost
    while a cheaper one sat beside it.

    An indicator the active context gives no direction to is left out. Whatever
    the model does with it cannot be stated as better or worse.
    """
    effect: dict[str, float] = {}
    for head in comparisons:
        for group in ("decisive", "contributing", "rival_wins"):
            for entry in head[group]:
                key = entry["indicator"]
                effect[key] = max(effect.get(key, 0.0), abs(entry["closes"]))

    wins: dict[int, list[tuple[float, str]]] = {i: [] for i in range(len(alternatives))}

    for key, size in effect.items():
        direction, _ = registry.resolve_direction(category_key, key, context_keys)
        if direction == 0:
            continue

        ranked = [
            (index, _comparable(registry, key, _value_of(alternative, key)))
            for index, alternative in enumerate(alternatives)
        ]
        ranked = [(i, v) for i, v in ranked if v is not None]
        if len(ranked) < 2:
            continue

        best_value = max(v for _, v in ranked) if direction > 0 else min(v for _, v in ranked)
        holders = [i for i, v in ranked if v == best_value]
        if len(holders) != 1 or len(ranked) < len(alternatives):
            continue

        wins[holders[0]].append((size, registry.indicator(key).display_name))

    return {
        str(index): [name for _, name in sorted(entries, key=lambda e: -e[0])]
        for index, entries in wins.items()
        if entries
    }


def compare(
    scorer: Scorer,
    registry: Registry,
    category_key: str,
    alternatives: Sequence[AlternativeInput],
    context_keys: Sequence[str],
    stakeholder_keys: Sequence[str],
) -> dict:
    if len(alternatives) < 2:
        raise ValueError("a comparison needs at least two alternatives")

    scores = score(
        scorer, registry, category_key, alternatives, context_keys, stakeholder_keys
    )
    lead = int(np.argmax(scores))
    leader = alternatives[lead]
    category = registry.category(category_key)

    comparisons: list[dict] = []
    for index, rival in enumerate(alternatives):
        if index == lead:
            continue

        differing = [
            key
            for key in category.token_order
            if not registry.indicator(key).is_derived
            and _value_of(leader, key) != _value_of(rival, key)
        ]

        gap = float(scores[lead] - scores[index])
        differences: list[Difference] = []

        if differing:
            variants = []
            for key in differing:
                shortlist = list(alternatives)
                shortlist[lead] = _swap(leader, rival, key)
                variants.append(shortlist)

            lead_after = score_many(
                scorer, registry, category_key, variants, context_keys,
                stakeholder_keys, lead,
            )
            rival_after = score_many(
                scorer, registry, category_key, variants, context_keys,
                stakeholder_keys, index,
            )

            for key, new_gap in zip(differing, lead_after - rival_after):
                indicator = registry.indicator(key)
                closes = gap - float(new_gap)
                differences.append(
                    Difference(
                        indicator=key,
                        display_name=indicator.display_name,
                        family=indicator.family_key,
                        unit=indicator.unit,
                        leader_value=_display(registry, key, _value_of(leader, key)),
                        rival_value=_display(registry, key, _value_of(rival, key)),
                        new_gap=round(float(new_gap), 4),
                        closes=round(closes, 4),
                        flips=bool(new_gap < 0),
                        favours="leader" if closes > 0 else "rival",
                    )
                )

        decisive = sorted(
            (d for d in differences if d.flips), key=lambda d: -d.closes
        )
        contributing = sorted(
            (d for d in differences if not d.flips and d.closes > NEGLIGIBLE),
            key=lambda d: -d.closes,
        )
        rival_wins = sorted(
            (d for d in differences if d.closes < -NEGLIGIBLE), key=lambda d: d.closes
        )

        head = {
            "rival": rival.key,
            "rival_index": index,
            "rival_score": round(float(scores[index]), 4),
            "gap": round(gap, 4),
            "decisive": [asdict(d) for d in decisive],
            "contributing": [asdict(d) for d in contributing],
            "rival_wins": [asdict(d) for d in rival_wins],
        }
        head["summary"] = _summarise(rival.key, leader.key, head)
        comparisons.append(head)

    comparisons.sort(key=lambda c: -c["rival_score"])

    return {
        "wins": _wins(registry, category_key, alternatives, context_keys, comparisons),
        "leader": leader.key,
        "leader_index": lead,
        "leader_score": round(float(scores[lead]), 4),
        "scores": [round(float(v), 4) for v in scores],
        "comparisons": comparisons,
    }
