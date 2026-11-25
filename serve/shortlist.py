"""A shortlist request, checked against the served registry before it reaches the model."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from fastapi import HTTPException
from pydantic import BaseModel, Field

from core.encoding import AlternativeInput
from core.registry import CategorySpec, Registry
from serve import limits


class AlternativePayload(BaseModel):
    id: str = Field(description="Caller's identifier for this alternative, unique in the shortlist")
    values: dict[str, float] = Field(
        default_factory=dict,
        description="Numeric indicator values. Omit an indicator you have no "
        "value for; omission means unknown and is never imputed.",
    )
    levels: dict[str, str] = Field(
        default_factory=dict,
        description="Level keys for indicators declared on an ordered or "
        "unordered scale.",
    )


class ShortlistRequest(BaseModel):
    category: str
    context: list[str] = Field(
        default_factory=list,
        description="Active application contexts, at most the category's "
        "max_contexts from /api/explore/form. Empty means the category's "
        "baseline, which never combines with another context.",
    )
    stakeholders: list[str] = Field(
        min_length=1,
        description="Whose priorities count, as keys from /api/stakeholders; at most the "
        "category's max_stakeholders from /api/explore/form.",
    )
    alternatives: list[AlternativePayload] = Field(
        min_length=limits.MIN_ALTERNATIVES,
        max_length=limits.MAX_ALTERNATIVES,
        description="The shortlist to rank.",
    )


@dataclass(frozen=True)
class Shortlist:
    category_key: str
    category: CategorySpec
    contexts: list[str]
    stakeholders: list[str]
    alternatives: list[AlternativeInput]
    missing: list[list[str]]
    disqualifying: list[list[str]]


def resolve(
    registry: Registry,
    request: ShortlistRequest,
    max_contexts: Mapping[str, int],
    max_stakeholders: Mapping[str, int],
) -> Shortlist:
    try:
        category = registry.category(request.category)
    except LookupError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown category {request.category!r}; "
                f"available: {sorted(registry.categories)}"
            ),
        )

    contexts = checked_contexts(category, request.context, max_contexts[request.category])
    _refuse_unlisted("stakeholder(s)", request.stakeholders, registry.stakeholders, request.category)
    checked_stakeholder_count(request.category, request.stakeholders, max_stakeholders[request.category])

    held = set(category.token_order)
    supplied = [set(a.values) | set(a.levels) for a in request.alternatives]
    _refuse_unlisted("indicator(s)", set().union(*supplied), held, request.category)

    repeated = sorted(key for key, n in Counter(a.id for a in request.alternatives).items() if n > 1)
    if repeated:
        raise HTTPException(
            status_code=400, detail=f"alternative ids must be unique; repeated: {repeated}"
        )

    derived = {key for key in held if registry.indicator(key).is_derived}
    return Shortlist(
        category_key=request.category,
        category=category,
        contexts=list(contexts),
        stakeholders=list(request.stakeholders),
        alternatives=[
            AlternativeInput(key=a.id, values=dict(a.values), levels=dict(a.levels))
            for a in request.alternatives
        ],
        missing=[sorted(held - given - derived) for given in supplied],
        disqualifying=[_disqualifying(registry, a.levels) for a in request.alternatives],
    )


def checked_contexts(category: CategorySpec, given: Sequence[str], limit: int) -> list[str]:
    """The contexts to score under: the baseline when none are given, otherwise as trained."""
    if not given:
        return [category.default_context_key]
    _refuse_unlisted("context(s)", given, category.available_contexts, category.key)
    if len(set(given)) < len(given):
        raise HTTPException(status_code=400, detail=f"a context is repeated: {list(given)}")
    if category.default_context_key in given and len(given) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"the baseline {category.default_context_key!r} stands alone; "
            "leave the context list empty for it",
        )
    if len(given) > limit:
        raise HTTPException(
            status_code=400,
            detail=f"{category.key!r} combines at most {limit} context(s), the most "
            f"its training data combined; got {len(given)}",
        )
    return list(given)


def checked_stakeholder_count(category_key: str, given: Sequence[str], limit: int) -> None:
    if len(set(given)) < len(given):
        raise HTTPException(status_code=400, detail=f"a stakeholder is repeated: {list(given)}")
    if len(given) > limit:
        raise HTTPException(
            status_code=400,
            detail=f"{category_key!r} combines at most {limit} stakeholder(s), the most "
            f"its training data combined; got {len(given)}",
        )


def _refuse_unlisted(
    noun: str, given: Iterable[str], allowed: Iterable[str], category_key: str
) -> None:
    allowed = set(allowed)
    unlisted = sorted(set(given) - allowed)
    if unlisted:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{noun} {unlisted} not available for {category_key!r}; "
                f"available: {sorted(allowed)}"
            ),
        )


def _disqualifying(registry: Registry, levels: dict[str, str]) -> list[str]:
    found = []
    for indicator_key, level_key in levels.items():
        indicator = registry.indicator(indicator_key)
        try:
            level = indicator.level(level_key)
        except LookupError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{indicator_key} has no level {level_key!r}; "
                    f"declared levels: {[lv.key for lv in indicator.levels]}"
                ),
            )
        if level.is_disqualifying:
            found.append(f"{indicator_key}={level_key}")
    return found
