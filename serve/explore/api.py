"""Endpoints behind the comparison page."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from serve.explore import analysis, compare as compare_module
from serve.explore.compare import Difference
from serve.shortlist import ShortlistRequest, resolve


class Level(BaseModel):
    key: str
    display_name: str
    disqualifying: bool


class ReferenceRange(BaseModel):
    low: float
    high: float


class FormField(BaseModel):
    key: str
    display_name: str
    family: str
    unit: str | None
    value_type: str
    definition: str
    direction: int
    range: ReferenceRange | None
    levels: list[Level]


class FormContext(BaseModel):
    key: str
    display_name: str
    definition: str


class FormCategory(BaseModel):
    key: str
    display_name: str
    preview: bool = Field(
        description="Trained partly on synthetic labels; scores restate the declared rules."
    )
    functional_unit: str
    functional_unit_symbol: str
    eligibility_precondition: str
    default_context: str
    contexts: list[FormContext]
    fields: list[FormField]


class FormStakeholder(BaseModel):
    key: str
    display_name: str
    definition: str


class Form(BaseModel):
    registry_version: str | None
    snapshot: str | None
    categories: list[FormCategory]
    stakeholders: list[FormStakeholder]
    families: dict[str, str]


class HeadToHead(BaseModel):
    rival: str
    rival_index: int
    rival_score: float
    gap: float
    decisive: list[Difference]
    contributing: list[Difference]
    rival_wins: list[Difference]
    summary: str


class Comparison(BaseModel):
    wins: dict[str, list[str]] = Field(
        description="Per alternative index, the indicators it is credited with winning on."
    )
    leader: str
    leader_index: int
    leader_score: float
    scores: list[float]
    comparisons: list[HeadToHead]


class SensitivityRow(BaseModel):
    label: str = Field(description="The context or stakeholder key this row varies.")
    scores: list[float]
    winner: int


class Sensitivity(BaseModel):
    axis: str
    alternatives: list[str]
    rows: list[SensitivityRow]
    changes_winner: bool


def build_router(service_getter, metered: Any) -> APIRouter:
    router = APIRouter(prefix="/explore", tags=["explore"])
    costly = [metered]

    @router.get("/form")
    def form() -> Form:
        """Everything needed to draw the input grid and the selectors."""
        service = service_getter()
        registry = service.registry

        categories = []
        for key, category in sorted(registry.categories.items()):
            fields = []
            for indicator_key in category.token_order:
                indicator = registry.indicator(indicator_key)
                if indicator.is_derived:
                    continue
                spec = category.members[indicator_key].reference_range
                fields.append(
                    {
                        "key": indicator_key,
                        "display_name": indicator.display_name,
                        "family": indicator.family_key,
                        "unit": category.indicator_unit(indicator.unit),
                        "value_type": indicator.value_type,
                        "definition": indicator.definition_text,
                        "direction": indicator.default_direction,
                        "range": None
                        if spec is None
                        else {"low": spec.ref_low, "high": spec.ref_high},
                        "levels": [
                            {
                                "key": level.key,
                                "display_name": level.display_name,
                                "disqualifying": level.is_disqualifying,
                            }
                            for level in indicator.levels
                        ],
                    }
                )
            categories.append(
                {
                    "key": key,
                    "display_name": category.display_name,
                    "preview": category.is_preview,
                    "functional_unit": category.functional_unit_display,
                    "functional_unit_symbol": category.functional_unit_symbol,
                    "eligibility_precondition": category.eligibility_precondition_text,
                    "default_context": category.default_context_key,
                    "contexts": [
                        {
                            "key": context_key,
                            "display_name": registry.contexts[context_key].display_name,
                            "definition": registry.contexts[context_key].definition_text,
                        }
                        for context_key in sorted(category.available_contexts)
                    ],
                    "fields": fields,
                }
            )

        return {
            "registry_version": service.registry_version,
            "snapshot": service.snapshot_hash,
            "categories": categories,
            "stakeholders": [
                {
                    "key": key,
                    "display_name": stakeholder.display_name,
                    "definition": stakeholder.definition_text,
                }
                for key, stakeholder in sorted(registry.stakeholders.items())
            ],
            "families": {key: key.replace("_", " ") for key in registry.families},
        }

    @router.post("/compare", dependencies=costly)
    def compare(request: ShortlistRequest) -> Comparison:
        """Why the leading alternative is ahead of each of the others."""
        service = service_getter()
        shortlist = resolve(service.registry, request)
        return compare_module.compare(
            service.scorer, service.registry, shortlist.category_key,
            shortlist.alternatives, shortlist.contexts, shortlist.stakeholders,
        )

    @router.get("/response", dependencies=costly)
    def response(
        category: str,
        indicator: str,
        context: str | None = None,
        stakeholders: str | None = None,
        steps: int = analysis.DEFAULT_STEPS,
    ) -> dict[str, Any]:
        service = service_getter()
        registry = service.registry
        try:
            spec = registry.category(category)
        except LookupError:
            raise HTTPException(status_code=404, detail=f"unknown category {category!r}")
        if indicator not in spec.members:
            raise HTTPException(
                status_code=404,
                detail=f"{category!r} has no indicator {indicator!r}",
            )

        contexts = [context] if context else [spec.default_context_key]
        if contexts[0] not in spec.available_contexts:
            raise HTTPException(
                status_code=400, detail=f"context {contexts[0]!r} is not available"
            )

        chosen = (
            [s for s in stakeholders.split(",") if s]
            if stakeholders
            else sorted(registry.stakeholders)
        )
        unknown = [s for s in chosen if s not in registry.stakeholders]
        if unknown:
            raise HTTPException(
                status_code=400, detail=f"unknown stakeholder(s) {unknown}"
            )

        return analysis.response_curve(
            service.scorer,
            registry,
            category,
            indicator,
            contexts,
            chosen,
            steps=max(3, min(steps, 61)),
        )

    @router.post("/context-sensitivity", dependencies=costly)
    def context_sensitivity(request: ShortlistRequest) -> Sensitivity:
        service = service_getter()
        shortlist = resolve(service.registry, request)
        return analysis.context_sensitivity(
            service.scorer, service.registry, shortlist.category_key,
            shortlist.alternatives, shortlist.stakeholders,
        )

    @router.post("/stakeholder-sensitivity", dependencies=costly)
    def stakeholder_sensitivity(request: ShortlistRequest) -> Sensitivity:
        service = service_getter()
        shortlist = resolve(service.registry, request)
        return analysis.stakeholder_sensitivity(
            service.scorer, service.registry, shortlist.category_key,
            shortlist.alternatives, shortlist.contexts,
        )

    return router
