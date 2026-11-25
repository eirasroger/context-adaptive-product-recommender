"""Endpoints behind the comparison page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.scoring import TIE_EPSILON
from serve.explore import analysis, compare as compare_module
from serve.explore.compare import Difference
from serve.shortlist import ShortlistRequest, checked_contexts, resolve


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
    theme: str | None = Field(description="The group the page shows this context under.")


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
    max_contexts: int = Field(description="Most contexts one request may combine.")
    max_stakeholders: int = Field(description="Most stakeholders one request may combine.")
    contexts: list[FormContext] = Field(description="In display order.")
    fields: list[FormField]


class FormStakeholder(BaseModel):
    key: str
    display_name: str
    definition: str


EXAMPLES = Path(__file__).parent / "examples"


class Example(BaseModel):
    title: str
    request: ShortlistRequest


def examples() -> list[Example]:
    return [
        Example.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(EXAMPLES.glob("*.json"))
    ]


class Form(BaseModel):
    registry_version: str | None
    snapshot: str | None
    tie_epsilon: float = Field(
        description="Score difference below which two alternatives count as tied."
    )
    categories: list[FormCategory]
    stakeholders: list[FormStakeholder]
    examples: list[Example] = Field(description="Worked shortlists the page offers to load.")
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
                    "max_contexts": service.max_contexts[key],
                    "max_stakeholders": service.max_stakeholders[key],
                    "contexts": [
                        {
                            "key": context_key,
                            "display_name": shown.display_name,
                            "definition": shown.definition_text,
                            "theme": shown.theme,
                        }
                        for context_key, shown in category.context_presentation.items()
                    ],
                    "fields": fields,
                }
            )

        return {
            "registry_version": service.registry_version,
            "snapshot": service.snapshot_hash,
            "tie_epsilon": TIE_EPSILON,
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
            "examples": examples(),
        }

    @router.post("/compare", dependencies=costly)
    def compare(request: ShortlistRequest) -> Comparison:
        """Why the leading alternative is ahead of each of the others."""
        service = service_getter()
        shortlist = resolve(service.registry, request, service.max_contexts, service.max_stakeholders)
        return compare_module.compare(
            service.scorer, service.registry, shortlist.category_key,
            shortlist.alternatives, shortlist.contexts, shortlist.stakeholders,
        )

    @router.get("/response", dependencies=costly)
    def response(
        category: str,
        indicator: str,
        context: str | None = Query(
            None, description="Comma-separated context keys; empty means the baseline."
        ),
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

        contexts = checked_contexts(
            spec, [c for c in (context or "").split(",") if c], service.max_contexts[category]
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
        shortlist = resolve(service.registry, request, service.max_contexts, service.max_stakeholders)
        return analysis.context_sensitivity(
            service.scorer, service.registry, shortlist.category_key,
            shortlist.alternatives, shortlist.stakeholders,
        )

    @router.post("/stakeholder-sensitivity", dependencies=costly)
    def stakeholder_sensitivity(request: ShortlistRequest) -> Sensitivity:
        service = service_getter()
        shortlist = resolve(service.registry, request, service.max_contexts, service.max_stakeholders)
        return analysis.stakeholder_sensitivity(
            service.scorer, service.registry, shortlist.category_key,
            shortlist.alternatives, shortlist.contexts,
        )

    return router
