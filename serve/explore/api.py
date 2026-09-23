"""Endpoints behind the comparison tool.

Data is served as JSON and the page separately, so the same endpoints feed a
browser, a notebook and a figure script.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.encoding import AlternativeInput
from serve import limits
from serve.explore import analysis, compare as compare_module


class AlternativePayload(BaseModel):
    id: str
    values: dict[str, float] = Field(default_factory=dict)
    levels: dict[str, str] = Field(default_factory=dict)


class ShortlistRequest(BaseModel):
    category: str
    context: list[str] = Field(default_factory=list)
    stakeholders: list[str] = Field(default_factory=list)
    alternatives: list[AlternativePayload] = Field(
        default_factory=list, max_length=limits.MAX_ALTERNATIVES
    )


def build_router(
    service_getter, device: str = "cpu", metered: Any | None = None
) -> APIRouter:
    router = APIRouter(prefix="/explore", tags=["explore"])
    if metered is None:
        metered = Depends(limits.gate(limits.from_env()))
    costly = [metered]

    def _resolve(request: ShortlistRequest):
        service = service_getter()
        registry = service.registry
        try:
            category = registry.category(request.category)
        except LookupError:
            raise HTTPException(
                status_code=404, detail=f"unknown category {request.category!r}"
            )

        contexts = request.context or [category.default_context_key]
        unavailable = [c for c in contexts if c not in category.available_contexts]
        if unavailable:
            raise HTTPException(
                status_code=400,
                detail=f"context(s) {unavailable} are not available for "
                f"{request.category!r}",
            )

        stakeholders = request.stakeholders or [sorted(registry.stakeholders)[0]]
        unknown = [s for s in stakeholders if s not in registry.stakeholders]
        if unknown:
            raise HTTPException(
                status_code=400, detail=f"unknown stakeholder(s) {unknown}"
            )

        held = set(category.token_order)
        alternatives = [
            AlternativeInput(
                key=a.id,
                values={k: v for k, v in a.values.items() if k in held},
                levels={k: v for k, v in a.levels.items() if k in held},
            )
            for a in request.alternatives
        ]
        if not alternatives:
            alternatives = analysis.example_shortlist(
                registry, request.category, contexts
            )

        return service, registry, contexts, stakeholders, alternatives

    @router.get("/form")
    def form() -> dict[str, Any]:
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
                        "unit": indicator.unit,
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
                    "functional_unit": category.functional_unit_display,
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
            "registry_version": service.meta.registry_version,
            "snapshot": service.meta.snapshot_hash,
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
    def compare(request: ShortlistRequest) -> dict[str, Any]:
        """Why the leading alternative is ahead of each of the others."""
        service, registry, contexts, stakeholders, alternatives = _resolve(request)
        if len(alternatives) < 2:
            raise HTTPException(
                status_code=400, detail="a comparison needs at least two alternatives"
            )
        return compare_module.compare(
            service.model, registry, request.category, alternatives,
            contexts, stakeholders, device,
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
                detail=f"{category!r} does not hold indicator {indicator!r}",
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
            service.model,
            registry,
            category,
            indicator,
            contexts,
            chosen,
            steps=max(3, min(steps, 61)),
            device=device,
        )

    @router.post("/context-sensitivity", dependencies=costly)
    def context_sensitivity(request: ShortlistRequest) -> dict[str, Any]:
        service, registry, _, stakeholders, alternatives = _resolve(request)
        return analysis.context_sensitivity(
            service.model, registry, request.category, alternatives, stakeholders, device
        )

    @router.post("/stakeholder-sensitivity", dependencies=costly)
    def stakeholder_sensitivity(request: ShortlistRequest) -> dict[str, Any]:
        service, registry, contexts, _, alternatives = _resolve(request)
        return analysis.stakeholder_sensitivity(
            service.model, registry, request.category, alternatives, contexts, device
        )

    return router
