"""Inference API.

Submit a shortlist and a context, receive a score per alternative. Every
response echoes the category's eligibility precondition: the model ranks
options assumed to have passed regulatory checks.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from core.registry import Registry
from serve import limits, partners, release
from serve.engine import ServedModel
from serve.explore import analysis
from serve.shortlist import ShortlistRequest, resolve

MODEL_ENV = "RECOMMENDER_MODEL"
COMMIT_ENV = "VERCEL_GIT_COMMIT_SHA"
CORS_ENV = "RECOMMENDER_CORS_ORIGINS"


class ScoredAlternative(BaseModel):
    id: str
    score: float
    rank: int
    missing_indicators: list[str]
    disqualifying_levels: list[str]


class ScoreResponse(BaseModel):
    category: str
    functional_unit: str
    eligibility_precondition: str
    contexts: list[str]
    stakeholders: list[str]
    registry_version: str | None
    model_snapshot: str | None
    results: list[ScoredAlternative]
    notes: list[str]


class Service:

    def __init__(self, model_path: Path):
        served = ServedModel(model_path)
        self.scorer = served.scorer
        self.registry: Registry = served.registry
        self.registry_version = served.registry_version
        self.snapshot_hash = served.snapshot_hash

    def score(self, request: ShortlistRequest) -> ScoreResponse:
        shortlist = resolve(self.registry, request)
        scores = analysis.score(
            self.scorer,
            self.registry,
            shortlist.category_key,
            shortlist.alternatives,
            shortlist.contexts,
            shortlist.stakeholders,
        )

        order = np.argsort(-scores)
        rank_of = {int(index): rank + 1 for rank, index in enumerate(order)}

        results = [
            ScoredAlternative(
                id=alternative.key,
                score=round(float(scores[position]), 4),
                rank=rank_of[position],
                missing_indicators=shortlist.missing[position],
                disqualifying_levels=shortlist.disqualifying[position],
            )
            for position, alternative in enumerate(shortlist.alternatives)
        ]

        notes: list[str] = []
        if any(result.disqualifying_levels for result in results):
            notes.append(
                "One or more alternatives sit at a level the registry marks as "
                "never selectable. This is reported, not enforced: the score is "
                "the model's, the judgement is yours."
            )

        category = shortlist.category
        return ScoreResponse(
            category=shortlist.category_key,
            functional_unit=category.functional_unit_display,
            eligibility_precondition=category.eligibility_precondition_text,
            contexts=shortlist.contexts,
            stakeholders=shortlist.stakeholders,
            registry_version=self.registry_version,
            model_snapshot=self.snapshot_hash,
            results=results,
            notes=notes,
        )


DEFAULT_MODEL = release.RELEASE_DIR / release.SERVED_MODEL_FILE
FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "dist"


def create_app(
    model_path: Path | str | None = None,
    frontend: Path | None = FRONTEND,
) -> FastAPI:
    """The API under /api, and the frontend build at every other path."""
    path = Path(model_path or os.environ.get(MODEL_ENV) or DEFAULT_MODEL)
    service: dict[str, Service] = {}
    metered = Depends(
        limits.gate(
            limits.from_env(), limits.partner_limiter_from_env(), partners.from_env()
        )
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service["instance"] = Service(path)
        yield
        service.clear()

    app = FastAPI(
        title="Context-adaptive product recommender",
        description=__doc__,
        version="0.1.0",
        lifespan=lifespan,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        generate_unique_id_function=lambda route: route.name,
    )
    origins = [o.strip() for o in os.environ.get(CORS_ENV, "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
            expose_headers=["Retry-After"],
        )

    api = APIRouter(prefix="/api")

    def _service() -> Service:
        if "instance" not in service:
            raise HTTPException(status_code=503, detail="model not loaded")
        return service["instance"]

    @app.get("/explore/", include_in_schema=False)
    def former_page_address() -> RedirectResponse:
        return RedirectResponse(url="/", status_code=301)

    @api.get("/health")
    def health() -> dict[str, Any]:
        instance = _service()
        return {
            "status": "ok",
            "registry_version": instance.registry_version,
            "snapshot": instance.snapshot_hash,
            "categories": sorted(instance.registry.categories),
            "commit": os.environ.get(COMMIT_ENV),
        }

    @api.get("/categories")
    def categories() -> list[dict[str, Any]]:
        """What can be compared, and what each category assumes about its input."""
        registry = _service().registry
        return [
            {
                "key": key,
                "display_name": category.display_name,
                "preview": category.is_preview,
                "functional_unit": category.functional_unit_display,
                "functional_unit_symbol": category.functional_unit_symbol,
                "eligibility_precondition": category.eligibility_precondition_text,
                "contexts": sorted(category.available_contexts),
                "default_context": category.default_context_key,
                "indicators": list(category.token_order),
            }
            for key, category in sorted(registry.categories.items())
        ]

    @api.get("/categories/{category_key}/indicators")
    def indicators(category_key: str) -> list[dict[str, Any]]:
        registry = _service().registry
        try:
            category = registry.category(category_key)
        except LookupError:
            raise HTTPException(status_code=404, detail=f"unknown category {category_key!r}")

        out = []
        for key in category.token_order:
            indicator = registry.indicator(key)
            member = category.members[key]
            spec = member.reference_range
            out.append(
                {
                    "key": key,
                    "display_name": indicator.display_name,
                    "family": indicator.family_key,
                    "value_type": indicator.value_type,
                    "unit": indicator.unit,
                    "definition": indicator.definition_text,
                    "derived_from": [source for source, _ in indicator.sources],
                    "reference_range": None
                    if spec is None
                    else {"low": spec.ref_low, "high": spec.ref_high, "scale": spec.scale},
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
        return out

    @api.get("/stakeholders")
    def stakeholders() -> list[dict[str, Any]]:
        registry = _service().registry
        return [
            {
                "key": key,
                "display_name": stakeholder.display_name,
                "definition": stakeholder.definition_text,
                "domain_scope": stakeholder.domain_scope_text,
            }
            for key, stakeholder in sorted(registry.stakeholders.items())
        ]

    @api.post("/score", response_model=ScoreResponse, dependencies=[metered])
    def score(request: ShortlistRequest) -> ScoreResponse:
        return _service().score(request)

    from serve.explore.api import build_router

    api.include_router(build_router(_service, metered=metered))

    @api.get("/{path:path}", include_in_schema=False)
    def unknown(path: str) -> None:
        """Keeps the frontend's fallback page off /api."""
        raise HTTPException(status_code=404, detail=f"no endpoint /api/{path}")

    app.include_router(api)

    if frontend is not None:
        app.frontend("/", directory=frontend, fallback="index.html", check_dir=True)

    return app
