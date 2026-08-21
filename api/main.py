"""
FastAPI wrapper around the real Random Forest tire-degradation model.

Loads an ALREADY-FITTED model + precomputed defaults from disk (see
scripts/export_model.py) rather than fitting from raw data on every cold
boot -- see get_model_and_defaults() below for why that changed.

Verified against the actual project source you provided
(f1_tire_model.models.random_forest, f1_tire_model.models.tire_report,
f1_tire_model.models.tire_life, f1_tire_model.datasets.storage) -- the
prediction logic mirrors generate_tire_report()'s exact precedence rules
(circuit+compound-specific defaults, explicit request values always
override those defaults, is_raining injected the same way
generate_tire_report injects extra_covariates), just computed once ahead
of time via scripts/export_model.py instead of live per-request.

Run it locally:
    pip install -r api/requirements.txt
    python scripts/export_model.py   # from your project root, once
    uvicorn api.main:app --reload --port 8000

Then point the frontend at it:
    # .env.local
    NEXT_PUBLIC_API_URL=http://localhost:8000
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from f1_tire_model.config import DEFAULT_SETTINGS
from f1_tire_model.models.random_forest import RandomForestTireDegradationModel  # noqa: F401 -- needed so joblib can unpickle the model class

MAX_AGE_LAPS = 40

CompoundId = Literal["SOFT", "MEDIUM", "HARD"]

ARTIFACT_DIR = Path(__file__).parent
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
DEFAULTS_PATH = ARTIFACT_DIR / "defaults.joblib"


# ---------------------------------------------------------------------------
# Request / response schema -- mirrors types/circuit.ts exactly. Don't rename
# fields here without updating lib/api.ts on the frontend to match.
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    circuit: str
    track_temp_c: float = Field(ge=0, le=60)
    wet: bool
    threshold_s: float = Field(gt=0, le=10)
    compounds: list[CompoundId]


class CurvePoint(BaseModel):
    tire_age_laps: int
    delta_s: float
    extrapolating: bool


class CompoundResult(BaseModel):
    compound: CompoundId
    curve: list[CurvePoint]
    life_estimate_laps: int | None
    durable: bool
    warning: str | None


class PredictionResponse(BaseModel):
    circuit: str
    track_temp_c: float
    wet: bool
    threshold_s: float
    compounds: list[CompoundResult]


# ---------------------------------------------------------------------------
# Model loading -- loads an ALREADY-FITTED model + precomputed defaults from
# disk (see scripts/export_model.py), rather than refitting from raw parquet
# data on every cold boot. That refit-on-boot approach worked locally but was
# far too slow on a free-tier host's single shared vCPU -- slow enough that
# requests silently exceeded even a 50s client timeout with no server error
# at all (uvicorn only logs a request line after it finishes responding, so
# a stuck fit() produces zero log evidence, which is what made this tricky
# to diagnose). Loading a pre-fitted model is near-instant by comparison.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_model_and_defaults() -> tuple[RandomForestTireDegradationModel, dict]:
    if not MODEL_PATH.exists() or not DEFAULTS_PATH.exists():
        raise RuntimeError(
            f"{MODEL_PATH.name} / {DEFAULTS_PATH.name} not found in {ARTIFACT_DIR}. "
            "Run scripts/export_model.py locally (where your real dataset lives) and "
            "commit the two output files into this repo's api/ folder."
        )
    model = joblib.load(MODEL_PATH)
    lookup = joblib.load(DEFAULTS_PATH)  # {"circuits": [...], "defaults": {circuit: {compound: {...}}}}
    return model, lookup


app = FastAPI(title="F1 Tire Degradation Model API")

# Allow the Next.js dev server (and your deployed frontend) to call this.
# Allow the Next.js dev server and your deployed frontend to call this.
# NOTE: allow_origins requires EXACT origin strings -- it does not support
# wildcards like "https://*.netlify.app" (that was silently never matching
# anything in an earlier version of this file). Use allow_origin_regex for
# pattern matching, or just list your final Netlify URL explicitly once you
# know it -- exact-match is safer for a public API anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_origin_regex=r"https://.*\.netlify\.app",  # TODO: replace with your exact production domain once deployed
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    model, lookup = get_model_and_defaults()

    if req.circuit not in lookup["circuits"]:
        raise HTTPException(
            status_code=400,
            detail=f"Circuit '{req.circuit}' was not seen in training -- this model can only "
            "predict for circuits it has multi-season history for.",
        )

    ages = np.arange(1, MAX_AGE_LAPS + 1)
    results: list[CompoundResult] = []

    for compound in req.compounds:
        # Precomputed circuit+compound-specific defaults (see
        # scripts/export_model.py) -- same values default_covariates()
        # would compute live, just done once at export time instead of on
        # every request.
        covariates = dict(lookup["defaults"][req.circuit][compound])

        # Explicit request values always win over the auto-computed
        # defaults above -- same precedence generate_tire_report() uses
        # for extra_covariates.
        covariates["track_temp_c"] = req.track_temp_c
        covariates["circuit"] = req.circuit
        if "is_raining" in model.feature_columns:
            covariates["is_raining"] = req.wet

        query_df = pd.DataFrame(
            {
                **{k: [v] * len(ages) for k, v in covariates.items()},
                "tire_compound": [compound] * len(ages),
                "tire_age_laps": ages,
            }
        )

        # fuel_load_estimate_kg is excluded from the production feature set
        # above (see random_forest.py's module docstring), so this branch
        # shouldn't fire -- kept only so this endpoint stays correct if
        # that config ever changes. Mirrors generate_tire_report()'s
        # age-dependent fuel-load handling exactly: fuel decreases with
        # age using the same linear burn assumption the dataset itself
        # was built with, rather than being held at one static value
        # (the bug documented in docs/VALIDATION.md).
        if "fuel_load_estimate_kg" in model.feature_columns:
            query_df["fuel_load_estimate_kg"] = np.maximum(
                DEFAULT_SETTINGS.initial_fuel_kg - ages * DEFAULT_SETTINGS.fuel_burn_rate_kg_per_lap, 0.0
            )

        deltas = model.predict(query_df)
        extrap_mask = model.extrapolation_mask(query_df)

        curve = [
            CurvePoint(tire_age_laps=int(age), delta_s=round(float(delta), 3), extrapolating=bool(is_extrap))
            for age, delta, is_extrap in zip(ages, deltas, extrap_mask)
        ]

        life = next((p.tire_age_laps for p in curve if p.delta_s >= req.threshold_s), None)
        durable = life is None
        any_extrap = any(p.extrapolating for p in curve if life is None or p.tire_age_laps <= life)
        warning = (
            f"Sparse training data for {compound} at {req.circuit} in this condition range -- "
            "treat as a rough estimate."
            if any_extrap
            else None
        )

        results.append(
            CompoundResult(
                compound=compound,
                curve=curve,
                life_estimate_laps=life,
                durable=durable,
                warning=warning,
            )
        )

    return PredictionResponse(
        circuit=req.circuit,
        track_temp_c=req.track_temp_c,
        wet=req.wet,
        threshold_s=req.threshold_s,
        compounds=results,
    )
