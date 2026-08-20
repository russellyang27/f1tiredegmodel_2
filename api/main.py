"""
FastAPI wrapper around the real Random Forest tire-degradation model.

Verified against the actual project source you provided
(f1_tire_model.models.random_forest, f1_tire_model.models.tire_report,
f1_tire_model.models.tire_life) -- the signatures and logic below are
written directly against that source, not guesswork, with two exceptions
flagged below.

Design note: generate_tire_report() / TireLifeEstimator are your validated
"one number" report layer (a single life-estimate per compound, with
fractional-lap interpolation). This endpoint needs the FULL per-age curve
for the frontend chart, which that report layer doesn't expose -- so this
does its own single age-sweep per compound instead of calling
generate_tire_report(), but deliberately mirrors its exact precedence
rules (default_covariates() for circuit+compound-specific defaults,
explicit request values always override those defaults, is_raining
injected the same way generate_tire_report injects extra_covariates) so
this endpoint's numbers should match what generate_tire_report would say
for the same inputs.

Confirmed against your actual source for all four modules this touches
(random_forest.py, tire_report.py, tire_life.py, and now storage.py) --
nothing below is guesswork anymore except your final deployed domain.

Run it:
    pip install fastapi "uvicorn[standard]" --break-system-packages
    uvicorn api.main:app --reload --port 8000

Then point the frontend at it:
    # .env.local
    NEXT_PUBLIC_API_URL=http://localhost:8000
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from f1_tire_model.config import DEFAULT_SETTINGS
from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.models.random_forest import (
    DEFAULT_CONTINUOUS_FEATURES,
    RandomForestTireDegradationModel,
)
from f1_tire_model.models.tire_report import default_covariates

MAX_AGE_LAPS = 40
DRIVER_FEATURES = ("avg_braking_decel_ms2", "peak_braking_decel_ms2", "avg_throttle_pct")

CompoundId = Literal["SOFT", "MEDIUM", "HARD"]


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
# Model loading -- fit once at process startup, cache in memory. Matches the
# validated production config from random_forest.py's own module docstring:
# fuel_load_estimate_kg excluded (age-inconsistency bug), driver-behavior
# features included, fit on all data (no held-out split here -- evaluation
# happens separately/offline per your project's own workflow).
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_model_and_data() -> tuple[RandomForestTireDegradationModel, pd.DataFrame]:
    df = load_dataset()  # defaults to Settings.dataset_output_dir -- confirmed correct against your real storage.py

    if df.empty:
        # load_dataset() returns an empty frame rather than raising when no
        # .parquet fragments exist (see its own docstring) -- almost always
        # means the dataset directory wasn't included in this deployment.
        # Fail loudly here instead of letting sklearn raise a confusing
        # error several calls deeper.
        raise RuntimeError(
            "load_dataset() returned no rows -- no .parquet fragments found in "
            "Settings.dataset_output_dir. Make sure the built dataset directory is "
            "actually present in this deployment (it won't be included automatically "
            "by a typical git-based deploy unless committed or fetched at build time)."
        )

    continuous_no_fuel = tuple(f for f in DEFAULT_CONTINUOUS_FEATURES if f != "fuel_load_estimate_kg")
    model = RandomForestTireDegradationModel(
        continuous_features=continuous_no_fuel + DRIVER_FEATURES,
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
    ).fit(df)

    return model, df


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
    model, df = get_model_and_data()

    if req.circuit not in df["circuit"].unique():
        raise HTTPException(
            status_code=400,
            detail=f"Circuit '{req.circuit}' was not seen in training -- this model can only "
            "predict for circuits it has multi-season history for.",
        )

    ages = np.arange(1, MAX_AGE_LAPS + 1)
    results: list[CompoundResult] = []

    for compound in req.compounds:
        # Circuit+compound-specific defaults for whatever the model needs
        # beyond track_temp_c/circuit/compound/age (driver-behavior
        # features) -- see default_covariates()'s own docstring for why
        # this is restricted to circuit+compound rather than blended
        # across all compounds (the same class of bug as fuel load).
        covariates = default_covariates(df, model, circuit=req.circuit, compound=compound)

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
