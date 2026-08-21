"""
Fits the production RandomForestTireDegradationModel on your real dataset
and serializes it, along with a precomputed circuit+compound
default-covariates lookup table, so the deployed API (api/main.py) can just
LOAD these two files at startup instead of re-fitting a 200-tree forest
from raw parquet data on every cold boot.

Why this matters: Render's free tier gives you one shared, limited vCPU
(WEB_CONCURRENCY=1). Fitting from scratch there -- as api/main.py originally
did -- was slow enough to blow past even a 50-second client timeout, with
zero server-side error, just an endless wait. This removes the fit() call
from the request path entirely: the deployed process just deserializes an
already-fitted model, which takes a fraction of a second.

Run this locally, from your project root, whenever your dataset or model
config changes:
    python scripts/export_model.py

Then copy the two output files (api/model.joblib, api/defaults.joblib)
into your backend repo (the one deployed to Render) and commit them.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.models.random_forest import DEFAULT_CONTINUOUS_FEATURES, RandomForestTireDegradationModel
from f1_tire_model.models.tire_report import default_covariates

# Must exactly match the production config in api/main.py -- if you change
# one, change the other, or the deployed API will silently serve
# predictions from a differently-configured model than you think.
DRIVER_FEATURES = ("avg_braking_decel_ms2", "peak_braking_decel_ms2", "avg_throttle_pct")
COMPOUNDS = ("SOFT", "MEDIUM", "HARD")

OUTPUT_DIR = Path("api")


def main() -> None:
    df = load_dataset()
    if df.empty:
        raise RuntimeError(
            "load_dataset() returned no rows -- check that your dataset's .parquet "
            "fragments are actually present before running this."
        )

    continuous_no_fuel = tuple(f for f in DEFAULT_CONTINUOUS_FEATURES if f != "fuel_load_estimate_kg")
    model = RandomForestTireDegradationModel(
        continuous_features=continuous_no_fuel + DRIVER_FEATURES,
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
    ).fit(df)

    circuits = sorted(df["circuit"].unique())
    defaults: dict[str, dict[str, dict]] = {}
    for circuit in circuits:
        defaults[circuit] = {}
        for compound in COMPOUNDS:
            defaults[circuit][compound] = default_covariates(df, model, circuit=circuit, compound=compound)

    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / "model.joblib")
    joblib.dump({"circuits": circuits, "defaults": defaults}, OUTPUT_DIR / "defaults.joblib")

    print(f"Wrote {OUTPUT_DIR / 'model.joblib'} and {OUTPUT_DIR / 'defaults.joblib'}")
    print(f"Covers {len(circuits)} circuits: {', '.join(circuits)}")


if __name__ == "__main__":
    main()
