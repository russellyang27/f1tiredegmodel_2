import numpy as np
import pandas as pd
from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.models.random_forest import RandomForestTireDegradationModel, DEFAULT_CONTINUOUS_FEATURES
from f1_tire_model.models.tire_report import default_covariates

df = load_dataset()

driver_features = ("avg_braking_decel_ms2", "peak_braking_decel_ms2", "avg_throttle_pct")
continuous_no_fuel = tuple(f for f in DEFAULT_CONTINUOUS_FEATURES if f != "fuel_load_estimate_kg")

model = RandomForestTireDegradationModel(
    continuous_features=continuous_no_fuel + driver_features,
    n_estimators=200, max_depth=10, min_samples_leaf=5,
).fit(df)

print("model.feature_columns:", model.feature_columns)

defaults = default_covariates(df, model, circuit="Sakhir")
print("defaults:", defaults)

ages = [1, 5, 10, 15, 20, 30, 40, 50, 60]
query = pd.DataFrame({
    "tire_compound": ["HARD"]*len(ages), "circuit": ["Sakhir"]*len(ages),
    "tire_age_laps": ages, "track_temp_c": [35.0]*len(ages),
    **{k: [v]*len(ages) for k, v in defaults.items()},
})
print("query columns:", list(query.columns))
preds = model.predict(query)
print("predictions:", preds)
