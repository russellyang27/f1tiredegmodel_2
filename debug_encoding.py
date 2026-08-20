import pandas as pd
from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.models.random_forest import RandomForestTireDegradationModel, DEFAULT_CONTINUOUS_FEATURES

df = load_dataset()

driver_features = ("avg_braking_decel_ms2", "peak_braking_decel_ms2", "avg_throttle_pct")
continuous_no_fuel = tuple(f for f in DEFAULT_CONTINUOUS_FEATURES if f != "fuel_load_estimate_kg")

model = RandomForestTireDegradationModel(
    continuous_features=continuous_no_fuel + driver_features,
    n_estimators=200, max_depth=10, min_samples_leaf=5,
).fit(df)

print("Learned tire_compound categories:", model._categories.get("tire_compound"))
print("Learned circuit categories (first 5):", model._categories.get("circuit")[:5])
print("")

query = pd.DataFrame({
    "tire_compound": ["SOFT", "MEDIUM", "HARD"],
    "circuit": ["Sakhir", "Sakhir", "Sakhir"],
    "tire_age_laps": [10, 10, 10],
    "track_temp_c": [35.0, 35.0, 35.0],
    "air_temp_c": [23.8, 23.8, 23.8],
    "avg_braking_decel_ms2": [14.8, 14.8, 14.8],
    "peak_braking_decel_ms2": [48.0, 48.0, 48.0],
    "avg_throttle_pct": [62.4, 62.4, 62.4],
    "is_raining": [False, False, False],
})

encoded = model._encode(query)
tire_compound_cols = [c for c in encoded.columns if c.startswith("tire_compound__")]
print("tire_compound one-hot columns:", tire_compound_cols)
print(encoded[tire_compound_cols])
