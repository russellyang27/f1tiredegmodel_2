from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.models.random_forest import RandomForestTireDegradationModel, DEFAULT_CONTINUOUS_FEATURES

df = load_dataset()
yas = df[df["circuit"] == "Yas Island"]

print("--- lap_time_delta_s stats for Yas Island ---")
print(yas["lap_time_delta_s"].describe())

print("\n--- max tire_age_laps seen for Yas Island ---")
print(yas["tire_age_laps"].max())

print("\n--- mean lap_time_delta_s by compound ---")
print(yas.groupby("tire_compound")["lap_time_delta_s"].mean())

print("\n--- all tire_compound values seen in the FULL dataset ---")
print(df["tire_compound"].unique())

DRIVER_FEATURES = ("avg_braking_decel_ms2", "peak_braking_decel_ms2", "avg_throttle_pct")
continuous_no_fuel = tuple(f for f in DEFAULT_CONTINUOUS_FEATURES if f != "fuel_load_estimate_kg")
model = RandomForestTireDegradationModel(
    continuous_features=continuous_no_fuel + DRIVER_FEATURES,
    n_estimators=200, max_depth=10, min_samples_leaf=5,
).fit(df)

print("\n--- feature importance ---")
print(model.feature_importance_summary())
