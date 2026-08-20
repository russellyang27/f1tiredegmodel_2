from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.random_forest import RandomForestTireDegradationModel, DEFAULT_CONTINUOUS_FEATURES
from f1_tire_model.models.baseline import LinearTireDegradationModel

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)
test_valid = test_df[test_df["is_valid_lap"]]

driver_features = ("avg_braking_decel_ms2", "peak_braking_decel_ms2", "avg_throttle_pct")
continuous_no_fuel = tuple(f for f in DEFAULT_CONTINUOUS_FEATURES if f != "fuel_load_estimate_kg")

model = RandomForestTireDegradationModel(
    continuous_features=continuous_no_fuel + driver_features,
    n_estimators=200, max_depth=10, min_samples_leaf=5,
).fit(train_df)

baseline = LinearTireDegradationModel().fit(train_df)
print("Baseline:                    ", baseline.evaluate(test_valid))
print("RF without fuel_load:        ", model.evaluate(test_valid))
print("")
print(model.feature_importance_summary().to_string(index=False))
