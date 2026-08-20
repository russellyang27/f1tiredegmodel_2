from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.random_forest import (
    RandomForestTireDegradationModel,
    DEFAULT_CONTINUOUS_FEATURES,
)
from f1_tire_model.models.baseline import LinearTireDegradationModel

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)
tune_train_df, val_df = split_chronological(train_df, test_size=0.2)
val_valid = val_df[val_df["is_valid_lap"]]
test_valid = test_df[test_df["is_valid_lap"]]

per_lap_driver_features = ("avg_braking_decel_ms2", "peak_braking_decel_ms2", "avg_throttle_pct")

print("--- Re-checking on VALIDATION first (not test yet) ---")
for max_depth in [8, 10, 12]:
    model = RandomForestTireDegradationModel(
        continuous_features=DEFAULT_CONTINUOUS_FEATURES + per_lap_driver_features,
        n_estimators=200, max_depth=max_depth, min_samples_leaf=5,
    ).fit(tune_train_df)
    print(f"depth={max_depth}:", model.evaluate(val_valid))

print("")
print("--- FINAL, one-time test check (best depth from above) ---")
final_model = RandomForestTireDegradationModel(
    continuous_features=DEFAULT_CONTINUOUS_FEATURES + per_lap_driver_features,
    n_estimators=200, max_depth=10, min_samples_leaf=5,
).fit(train_df)
baseline = LinearTireDegradationModel().fit(train_df)

print("Baseline:     ", baseline.evaluate(test_valid))
print("Random Forest:", final_model.evaluate(test_valid))
print("")
print(final_model.feature_importance_summary().to_string(index=False))
