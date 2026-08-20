from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.random_forest import (
    RandomForestTireDegradationModel,
    DEFAULT_CONTINUOUS_FEATURES,
)
from f1_tire_model.models.baseline import LinearTireDegradationModel

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)
test_valid = test_df[test_df["is_valid_lap"]]

driver_features = ("avg_braking_decel_ms2", "peak_braking_decel_ms2", "avg_throttle_pct", "driver_consistency_std_s")

final_model = RandomForestTireDegradationModel(
    continuous_features=DEFAULT_CONTINUOUS_FEATURES + driver_features,
    n_estimators=200, max_depth=10, min_samples_leaf=5,
).fit(train_df)

baseline = LinearTireDegradationModel().fit(train_df)

print("=== FINAL, one-time held-out comparison ===")
print("Baseline (compound + age only):     ", baseline.evaluate(test_valid))
print("Random Forest (final, tuned model): ", final_model.evaluate(test_valid))
print("")
print("Feature importances:")
print(final_model.feature_importance_summary().to_string(index=False))
