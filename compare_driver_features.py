from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological, split_summary
from f1_tire_model.models.random_forest import (
    RandomForestTireDegradationModel,
    DEFAULT_CONTINUOUS_FEATURES,
)

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)
tune_train_df, val_df = split_chronological(train_df, test_size=0.2)
val_valid = val_df[val_df["is_valid_lap"]]

driver_features = ("avg_braking_decel_ms2", "peak_braking_decel_ms2", "avg_throttle_pct", "driver_consistency_std_s")

baseline_model = RandomForestTireDegradationModel(
    n_estimators=200, max_depth=10, min_samples_leaf=5,
).fit(tune_train_df)
print("Without driver features:", baseline_model.evaluate(val_valid))
print("  training rows used:", len(tune_train_df[tune_train_df["is_valid_lap"]].dropna(subset=list(baseline_model.feature_columns))))
print("")

for max_depth in [8, 10, 12]:
    model = RandomForestTireDegradationModel(
        continuous_features=DEFAULT_CONTINUOUS_FEATURES + driver_features,
        n_estimators=200, max_depth=max_depth, min_samples_leaf=5,
    ).fit(tune_train_df)
    result = model.evaluate(val_valid)
    n_rows = len(tune_train_df[tune_train_df["is_valid_lap"]].dropna(subset=list(model.feature_columns)))
    print(f"With driver features, depth={max_depth}:", result, " training rows used:", n_rows)
