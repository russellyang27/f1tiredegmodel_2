import numpy as np
from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.baseline import LinearTireDegradationModel
from f1_tire_model.models.circuit_aware import CircuitAwareDegradationModel

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)

print("Per-circuit avg_corner_speed_kph / track_temp_c:")
circuit_features = df.groupby("circuit", observed=True)[["avg_corner_speed_kph", "track_temp_c"]].mean()
print(circuit_features.sort_values("avg_corner_speed_kph").to_string())
print("")

train_range_speed = (train_df["avg_corner_speed_kph"].min(), train_df["avg_corner_speed_kph"].max())
train_range_temp = (train_df["track_temp_c"].min(), train_df["track_temp_c"].max())
print("Training range -- corner speed:", train_range_speed, " track temp:", train_range_temp)

test_range_speed = (test_df["avg_corner_speed_kph"].min(), test_df["avg_corner_speed_kph"].max())
test_range_temp = (test_df["track_temp_c"].min(), test_df["track_temp_c"].max())
print("Test range     -- corner speed:", test_range_speed, " track temp:", test_range_temp)
print("")

baseline = LinearTireDegradationModel().fit(train_df)
circuit_aware = CircuitAwareDegradationModel().fit(train_df)

test_valid = test_df[test_df["is_valid_lap"]]
for compound in ["HARD", "MEDIUM"]:
    subset = test_valid[test_valid["tire_compound"] == compound]
    if subset.empty:
        continue
    print(f"--- {compound} (n={len(subset)}) ---")
    print("Baseline:     ", baseline.evaluate(subset))
    print("Circuit-aware:", circuit_aware.evaluate(subset))
