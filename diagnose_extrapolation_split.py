from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.baseline import LinearTireDegradationModel
from f1_tire_model.models.circuit_aware import CircuitAwareDegradationModel

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)
test_valid = test_df[test_df["is_valid_lap"]]

baseline = LinearTireDegradationModel().fit(train_df)
circuit_aware = CircuitAwareDegradationModel().fit(train_df)

extrapolating_mask = circuit_aware.extrapolation_mask(test_valid)
in_range = test_valid[~extrapolating_mask]
out_of_range = test_valid[extrapolating_mask]

print(f"In-range (interpolating) rows: {len(in_range)}")
print(f"Out-of-range (extrapolating) rows: {len(out_of_range)}")
print("")

print("--- IN-RANGE ONLY (the fair comparison) ---")
print("Baseline:     ", baseline.evaluate(in_range))
print("Circuit-aware:", circuit_aware.evaluate(in_range))
print("")

print("--- OUT-OF-RANGE ONLY (known-unreliable, for reference) ---")
print("Baseline:     ", baseline.evaluate(out_of_range))
print("Circuit-aware:", circuit_aware.evaluate(out_of_range))
print("")

print("Which circuit(s) are driving the extrapolation:")
print(out_of_range.groupby("circuit", observed=True)[["avg_corner_speed_kph", "track_temp_c"]].mean())
