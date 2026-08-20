import numpy as np
from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.baseline import LinearTireDegradationModel

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)

print("Train circuits:", sorted(train_df["circuit"].dropna().unique()))
print("Test circuits: ", sorted(test_df["circuit"].dropna().unique()))
print("")

linear = LinearTireDegradationModel().fit(train_df)

test_valid = test_df[test_df["is_valid_lap"]].dropna(subset=["lap_time_delta_s", "tire_age_laps"]).copy()
test_valid["predicted"] = linear.predict(test_valid)
test_valid = test_valid.dropna(subset=["predicted"])
test_valid["residual"] = test_valid["lap_time_delta_s"] - test_valid["predicted"]

print("Residuals by circuit + compound (mean, std, count):")
print(
    test_valid.groupby(["circuit", "tire_compound"], observed=True)["residual"]
    .agg(["mean", "std", "count"])
    .to_string()
)
print("")

train_valid = train_df[train_df["is_valid_lap"]].dropna(subset=["lap_time_delta_s"])
compound_means = train_valid.groupby("tire_compound", observed=True)["lap_time_delta_s"].mean()

test_valid["baseline_predicted"] = test_valid["tire_compound"].astype(str).map(compound_means)
baseline_valid = test_valid.dropna(subset=["baseline_predicted"])

baseline_residuals = baseline_valid["lap_time_delta_s"] - baseline_valid["baseline_predicted"]
baseline_mae = baseline_residuals.abs().mean()
baseline_rmse = np.sqrt((baseline_residuals ** 2).mean())
ss_res = (baseline_residuals ** 2).sum()
ss_tot = ((baseline_valid["lap_time_delta_s"] - baseline_valid["lap_time_delta_s"].mean()) ** 2).sum()
baseline_r2 = 1 - ss_res / ss_tot

print("Trivial baseline (compound mean only, ignores tire_age):")
print(f"MAE={baseline_mae:.4f}  RMSE={baseline_rmse:.4f}  R2={baseline_r2:.4f}  (n={len(baseline_valid)})")
