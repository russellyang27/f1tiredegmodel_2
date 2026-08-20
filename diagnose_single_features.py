import numpy as np
import pandas as pd
from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.circuit_aware import CircuitAwareDegradationModel
from f1_tire_model.models.baseline import LinearTireDegradationModel


def fit_single_feature_model(df, feature, min_points=30, min_circuits=5):
    means, stds, coeffs_by_compound = {}, {}, {}
    mean, std = df[feature].mean(), df[feature].std()
    means[feature], stds[feature] = mean, (std if std > 1e-8 else 1.0)

    for compound, group in df.groupby("tire_compound", observed=True):
        if len(group) < min_points or group["circuit"].nunique() < min_circuits:
            continue
        age = group["tire_age_laps"].to_numpy(dtype=float)
        age_std = (age - age.mean()) / age.std()
        f_std = (group[feature].to_numpy(dtype=float) - means[feature]) / stds[feature]
        X = np.column_stack([np.ones(len(group)), age_std, age_std * f_std, f_std])
        y = group["lap_time_delta_s"].to_numpy(dtype=float)
        coeffs_by_compound[str(compound)] = (age.mean(), age.std(), np.linalg.lstsq(X, y, rcond=None)[0])
    return means, stds, coeffs_by_compound


def predict_single_feature(row, feature, means, stds, coeffs_by_compound):
    compound = str(row["tire_compound"])
    if compound not in coeffs_by_compound:
        return np.nan
    age_mean, age_std_scale, coeffs = coeffs_by_compound[compound]
    age_std = (row["tire_age_laps"] - age_mean) / age_std_scale
    f_std = (row[feature] - means[feature]) / stds[feature]
    return coeffs[0] + coeffs[1] * age_std + coeffs[2] * age_std * f_std + coeffs[3] * f_std


def evaluate(df, feature, means, stds, coeffs_by_compound):
    preds = df.apply(lambda r: predict_single_feature(r, feature, means, stds, coeffs_by_compound), axis=1)
    valid = df["lap_time_delta_s"].notna() & preds.notna()
    residuals = df.loc[valid, "lap_time_delta_s"].to_numpy() - preds[valid].to_numpy()
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    y = df.loc[valid, "lap_time_delta_s"].to_numpy()
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    return f"MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  (n={valid.sum()})"


df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)
test_valid = test_df[test_df["is_valid_lap"]]

full_model = CircuitAwareDegradationModel().fit(train_df)
in_range = test_valid[~full_model.extrapolation_mask(test_valid)]
print(f"Evaluating on the same {len(in_range)}-row in-range set as before, for comparability.\n")

baseline = LinearTireDegradationModel().fit(train_df)
print("Baseline (compound + age only):      ", baseline.evaluate(in_range))

train_prepped = train_df[train_df["is_valid_lap"]].dropna(
    subset=["lap_time_delta_s", "tire_age_laps", "tire_compound", "avg_corner_speed_kph", "track_temp_c"]
)

means, stds, coeffs = fit_single_feature_model(train_prepped, "avg_corner_speed_kph")
print("Corner-speed ONLY (no temp):          ", evaluate(in_range, "avg_corner_speed_kph", means, stds, coeffs))

means, stds, coeffs = fit_single_feature_model(train_prepped, "track_temp_c")
print("Track-temp ONLY (no corner speed):    ", evaluate(in_range, "track_temp_c", means, stds, coeffs))

print("Full two-feature model (for reference): R2=-1.0816 (from the earlier run)")
