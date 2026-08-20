import numpy as np
from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.baseline import LinearTireDegradationModel

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)

print("Test race(s):", test_df[["season", "round_number", "circuit"]].drop_duplicates().to_string(index=False))
print("")

linear = LinearTireDegradationModel().fit(train_df)

scored = test_df.dropna(subset=["lap_time_delta_s", "tire_age_laps"]).copy()
scored["predicted"] = linear.predict(scored)
scored = scored.dropna(subset=["predicted"])
scored["residual"] = scored["lap_time_delta_s"] - scored["predicted"]

print("Residual summary by compound (mean, std, count):")
print(scored.groupby("tire_compound", observed=True)["residual"].agg(["mean", "std", "count"]))
print("")

print("10 worst-magnitude residuals:")
worst = scored.reindex(scored["residual"].abs().sort_values(ascending=False).index)
print(worst[["driver_code", "tire_compound", "tire_age_laps", "lap_time_delta_s", "predicted", "residual", "is_safety_car_lap"]].head(10).to_string(index=False))
