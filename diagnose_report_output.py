import numpy as np
import pandas as pd
from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.random_forest import RandomForestTireDegradationModel, DEFAULT_CONTINUOUS_FEATURES
from f1_tire_model.models.tire_report import default_covariates

df = load_dataset()
print("Circuits in dataset:", sorted(df["circuit"].dropna().unique()))
print("")

print("Monaco compound counts:")
print(df[df["circuit"] == "Monaco"].groupby("tire_compound", observed=True).size())
print("")

train_df, _ = split_chronological(df, test_size=0.2)
driver_features = ("avg_braking_decel_ms2", "peak_braking_decel_ms2", "avg_throttle_pct")
model = RandomForestTireDegradationModel(
    continuous_features=DEFAULT_CONTINUOUS_FEATURES + driver_features,
    n_estimators=200, max_depth=10, min_samples_leaf=5,
).fit(df)

defaults = default_covariates(df, model, circuit="Monaco")
ages = [1, 5, 10, 15, 20, 25, 30]
for compound in ["SOFT", "MEDIUM", "HARD"]:
    query = pd.DataFrame({
        "tire_compound": [compound]*len(ages), "circuit": ["Monaco"]*len(ages),
        "tire_age_laps": ages, "track_temp_c": [35.0]*len(ages),
        **{k: [v]*len(ages) for k, v in defaults.items() if k != "fuel_load_estimate_kg"},
        "fuel_load_estimate_kg": [max(100.0 - a*1.6, 0.0) for a in ages],
    })
    preds = model.predict(query)
    print(f"{compound}: predictions at ages {ages} =", np.round(preds, 3))
