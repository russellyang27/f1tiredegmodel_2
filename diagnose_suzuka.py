from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.models.circuit_aware import CircuitAwareDegradationModel

df = load_dataset()

print("Full-season model coefficients:")
model = CircuitAwareDegradationModel().fit(df)
print(model.degradation_rate_summary().to_string(index=False))
print("")

suzuka = df[(df["circuit"] == "Suzuka") & (df["is_valid_lap"])]
for compound in ["MEDIUM", "HARD"]:
    subset = suzuka[suzuka["tire_compound"] == compound].dropna(subset=["tire_age_laps", "lap_time_delta_s"])
    if subset.empty:
        print(compound, "-- no valid laps at Suzuka")
        continue
    print(f"--- Suzuka {compound} (n={len(subset)} valid laps) ---")
    print(subset[["driver_code", "stint_number", "tire_age_laps", "lap_time_delta_s"]].sort_values(
        ["stint_number", "tire_age_laps"]
    ).to_string(index=False))
    print("")
