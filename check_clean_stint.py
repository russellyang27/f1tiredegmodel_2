from f1_tire_model.datasets.storage import load_dataset

df = load_dataset()

dry_compounds = {"SOFT", "MEDIUM", "HARD"}
dry = df[df["tire_compound"].isin(dry_compounds)]

valid_counts = (
    dry[dry["is_valid_lap"]]
    .groupby(["driver_code", "stint_number"], observed=True)
    .size()
    .sort_values(ascending=False)
)

print("Stints with the most valid (clean) laps:")
print(valid_counts.head(5))

best_driver, best_stint = valid_counts.index[0]
example = df[(df["driver_code"] == best_driver) & (df["stint_number"] == best_stint)]
compound = example["tire_compound"].iloc[0]

print("")
print(best_driver, "stint", best_stint, "compound:", compound)
print(example[["tire_age_laps", "lap_time_s", "lap_time_delta_s", "is_valid_lap"]].to_string(index=False))
