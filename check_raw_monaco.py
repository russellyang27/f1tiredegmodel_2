from f1_tire_model.datasets.storage import load_dataset

df = load_dataset()
monaco = df[(df["circuit"] == "Monaco") & (df["is_valid_lap"]) & (df["tire_age_laps"] >= 20) & (df["tire_age_laps"] <= 35)]

print("Real Monaco lap_time_delta_s, ages 20-35, by compound:")
print(monaco.groupby("tire_compound", observed=True)["lap_time_delta_s"].agg(["mean", "std", "min", "max", "count"]))
print("")

print("Same, broken out age by age for HARD (most sample-rich compound):")
hard = monaco[monaco["tire_compound"] == "HARD"]
print(hard.groupby("tire_age_laps")["lap_time_delta_s"].agg(["mean", "count"]))
