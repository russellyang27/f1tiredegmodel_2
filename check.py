from f1_tire_model.datasets.storage import load_dataset
df = load_dataset()

sub = df[(df["driver_code"] == "BEA") & (df["stint_number"] == 5)]
print(sub[["tire_age_laps", "lap_time_s", "is_valid_lap"]])

pct_missing = df["tire_age_laps"].isna().mean()
print(f"{pct_missing:.1%} of rows have missing tire_age_laps")
