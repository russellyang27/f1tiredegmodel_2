from f1_tire_model.datasets.storage import load_dataset
df = load_dataset()
sub = df[(df["driver_code"] == "ALB") & (df["stint_number"] == 5)]
print(sub[["tire_age_laps", "lap_time_s", "is_pit_lap", "is_safety_car_lap"]])
