from f1_tire_model.datasets.storage import load_dataset

df = load_dataset()

example = df[(df["driver_code"] == "LAW") & (df["stint_number"] == 5)]
print(example[["tire_age_laps", "lap_time_s", "is_safety_car_lap", "is_pit_lap"]].to_string(index=False))
