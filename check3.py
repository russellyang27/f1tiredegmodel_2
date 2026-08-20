from f1_tire_model.datasets.storage import load_dataset
df = load_dataset()
print(df["driver_code"].nunique(), "drivers:", sorted(df["driver_code"].unique()))
