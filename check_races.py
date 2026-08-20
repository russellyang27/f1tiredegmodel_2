from f1_tire_model.datasets.storage import load_dataset
df = load_dataset()
print("Total rows:", len(df))
print(df[["season", "round_number"]].drop_duplicates().sort_values(["season", "round_number"]))
