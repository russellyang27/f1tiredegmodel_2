import pandas as pd
from f1_tire_model.datasets.storage import load_dataset

df = load_dataset()
monaco = df[(df["circuit"] == "Monaco") & (df["is_valid_lap"])]

print("Real Monaco lap counts by tire_age_laps bracket:")
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 100]
labels = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41+"]
monaco["age_bucket"] = pd.cut(monaco["tire_age_laps"], bins=bins, labels=labels)
print(monaco.groupby("age_bucket", observed=True).size())
print("")
print("Max tire_age_laps actually seen at Monaco:", monaco["tire_age_laps"].max())
