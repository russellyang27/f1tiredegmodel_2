from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.manifest import load_manifest

df = load_dataset()
print("Rounds currently in the dataset:")
print(sorted(df["round_number"].unique()))
print("")
print("Manifest (fully completed sessions):")
print(sorted(load_manifest()))
