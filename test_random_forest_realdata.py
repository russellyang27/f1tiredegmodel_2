from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.models.random_forest import RandomForestTireDegradationModel

df = load_dataset()
df_valid = df[df["is_valid_lap"]]

model = RandomForestTireDegradationModel(n_estimators=200).fit(df_valid)
print(model.feature_importance_summary())
print("")
print(model.evaluate(df_valid))  # IN-SAMPLE ONLY -- not a real generalization test, see caveat above
