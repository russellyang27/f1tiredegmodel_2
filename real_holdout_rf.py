from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological, split_summary
from f1_tire_model.models.baseline import LinearTireDegradationModel
from f1_tire_model.models.random_forest import RandomForestTireDegradationModel

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)
print("Split summary:", split_summary(train_df, test_df))

train_circuits = set(train_df["circuit"].dropna().unique())
test_circuits = set(test_df["circuit"].dropna().unique())
seen_before = test_circuits & train_circuits
unseen = test_circuits - train_circuits
print(f"Test circuits also in training: {len(seen_before)}/{len(test_circuits)}")
print(f"Genuinely unseen test circuits: {sorted(unseen)}")
print("")

test_valid = test_df[test_df["is_valid_lap"]]

baseline = LinearTireDegradationModel().fit(train_df)
rf_model = RandomForestTireDegradationModel(n_estimators=300).fit(train_df)

print("Baseline (compound + age only): ", baseline.evaluate(test_valid))
print("Random Forest (full features):  ", rf_model.evaluate(test_valid))
print("")
print("Feature importances (from training data):")
print(rf_model.feature_importance_summary().to_string(index=False))
