from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological, split_summary
from f1_tire_model.models.baseline import LinearTireDegradationModel
from f1_tire_model.models.circuit_aware import CircuitAwareDegradationModel

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)
print("Split summary:", split_summary(train_df, test_df))

test_valid = test_df[test_df["is_valid_lap"]]

baseline = LinearTireDegradationModel().fit(train_df)
circuit_aware = CircuitAwareDegradationModel().fit(train_df)

print("")
print("Baseline (compound + age only):    ", baseline.evaluate(test_valid))
print("Circuit-aware (+ track features):  ", circuit_aware.evaluate(test_valid))
