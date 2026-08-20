from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.circuit_aware import CircuitAwareDegradationModel

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)

model = CircuitAwareDegradationModel().fit(train_df)
print(model.degradation_rate_summary().to_string(index=False))

print("")
print("Distinct circuits per compound in training data:")
print(train_df[train_df["is_valid_lap"]].groupby("tire_compound", observed=True)["circuit"].nunique())
