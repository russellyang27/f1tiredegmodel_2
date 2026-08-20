from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.circuit_aware import CircuitAwareDegradationModel
from f1_tire_model.models.circuit_aware_ridge import RidgeCircuitAwareDegradationModel

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)
test_valid = test_df[test_df["is_valid_lap"]]

ols_model = CircuitAwareDegradationModel().fit(train_df)
in_range_mask = ~ols_model.extrapolation_mask(test_valid)
in_range = test_valid[in_range_mask]

print(f"Fair (in-range) comparison set: {len(in_range)} rows")
print("")
print("OLS (unregularized):", ols_model.evaluate(in_range))
print("")

for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
    ridge_model = RidgeCircuitAwareDegradationModel(alpha=alpha).fit(train_df)
    result = ridge_model.evaluate(in_range)
    print(f"Ridge alpha={alpha:6.1f}:", result)
