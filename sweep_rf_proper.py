from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological, split_summary
from f1_tire_model.models.random_forest import RandomForestTireDegradationModel

df = load_dataset()

train_df, test_df = split_chronological(df, test_size=0.2)

tune_train_df, val_df = split_chronological(train_df, test_size=0.2)
print("Tuning split:", split_summary(tune_train_df, val_df))
val_valid = val_df[val_df["is_valid_lap"]]
print("")

results = []
for n_estimators in [200, 400]:
    for max_depth in [4, 6, 8, 10, 12]:
        for min_samples_leaf in [5, 10, 20]:
            model = RandomForestTireDegradationModel(
                n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            ).fit(tune_train_df)
            r2 = model.evaluate(val_valid).r2
            results.append((n_estimators, max_depth, min_samples_leaf, r2))
            print(f"n_est={n_estimators:4d} depth={max_depth:3d} leaf={min_samples_leaf:3d}  VAL R2={r2:.4f}")

best = max(results, key=lambda r: r[3])
print("")
print("Best config on validation:", best)

n_estimators, max_depth, min_samples_leaf, _ = best
final_model = RandomForestTireDegradationModel(
    n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
).fit(train_df)
test_valid = test_df[test_df["is_valid_lap"]]
print("FINAL real held-out result:", final_model.evaluate(test_valid))
