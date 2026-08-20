from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.random_forest import RandomForestTireDegradationModel

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)  # test_df untouched below

results = []
for n_estimators in [200, 400]:
    for max_depth in [8, 12, None]:
        for min_samples_leaf in [3, 5, 10]:
            model = RandomForestTireDegradationModel(
                n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            ).fit(train_df)
            results.append((n_estimators, max_depth, min_samples_leaf, model.oob_score_))
            print(f"n_est={n_estimators:4d} depth={str(max_depth):5s} leaf={min_samples_leaf:3d}  OOB R2={model.oob_score_:.4f}")

best = max(results, key=lambda r: r[3])
print("")
print("Best config:", best)
