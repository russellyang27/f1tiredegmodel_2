from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.random_forest import RandomForestTireDegradationModel

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)
test_valid = test_df[test_df["is_valid_lap"]]

configs = [
    ("shallow (original default)", dict(n_estimators=300, max_depth=10, min_samples_leaf=5)),
    ("medium", dict(n_estimators=300, max_depth=12, min_samples_leaf=5)),
    ("deep (OOB-best)", dict(n_estimators=400, max_depth=None, min_samples_leaf=3)),
]

for label, params in configs:
    model = RandomForestTireDegradationModel(**params).fit(train_df)
    real_result = model.evaluate(test_valid)
    print(f"{label:28s} OOB R2={model.oob_score_:.4f}   REAL held-out R2={real_result.r2:.4f}")
