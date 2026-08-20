from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.random_forest import RandomForestTireDegradationModel, DEFAULT_CONTINUOUS_FEATURES
from f1_tire_model.models.tire_report import default_covariates, generate_tire_report

df = load_dataset()
train_df, _ = split_chronological(df, test_size=0.2)

driver_features = ("avg_braking_decel_ms2", "peak_braking_decel_ms2", "avg_throttle_pct")
model = RandomForestTireDegradationModel(
    continuous_features=DEFAULT_CONTINUOUS_FEATURES + driver_features,
    n_estimators=200, max_depth=10, min_samples_leaf=5,
).fit(df)  # production model: fit on ALL data, not just train_df, now that evaluation is done

for circuit in ["Monaco", "Singapore", "Silverstone"]:
    defaults = default_covariates(df, model, circuit=circuit)
    report = generate_tire_report(
        model, circuit=circuit, track_temp_c=35.0, degradation_threshold_s=1.0, extra_covariates=defaults,
    )
    print(report.summary())
    print("")
