from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.models.random_forest import RandomForestTireDegradationModel, DEFAULT_CONTINUOUS_FEATURES
from f1_tire_model.models.tire_report import default_covariates, generate_tire_report

df = load_dataset()

driver_features = ("avg_braking_decel_ms2", "peak_braking_decel_ms2", "avg_throttle_pct")
continuous_no_fuel = tuple(f for f in DEFAULT_CONTINUOUS_FEATURES if f != "fuel_load_estimate_kg")

model = RandomForestTireDegradationModel(
    continuous_features=continuous_no_fuel + driver_features,
    n_estimators=200, max_depth=10, min_samples_leaf=5,
).fit(df)

for circuit in ["Monaco", "Marina Bay", "Silverstone"]:
    defaults = default_covariates(df, model, circuit=circuit)
    report = generate_tire_report(
        model, circuit=circuit, track_temp_c=35.0, degradation_threshold_s=1.0, extra_covariates=defaults,
    )
    print(report.summary())
    print("")
