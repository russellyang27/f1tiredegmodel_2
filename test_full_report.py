from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological
from f1_tire_model.models.circuit_aware import CircuitAwareDegradationModel
from f1_tire_model.models.tire_report import generate_tire_report

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)

model = CircuitAwareDegradationModel().fit(train_df)

print("=== A fast, cool circuit (e.g. Suzuka-like) ===")
report = generate_tire_report(model, avg_corner_speed_kph=207.0, track_temp_c=20.0, degradation_threshold_s=1.0)
print(report.summary())

print("")
print("=== A slow, hot circuit (e.g. Singapore/Miami-like) ===")
report = generate_tire_report(model, avg_corner_speed_kph=140.0, track_temp_c=48.0, degradation_threshold_s=1.0)
print(report.summary())
