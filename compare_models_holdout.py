from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import (
    split_chronological,
    split_summary,
    check_split_compound_coverage,
)
from f1_tire_model.models.baseline import LinearTireDegradationModel, QuadraticTireDegradationModel
from f1_tire_model.visualization.report import build_validation_report

df = load_dataset()

train_df, test_df = split_chronological(df, test_size=0.2)
print("Split summary:", split_summary(train_df, test_df))

uncovered = check_split_compound_coverage(train_df, test_df)
if uncovered:
    print("WARNING -- compounds in test set with no training examples:", uncovered)
print("")

linear = LinearTireDegradationModel().fit(train_df)
quadratic = QuadraticTireDegradationModel().fit(train_df)

print("Held-out evaluation (fit on train races, scored on test races):")
print("Linear:   ", linear.evaluate(test_df))
print("Quadratic:", quadratic.evaluate(test_df))
print("")

print("Linear coefficients (from training data only):")
print(linear.degradation_rate_summary().to_string(index=False))
print("")
print("Quadratic coefficients (from training data only):")
print(quadratic.degradation_rate_summary().to_string(index=False))

fig = build_validation_report(
    test_df, {"Linear": linear, "Quadratic": quadratic}, save_path="validation_report_holdout.png"
)
print("")
print("Saved validation_report_holdout.png (built from the held-out test race(s) only).")
