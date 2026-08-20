from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.datasets.splitting import split_chronological, split_summary
from f1_tire_model.models.baseline import LinearTireDegradationModel, QuadraticTireDegradationModel

df = load_dataset()
train_df, test_df = split_chronological(df, test_size=0.2)
print("Split summary:", split_summary(train_df, test_df))
print("")

linear = LinearTireDegradationModel().fit(train_df)
quadratic = QuadraticTireDegradationModel().fit(train_df)

# WRONG (what we did before): score against every test row, including
# Safety Car / pit / outlier laps the model was never trained to predict.
print("Unfiltered (includes invalid laps -- this is what we ran last time):")
print("Linear:   ", linear.evaluate(test_df))
print("Quadratic:", quadratic.evaluate(test_df))
print("")

# RIGHT: score only against valid degradation laps, matching what the
# model was trained on and what it's actually meant to predict.
test_valid = test_df[test_df["is_valid_lap"]]
print("Filtered to is_valid_lap only (", len(test_valid), "of", len(test_df), "test rows ):")
print("Linear:   ", linear.evaluate(test_valid))
print("Quadratic:", quadratic.evaluate(test_valid))
