from f1_tire_model.datasets.storage import load_dataset
from f1_tire_model.models.baseline import LinearTireDegradationModel

df = load_dataset()

print("Rows per round:")
print(df["round_number"].value_counts())
print("")

for round_number in sorted(df["round_number"].unique()):
    race_df = df[df["round_number"] == round_number]
    n_caution = race_df["is_safety_car_lap"].sum()
    n_laps = len(race_df)
    compounds = race_df["tire_compound"].value_counts().to_dict()

    print("=" * 60)
    print("Round", round_number)
    print(n_laps, "total laps,", n_caution, "under caution")
    print("Compounds used:", compounds)

    try:
        model = LinearTireDegradationModel().fit(race_df)
        summary = model.degradation_rate_summary()
        print(summary.to_string(index=False))
    except ValueError as exc:
        print("Could not fit a model for this round:", exc)

    print("")
