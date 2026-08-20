"""
Core validation plots for a fitted TireDegradationModel:

- `plot_degradation_curve` — actual lap-time delta vs. tire age, with the
  model's fitted curve overlaid, per compound. This is the single most
  important plot for judging whether the baseline is physically sensible:
  a good fit should track the scatter, and the compound ordering (soft
  degrading faster than hard) should be visible by eye, not just in a
  coefficients table.
- `plot_residuals` — (actual - predicted) vs. tire age. A well-specified
  model should show residuals scattered evenly around zero with no visible
  trend; a trend (e.g. residuals drifting positive at high tire age) is a
  sign the model's functional form doesn't match reality at high tire
  age — exactly the kind of "cliff" a linear model can't capture and a
  quadratic one might.

Every function accepts an optional `ax` so these compose into a larger
multi-panel figure (see report.py) instead of only working as standalone
plots.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1_tire_model.models.base import TireDegradationModel
from f1_tire_model.visualization._utils import valid_rows
from f1_tire_model.visualization.style import (
    DEFAULT_FIGSIZE,
    INVALID_LAP_ALPHA,
    INVALID_LAP_COLOR,
    compound_color,
)


def plot_degradation_curve(
    features_df: pd.DataFrame,
    model: TireDegradationModel,
    *,
    compounds: list[str] | None = None,
    show_invalid_laps: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot actual lap_time_delta_s vs. tire_age_laps with the model's fitted
    curve overlaid, one color per compound.

    Parameters
    ----------
    features_df : a LapFeatureRecord-shaped DataFrame (or compatible subset)
        with at least tire_compound, tire_age_laps, lap_time_delta_s.
    model : an already-fitted TireDegradationModel.
    compounds : restrict to specific compounds; defaults to every compound
        present in the data.
    show_invalid_laps : if True and an `is_valid_lap` column is present,
        excluded laps (pit/caution/outlier) are still plotted, in grey and
        faded, for context — showing you what was excluded and why the
        clean-lap trend looks the way it does.
    ax : existing Axes to draw on; a new figure is created if omitted.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    clean_df = valid_rows(features_df)
    plot_compounds = compounds or sorted(clean_df["tire_compound"].dropna().unique().astype(str))

    if show_invalid_laps and "is_valid_lap" in features_df.columns:
        invalid_df = features_df[~features_df["is_valid_lap"]]
        if not invalid_df.empty:
            ax.scatter(
                invalid_df["tire_age_laps"],
                invalid_df["lap_time_delta_s"],
                color=INVALID_LAP_COLOR,
                alpha=INVALID_LAP_ALPHA,
                marker="x",
                label="Excluded laps (pit/caution/outlier)",
            )

    for compound in plot_compounds:
        compound_df = clean_df[clean_df["tire_compound"].astype(str) == compound].dropna(
            subset=["tire_age_laps", "lap_time_delta_s"]
        )
        if compound_df.empty:
            continue

        color = compound_color(compound)
        ax.scatter(
            compound_df["tire_age_laps"], compound_df["lap_time_delta_s"],
            color=color, alpha=0.7, label=f"{compound} (actual)", edgecolors="black", linewidths=0.3,
        )

        age_min, age_max = compound_df["tire_age_laps"].min(), compound_df["tire_age_laps"].max()
        age_range = np.linspace(age_min, age_max, 50)
        curve_input = pd.DataFrame({"tire_compound": [compound] * 50, "tire_age_laps": age_range})
        predicted = model.predict(curve_input)
        ax.plot(age_range, predicted, color=color, linewidth=2, linestyle="--", label=f"{compound} (fitted)")

    ax.set_xlabel("Tire age (laps)")
    ax.set_ylabel("Lap time delta (s)")
    ax.set_title("Tire degradation: actual vs. fitted")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    return ax


def plot_residuals(
    features_df: pd.DataFrame,
    model: TireDegradationModel,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot (actual - predicted) lap time delta vs. tire age, colored by compound.

    Only evaluated over valid laps (see visualization._utils.valid_rows) —
    residuals against excluded laps (e.g. a Safety Car lap) aren't a
    meaningful test of the degradation model's fit.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    df = valid_rows(features_df).dropna(subset=["tire_age_laps", "lap_time_delta_s", "tire_compound"])

    predicted = model.predict(df)
    residuals = df["lap_time_delta_s"].to_numpy() - predicted

    for compound in sorted(df["tire_compound"].astype(str).unique()):
        mask = df["tire_compound"].astype(str) == compound
        ax.scatter(
            df.loc[mask, "tire_age_laps"], residuals[mask.to_numpy()],
            color=compound_color(compound), alpha=0.7, label=compound, edgecolors="black", linewidths=0.3,
        )

    ax.axhline(0.0, color="black", linewidth=1, linestyle="-")
    ax.set_xlabel("Tire age (laps)")
    ax.set_ylabel("Residual: actual - predicted (s)")
    ax.set_title("Residuals vs. tire age")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    return ax
