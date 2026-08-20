"""
Group a driver's laps into stints (continuous runs on one set of tires).

FastF1 already tracks `Stint` (an integer that increments each pit stop)
and `TyreLife` (tire age in laps) per lap, so this module does not
recompute those from scratch — it validates them and wraps them into a
structured `StintInfo` object that the rest of the pipeline (features,
datasets) can rely on without re-deriving stint boundaries independently in
multiple places.

IMPORTANT — read before using `stint_length_laps` anywhere near a model:
`stint_length_laps` is only fully known once the stint has ended (the
driver has pitted, or the session ended). It is legitimate to compute as a
descriptive/summary statistic, and legitimate as a *label* if you are doing
retrospective analysis. It is NOT legitimate as a model *input feature* for
a model meant to predict remaining tire life mid-race, because the model
would not have that information at prediction time. `tire_age_laps`
(FastF1's `TyreLife`) is safe to use as an input feature because it only
depends on laps already completed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from f1_tire_model.preprocessing.lap_filters import is_under_caution

from f1_tire_model.preprocessing.lap_filters import is_valid_degradation_lap


@dataclass
class StintInfo:
    """One stint: a continuous run of laps on a single set of tires."""

    driver_code: str
    stint_number: int
    compound: str
    lap_numbers: list[int]
    is_complete: bool  # False if this is the driver's final (still-active) stint

    @property
    def stint_length_laps(self) -> int:
        """Total laps in this stint. See module docstring — hindsight-only quantity."""
        return len(self.lap_numbers)


def build_stints(laps_df: pd.DataFrame, driver_code: str) -> list[StintInfo]:
    """Build a list of StintInfo for one driver from a session's laps table.

    Parameters
    ----------
    laps_df : `session.laps` (or an equivalent DataFrame) containing at
        least Driver, Stint, Compound, LapNumber columns.
    driver_code : three-letter driver code to filter to.
    """
    driver_laps = laps_df[laps_df["Driver"] == driver_code].sort_values("LapNumber")

    if driver_laps.empty:
        return []

    max_stint_number = driver_laps["Stint"].max()

    stints: list[StintInfo] = []
    for stint_number, group in driver_laps.groupby("Stint"):
        compounds = group["Compound"].dropna().unique()
        compound = compounds[0] if len(compounds) > 0 else "UNKNOWN"

        if len(compounds) > 1:
            # A stint changing compound mid-way is not physically expected —
            # surface it loudly rather than silently picking the first value.
            raise ValueError(
                f"Stint {stint_number} for {driver_code} has multiple compounds: "
                f"{list(compounds)}. Check for a data quality issue in laps_df."
            )

        stints.append(
            StintInfo(
                driver_code=driver_code,
                stint_number=int(stint_number),
                compound=str(compound),
                lap_numbers=group["LapNumber"].astype(int).tolist(),
                is_complete=bool(int(stint_number) < max_stint_number),
            )
        )

    return stints


def stint_median_lap_time_s(laps_df: pd.DataFrame, stint: StintInfo) -> float:
    """Median lap time (seconds) for the given stint, over non-pit laps, non-caution laps only.

    Used as the `reference_lap_time_s` baseline in
    `lap_filters.is_valid_degradation_lap` / `is_lap_time_outlier` — using
    the stint's own median (rather than a whole-session median) means
    genuine degradation-driven slowdown isn't mistaken for an outlier.
    """
    stint_laps = laps_df[
        (laps_df["Driver"] == stint.driver_code)
        & (laps_df["LapNumber"].isin(stint.lap_numbers))
    ]
    is_pit = stint_laps["PitInTime"].notna() | stint_laps["PitOutTime"].notna()
    is_caution = stint_laps.apply(is_under_caution, axis=1)
    non_pit_laps = stint_laps[~is_pit & ~is_caution]

    lap_times = non_pit_laps["LapTime"].dropna()
    if lap_times.empty:
        raise ValueError(
            f"No non-pit laps with a valid LapTime in stint {stint.stint_number} "
            f"for {stint.driver_code}; cannot compute a reference lap time."
        )

    lap_times_s = lap_times.apply(
        lambda t: t.total_seconds() if hasattr(t, "total_seconds") else t
    )
    return float(lap_times_s.median())


def valid_degradation_laps(
    laps_df: pd.DataFrame, stint: StintInfo, outlier_factor: float
) -> list[int]:
    """Return the subset of `stint.lap_numbers` usable as clean degradation samples.

    Tire age (from FastF1's TyreLife) still counts every lap in the stint,
    including ones excluded here — only the *feature/target row* for an
    excluded lap should be dropped, not the tire-age counter itself.
    """
    reference = stint_median_lap_time_s(laps_df, stint)

    stint_laps = laps_df[
        (laps_df["Driver"] == stint.driver_code)
        & (laps_df["LapNumber"].isin(stint.lap_numbers))
    ]

    valid_laps = [
        int(lap["LapNumber"])
        for _, lap in stint_laps.iterrows()
        if is_valid_degradation_lap(lap, reference, outlier_factor)
    ]
    return valid_laps
