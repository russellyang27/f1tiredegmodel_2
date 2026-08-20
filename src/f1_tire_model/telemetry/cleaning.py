"""
Cleaning and resampling for a single lap's telemetry.

Two distinct operations happen here, kept as separate functions on purpose:

1. `clean_telemetry` — fixes data quality problems (duplicate timestamps,
   small gaps, out-of-range values) WITHOUT changing the sampling grid.
   This is the right output for anything that cares about raw time-series
   shape (e.g. plotting a speed trace).

2. `resample_by_distance` — re-samples cleaned telemetry onto a uniform
   distance grid. This matters because FastF1's raw sampling is roughly
   uniform in *time*, not distance: a car covers more distance per sample
   on a straight (high speed) than in a slow corner. Comparing "speed at
   the apex" across laps/drivers only makes sense if you're comparing at
   the same *distance* into the lap, on a consistent grid — otherwise two
   laps can have samples at slightly different physical points on track,
   which is exactly the kind of subtle bug that would silently corrupt the
   corner-speed features listed in the project spec (corner entry/apex/exit
   speed). Feature extraction (features/) should build on this resampled
   output, not the raw one.

Both functions return a *new* DataFrame; neither mutates its input, which
matters once multiple feature extractors read the same telemetry object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_tire_model.telemetry import schema
from f1_tire_model.telemetry.exceptions import TelemetryNotAvailableError

_NUMERIC_CHANNELS_TO_INTERPOLATE = (
    schema.SPEED,
    schema.THROTTLE,
    schema.BRAKE,
    schema.N_GEAR,
    schema.RPM,
    schema.X,
    schema.Y,
    schema.Z,
)


def clean_telemetry(tel: pd.DataFrame, *, max_interpolate_gap: int = 5) -> pd.DataFrame:
    """Clean one lap's raw telemetry DataFrame.

    Steps (in order — order matters: dedupe before interpolate, interpolate
    before clipping, so clipping sees the final values):
      1. Validate required channels are present.
      2. Sort by time and drop duplicate timestamps (FastF1 occasionally
         returns repeated samples at session/lap boundaries).
      3. Linearly interpolate short gaps (<= max_interpolate_gap consecutive
         missing samples) in numeric channels; longer gaps are left as NaN
         rather than guessed, since interpolating across a large gap could
         fabricate a corner that didn't happen.
      4. Clip physically implausible Speed values (sensor glitches) rather
         than dropping the whole row, to keep the time series intact.
      5. Enforce non-decreasing Distance (a rare telemetry glitch can produce
         a small backwards jump; degradation features assume the lap
         progresses monotonically).

    Raises
    ------
    TelemetryNotAvailableError
        If the input is empty or missing required channels.
    """
    if tel is None or tel.empty:
        raise TelemetryNotAvailableError("Telemetry DataFrame is empty.")

    missing = set(schema.REQUIRED_CHANNELS) - set(tel.columns)
    if missing:
        raise TelemetryNotAvailableError(f"Telemetry missing required channels: {sorted(missing)}")

    df = tel.sort_values(schema.TIME).drop_duplicates(subset=schema.TIME, keep="first")
    df = df.reset_index(drop=True)

    for col in _NUMERIC_CHANNELS_TO_INTERPOLATE:
        if col in df.columns:
            df[col] = df[col].interpolate(
                method="linear", limit=max_interpolate_gap, limit_area="inside"
            )

    df[schema.SPEED] = df[schema.SPEED].clip(
        lower=schema.MIN_PLAUSIBLE_SPEED_KPH, upper=schema.MAX_PLAUSIBLE_SPEED_KPH
    )

    # Enforce monotonic, non-decreasing distance via cumulative max — this
    # removes small backwards glitches while never moving a sample forward.
    df[schema.DISTANCE] = df[schema.DISTANCE].cummax()

    return df


def resample_by_distance(df: pd.DataFrame, *, step_meters: float = 5.0) -> pd.DataFrame:
    """Resample cleaned telemetry onto a uniform distance grid.

    Parameters
    ----------
    df : output of `clean_telemetry` (must have monotonic Distance).
    step_meters : grid spacing. 5m is a reasonable default for corner-speed
        analysis (fine enough to localize an apex, coarse enough to keep
        dataset size manageable across a full season).

    Notes
    -----
    Uses `np.interp` per-channel rather than `pandas.DataFrame.resample`
    because resampling is being done on the `Distance` axis, not the `Time`
    axis — `Distance` is not evenly spaced in the raw data, so this is a
    distance-domain interpolation, not a time-domain resample.
    """
    if not df[schema.DISTANCE].is_monotonic_increasing:
        raise TelemetryNotAvailableError(
            "Distance column must be non-decreasing before resampling; run clean_telemetry first."
        )

    distance = df[schema.DISTANCE].to_numpy()
    grid = np.arange(distance.min(), distance.max(), step_meters)

    resampled = {schema.DISTANCE: grid}
    for col in _NUMERIC_CHANNELS_TO_INTERPOLATE:
        if col in df.columns:
            resampled[col] = np.interp(grid, distance, df[col].to_numpy())

    return pd.DataFrame(resampled)
