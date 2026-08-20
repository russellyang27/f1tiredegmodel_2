"""
Lap-level validity checks: deciding which laps are usable for degradation
analysis.

This module operates on rows of FastF1's `session.laps` DataFrame (each row
is one lap, with columns like `Deleted`, `PitInTime`, `PitOutTime`,
`TrackStatus`, `LapTime`). It does NOT touch telemetry — telemetry cleaning
(telemetry/cleaning.py) and lap-level validity are separate concerns:
a lap can have perfectly clean telemetry and still be invalid for
degradation analysis (e.g. it was driven behind the Safety Car).

Engineering assumptions (please validate against real data — see notes at
bottom of this file):
- FastF1's `TrackStatus` is a string of concatenated single-digit codes
  representing every status active at some point during the lap
  (e.g. "12" = AllClear then Yellow). We treat a lap as "under caution"
  if ANY of the Safety Car (4), Virtual Safety Car (6/7), or Red Flag (5)
  codes appear anywhere in that string — a conservative choice, since even
  a partial-lap Safety Car period changes tire/brake temperatures enough to
  distort a "clean" degradation reading.
- A lap is a "pit lap" if either PitInTime or PitOutTime is not null —
  both in-laps and out-laps have artificially slower pace and shouldn't be
  used as clean degradation samples, even though they still belong to a
  stint for tire-age counting purposes.
"""

from __future__ import annotations

import pandas as pd

# TrackStatus codes per FastF1 documentation that indicate non-representative
# conditions for a "clean" degradation lap.
_CAUTION_TRACK_STATUS_CODES = {"4", "5", "6", "7"}  # SC, Red Flag, VSC, VSC Ending


def is_deleted(lap: pd.Series) -> bool:
    """True if the lap time was deleted (e.g. for a track limits violation)."""
    return bool(lap.get("Deleted", False))


def is_pit_lap(lap: pd.Series) -> bool:
    """True if this lap includes a pit entry or exit."""
    return pd.notna(lap.get("PitInTime")) or pd.notna(lap.get("PitOutTime"))


def is_under_caution(lap: pd.Series) -> bool:
    """True if any Safety Car / VSC / Red Flag status code is present for this lap."""
    track_status = lap.get("TrackStatus")
    if pd.isna(track_status):
        return False
    codes = set(str(track_status))
    return bool(codes & _CAUTION_TRACK_STATUS_CODES)


def is_lap_time_outlier(lap: pd.Series, reference_lap_time_s: float, factor: float) -> bool:
    """True if this lap's time exceeds `reference_lap_time_s * factor`.

    `reference_lap_time_s` is typically the stint's median clean lap time
    (computed by the caller — see `segmentation.stint_median_lap_time`),
    not a global session median, since degradation itself makes later laps
    in a stint slower; comparing against the *stint's own* median avoids
    flagging genuine degradation as an outlier.
    """
    lap_time = lap.get("LapTime")
    if pd.isna(lap_time):
        return True  # missing lap time cannot be used regardless

    lap_time_s = lap_time.total_seconds() if hasattr(lap_time, "total_seconds") else lap_time
    return lap_time_s > reference_lap_time_s * factor


def is_valid_degradation_lap(
    lap: pd.Series, reference_lap_time_s: float, outlier_factor: float
) -> bool:
    """Combined check: is this lap usable as a clean degradation data point?

    A lap can still be *part of a stint* (for tire-age counting) while being
    excluded here — segmentation and validity are deliberately separate so
    tire age stays continuous even when individual laps are filtered out of
    the feature/target dataset.
    """
    return not (
        is_deleted(lap)
        or is_pit_lap(lap)
        or is_under_caution(lap)
        or is_race_start_lap(lap)
        or is_lap_time_outlier(lap, reference_lap_time_s, outlier_factor)
    )

def is_race_start_lap(lap: pd.Series) -> bool:
    """True if this is the very first lap of the session (LapNumber == 1).

    The opening lap of a race is systematically slower than normal
    green-flag pace due to the standing start and cars bunched through the
    first corners -- not tire degradation. Confirmed via real 2025 data:
    a stint 1 showed enormous (-6 to -13s) apparent 'improvement' in
    lap_time_delta_s that vanished in stint 2+ (pit-exit stints, not a
    standing start), because LapNumber==1 was being used as the zero
    baseline despite being artificially slow. Treated the same way as pit
    in/out laps: excluded from being either a valid degradation sample or
    a baseline reference lap.
    """
    lap_number = lap.get("LapNumber")
    return pd.notna(lap_number) and int(lap_number) == 1


# --- Validation notes for real data (not yet verified against a live session) ---
#
# TODO before trusting this on real telemetry:
# 1. Confirm FastF1's `TrackStatus` semantics against a session you know had
#    a Safety Car (e.g. search race history for a race with a known SC
#    period) and check the codes present line up with codes 4/6/7.
# 2. Confirm `Deleted` is populated the way expected in both Qualifying and
#    Race sessions — FastF1 fills lap-invalidation flags slightly differently
#    across session types.
# 3. Spot-check a few `is_pit_lap` results against the session's official
#    pit stop log (FastF1 exposes `session.laps.pick_box_laps()` for
#    cross-checking).
