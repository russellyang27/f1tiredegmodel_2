"""
Retrieve a single lap's telemetry from a loaded FastF1 session and run it
through the cleaning pipeline.

This is the one place that calls FastF1's `Lap.get_telemetry()` — if a
future FastF1 version changes that API, this function is the only thing
that needs updating.
"""

from __future__ import annotations

import fastf1.core
import pandas as pd

from f1_tire_model.telemetry.cleaning import clean_telemetry
from f1_tire_model.telemetry.exceptions import TelemetryNotAvailableError


def get_lap_telemetry(
    session: fastf1.core.Session,
    driver_code: str,
    lap_number: int,
    *,
    clean: bool = True,
) -> pd.DataFrame:
    """Return telemetry for one driver's one lap.

    Parameters
    ----------
    session : a FastF1 session already loaded with `laps=True, telemetry=True`
        (see `data.session_loader.load_session`).
    driver_code : three-letter driver code, e.g. "VER".
    lap_number : lap number within the session.
    clean : if True (default), run `clean_telemetry` before returning.
        Set False only for debugging/inspecting raw FastF1 output.

    Raises
    ------
    TelemetryNotAvailableError
        If the lap doesn't exist for this driver, or its telemetry is
        empty/unusable.
    """
    # NOTE: `pick_drivers` (plural) is the current FastF1 API (>=3.3). Older
    # FastF1 versions used the singular `pick_driver`. If you pin an older
    # FastF1 version, this is the line to change.
    driver_laps = session.laps.pick_drivers(driver_code)
    lap_rows = driver_laps[driver_laps["LapNumber"] == lap_number]

    if lap_rows.empty:
        raise TelemetryNotAvailableError(
            f"No lap {lap_number} found for driver {driver_code} in this session."
        )

    lap = lap_rows.iloc[0]

    try:
        tel = lap.get_telemetry()
    except Exception as exc:  # FastF1 can raise several internal exception types
        raise TelemetryNotAvailableError(
            f"Failed to retrieve telemetry for {driver_code} lap {lap_number}: {exc}"
        ) from exc

    if not clean:
        return tel

    return clean_telemetry(tel)
