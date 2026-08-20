"""
Adapter between a live FastF1 `Session` object and the plain data structures
`datasets.builder` actually consumes.

Why this exists as a separate layer instead of having the builder call
FastF1 directly: it means `datasets.builder` — the piece with all the
actual dataset-assembly logic — never imports `fastf1` at all. Tests for
the builder can construct a `SessionInputs` from plain pandas DataFrames
with no FastF1 session, no cache, no network. If FastF1's `Session` API
changes in a future version, this is the only file that needs updating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import fastf1.core
import pandas as pd

from f1_tire_model.telemetry.loader import get_lap_telemetry

logger = logging.getLogger(__name__)


@dataclass
class SessionInputs:
    """Everything datasets.builder needs from a session, decoupled from FastF1."""

    laps_df: pd.DataFrame
    weather_df: pd.DataFrame
    corners: pd.DataFrame | None
    circuit_name: str
    reference_driver: str | None
    reference_lap_number: int | None
    #: Given (driver_code, lap_number), return cleaned telemetry or raise
    #: TelemetryNotAvailableError. Injected as a callable (rather than the
    #: builder holding a `Session` reference) purely for testability.
    telemetry_fetcher: Callable[[str, int], pd.DataFrame]


def build_session_inputs(session: fastf1.core.Session) -> SessionInputs:
    """Construct SessionInputs from a loaded FastF1 session."""
    laps_df = session.laps
    weather_df = session.weather_data

    try:
        corners = session.get_circuit_info().corners
    except Exception as exc:  # FastF1 circuit info isn't available for every track/session
        logger.warning("Could not retrieve circuit info: %s", exc)
        corners = None

    circuit_name = _extract_circuit_name(session)
    reference_driver, reference_lap_number = _pick_reference_lap(laps_df)

    def telemetry_fetcher(driver_code: str, lap_number: int) -> pd.DataFrame:
        return get_lap_telemetry(session, driver_code, lap_number)

    return SessionInputs(
        laps_df=laps_df,
        weather_df=weather_df,
        corners=corners,
        circuit_name=circuit_name,
        reference_driver=reference_driver,
        reference_lap_number=reference_lap_number,
        telemetry_fetcher=telemetry_fetcher,
    )


def _extract_circuit_name(session: fastf1.core.Session) -> str:
    try:
        return str(session.event["Location"])
    except Exception:
        return "Unknown"


def _pick_reference_lap(laps_df: pd.DataFrame) -> tuple[str | None, int | None]:
    """Pick the fastest non-deleted lap in the session to use for
    session-level track features (corner speeds, DRS zones).

    A Qualifying session's fastest lap is the ideal reference (DRS enabled,
    representative pace); this function doesn't know the session type, so
    it just picks the fastest valid lap available, whatever session this is.
    """
    if laps_df.empty:
        return None, None

    candidates = laps_df[~laps_df.get("Deleted", pd.Series(False, index=laps_df.index)).fillna(False)]
    candidates = candidates.dropna(subset=["LapTime"])

    if candidates.empty:
        return None, None

    fastest = candidates.loc[candidates["LapTime"].idxmin()]
    return str(fastest["Driver"]), int(fastest["LapNumber"])
