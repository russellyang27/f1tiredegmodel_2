"""
Small numeric helpers shared by feature extractors, kept separate from
telemetry/cleaning.py because these are feature-computation math (physics
derivations), not data-quality cleaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_tire_model.telemetry import schema

KPH_TO_MS = 1000.0 / 3600.0


def longitudinal_acceleration_ms2(tel: pd.DataFrame) -> np.ndarray:
    """Per-sample longitudinal acceleration (m/s^2) from time-indexed telemetry.

    Computed as d(speed)/d(time) using consecutive samples. Requires
    `tel` to be cleaned (clean_telemetry) so Time is monotonic and gaps are
    already interpolated — differentiating raw, un-cleaned telemetry would
    amplify noise and duplicate-timestamp glitches into huge spurious
    acceleration spikes.

    Returns an array one shorter than len(tel) (there's no acceleration
    defined for the very first sample); callers should align accordingly.
    """
    speed_ms = tel[schema.SPEED].to_numpy() * KPH_TO_MS
    time_s = tel[schema.TIME].dt.total_seconds().to_numpy()

    dt = np.diff(time_s)
    dv = np.diff(speed_ms)

    # Guard against division by ~0 for any residual duplicate/near-duplicate
    # timestamps that survived cleaning.
    dt_safe = np.where(dt <= 1e-6, np.nan, dt)
    return dv / dt_safe


def is_braking(tel: pd.DataFrame) -> np.ndarray:
    """Boolean mask of samples where the driver is braking.

    FastF1's Brake channel is either a 0-100 percentage or a bool depending
    on data source/version; treating any value > 0 as "braking" handles
    both representations without needing to know which one a given session
    returned.
    """
    return tel[schema.BRAKE].to_numpy() > 0
