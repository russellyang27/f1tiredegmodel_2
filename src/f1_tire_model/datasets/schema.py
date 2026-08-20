"""
Structured schema for the dataset this pipeline produces.

Why a dataclass schema instead of building dicts/DataFrames ad hoc in each
script:
- Autocomplete + type checking while writing feature extraction code.
- One authoritative list of every column the dataset will ever have — as you
  add features over the coming months, you add a field here once, and both
  the physics model and (later) every ML model see it consistently.
- `to_frame()` gives a single, obvious conversion point from "one record" to
  "one row of a pandas DataFrame" — the DataFrame is the shared currency
  between features/, models/, and visualization/.

This is intentionally flat (no nested objects) because a flat structure
maps 1:1 to a DataFrame row, which is what every downstream model
(physics or ML) will actually consume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import ClassVar

import pandas as pd


@dataclass
class LapFeatureRecord:
    """One row = one lap, fully described for degradation modeling.

    Grouped by comment, matching the feature categories in the project spec
    (tire / driver / vehicle / track / environment / strategy) plus
    identifying keys and the eventual prediction target(s).
    """

    # --- Identifiers ---
    season: int
    round_number: int
    session_name: str  # e.g. "R", "Q", "FP1"
    driver_code: str
    lap_number: int

    # --- Tire features ---
    tire_compound: str
    tire_age_laps: int
    stint_number: int
    stint_length_laps: int | None = None  # None until stint is complete

    # --- Driver features ---
    avg_braking_decel_ms2: float | None = None
    peak_braking_decel_ms2: float | None = None
    avg_throttle_pct: float | None = None
    max_acceleration_ms2: float | None = None
    driver_consistency_std_s: float | None = None

    # --- Vehicle features ---
    avg_speed_kph: float | None = None
    max_speed_kph: float | None = None
    brake_energy_estimate_kj: float | None = None
    longitudinal_accel_ms2: float | None = None
    corner_entry_speed_kph: float | None = None
    apex_speed_kph: float | None = None
    corner_exit_speed_kph: float | None = None

    # --- Track features ---
    circuit: str | None = None
    num_corners: int | None = None
    avg_corner_speed_kph: float | None = None
    is_high_speed_circuit: bool | None = None
    num_drs_zones: int | None = None

    # --- Environment features ---
    air_temp_c: float | None = None
    track_temp_c: float | None = None
    weather: str | None = None
    is_raining: bool | None = None

    # --- Strategy features ---
    is_pit_lap: bool = False
    fuel_load_estimate_kg: float | None = None
    is_safety_car_lap: bool = False

    # --- Data quality flag (set by the dataset builder, not a feature extractor) ---
    # True if this lap passed preprocessing.lap_filters.is_valid_degradation_lap
    # (not deleted, not a pit lap, not under caution, not a lap-time outlier
    # relative to its stint). Rows with is_valid_lap=False are kept (useful
    # for descriptive plots of a full stint/race) but should be filtered out
    # before training any degradation model — see NON_FEATURE_COLUMNS below,
    # which does NOT include this column on purpose: it's a row filter, not
    # a per-row input feature.
    is_valid_lap: bool = True

    # --- Targets (populated by preprocessing, not by feature extractors) ---
    lap_time_s: float | None = None
    lap_time_delta_s: float | None = None  # vs. stint's first clean lap
    remaining_tire_life_laps: int | None = None

    #: Columns that should never be used as model *input* features —
    #: identifiers and targets. Kept here (not recomputed elsewhere) so
    #: feature-selection code has one source of truth.
    NON_FEATURE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "season",
        "round_number",
        "session_name",
        "driver_code",
        "lap_number",
        "is_valid_lap",
        "lap_time_s",
        "lap_time_delta_s",
        "remaining_tire_life_laps",
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def candidate_feature_columns(cls) -> list[str]:
        """All fields except identifiers/targets — a sensible default
        feature set for a first model, before doing any feature selection."""
        return [f for f in cls.field_names() if f not in cls.NON_FEATURE_COLUMNS]


def records_to_frame(records: list[LapFeatureRecord]) -> pd.DataFrame:
    """Convert a list of records into the canonical DataFrame representation.

    This is the single conversion point between "list of typed records"
    (what feature extractors produce) and "DataFrame" (what models and
    visualization consume). Keeping it in one function means storage
    format changes (e.g. dtype tuning for parquet) happen in one place.
    """
    df = pd.DataFrame([r.to_dict() for r in records])

    # Explicit dtypes matter for parquet storage and for categorical
    # handling in future ML models (e.g. one-hot/embedding of compound).
    categorical_cols = ["session_name", "driver_code", "tire_compound", "circuit", "weather"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df
