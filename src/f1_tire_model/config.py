"""
Central configuration for the pipeline.

Why a single settings object instead of scattering constants across files:
- One place to change the FastF1 cache directory, target seasons, or
  physical thresholds (e.g., what counts as a "high-speed corner").
- Makes the pipeline testable: tests can construct a Settings() with a
  temp cache dir instead of touching your real cache.
- Every module that needs configuration takes a `settings: Settings`
  argument (dependency injection) rather than importing globals directly.
  This keeps modules unit-testable in isolation.

If the project grows to need environment-specific overrides (e.g. a
different cache path on a lab machine vs. a laptop), this is the natural
place to add environment-variable overrides later — the rest of the
codebase would not need to change.
"""

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Immutable pipeline configuration.

    frozen=True prevents accidental mutation of shared config at runtime,
    which matters once multiple pipeline stages read the same object.
    """

    # --- Storage ---
    cache_dir: Path = REPO_ROOT / "data_cache"
    dataset_output_dir: Path = REPO_ROOT / "src" / "f1_tire_model" / "datasets" / "output"

    # --- Scope of data to pull ---
    seasons: tuple[int, ...] = (2025, )

    # --- Track classification thresholds ---
    # Corners with an average speed above this are treated as "high speed"
    # when computing the track-level "high-speed vs low-speed circuit" feature.
    high_speed_corner_threshold_kph: float = 200.0

    # --- Lap filtering ---
    # Laps slower than (median_lap_time * this factor) are treated as outliers
    # (e.g., laps under Safety Car / VSC, or laps with an incident) and
    # excluded from degradation feature computation by default.
    outlier_lap_time_factor: float = 1.15

    # --- Physics assumptions used by feature extractors ---
    # These are ESTIMATES, not measured values — FastF1 does not expose car
    # mass or actual fuel load. They should be treated as tunable parameters,
    # not ground truth, and revisited once real feature values are validated
    # against known race behavior (e.g. known heavy-fuel opening stints).

    # Approximate F1 car + driver + fuel mass (kg). Current-generation cars
    # have a minimum mass around 798kg (car+driver, excluding fuel); this
    # adds a rough mid-race fuel load. Used only for brake energy estimation,
    # where being off by ~5% has a small effect on the feature's *ranking*
    # usefulness even if the absolute kJ value isn't exact.
    car_mass_kg: float = 860.0

    # Starting fuel load estimate for a full race distance (kg). Actual
    # value varies by team strategy and track (fuel-heavy vs fuel-light
    # circuits); this is a single global approximation for V1.
    initial_fuel_kg: float = 100.0

    # Approximate fuel burned per lap (kg/lap), assuming roughly linear
    # burn across a race distance of ~60 laps. Real burn rate varies lap to
    # lap (fuel-saving laps, safety car laps burn less); linear is a
    # deliberate V1 simplification to revisit once real fuel data/telemetry
    # (if available) can validate it.
    fuel_burn_rate_kg_per_lap: float = 1.6

    # --- Corner speed feature extraction ---
    # Window (meters) around each circuit-info corner marker used to define
    # "entry" (before marker), "apex" (min speed within window), and "exit"
    # (after marker). FastF1's corner markers are approximate reference
    # points, not exact apex geometry, so these are proxies, not precise
    # entry/apex/exit measurements.
    corner_entry_offset_m: float = 50.0
    corner_exit_offset_m: float = 50.0
    corner_apex_window_m: float = 15.0

    def ensure_dirs(self) -> None:
        """Create cache/output directories if they don't exist yet."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_output_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = Settings()
