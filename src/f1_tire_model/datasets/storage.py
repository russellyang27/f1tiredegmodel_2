"""
Storage for the dataset the builder produces.

Why per-session parquet fragments instead of one growing file:
- A season-long build is many separate FastF1 API calls; writing one
  fragment per session means a build that gets interrupted (network issue,
  rate limit, laptop sleeps) has already-saved progress, and — combined
  with the manifest (manifest.py) — doesn't need to reprocess sessions it
  already has.
- Parquet preserves dtypes (categoricals stay categorical) across
  save/load, which matters once you're feeding this into ML models later.
- Concatenating fragments back into one DataFrame for modeling is a single
  function call (`load_dataset`) — the fragmentation is an implementation
  detail, not something callers need to think about.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1_tire_model.config import DEFAULT_SETTINGS, Settings
from f1_tire_model.datasets.schema import LapFeatureRecord, records_to_frame


def fragment_path(
    season: int, round_number: int, session_name: str, *, settings: Settings = DEFAULT_SETTINGS
) -> Path:
    """Deterministic path for one session's dataset fragment."""
    filename = f"{season}_{round_number:02d}_{session_name}.parquet"
    return settings.dataset_output_dir / filename


def write_fragment(
    records: list[LapFeatureRecord],
    *,
    season: int,
    round_number: int,
    session_name: str,
    settings: Settings = DEFAULT_SETTINGS,
) -> Path:
    """Write one session's records to its parquet fragment. Returns the path written."""
    settings.ensure_dirs()
    path = fragment_path(season, round_number, session_name, settings=settings)

    df = records_to_frame(records)
    df.to_parquet(path, index=False)
    return path


def load_fragment(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_dataset(settings: Settings = DEFAULT_SETTINGS) -> pd.DataFrame:
    """Load and concatenate every fragment in the dataset output directory.

    Returns an empty DataFrame (not an error) if no fragments exist yet —
    callers building tooling around this (e.g. a notebook that runs before
    any data has been built) shouldn't need a special case for "no data yet".
    """
    fragment_paths = sorted(settings.dataset_output_dir.glob("*.parquet"))
    if not fragment_paths:
        return pd.DataFrame()

    frames = [load_fragment(p) for p in fragment_paths]
    return pd.concat(frames, ignore_index=True)
