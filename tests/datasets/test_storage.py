from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from f1_tire_model.config import Settings
from f1_tire_model.datasets.schema import LapFeatureRecord
from f1_tire_model.datasets.storage import fragment_path, load_dataset, write_fragment


def _sample_record(lap_number: int, round_number: int = 1) -> LapFeatureRecord:
    return LapFeatureRecord(
        season=2024,
        round_number=round_number,
        session_name="R",
        driver_code="VER",
        lap_number=lap_number,
        tire_compound="SOFT",
        tire_age_laps=lap_number,
        stint_number=1,
        lap_time_s=90.0 + lap_number,
    )


def test_write_and_load_fragment_roundtrip(tmp_path):
    settings = Settings(cache_dir=tmp_path / "cache", dataset_output_dir=tmp_path / "output")
    records = [_sample_record(1), _sample_record(2)]

    path = write_fragment(records, season=2024, round_number=1, session_name="R", settings=settings)

    assert path.exists()
    assert path == fragment_path(2024, 1, "R", settings=settings)

    df = pd.read_parquet(path)
    assert len(df) == 2
    assert set(df["lap_number"]) == {1, 2}


def test_load_dataset_returns_empty_frame_when_no_fragments(tmp_path):
    settings = Settings(cache_dir=tmp_path / "cache", dataset_output_dir=tmp_path / "output")

    df = load_dataset(settings=settings)

    assert df.empty


def test_load_dataset_concatenates_multiple_fragments(tmp_path):
    settings = Settings(cache_dir=tmp_path / "cache", dataset_output_dir=tmp_path / "output")

    write_fragment([_sample_record(1, round_number=1)], season=2024, round_number=1, session_name="R", settings=settings)
    write_fragment([_sample_record(1, round_number=2)], season=2024, round_number=2, session_name="R", settings=settings)

    df = load_dataset(settings=settings)

    assert len(df) == 2
    assert set(df["round_number"]) == {1, 2}
