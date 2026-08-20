from __future__ import annotations

import pandas as pd
import pytest

from f1_tire_model.preprocessing.segmentation import (
    build_stints,
    stint_median_lap_time_s,
    valid_degradation_laps,
)


def _synthetic_laps_df() -> pd.DataFrame:
    """Two drivers, VER with 2 stints (5 laps then 3 laps), HAM with 1 stint.

    Lap 3 for VER's first stint is a pit-in lap (should be excluded from
    degradation laps but still counted for tire age / stint membership).
    Lap 8 (VER's second stint, lap 2) is under Safety Car.
    """
    rows = []
    # VER stint 1: laps 1-5, laps 1,2,4,5 clean; lap 3 is a pit-in lap
    for lap_num in range(1, 6):
        rows.append(
            {
                "Driver": "VER",
                "LapNumber": lap_num,
                "Stint": 1,
                "Compound": "MEDIUM",
                "TyreLife": lap_num,
                "LapTime": pd.Timedelta(seconds=90.0 + lap_num * 0.2),
                "PitInTime": pd.Timestamp("2024-01-01") if lap_num == 3 else pd.NaT,
                "PitOutTime": pd.NaT,
                "TrackStatus": "1",
                "Deleted": False,
            }
        )
    # VER stint 2: laps 6-8, lap 8 under Safety Car (TrackStatus contains '4')
    for i, lap_num in enumerate(range(6, 9)):
        rows.append(
            {
                "Driver": "VER",
                "LapNumber": lap_num,
                "Stint": 2,
                "Compound": "HARD",
                "TyreLife": i + 1,
                "LapTime": pd.Timedelta(seconds=91.0),
                "PitInTime": pd.NaT,
                "PitOutTime": pd.Timestamp("2024-01-01") if lap_num == 6 else pd.NaT,
                "TrackStatus": "4" if lap_num == 8 else "1",
                "Deleted": False,
            }
        )
    # HAM stint 1: laps 1-4, all clean
    for lap_num in range(1, 5):
        rows.append(
            {
                "Driver": "HAM",
                "LapNumber": lap_num,
                "Stint": 1,
                "Compound": "SOFT",
                "TyreLife": lap_num,
                "LapTime": pd.Timedelta(seconds=89.0),
                "PitInTime": pd.NaT,
                "PitOutTime": pd.NaT,
                "TrackStatus": "1",
                "Deleted": False,
            }
        )

    return pd.DataFrame(rows)


def test_build_stints_groups_correctly():
    laps_df = _synthetic_laps_df()

    stints = build_stints(laps_df, "VER")

    assert len(stints) == 2
    assert stints[0].compound == "MEDIUM"
    assert stints[0].lap_numbers == [1, 2, 3, 4, 5]
    assert stints[0].stint_length_laps == 5
    assert stints[0].is_complete is True  # a later stint exists

    assert stints[1].compound == "HARD"
    assert stints[1].is_complete is False  # this is VER's last stint in the data


def test_build_stints_empty_for_unknown_driver():
    laps_df = _synthetic_laps_df()
    assert build_stints(laps_df, "NOBODY") == []


def test_build_stints_raises_on_mixed_compound_within_stint():
    laps_df = _synthetic_laps_df()
    laps_df.loc[(laps_df["Driver"] == "VER") & (laps_df["LapNumber"] == 2), "Compound"] = "HARD"

    with pytest.raises(ValueError, match="multiple compounds"):
        build_stints(laps_df, "VER")


def test_stint_median_lap_time_excludes_pit_laps():
    laps_df = _synthetic_laps_df()
    stint = build_stints(laps_df, "VER")[0]

    median = stint_median_lap_time_s(laps_df, stint)

    # Median should be computed only over laps 1,2,4,5 (lap 3 is a pit lap)
    expected_times = [90.2, 90.4, 90.8, 91.0]
    assert median == pytest.approx(pd.Series(expected_times).median())


def test_valid_degradation_laps_excludes_pit_and_caution_laps():
    laps_df = _synthetic_laps_df()
    ver_stints = build_stints(laps_df, "VER")

    stint1_valid = valid_degradation_laps(laps_df, ver_stints[0], outlier_factor=1.15)
    assert 3 not in stint1_valid  # pit-in lap excluded
    # Lap 1 is now also excluded -- it's the race-start (standing start) lap,
    # not just "not a pit lap". See lap_filters.is_race_start_lap.
    assert set(stint1_valid) == {2, 4, 5}

    stint2_valid = valid_degradation_laps(laps_df, ver_stints[1], outlier_factor=1.15)
    assert 8 not in stint2_valid  # Safety Car lap excluded
    assert 6 not in stint2_valid  # pit-out lap excluded
    assert stint2_valid == [7]

def test_stint_median_lap_time_excludes_safety_car_laps():
    """Regression test for a real-data bug: the median used as the outlier
    reference must exclude Safety Car laps, not just pit laps. Without this,
    a stint that starts under caution (very slow laps) inflates the median,
    which inflates the outlier threshold, letting genuinely anomalous
    green-flag laps slip through as 'valid'.
    """
    rows = []
    # Laps 1-3: under Safety Car, very slow (would blow up the median if included).
    for lap_num, lap_time in [(1, 130.0), (2, 128.0), (3, 125.0)]:
        rows.append(
            {
                "Driver": "HAM", "LapNumber": lap_num, "Stint": 1, "Compound": "HARD",
                "TyreLife": lap_num, "LapTime": pd.Timedelta(seconds=lap_time),
                "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "TrackStatus": "4", "Deleted": False,
            }
        )
    # Laps 4-6: clean green-flag racing, genuine pace ~85s.
    for lap_num, lap_time in [(4, 85.0), (5, 85.5), (6, 86.0)]:
        rows.append(
            {
                "Driver": "HAM", "LapNumber": lap_num, "Stint": 1, "Compound": "HARD",
                "TyreLife": lap_num, "LapTime": pd.Timedelta(seconds=lap_time),
                "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "TrackStatus": "1", "Deleted": False,
            }
        )
    laps_df = pd.DataFrame(rows)
    stint = build_stints(laps_df, "HAM")[0]

    median = stint_median_lap_time_s(laps_df, stint)

    # Median must reflect only the clean 85.0/85.5/86.0 laps, not the SC laps.
    assert median == pytest.approx(85.5)

    valid_laps = valid_degradation_laps(laps_df, stint, outlier_factor=1.15)
    assert set(valid_laps) == {4, 5, 6}

    