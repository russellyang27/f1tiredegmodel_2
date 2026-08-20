from __future__ import annotations

import pandas as pd
import pytest

from f1_tire_model.datasets.splitting import (
    check_split_compound_coverage,
    split_by_race,
    split_by_season,
    split_chronological,
    split_summary,
)


def _synthetic_multi_race_df() -> pd.DataFrame:
    """5 races across 2 seasons, 3 laps each, all SOFT compound except the
    very last race, which introduces INTERMEDIATE (simulating a wet race)."""
    rows = []
    races = [
        (2023, 1, "R"),
        (2023, 2, "R"),
        (2023, 3, "R"),
        (2024, 1, "R"),
        (2024, 2, "R"),
    ]
    for season, round_number, session_name in races:
        compound = "INTERMEDIATE" if (season, round_number) == (2024, 2) else "SOFT"
        for lap in range(1, 4):
            rows.append(
                {
                    "season": season,
                    "round_number": round_number,
                    "session_name": session_name,
                    "driver_code": "VER",
                    "lap_number": lap,
                    "tire_compound": compound,
                    "tire_age_laps": lap,
                    "lap_time_delta_s": 0.1 * lap,
                }
            )
    return pd.DataFrame(rows)


def test_split_by_race_keeps_races_disjoint():
    df = _synthetic_multi_race_df()

    train_df, test_df = split_by_race(df, test_size=0.4, random_state=42)

    train_keys = set(zip(train_df["season"], train_df["round_number"], train_df["session_name"]))
    test_keys = set(zip(test_df["season"], test_df["round_number"], test_df["session_name"]))

    assert train_keys.isdisjoint(test_keys)
    assert len(train_keys) + len(test_keys) == 5


def test_split_by_race_is_reproducible_with_same_seed():
    df = _synthetic_multi_race_df()

    train1, test1 = split_by_race(df, test_size=0.4, random_state=7)
    train2, test2 = split_by_race(df, test_size=0.4, random_state=7)

    assert set(test1["round_number"]) == set(test2["round_number"])


def test_split_by_race_raises_with_single_race():
    df = _synthetic_multi_race_df()
    one_race = df[(df["season"] == 2023) & (df["round_number"] == 1)]

    with pytest.raises(ValueError, match="at least 2 distinct races"):
        split_by_race(one_race)


def test_split_chronological_holds_out_latest_races():
    df = _synthetic_multi_race_df()

    train_df, test_df = split_chronological(df, test_size=0.4)

    # The 2 most recent races (2024 round 1, 2024 round 2) should be in test.
    test_keys = set(zip(test_df["season"], test_df["round_number"]))
    assert test_keys == {(2024, 1), (2024, 2)}

    train_keys = set(zip(train_df["season"], train_df["round_number"]))
    assert train_keys == {(2023, 1), (2023, 2), (2023, 3)}


def test_split_chronological_raises_with_single_race():
    df = _synthetic_multi_race_df()
    one_race = df[(df["season"] == 2023) & (df["round_number"] == 1)]

    with pytest.raises(ValueError, match="at least 2 distinct races"):
        split_chronological(one_race)


def test_split_by_season_assigns_correctly():
    df = _synthetic_multi_race_df()

    train_df, test_df = split_by_season(df, train_seasons=[2023], test_seasons=[2024])

    assert set(train_df["season"]) == {2023}
    assert set(test_df["season"]) == {2024}


def test_split_by_season_raises_on_empty_result():
    df = _synthetic_multi_race_df()

    with pytest.raises(ValueError, match="No rows found for test_seasons"):
        split_by_season(df, train_seasons=[2023], test_seasons=[2099])


def test_check_split_compound_coverage_detects_uncovered_compound():
    df = _synthetic_multi_race_df()
    train_df, test_df = split_by_season(df, train_seasons=[2023], test_seasons=[2024])

    uncovered = check_split_compound_coverage(train_df, test_df)

    # 2024 round 2 introduces INTERMEDIATE, never seen in the 2023 training data.
    assert uncovered == ["INTERMEDIATE"]


def test_check_split_compound_coverage_empty_when_fully_covered():
    df = _synthetic_multi_race_df()
    # Exclude the wet race so both splits only ever see SOFT.
    df = df[~((df["season"] == 2024) & (df["round_number"] == 2))]

    train_df, test_df = split_chronological(df, test_size=0.5)

    assert check_split_compound_coverage(train_df, test_df) == []


def test_split_summary_reports_expected_counts():
    df = _synthetic_multi_race_df()
    train_df, test_df = split_chronological(df, test_size=0.4)

    summary = split_summary(train_df, test_df)

    assert summary["train_races"] == 3
    assert summary["test_races"] == 2
    assert summary["train_laps"] == 9  # 3 races * 3 laps
    assert summary["test_laps"] == 6  # 2 races * 3 laps
