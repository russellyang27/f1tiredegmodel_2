"""
Train/test splitting for the tire degradation dataset.

CRITICAL RULE, read before using any function here: splits must happen at
the RACE level, never at the individual-lap level. Laps within one race
share track surface, weather, and compound allocation — a random lap-level
split lets information from a race leak between train and test (the model
effectively gets to "see" that race during training), which inflates
apparent performance without testing real generalization. Every splitter
below groups by (season, round_number, session_name) and assigns whole
groups to train or test.

Three splitters are provided because they answer different validation
questions — see each function's docstring for when to prefer it. If you
only use one, use `split_chronological`: it's the closest match to how
this model would actually be deployed (train on races you have, predict a
race you don't), and F1 seasons are not statistically uniform across
rounds (track evolution, compound allocation, sprint format changes), so a
random split can overstate real-world generalization.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_GROUP_COLS: tuple[str, ...] = ("season", "round_number", "session_name")


def _race_keys(df: pd.DataFrame, group_cols: tuple[str, ...]) -> pd.DataFrame:
    missing = set(group_cols) - set(df.columns)
    if missing:
        raise ValueError(f"features_df is missing grouping columns: {sorted(missing)}")
    return df[list(group_cols)].drop_duplicates().reset_index(drop=True)


def _split_by_keys(
    df: pd.DataFrame, train_keys: pd.DataFrame, test_keys: pd.DataFrame, group_cols: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.merge(train_keys, on=list(group_cols), how="inner")
    test_df = df.merge(test_keys, on=list(group_cols), how="inner")
    return train_df, test_df


def split_by_race(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int | None = None,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly assign whole races to train or test.

    Use when: you want a generic, data-efficient estimate of cross-race
    generalization and aren't specifically testing forward-in-time
    prediction. Weakest of the three at detecting whether the model
    depends on season-specific or chronological patterns, since test races
    can be interleaved with training races in time.

    Raises
    ------
    ValueError
        If there are fewer than 2 distinct races (can't split), or
        `test_size` would produce an empty train or test set.
    """
    keys = _race_keys(df, group_cols)
    if len(keys) < 2:
        raise ValueError(
            f"Need at least 2 distinct races to split; found {len(keys)}. "
            "Build a larger dataset before splitting."
        )

    shuffled = keys.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n_test = round(len(shuffled) * test_size)
    n_test = max(1, min(n_test, len(shuffled) - 1))  # keep both sides non-empty

    test_keys = shuffled.iloc[:n_test]
    train_keys = shuffled.iloc[n_test:]

    return _split_by_keys(df, train_keys, test_keys, group_cols)


def split_chronological(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort races chronologically and hold out the most recent ones as test.

    Use when: you want to know how well this would have predicted the next
    race(s) using only data available before them — the realistic
    deployment scenario for this project. Recommended default.

    Raises
    ------
    ValueError
        If there are fewer than 2 distinct races.
    """
    keys = _race_keys(df, group_cols).sort_values(list(group_cols)).reset_index(drop=True)
    if len(keys) < 2:
        raise ValueError(
            f"Need at least 2 distinct races to split; found {len(keys)}. "
            "Build a larger dataset before splitting."
        )

    n_test = round(len(keys) * test_size)
    n_test = max(1, min(n_test, len(keys) - 1))

    train_keys = keys.iloc[: len(keys) - n_test]
    test_keys = keys.iloc[len(keys) - n_test :]

    return _split_by_keys(df, train_keys, test_keys, group_cols)


def split_by_season(
    df: pd.DataFrame,
    *,
    train_seasons: list[int],
    test_seasons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explicit season-based split: train on some seasons, test on others.

    Use when: you want the strongest, most conservative generalization
    test — does a model trained on one year's tire construction and
    regulations still work the next year? Expect worse performance here
    than the other two splitters; that's the point; Pirelli tire
    construction and compound naming/allocation change between seasons.

    Raises
    ------
    ValueError
        If either resulting split is empty (e.g. a requested season isn't
        actually present in `df`).
    """
    train_df = df[df["season"].isin(train_seasons)]
    test_df = df[df["season"].isin(test_seasons)]

    if train_df.empty:
        raise ValueError(f"No rows found for train_seasons={train_seasons}.")
    if test_df.empty:
        raise ValueError(f"No rows found for test_seasons={test_seasons}.")

    return train_df, test_df


def check_split_compound_coverage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    """Return compounds present in test_df but absent from train_df.

    A non-empty result is a warning sign, not necessarily a fatal error: a
    per-compound model (see models.baseline) will silently fall back to its
    global fit for these compounds, which is a fair thing to do at
    prediction time but means your evaluation on those rows is measuring
    the fallback path, not a compound-specific fit — worth knowing before
    you interpret an evaluation metric.
    """
    train_compounds = set(train_df["tire_compound"].dropna().astype(str).unique())
    test_compounds = set(test_df["tire_compound"].dropna().astype(str).unique())

    uncovered = sorted(test_compounds - train_compounds)
    if uncovered:
        logger.warning(
            "Compounds in test set with no training examples (will use global "
            "fallback fit): %s",
            uncovered,
        )
    return uncovered


def split_summary(
    train_df: pd.DataFrame, test_df: pd.DataFrame, group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS
) -> dict:
    """Quick sanity-check counts for a split — race counts and lap counts per side.

    Useful to eyeball immediately after splitting, before running any
    model: an unexpectedly tiny test set, or a test set with far fewer laps
    per race than train (e.g. mostly short/wet sessions), is worth noticing
    before you trust the evaluation numbers that follow.
    """
    return {
        "train_races": len(_race_keys(train_df, group_cols)),
        "test_races": len(_race_keys(test_df, group_cols)),
        "train_laps": len(train_df),
        "test_laps": len(test_df),
    }
