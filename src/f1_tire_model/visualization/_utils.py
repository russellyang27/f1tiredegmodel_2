"""Internal helpers shared across visualization functions."""

from __future__ import annotations

import pandas as pd


def valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows usable for degradation analysis.

    If the DataFrame carries an `is_valid_lap` column (produced by
    datasets.builder), filters to True rows. If not present (e.g. a hand-built
    DataFrame in a notebook experiment), returns the DataFrame unchanged
    rather than raising — plotting code should degrade gracefully, not
    force every caller to have the full pipeline's column set.
    """
    if "is_valid_lap" in df.columns:
        return df[df["is_valid_lap"]]
    return df
