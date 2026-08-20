"""
Compare multiple fitted models (e.g. linear vs. quadratic baseline today;
baseline vs. ML models later) on the same metrics, using
`TireDegradationModel.evaluate()` so every model — regardless of its
internal implementation — is scored identically.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1_tire_model.models.base import TireDegradationModel
from f1_tire_model.visualization._utils import valid_rows
from f1_tire_model.visualization.style import DEFAULT_FIGSIZE


def plot_model_comparison(
    features_df: pd.DataFrame,
    models: dict[str, TireDegradationModel],
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart of RMSE for each model, annotated with MAE and R^2.

    Parameters
    ----------
    features_df : evaluation data — pass a held-out test split, e.g. from
        `datasets.splitting.split_chronological`, not the same data the
        model was fit on (see datasets/splitting.py for why race-level,
        not lap-level, splitting matters here).
    models : {label: fitted_model}. Every model must already be fitted;
        this function only evaluates, it doesn't call fit().
    """
    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    df = valid_rows(features_df)

    labels, rmses, annotations = [], [], []
    for label, model in models.items():
        result = model.evaluate(df)
        labels.append(label)
        rmses.append(result.rmse)
        annotations.append(f"MAE={result.mae:.3f}\nR2={result.r2:.3f}")

    bars = ax.bar(labels, rmses, color="#4C72B0")
    for bar, annotation in zip(bars, annotations):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            annotation, ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel("RMSE (s)")
    ax.set_title("Model comparison")
    ax.grid(axis="y", alpha=0.3)

    return ax
