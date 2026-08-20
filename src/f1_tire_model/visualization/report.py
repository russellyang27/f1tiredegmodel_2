"""
Assembles the individual plots (degradation curves, residuals, model
comparison) into one multi-panel validation report — the thing you'll
actually run every time you touch the pipeline or add a new model, rather
than re-running three separate plotting calls by hand.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from f1_tire_model.models.base import TireDegradationModel
from f1_tire_model.visualization.degradation_plots import plot_degradation_curve, plot_residuals
from f1_tire_model.visualization.model_comparison import plot_model_comparison


def build_validation_report(
    features_df: pd.DataFrame,
    models: dict[str, TireDegradationModel],
    *,
    primary_model_label: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Build a 3-panel validation report: degradation curve, residuals, model comparison.

    Parameters
    ----------
    features_df : evaluation dataset.
    models : {label: fitted_model}, e.g. {"Linear": linear_model, "Quadratic": quadratic_model}.
    primary_model_label : which model's curve/residuals to feature in the
        top two panels. Defaults to the first model in `models` (dict
        insertion order) if not given.
    save_path : if given, the figure is saved here (e.g. "reports/v1_validation.png")
        in addition to being returned — useful for a portfolio README or a
        quick "did this get better" check between pipeline changes.
    """
    if not models:
        raise ValueError("`models` must contain at least one fitted model.")

    primary_label = primary_model_label or next(iter(models))
    if primary_label not in models:
        raise ValueError(f"primary_model_label '{primary_label}' not found in models: {list(models)}")
    primary_model = models[primary_label]

    fig, (ax_curve, ax_resid, ax_compare) = plt.subplots(1, 3, figsize=(18, 5))

    plot_degradation_curve(features_df, primary_model, ax=ax_curve)
    ax_curve.set_title(f"Degradation curve ({primary_label})")

    plot_residuals(features_df, primary_model, ax=ax_resid)
    ax_resid.set_title(f"Residuals ({primary_label})")

    plot_model_comparison(features_df, models, ax=ax_compare)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
