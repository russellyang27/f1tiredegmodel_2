from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend — no display needed for tests or CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from f1_tire_model.models.baseline import LinearTireDegradationModel, QuadraticTireDegradationModel
from f1_tire_model.visualization.degradation_plots import plot_degradation_curve, plot_residuals
from f1_tire_model.visualization.model_comparison import plot_model_comparison
from f1_tire_model.visualization.report import build_validation_report
from f1_tire_model.visualization.style import compound_color


def _synthetic_features_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for compound, rate in [("SOFT", 0.15), ("MEDIUM", 0.08), ("HARD", 0.04)]:
        for age in range(1, 15):
            rows.append(
                {
                    "tire_compound": compound,
                    "tire_age_laps": age,
                    "lap_time_delta_s": rate * age + rng.normal(scale=0.02),
                    "is_valid_lap": True,
                }
            )
    # A couple of excluded (invalid) laps for context in the plots.
    rows.append({"tire_compound": "SOFT", "tire_age_laps": 6, "lap_time_delta_s": 5.0, "is_valid_lap": False})
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_linear_model():
    df = _synthetic_features_df()
    return LinearTireDegradationModel().fit(df), df


def test_compound_color_known_and_unknown():
    assert compound_color("SOFT").startswith("#")
    assert compound_color("soft") == compound_color("SOFT")  # case-insensitive
    assert compound_color("NOT_A_COMPOUND") == "#888888"


def test_plot_degradation_curve_returns_axes_with_content(fitted_linear_model):
    model, df = fitted_linear_model

    ax = plot_degradation_curve(df, model)

    assert isinstance(ax, plt.Axes)
    assert len(ax.collections) > 0  # scatter points were drawn
    assert len(ax.lines) > 0  # fitted curves were drawn
    plt.close(ax.figure)


def test_plot_degradation_curve_respects_compounds_filter(fitted_linear_model):
    model, df = fitted_linear_model

    ax = plot_degradation_curve(df, model, compounds=["SOFT"])

    # Only SOFT's legend entries should be present (2 lines: actual + fitted... scatter
    # legend entries counted via legend handles)
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("SOFT" in label for label in legend_labels)
    assert not any("HARD" in label for label in legend_labels)
    plt.close(ax.figure)


def test_plot_residuals_returns_axes_with_content(fitted_linear_model):
    model, df = fitted_linear_model

    ax = plot_residuals(df, model)

    assert isinstance(ax, plt.Axes)
    assert len(ax.collections) > 0
    plt.close(ax.figure)


def test_plot_model_comparison_shows_all_models(fitted_linear_model):
    _, df = fitted_linear_model
    linear = LinearTireDegradationModel().fit(df)
    quadratic = QuadraticTireDegradationModel().fit(df)

    ax = plot_model_comparison(df, {"Linear": linear, "Quadratic": quadratic})

    assert len(ax.patches) == 2  # one bar per model
    plt.close(ax.figure)


def test_build_validation_report_returns_figure_and_saves(tmp_path, fitted_linear_model):
    _, df = fitted_linear_model
    linear = LinearTireDegradationModel().fit(df)
    quadratic = QuadraticTireDegradationModel().fit(df)

    save_path = tmp_path / "report.png"
    fig = build_validation_report(df, {"Linear": linear, "Quadratic": quadratic}, save_path=save_path)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3
    assert save_path.exists()
    plt.close(fig)


def test_build_validation_report_raises_on_empty_models(fitted_linear_model):
    _, df = fitted_linear_model

    with pytest.raises(ValueError, match="at least one fitted model"):
        build_validation_report(df, {})


def test_build_validation_report_raises_on_unknown_primary_label(fitted_linear_model):
    _, df = fitted_linear_model
    linear = LinearTireDegradationModel().fit(df)

    with pytest.raises(ValueError, match="not found in models"):
        build_validation_report(df, {"Linear": linear}, primary_model_label="Nonexistent")
