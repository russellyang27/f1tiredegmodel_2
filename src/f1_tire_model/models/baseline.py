"""
V1 baseline: per-compound polynomial fits of lap time delta vs. tire age.

Two concrete models are provided, both implementing `TireDegradationModel`:

- `LinearTireDegradationModel` (degree=1): the classic assumption that lap
  time increases roughly linearly with tire age. Simple, interpretable,
  and a reasonable first baseline.
- `QuadraticTireDegradationModel` (degree=2): allows for the "cliff" effect
  documented in real F1 tire behavior, where pace holds relatively steady
  for many laps and then degrades sharply near end of life. A quadratic
  term can capture this curvature; a straight line cannot.

Both share `_PerCompoundPolynomialModel`, differing only in polynomial
degree — this is a deliberate demonstration that the `TireDegradationModel`
interface supports multiple interchangeable implementations *before* any
ML model exists, and both can be scored with the exact same `evaluate()`
from models/base.py for direct comparison.

Neither model uses machine learning — both are `numpy.polyfit` per
compound group, which is a least-squares regression, consistent with the
project's "physics/statistical baseline, no ML yet" requirement for V1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_tire_model.models.base import TireDegradationModel


class _PerCompoundPolynomialModel(TireDegradationModel):
    """Shared fitting/prediction logic for per-compound polynomial degradation models.

    Subclasses set the class attribute `degree`. This is not meant to be
    instantiated directly — use `LinearTireDegradationModel` or
    `QuadraticTireDegradationModel`.
    """

    feature_columns = ("tire_compound", "tire_age_laps")
    target_column = "lap_time_delta_s"
    degree: int = 1

    def __init__(self):
        self._coeffs_by_compound: dict[str, np.ndarray] = {}
        self._global_coeffs: np.ndarray | None = None
        self._min_points = self.degree + 1  # a degree-N polynomial needs >= N+1 points

    def fit(self, features_df: pd.DataFrame) -> "_PerCompoundPolynomialModel":
        self._validate_features(features_df)
        df = self._prepare_training_frame(features_df)

        if df.empty:
            raise ValueError(
                "No usable training rows after dropping NaNs and invalid laps — "
                "check that features_df has valid 'lap_time_delta_s', 'tire_age_laps', "
                "'tire_compound' values and at least some is_valid_lap=True rows."
            )

        self._coeffs_by_compound = {}
        for compound, group in df.groupby("tire_compound", observed=True):
            if len(group) < self._min_points:
                continue  # not enough points for this compound to fit a degree-N curve
            self._coeffs_by_compound[str(compound)] = np.polyfit(
                group["tire_age_laps"], group[self.target_column], deg=self.degree
            )

        if len(df) >= self._min_points:
            self._global_coeffs = np.polyfit(df["tire_age_laps"], df[self.target_column], deg=self.degree)

        if not self._coeffs_by_compound and self._global_coeffs is None:
            raise ValueError(
                f"Not enough data points to fit a degree-{self.degree} model "
                f"(need >= {self._min_points} points total)."
            )

        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        self._validate_features(features_df)

        tire_ages = features_df["tire_age_laps"].to_numpy(dtype=float)
        compounds = features_df["tire_compound"].astype(str).to_numpy()

        predictions = np.full(len(features_df), np.nan)
        for compound in np.unique(compounds):
            mask = compounds == compound
            coeffs = self._coeffs_by_compound.get(compound, self._global_coeffs)
            if coeffs is None:
                continue  # no fit available at all (e.g. fit() was never called) — leave as NaN
            predictions[mask] = np.polyval(coeffs, tire_ages[mask])

        return predictions

    def _prepare_training_frame(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Filter to rows usable for fitting a clean degradation curve.

        Drops rows missing the target or key inputs, and — if the dataset
        has been through the builder (datasets/builder.py) and carries an
        `is_valid_lap` column — excludes pit/caution/outlier laps, which
        would otherwise distort the fitted curve with non-representative
        pace.
        """
        df = features_df.dropna(subset=[self.target_column, "tire_age_laps", "tire_compound"])
        if "is_valid_lap" in df.columns:
            df = df[df["is_valid_lap"]]
        return df

    def degradation_rate_summary(self) -> pd.DataFrame:
        """Interpretable per-compound fit summary: one row per compound with
        its fitted polynomial coefficients (numpy convention: highest degree
        first) and, for degree=1, a directly-readable 'seconds per lap of
        tire age' rate.

        This kind of direct interpretability — "this compound loses X
        seconds/lap" — is something a physics baseline gives you for free
        that a black-box ML model does not. Worth keeping as a sanity check
        and comparison tool even after ML models are added later.
        """
        rows = []
        for compound, coeffs in self._coeffs_by_compound.items():
            row = {"compound": compound}
            for power, coeff in zip(range(self.degree, -1, -1), coeffs):
                row[f"coeff_x^{power}"] = coeff
            if self.degree == 1:
                row["degradation_rate_s_per_lap"] = coeffs[0]
            rows.append(row)
        return pd.DataFrame(rows)


class LinearTireDegradationModel(_PerCompoundPolynomialModel):
    """Baseline: lap_time_delta_s ~ a + b * tire_age_laps, fit per compound."""

    degree = 1


class QuadraticTireDegradationModel(_PerCompoundPolynomialModel):
    """Baseline: lap_time_delta_s ~ a + b*age + c*age^2, fit per compound.

    Allows curvature to capture a late-stint "cliff" that a linear fit
    cannot represent. Requires at least 3 valid laps per compound to fit;
    compounds with fewer laps fall back to the global (all-compounds) fit,
    same as the linear model.
    """

    degree = 2
