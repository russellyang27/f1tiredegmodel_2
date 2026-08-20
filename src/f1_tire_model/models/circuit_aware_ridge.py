"""
Ridge-regularized version of the circuit-aware degradation model.

Motivation: real-data validation showed the unregularized
CircuitAwareDegradationModel overfits -- it underperforms the plain
compound-only baseline even on in-range (interpolating) held-out data, not
just on genuine extrapolation. With only ~N distinct circuits behind each
compound's fit, an ordinary least squares regression with 6 free
parameters per compound has very little independent data to constrain
those parameters against, and happily fits training noise as if it were
signal.

Ridge regression adds an L2 penalty on the coefficients (except the
intercept), shrinking weak/noisy coefficients toward zero rather than
letting them take on large, unstable values chasing every wiggle in the
training data. This is still NOT machine learning in the sense of this
project's stated V1 scope -- it's a well-established, closed-form
statistical technique (Tikhonov regularization), computed with the same
kind of linear algebra as the unregularized model, plus one matrix term.

Design differences from CircuitAwareDegradationModel:
- ALL non-intercept design columns are standardized before fitting,
  INCLUDING tire_age_laps itself (which the unregularized model left on
  its raw 0-60 scale). Ridge's penalty must apply fairly across features
  on different natural scales -- otherwise it shrinks whichever feature
  happens to have a naturally larger coefficient just because of its raw
  units, not because it's actually less informative.
- `alpha` (regularization strength) is a required, explicit hyperparameter.
  There's no universally correct value -- it should be chosen via
  held-out validation (see the comparison script this ships with), the
  same way you already chose degradation_threshold_s for TireLifeEstimator
  deliberately rather than accepting a hidden default.

Interface compatibility: this class implements the same public methods as
CircuitAwareDegradationModel (predict, effective_rate_s_per_lap,
has_dedicated_fit, extrapolation_mask, degradation_rate_summary), so
TireLifeEstimator and generate_tire_report work with it unchanged -- they
were built against these methods, not against a specific class.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from f1_tire_model.models.base import TireDegradationModel

_CIRCUIT_FEATURES: tuple[str, ...] = ("avg_corner_speed_kph", "track_temp_c")


class RidgeCircuitAwareDegradationModel(TireDegradationModel):
    """Same structure as CircuitAwareDegradationModel (per-compound linear
    interaction model), fit via ridge regression instead of plain OLS.
    """

    feature_columns = ("tire_compound", "tire_age_laps") + _CIRCUIT_FEATURES
    target_column = "lap_time_delta_s"

    _MIN_POINTS_PER_COMPOUND = 30
    _MIN_CIRCUITS_PER_COMPOUND = 5

    def __init__(self, alpha: float = 1.0):
        if alpha < 0:
            raise ValueError(f"alpha (ridge regularization strength) must be non-negative, got {alpha}")
        self.alpha = alpha
        self._coeffs_by_compound: dict[str, np.ndarray] = {}
        self._global_coeffs: np.ndarray | None = None
        self._feature_means: dict[str, float] = {}
        self._feature_stds: dict[str, float] = {}
        self._feature_ranges: dict[str, tuple[float, float]] = {}
        self._design_columns: list[str] = []

    def fit(self, features_df: pd.DataFrame) -> "RidgeCircuitAwareDegradationModel":
        self._validate_features(features_df)
        df = self._prepare_training_frame(features_df)

        if df.empty:
            raise ValueError(
                "No usable training rows after dropping NaNs and invalid laps -- "
                "check that features_df has valid target, tire_age_laps, "
                "tire_compound, and circuit feature columns."
            )
        if "circuit" not in df.columns:
            raise ValueError("features_df must include a 'circuit' column to check circuit diversity.")

        overall_circuits = df["circuit"].nunique()
        if overall_circuits < self._MIN_CIRCUITS_PER_COMPOUND:
            raise ValueError(
                f"Only {overall_circuits} distinct circuit(s) in training data; "
                f"need at least {self._MIN_CIRCUITS_PER_COMPOUND}."
            )

        self._fit_standardization(df)

        self._coeffs_by_compound = {}
        skipped_compounds = []
        for compound, group in df.groupby("tire_compound", observed=True):
            n_circuits = group["circuit"].nunique()
            if len(group) < self._MIN_POINTS_PER_COMPOUND or n_circuits < self._MIN_CIRCUITS_PER_COMPOUND:
                skipped_compounds.append((str(compound), len(group), n_circuits))
                continue
            X, y = self._design_matrix_and_target(group)
            self._coeffs_by_compound[str(compound)] = self._ridge_fit(X, y)

        if skipped_compounds:
            warnings.warn(
                "RidgeCircuitAwareDegradationModel: skipped per-compound fit for "
                f"{skipped_compounds} (rows, distinct circuits) -- using global fallback.",
                stacklevel=2,
            )

        X_all, y_all = self._design_matrix_and_target(df)
        self._global_coeffs = self._ridge_fit(X_all, y_all)

        return self

    def _ridge_fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Closed-form ridge: beta = (X^T X + alpha*D)^-1 X^T y, where D is
        identity except a 0 at the intercept's diagonal entry (column 0) --
        the intercept is never penalized, since shrinking it toward zero
        has no principled justification the way shrinking a slope does.
        """
        n_features = X.shape[1]
        penalty = np.eye(n_features) * self.alpha
        penalty[0, 0] = 0.0
        return np.linalg.solve(X.T @ X + penalty, X.T @ y)

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        self._validate_features(features_df)

        predictions = np.full(len(features_df), np.nan)
        row_mask = self._rows_with_complete_features(features_df)
        self._warn_if_extrapolating(features_df.loc[row_mask])

        compounds = features_df["tire_compound"].astype(str).to_numpy()
        for compound in np.unique(compounds[row_mask]):
            compound_mask = row_mask & (compounds == compound)
            coeffs = self._coeffs_by_compound.get(compound, self._global_coeffs)
            if coeffs is None:
                continue
            X = self._design_matrix(features_df.loc[compound_mask])
            predictions[compound_mask] = X @ coeffs

        return predictions

    def extrapolation_mask(self, features_df: pd.DataFrame) -> np.ndarray:
        mask = np.zeros(len(features_df), dtype=bool)
        for feature in _CIRCUIT_FEATURES:
            if feature not in self._feature_ranges:
                continue
            low, high = self._feature_ranges[feature]
            values = features_df[feature].to_numpy(dtype=float)
            mask |= (values < low) | (values > high)
        return mask

    def _warn_if_extrapolating(self, features_df: pd.DataFrame) -> None:
        if features_df.empty or not self._feature_ranges:
            return
        mask = self.extrapolation_mask(features_df)
        n_extrapolating = int(mask.sum())
        if n_extrapolating == 0:
            return
        warnings.warn(
            f"RidgeCircuitAwareDegradationModel.predict(): {n_extrapolating}/{len(features_df)} "
            "rows have a circuit feature outside the training range -- extrapolating.",
            stacklevel=3,
        )

    def _prepare_training_frame(self, features_df: pd.DataFrame) -> pd.DataFrame:
        required = [self.target_column, "tire_age_laps", "tire_compound", *_CIRCUIT_FEATURES]
        df = features_df.dropna(subset=required)
        if "is_valid_lap" in df.columns:
            df = df[df["is_valid_lap"]]
        return df

    def _rows_with_complete_features(self, features_df: pd.DataFrame) -> np.ndarray:
        required = ["tire_age_laps", *_CIRCUIT_FEATURES]
        return features_df[required].notna().all(axis=1).to_numpy()

    def _fit_standardization(self, df: pd.DataFrame) -> None:
        self._feature_means = {}
        self._feature_stds = {}
        self._feature_ranges = {}
        # Standardize tire_age_laps too (unlike the unregularized model) --
        # see module docstring for why this matters specifically for ridge.
        all_features = ("tire_age_laps",) + _CIRCUIT_FEATURES
        for feature in all_features:
            mean = float(df[feature].mean())
            std = float(df[feature].std())
            self._feature_means[feature] = mean
            self._feature_stds[feature] = std if std > 1e-8 else 1.0
            if feature in _CIRCUIT_FEATURES:
                self._feature_ranges[feature] = (float(df[feature].min()), float(df[feature].max()))

    def _standardize(self, feature: str, raw_values: np.ndarray) -> np.ndarray:
        return (raw_values - self._feature_means[feature]) / self._feature_stds[feature]

    def _design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        age_std = self._standardize("tire_age_laps", df["tire_age_laps"].to_numpy(dtype=float))
        circuit_std = {f: self._standardize(f, df[f].to_numpy(dtype=float)) for f in _CIRCUIT_FEATURES}

        columns = [np.ones(n), age_std]
        names = ["intercept", "age_std"]
        for f in _CIRCUIT_FEATURES:
            columns.append(age_std * circuit_std[f])
            names.append(f"age_std_x_{f}_std")
        for f in _CIRCUIT_FEATURES:
            columns.append(circuit_std[f])
            names.append(f"{f}_std")

        self._design_columns = names
        return np.column_stack(columns)

    def _design_matrix_and_target(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self._design_matrix(df)
        y = df[self.target_column].to_numpy(dtype=float)
        return X, y

    def has_dedicated_fit(self, compound: str) -> bool:
        return compound in self._coeffs_by_compound

    def degradation_rate_summary(self) -> pd.DataFrame:
        rows = []
        for compound, coeffs in self._coeffs_by_compound.items():
            row = {"compound": compound}
            for name, value in zip(self._design_columns, coeffs):
                row[name] = value
            rows.append(row)
        return pd.DataFrame(rows)

    def effective_rate_s_per_lap(
        self, compound: str, *, avg_corner_speed_kph: float, track_temp_c: float
    ) -> float | None:
        """Converts the standardized-age coefficient back to an
        interpretable raw seconds/lap rate via the chain rule:
        d(delta)/d(raw_age) = d(delta)/d(age_std) * d(age_std)/d(raw_age)
                             = coeff_on_age_std / age_std_scale
        """
        coeffs = self._coeffs_by_compound.get(compound, self._global_coeffs)
        if coeffs is None:
            return None

        raw_values = {"avg_corner_speed_kph": avg_corner_speed_kph, "track_temp_c": track_temp_c}
        standardized = {f: self._standardize(f, raw_values[f]) for f in _CIRCUIT_FEATURES}
        coeff_by_name = dict(zip(self._design_columns, coeffs))

        rate_in_std_units = coeff_by_name["age_std"]
        for f in _CIRCUIT_FEATURES:
            rate_in_std_units += coeff_by_name[f"age_std_x_{f}_std"] * standardized[f]

        return float(rate_in_std_units / self._feature_stds["tire_age_laps"])