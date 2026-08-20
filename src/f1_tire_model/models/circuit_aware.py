"""
Circuit-aware baseline: extends the per-compound polynomial degradation
model by letting the degradation RATE (not just an additive offset) vary
with track characteristics.

Motivation (see docs/VALIDATION.md for the concrete evidence): held-out
testing of the compound-only baseline (models.baseline) across 10 real
2025 races showed negative R^2 on unseen circuits, and a trivial "predict
the compound's average delta" baseline beat the fitted linear/quadratic
models. Residuals were opposite-signed on the same compound across
different held-out circuits (Barcelona vs Montreal) -- the specific
signature of one global slope being wrong for both. No amount of extra
training data fixes that; the model needs to know something about the
circuit it's predicting for.

Still a purely statistical model -- ordinary least squares via numpy, no
ML libraries -- consistent with the project's "no ML yet" scope for V1.
Track features used are already produced by the existing feature
extractors (features/track.py, features/environment.py); no new data
collection is needed.

Design choice: interaction terms, not additive-only circuit features.
lap_time_delta_s is already computed relative to each stint's own first
clean lap (see datasets/builder.py), so a circuit's effect on the general
LEVEL of lap time is already mostly normalized away by that construction.
What differed between Barcelona and Montreal was the RATE tires degrade,
not a flat offset -- so this model lets circuit features modulate the
tire_age slope via interaction terms (tire_age * track_feature), which an
additive-only linear model structurally cannot represent.

TWO LESSONS FROM REAL-DATA VALIDATION baked into this version (see
docs/VALIDATION.md for the full story) -- both worth reading before
touching this file again:

1. `is_high_speed_circuit` is deliberately NOT included as a circuit
   feature. It's computed in features/track.py by thresholding
   avg_corner_speed_kph -- including both as separate design-matrix
   columns is near-perfect multicollinearity, which produced wildly
   unstable coefficients (seen: values in the tens on real data) the first
   time this was tried. avg_corner_speed_kph alone already carries that
   information.

2. Per-compound fitting requires a MINIMUM NUMBER OF DISTINCT CIRCUITS,
   not just a minimum number of rows. A compound with 5,000 laps all at
   the SAME circuit still can't identify a circuit-dependent slope --
   there's no variation in the circuit features to fit against. On real
   2025 data, INTERMEDIATE appeared at only 1 training circuit and SOFT at
   only 3, and both produced nonsensical coefficients (an R^2 of -294 on
   held-out data) before this check was added.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_tire_model.models.base import TireDegradationModel

# Circuit features used as regression inputs, standardized (z-scored)
# before use. See lesson #1 above for why is_high_speed_circuit is
# deliberately excluded -- it's redundant with avg_corner_speed_kph.
_CIRCUIT_FEATURES: tuple[str, ...] = ("avg_corner_speed_kph", "track_temp_c")


class CircuitAwareDegradationModel(TireDegradationModel):
    """lap_time_delta_s ~ a + b*age + sum_i(c_i * age * circuit_feature_i) + sum_i(d_i * circuit_feature_i),
    fit per compound via ordinary least squares.

    The `age * circuit_feature` interaction terms are what let the
    degradation RATE vary by track; the plain `circuit_feature` main
    effects are included alongside them per standard regression practice
    (including an interaction without its main effect biases the
    interaction coefficient).
    """

    feature_columns = ("tire_compound", "tire_age_laps") + _CIRCUIT_FEATURES
    target_column = "lap_time_delta_s"

    #: Minimum training rows required to fit one compound's regression.
    _MIN_POINTS_PER_COMPOUND = 30

    #: Minimum DISTINCT circuits required to fit one compound's regression.
    #: This is the check that was missing before -- see lesson #2 above.
    #: With len(_CIRCUIT_FEATURES)*2 + 2 = 6 design columns, 5 distinct
    #: circuits is already a thin margin; treat this as a starting point to
    #: revisit once you have more races, not a permanently "correct" number.
    _MIN_CIRCUITS_PER_COMPOUND = 5

    def __init__(self):
        self._coeffs_by_compound: dict[str, np.ndarray] = {}
        self._global_coeffs: np.ndarray | None = None
        # Standardization stats, fit ONCE on training data and reused at
        # predict time -- never recomputed from data being predicted on,
        # or a model evaluated on a new circuit would silently leak
        # information about that circuit's own feature distribution.
        self._feature_means: dict[str, float] = {}
        self._feature_stds: dict[str, float] = {}
        #: (min, max) of each circuit feature seen during training. Used to
        #: detect and warn when predict() is asked to score a row outside
        #: the range the model was ever fit on -- a linear interaction
        #: model extrapolates as a straight line past its training range,
        #: which is unreliable if the real relationship isn't linear that
        #: far out (see the real-data finding in docs/VALIDATION.md: a
        #: held-out test set entirely above the training track_temp_c range
        #: coincided with this model underperforming the plain baseline).
        self._feature_ranges: dict[str, tuple[float, float]] = {}
        self._design_columns: list[str] = []

    def fit(self, features_df: pd.DataFrame) -> "CircuitAwareDegradationModel":
        self._validate_features(features_df)
        df = self._prepare_training_frame(features_df)

        if df.empty:
            raise ValueError(
                "No usable training rows after dropping NaNs and invalid laps -- "
                "check that features_df has valid target, tire_age_laps, "
                "tire_compound, and circuit feature columns."
            )
        if "circuit" not in df.columns:
            raise ValueError(
                "features_df must include a 'circuit' column to check circuit "
                "diversity per compound (see _MIN_CIRCUITS_PER_COMPOUND)."
            )

        overall_circuits = df["circuit"].nunique()
        if overall_circuits < self._MIN_CIRCUITS_PER_COMPOUND:
            raise ValueError(
                f"Only {overall_circuits} distinct circuit(s) in training data; "
                f"need at least {self._MIN_CIRCUITS_PER_COMPOUND} for a circuit-aware "
                "fit to be numerically stable. Use models.baseline instead until "
                "you have more races, or lower _MIN_CIRCUITS_PER_COMPOUND at your "
                "own risk (expect unstable coefficients -- see the module docstring)."
            )

        self._fit_standardization(df)

        self._coeffs_by_compound = {}
        skipped_compounds = []
        for compound, group in df.groupby("tire_compound", observed=True):
            n_circuits = group["circuit"].nunique()
            if len(group) < self._MIN_POINTS_PER_COMPOUND or n_circuits < self._MIN_CIRCUITS_PER_COMPOUND:
                skipped_compounds.append((str(compound), len(group), n_circuits))
                continue  # falls back to the global fit at predict time
            X, y = self._design_matrix_and_target(group)
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            self._coeffs_by_compound[str(compound)] = coeffs

        if skipped_compounds:
            import warnings

            warnings.warn(
                "CircuitAwareDegradationModel: skipped per-compound fit for "
                f"{skipped_compounds} (rows, distinct circuits) -- not enough "
                "circuit diversity or data; these compounds will use the global "
                "fallback fit at predict time.",
                stacklevel=2,
            )

        X_all, y_all = self._design_matrix_and_target(df)
        self._global_coeffs, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)

        return self

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
        """True for rows where ANY circuit feature falls outside the range
        this model was trained on -- i.e. rows where predict() is
        extrapolating rather than interpolating. Exposed publicly so a
        caller can filter these out, flag them in a report, or decide to
        trust them anyway with eyes open, rather than only finding out via
        a warning buried in logs.
        """
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

        details = []
        for feature in _CIRCUIT_FEATURES:
            low, high = self._feature_ranges[feature]
            values = features_df.loc[mask, feature].to_numpy(dtype=float)
            out_of_range = values[(values < low) | (values > high)]
            if len(out_of_range) > 0:
                details.append(
                    f"{feature}: trained on [{low:.1f}, {high:.1f}], "
                    f"predicting for values as far as {out_of_range.min():.1f}-{out_of_range.max():.1f}"
                )

        import warnings

        warnings.warn(
            f"CircuitAwareDegradationModel.predict(): {n_extrapolating}/{len(features_df)} rows "
            "have a circuit feature outside the training range -- predictions for these rows "
            "are EXTRAPOLATIONS (linear, past the range this model was ever fit on) and may be "
            f"unreliable. {' | '.join(details)}",
            stacklevel=3,
        )

    # --- internals ---

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
        for feature in _CIRCUIT_FEATURES:
            mean = float(df[feature].mean())
            std = float(df[feature].std())
            self._feature_means[feature] = mean
            # Guard against a degenerate (zero-variance) feature -- fall
            # back to 1.0 so we divide by something safe instead of
            # producing inf/NaN coefficients.
            self._feature_stds[feature] = std if std > 1e-8 else 1.0
            self._feature_ranges[feature] = (float(df[feature].min()), float(df[feature].max()))

    def _design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Build [intercept, age, age*circuit_features..., circuit_features...] columns.

        Uses self._feature_means/_feature_stds (fit-time stats) to
        standardize continuous features -- these must already be set via
        _fit_standardization before this is called at predict time.
        """
        n = len(df)
        age = df["tire_age_laps"].to_numpy(dtype=float)

        standardized = {}
        for feature in _CIRCUIT_FEATURES:
            mean = self._feature_means[feature]
            std = self._feature_stds[feature]
            standardized[feature] = (df[feature].to_numpy(dtype=float) - mean) / std

        columns = [np.ones(n), age]
        column_names = ["intercept", "age"]
        for feature in _CIRCUIT_FEATURES:
            columns.append(age * standardized[feature])
            column_names.append(f"age_x_{feature}")
        for feature in _CIRCUIT_FEATURES:
            columns.append(standardized[feature])
            column_names.append(feature)

        self._design_columns = column_names
        return np.column_stack(columns)

    def _design_matrix_and_target(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self._design_matrix(df)
        y = df[self.target_column].to_numpy(dtype=float)
        return X, y

    def has_dedicated_fit(self, compound: str) -> bool:
        """True if this compound had enough data/circuit diversity for its
        own per-compound fit; False if predictions for it fall back to the
        pooled global fit. Public so other modules (e.g. a strategy report)
        can surface this as a caveat without reaching into private state.
        """
        return compound in self._coeffs_by_compound

    def degradation_rate_summary(self) -> pd.DataFrame:
        """Interpretable per-compound coefficient table.

        Unlike models.baseline's single 'seconds per lap' number, the
        effective degradation rate here depends on the specific circuit's
        feature values (that's the whole point) -- use
        `effective_rate_s_per_lap()` for a single circuit's number, this
        table is for inspecting the raw fitted coefficients. Compounds
        fitted via the global fallback (see the fit() warning) are NOT
        included here -- only compounds with their own stable per-compound
        fit are shown.
        """
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
        """The degradation rate (seconds/lap) this model predicts for a
        SPECIFIC circuit's characteristics -- the interpretable, "what does
        this model actually believe about this track" number, computed by
        holding circuit features fixed and reading off the coefficient on
        tire_age at those values.
        """
        coeffs = self._coeffs_by_compound.get(compound, self._global_coeffs)
        if coeffs is None:
            return None

        raw_values = {"avg_corner_speed_kph": avg_corner_speed_kph, "track_temp_c": track_temp_c}
        standardized = {
            feature: (raw_values[feature] - self._feature_means[feature]) / self._feature_stds[feature]
            for feature in _CIRCUIT_FEATURES
        }

        coeff_by_name = dict(zip(self._design_columns, coeffs))
        rate = coeff_by_name["age"]
        for feature in _CIRCUIT_FEATURES:
            rate += coeff_by_name[f"age_x_{feature}"] * standardized[feature]
        return float(rate)