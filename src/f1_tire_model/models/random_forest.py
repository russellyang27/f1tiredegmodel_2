"""
First ML model in this project: a Random Forest regressor predicting
lap_time_delta_s from tire compound, tire age, circuit IDENTITY, weather,
fuel load, and (optionally) driver behavior features.

Why circuit identity (not descriptive track features): two independent
statistical experiments (models.circuit_aware + models.circuit_aware_ridge,
see docs/VALIDATION.md) showed hand-picked descriptive features
(avg_corner_speed_kph, track_temp_c) don't carry enough signal about WHY
circuits differ, under a linear functional form. Circuit identity
sidesteps needing to explain WHY -- the model learns a pattern per circuit
directly, matching the actual product interface (choosing from real,
recurring circuits).

Feature lists are configurable (not hardcoded) specifically so accuracy
experiments -- e.g. "do driver behavior features actually help" -- don't
require touching the class itself, just constructing it differently.

Hyperparameter/feature tuning should use `oob_score_` WITH CAUTION -- see
docs/VALIDATION.md for a confirmed real-data finding that scikit-learn's
OOB score can be badly misleading here: it holds out individual ROWS, not
whole races, so it doesn't detect the within-race correlation this project
has been careful about since preprocessing/segmentation.py. A hyperparameter
sweep trusting OOB alone picked the single WORST-performing config on a
genuine held-out test (deeper trees showed OOB climbing while real
held-out R^2 fell to -0.23). Use race-level validation splits
(datasets.splitting) for real tuning; OOB is only safe as a rough sanity
check, not a tuning signal.

VALIDATION CAVEAT: because this model uses circuit IDENTITY, it can only
make a meaningful prediction for a circuit it saw during training. Confirmed
valid on real data: with three seasons built (2023-2025), a held-out
chronological split showed 100% of test-set circuits also present in
training, and Random Forest (R2=0.20) beat the plain compound-only baseline
(R2=0.077) on that fair comparison.

FEATURE NOTE: fuel_load_estimate_kg is deliberately excluded from the
DEFAULT feature set. It genuinely depends on lap number (race position),
not tire age (stint position) -- these only coincide in a driver's first
stint. A real bug was found (see docs/VALIDATION.md) where the interactive
report's age-sweep, unable to know lap number, produced a physically
impossible combination (a fresh tire paired with a half-empty fuel tank)
for any query beyond the first stint, causing a sign-flipped, nonsensical
prediction. Similarly, driver_consistency_std_s (NOT included here) was
found to be a stint-level aggregate broadcast across every lap in a
stint, acting as a near-identifier that let the model memorize training
stints rather than generalize -- catastrophic on held-out data (R2=-0.68)
despite looking great in validation. Both are real, confirmed pitfalls;
don't re-add either without re-reading their full writeups in
docs/VALIDATION.md first.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from f1_tire_model.models.base import TireDegradationModel

DEFAULT_CATEGORICAL_FEATURES: tuple[str, ...] = ("tire_compound", "circuit")
DEFAULT_CONTINUOUS_FEATURES: tuple[str, ...] = (
    "tire_age_laps", "track_temp_c", "air_temp_c", "fuel_load_estimate_kg",
)
DEFAULT_BOOLEAN_FEATURES: tuple[str, ...] = ("is_raining",)


class RandomForestTireDegradationModel(TireDegradationModel):
    """Random Forest regression for lap_time_delta_s, with a configurable
    feature set (defaults to the validated circuit-identity + weather set).

    Categorical features are one-hot encoded using categories observed at
    fit time; a category never seen during training produces an all-zero
    encoding at predict time and triggers a warning.
    """

    target_column = "lap_time_delta_s"

    def __init__(
        self,
        *,
        categorical_features: tuple[str, ...] | None = None,
        continuous_features: tuple[str, ...] | None = None,
        boolean_features: tuple[str, ...] | None = None,
        n_estimators: int = 300,
        max_depth: int | None = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        min_confidence_samples: float = 10.0,
    ):
        self.categorical_features = categorical_features or DEFAULT_CATEGORICAL_FEATURES
        self.continuous_features = continuous_features or DEFAULT_CONTINUOUS_FEATURES
        self.boolean_features = boolean_features or DEFAULT_BOOLEAN_FEATURES
        self.feature_columns = self.categorical_features + self.continuous_features + self.boolean_features

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        #: Minimum average training-sample count across the leaves a query
        #: lands in (see extrapolation_mask()) before predict() warns that
        #: this specific combination of features has very little real
        #: evidence behind it -- distinct from min_samples_leaf, which
        #: only guarantees this at TRAINING time per individual tree, not
        #: per query, and not averaged across the ensemble.
        self.min_confidence_samples = min_confidence_samples

        self._model = None
        self._categories: dict[str, list[str]] = {}
        self._encoded_columns: list[str] = []
        self.oob_score_: float | None = None

    def fit(self, features_df: pd.DataFrame) -> "RandomForestTireDegradationModel":
        self._validate_features(features_df)
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for RandomForestTireDegradationModel. "
                'Install it with: pip install -e ".[ml]"'
            ) from exc

        df = self._prepare_training_frame(features_df)
        if df.empty:
            raise ValueError(
                "No usable training rows after dropping NaNs and invalid laps -- "
                "check that features_df has valid values for all feature_columns."
            )

        for feature in self.categorical_features:
            self._categories[feature] = sorted(df[feature].astype(str).unique())

        X = self._encode(df)
        y = df[self.target_column].to_numpy(dtype=float)

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
            oob_score=True,
        )
        self._model.fit(X, y)
        self.oob_score_ = float(self._model.oob_score_)
        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        self._validate_features(features_df)
        if self._model is None:
            raise ValueError("Model has not been fit yet.")

        predictions = np.full(len(features_df), np.nan)
        row_mask = self._rows_with_complete_features(features_df)
        self._warn_if_unseen_categories(features_df.loc[row_mask])
        self._warn_if_low_confidence(features_df.loc[row_mask])

        if row_mask.any():
            X = self._encode(features_df.loc[row_mask])
            predictions[row_mask] = self._model.predict(X)

        return predictions

    def extrapolation_mask(self, features_df: pd.DataFrame) -> np.ndarray:
        """True for rows whose prediction is based on very few real
        training samples -- computed from the tree structure itself
        (tree_.n_node_samples), not a min/max range check. A tree ensemble
        can have sparse or even empty pockets INSIDE the overall range of
        each individual feature (e.g. a specific circuit at an unusually
        hot temperature and a high tire age might each individually look
        "normal" while the COMBINATION is almost unprecedented) -- this
        catches that directly by asking how much real data actually backs
        the specific leaf(s) a query falls into, averaged across the whole
        forest. Named to match the linear models' extrapolation_mask() so
        generate_tire_report() picks this up automatically via hasattr,
        even though the underlying mechanism is different.
        """
        if self._model is None:
            raise ValueError("Model has not been fit yet.")

        avg_samples = self.avg_leaf_sample_count(features_df)
        with np.errstate(invalid="ignore"):
            return avg_samples < self.min_confidence_samples

    def avg_leaf_sample_count(self, features_df: pd.DataFrame) -> np.ndarray:
        """Average number of real training samples in the leaves each row
        lands in, across every tree in the forest. NaN for rows with
        incomplete features. Exposed publicly (not just as a boolean via
        extrapolation_mask) so a caller can inspect the actual confidence
        level, not just a threshold pass/fail.
        """
        if self._model is None:
            raise ValueError("Model has not been fit yet.")

        row_mask = self._rows_with_complete_features(features_df)
        avg_samples = np.full(len(features_df), np.nan)
        if row_mask.any():
            X = self._encode(features_df.loc[row_mask])
            leaf_indices = self._model.apply(X)  # shape (n_rows, n_estimators)
            sample_counts = np.empty(leaf_indices.shape, dtype=float)
            for tree_idx, tree in enumerate(self._model.estimators_):
                n_node_samples = tree.tree_.n_node_samples
                sample_counts[:, tree_idx] = n_node_samples[leaf_indices[:, tree_idx]]
            avg_samples[row_mask] = sample_counts.mean(axis=1)
        return avg_samples

    def _warn_if_low_confidence(self, features_df: pd.DataFrame) -> None:
        if features_df.empty or self._model is None:
            return
        avg_samples = self.avg_leaf_sample_count(features_df)
        valid = ~np.isnan(avg_samples)
        if not valid.any():
            return
        low_confidence = valid & (avg_samples < self.min_confidence_samples)
        n_low = int(low_confidence.sum())
        if n_low == 0:
            return
        worst = np.nanmin(avg_samples)
        warnings.warn(
            f"RandomForestTireDegradationModel.predict(): {n_low}/{len(features_df)} rows have "
            f"very little real training data behind this specific combination of features "
            f"(as few as {worst:.1f} average training samples in the relevant leaves, threshold "
            f"is {self.min_confidence_samples}) -- predictions for these rows are based on sparse "
            "evidence and may be unreliable, even though each individual feature value looks normal.",
            stacklevel=3,
        )

    def feature_importance_summary(self) -> pd.DataFrame:
        if self._model is None:
            raise ValueError("Model has not been fit yet.")

        raw = dict(zip(self._encoded_columns, self._model.feature_importances_))
        grouped: dict[str, float] = {}
        for column, importance in raw.items():
            original = self._original_feature_name(column)
            grouped[original] = grouped.get(original, 0.0) + importance

        return pd.DataFrame(
            sorted(grouped.items(), key=lambda kv: kv[1], reverse=True),
            columns=["feature", "importance"],
        )

    def _original_feature_name(self, encoded_column: str) -> str:
        for feature in self.categorical_features:
            if encoded_column.startswith(f"{feature}__"):
                return feature
        return encoded_column

    # --- internals ---

    def _prepare_training_frame(self, features_df: pd.DataFrame) -> pd.DataFrame:
        required = list(self.feature_columns) + [self.target_column]
        df = features_df.dropna(subset=required)
        if "is_valid_lap" in df.columns:
            df = df[df["is_valid_lap"]]
        return df

    def _rows_with_complete_features(self, features_df: pd.DataFrame) -> np.ndarray:
        return features_df[list(self.feature_columns)].notna().all(axis=1).to_numpy()

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        parts, columns = [], []
        for feature in self.categorical_features:
            for category in self._categories.get(feature, []):
                parts.append((df[feature].astype(str) == category).astype(float).to_numpy())
                columns.append(f"{feature}__{category}")
        for feature in self.continuous_features:
            parts.append(df[feature].to_numpy(dtype=float))
            columns.append(feature)
        for feature in self.boolean_features:
            parts.append(df[feature].astype(float).to_numpy())
            columns.append(feature)

        self._encoded_columns = columns
        return pd.DataFrame(np.column_stack(parts), columns=columns, index=df.index)

    def _warn_if_unseen_categories(self, df: pd.DataFrame) -> None:
        for feature in self.categorical_features:
            known = set(self._categories.get(feature, []))
            seen = set(df[feature].astype(str).unique())
            unseen = seen - known
            if unseen:
                warnings.warn(
                    f"RandomForestTireDegradationModel.predict(): unseen {feature} value(s) "
                    f"{sorted(unseen)} -- the model has no learned pattern for these; "
                    "predictions for these rows are unreliable.",
                    stacklevel=3,
                )