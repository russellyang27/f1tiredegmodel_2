"""
Derives a tire-life estimate (laps until predicted degradation crosses a
performance threshold) from ANY fitted TireDegradationModel, via numeric
search over predicted lap_time_delta_s across tire_age_laps.

Why numeric search instead of solving a model's equation directly: the
whole point of the TireDegradationModel interface (models/base.py) is that
predict() doesn't care whether the underlying model is linear
(models.baseline), circuit-conditioned (models.circuit_aware), or --
eventually -- an ML model with no closed form at all. This estimator works
identically for all of them by treating predict() as a black box and
searching over tire_age_laps, rather than hardcoding "solve a + b*age =
threshold" which only works for a linear model. Same design principle the
project has followed since models/base.py: build against the interface,
not a specific implementation.

Why this needs its own module rather than living inside models/base.py:
"how many laps until degradation crosses a threshold" is a DERIVED
question, not a model -- it doesn't fit/predict against training data
itself, it queries an already-fitted model. Keeping it separate also means
it composes with any current or future TireDegradationModel without that
model needing to know anything about "tire life" as a concept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_tire_model.models.base import TireDegradationModel


class TireLifeEstimator:
    """Estimates tire life (laps) as the first tire_age_laps where a fitted
    model's predicted lap_time_delta_s crosses a chosen threshold.

    Parameters
    ----------
    model : an already-FITTED TireDegradationModel (e.g. CircuitAwareDegradationModel).
    degradation_threshold_s : lap-time delta (seconds slower than a fresh
        tire) that defines "end of life". This is a judgment call, not a
        physical constant -- there is no universal "correct" value. A
        common starting point discussed in F1 strategy analysis is
        roughly 1.0-1.5s/lap, but the right number depends on what
        decision this estimate is meant to support (e.g. "when would a
        rational strategist pit" implies a lower threshold than "when does
        the tire become genuinely undriveable").
    max_age_laps : upper bound for the search. If the model never predicts
        a delta reaching the threshold within this range, estimate_life()
        returns None rather than guessing beyond where the model has
        anything meaningful to say.
    age_step : resolution of the search grid; the final estimate is
        linearly interpolated between the two bracketing grid points, so
        this mainly controls how finely curved thresholds get resolved,
        not the precision of the final answer.
    """

    def __init__(
        self,
        model: TireDegradationModel,
        degradation_threshold_s: float,
        *,
        max_age_laps: int = 60,
        age_step: float = 0.5,
    ):
        if degradation_threshold_s <= 0:
            raise ValueError(
                f"degradation_threshold_s must be positive (a lap-time delta "
                f"threshold in seconds), got {degradation_threshold_s}"
            )
        self.model = model
        self.degradation_threshold_s = degradation_threshold_s
        self.max_age_laps = max_age_laps
        self.age_step = age_step

    def estimate_life_laps(
        self, *, age_dependent_covariates: dict | None = None, **covariates
    ) -> float | None:
        """Estimate tire life given fixed covariates (everything the model
        needs EXCEPT tire_age_laps).

        Parameters
        ----------
        age_dependent_covariates : {feature_name: callable(ages_array) -> values_array}.
            For any feature that varies deterministically WITH tire age in
            reality (the clearest example: fuel load decreases as the race
            progresses) rather than being independent of it, holding that
            feature at one fixed value while sweeping age creates a
            physically inconsistent query -- e.g. a brand-new tire paired
            with a half-empty fuel tank, a combination the model never saw
            in training. This was a real bug found via testing (see
            docs/VALIDATION.md): predictions collapsed to a nonsensical
            constant across low tire ages because fuel_load_estimate_kg was
            held at its dataset-average value (implicitly consistent with a
            mid-race tire age) while tire_age_laps swept from 0. Supplying
            a function here lets that covariate track the swept age
            consistently instead. Overrides any static value passed for
            the same key in **covariates.

        Example
        -------
        >>> estimator.estimate_life_laps(
        ...     tire_compound="MEDIUM", avg_corner_speed_kph=180.0, track_temp_c=35.0
        ... )
        23.5

        Returns None if the model's prediction never reaches the threshold
        within max_age_laps.
        """
        ages = np.arange(1, self.max_age_laps + self.age_step, self.age_step)
        data = {key: [value] * len(ages) for key, value in covariates.items()}
        if age_dependent_covariates:
            for key, fn in age_dependent_covariates.items():
                data[key] = fn(ages)
        sweep_df = pd.DataFrame({**data, "tire_age_laps": ages})
        predicted = self.model.predict(sweep_df)

        for i in range(1, len(ages)):
            d0, d1 = predicted[i - 1], predicted[i]
            if np.isnan(d0) or np.isnan(d1):
                continue
            if d0 < self.degradation_threshold_s <= d1:
                age0, age1 = ages[i - 1], ages[i]
                frac = (self.degradation_threshold_s - d0) / (d1 - d0)
                return float(age0 + frac * (age1 - age0))

        return None  # never crosses the threshold within max_age_laps

    def estimate_life_laps_batch(self, covariates_df: pd.DataFrame) -> np.ndarray:
        """Row-wise version of estimate_life_laps for many different
        covariate combinations at once (e.g. one row per driver/compound
        combination you want to compare).

        NOTE: this loops row by row (one sweep per row), which is simple
        and correct but not vectorized -- fine for dozens or hundreds of
        rows (e.g. building a comparison table), but revisit with a
        vectorized/closed-form approach if this ever needs to run over an
        entire lap-level dataset.
        """
        results = np.full(len(covariates_df), np.nan)
        exclude = {"tire_age_laps"}
        for i, (_, row) in enumerate(covariates_df.iterrows()):
            covariates = {k: v for k, v in row.items() if k not in exclude}
            life = self.estimate_life_laps(**covariates)
            results[i] = life if life is not None else np.nan
        return results