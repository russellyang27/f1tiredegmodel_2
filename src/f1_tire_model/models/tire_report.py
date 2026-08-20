"""
Model-agnostic tire strategy report generation.

Works with ANY TireDegradationModel (LinearTireDegradationModel,
CircuitAwareDegradationModel, RandomForestTireDegradationModel, or a
future model) via TireLifeEstimator + predict(), rather than assuming a
specific model's extra methods. Model-specific diagnostics
(effective_rate_s_per_lap, has_dedicated_fit, extrapolation_mask) are used
OPPORTUNISTICALLY when the given model happens to support them (checked
via hasattr), so richer models get richer reports without breaking for
simpler ones.

For models needing MORE inputs than the primary user-facing ones (e.g.
RandomForestTireDegradationModel also needs fuel load and driver-style
features, not just track/weather/compound), missing required features are
auto-filled with sensible defaults (dataset averages, optionally
circuit-specific) via `default_covariates()` -- so the public interface
can stay "choose track + weather + compound" even as the underlying model
needs more than that under the hood.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from f1_tire_model.models.base import TireDegradationModel
from f1_tire_model.models.tire_life import TireLifeEstimator


@dataclass
class CompoundLifeEstimate:
    compound: str
    estimated_life_laps: float | None
    #: The following are populated only if the model supports the
    #: corresponding optional method (checked via hasattr) -- None means
    #: "not applicable for this model", not "unknown".
    degradation_rate_s_per_lap: float | None = None
    used_global_fallback: bool | None = None
    is_extrapolating: bool | None = None


@dataclass
class TireStrategyReport:
    conditions: dict
    degradation_threshold_s: float
    compounds: list[CompoundLifeEstimate] = field(default_factory=list)

    def best_compound_by_life(self) -> CompoundLifeEstimate | None:
        valid = [c for c in self.compounds if c.estimated_life_laps is not None]
        if not valid:
            return None
        return max(valid, key=lambda c: c.estimated_life_laps)

    def summary(self) -> str:
        lines = [
            "Conditions: " + ", ".join(f"{k}={v}" for k, v in self.conditions.items()),
            f"Degradation threshold: {self.degradation_threshold_s:.2f}s/lap",
            "",
        ]
        ranked = sorted(
            self.compounds, key=lambda c: (c.estimated_life_laps is None, -(c.estimated_life_laps or 0))
        )
        for c in ranked:
            if c.estimated_life_laps is not None:
                life_str = f"{c.estimated_life_laps:.1f} laps"
            else:
                life_str = "durable -- predicted degradation stays below threshold across the full search range"
            parts = [f"  {c.compound:12s} life: {life_str:28s}"]
            if c.degradation_rate_s_per_lap is not None:
                parts.append(f"rate: {c.degradation_rate_s_per_lap:.3f} s/lap")
            flags = []
            if c.used_global_fallback:
                flags.append("pooled fallback fit -- limited data for this compound")
            if c.is_extrapolating:
                flags.append("EXTRAPOLATING beyond training data -- treat with caution")
            if flags:
                parts.append(f"[{'; '.join(flags)}]")
            lines.append("   ".join(parts))

        best = self.best_compound_by_life()
        if best is not None:
            lines += ["", f"Longest estimated life: {best.compound} ({best.estimated_life_laps:.1f} laps)"]

        return "\n".join(lines)


def default_covariates(
    df: pd.DataFrame, model: TireDegradationModel, circuit: str | None = None, compound: str | None = None
) -> dict:
    """Sensible default values for any of the model's required
    feature_columns not explicitly supplied by the caller -- dataset
    averages, restricted to the given circuit and compound if provided.

    Restricting by COMPOUND as well as circuit matters: driver behavior
    features (braking, throttle) plausibly depend on which compound is
    fitted (e.g. more aggressive management on a high-grip Soft than a
    conservative Hard). Blending across all compounds to compute one
    average -- as an earlier version of this function did -- creates the
    same kind of physically-inconsistent query that caused the fuel-load
    bug (see docs/VALIDATION.md): querying "SOFT" but feeding it
    Wet/Intermediate-diluted braking behavior the model never saw paired
    with SOFT in training.

    Falls back gracefully if the circuit+compound combination has too
    little data: circuit+compound -> circuit only -> global average.

    NEVER defaults tire_compound, circuit, or tire_age_laps -- those are
    always explicitly provided by the caller (they're what the report is
    actually varying/comparing), never silently filled in.
    """
    subset = df
    if circuit is not None and "circuit" in df.columns:
        circuit_subset = df[df["circuit"] == circuit]
        if not circuit_subset.empty:
            subset = circuit_subset

    if compound is not None and "tire_compound" in subset.columns:
        compound_subset = subset[subset["tire_compound"] == compound]
        if not compound_subset.empty:
            subset = compound_subset
        # else: keep the circuit-only subset -- not enough data for this
        # exact circuit+compound combination, circuit-level average is the
        # next-best fallback rather than silently ignoring compound entirely.

    defaults: dict = {}
    for feature in model.feature_columns:
        if feature in ("tire_compound", "circuit", "tire_age_laps"):
            continue
        if feature not in subset.columns:
            continue
        series = subset[feature].dropna()
        if series.empty:
            continue
        if series.dtype == bool:
            defaults[feature] = bool(series.mode().iloc[0])
        else:
            defaults[feature] = float(series.mean())
    return defaults


def generate_tire_report(
    model: TireDegradationModel,
    *,
    circuit: str | None = None,
    track_temp_c: float,
    degradation_threshold_s: float,
    compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
    max_age_laps: int = 60,
    extra_covariates: dict | None = None,
    training_df: pd.DataFrame | None = None,
) -> TireStrategyReport:
    """Build a tire strategy report for one set of conditions, working with
    ANY TireDegradationModel.

    Parameters
    ----------
    model : an already-fitted model.
    circuit : circuit name, only relevant for models that take circuit
        IDENTITY as an input (e.g. RandomForestTireDegradationModel).
    track_temp_c : track temperature for the query -- shared across every
        compound (a genuine external condition, not something that should
        vary by compound the way driver behavior might).
    degradation_threshold_s : see TireLifeEstimator's docstring -- a
        judgment call about what "worn out" means, not a fixed constant.
    extra_covariates : explicit OVERRIDE values applied identically to
        every compound (e.g. forcing a specific hypothetical driving
        style). Takes precedence over any auto-computed per-compound
        default below.
    training_df : if provided, used to automatically compute PER-COMPOUND,
        circuit-specific defaults (see default_covariates()) for any
        feature the model needs beyond track_temp_c/circuit/compound/age.
        Strongly recommended whenever the model uses features that could
        plausibly correlate with compound choice (driver behavior, fuel
        load) -- without this, those features fall back to whatever's in
        extra_covariates applied identically across compounds, which risks
        the same physically-inconsistent-query problem documented in
        docs/VALIDATION.md (the fuel-load and driver-behavior bugs).
    """
    estimator = TireLifeEstimator(model, degradation_threshold_s, max_age_laps=max_age_laps)

    shared_covariates = dict(extra_covariates or {})
    shared_covariates["track_temp_c"] = track_temp_c
    if circuit is not None:
        shared_covariates["circuit"] = circuit

    conditions_for_display = {"track_temp_c": track_temp_c}
    if circuit is not None:
        conditions_for_display["circuit"] = circuit

    estimates = []
    for compound in compounds:
        covariates = dict(shared_covariates)
        if training_df is not None:
            auto_defaults = default_covariates(training_df, model, circuit=circuit, compound=compound)
            for key, value in auto_defaults.items():
                covariates.setdefault(key, value)  # explicit overrides always win

        # fuel_load_estimate_kg decreases deterministically with tire age in
        # reality (a proxy for lap number -- see features/strategy.py).
        # Holding it at one static value while sweeping age is physically
        # inconsistent and caused a real bug (see docs/VALIDATION.md) -- if
        # the model uses this feature, compute it AS A FUNCTION of the
        # swept age instead, using the same linear fuel-burn assumption the
        # dataset itself was built with (config.Settings).
        age_dependent_covariates = {}
        if "fuel_load_estimate_kg" in model.feature_columns:
            from f1_tire_model.config import DEFAULT_SETTINGS

            covariates.pop("fuel_load_estimate_kg", None)
            initial_fuel = DEFAULT_SETTINGS.initial_fuel_kg
            burn_rate = DEFAULT_SETTINGS.fuel_burn_rate_kg_per_lap
            age_dependent_covariates["fuel_load_estimate_kg"] = (
                lambda ages, _initial=initial_fuel, _rate=burn_rate: np.maximum(_initial - ages * _rate, 0.0)
            )

        life = estimator.estimate_life_laps(
            tire_compound=compound, age_dependent_covariates=age_dependent_covariates or None, **covariates
        )

        rate = None
        if hasattr(model, "effective_rate_s_per_lap"):
            try:
                rate = model.effective_rate_s_per_lap(compound, **covariates)
            except TypeError:
                rate = None  # this model's effective_rate_s_per_lap doesn't accept these covariates

        used_fallback = None
        if hasattr(model, "has_dedicated_fit"):
            used_fallback = not model.has_dedicated_fit(compound)

        is_extrapolating = None
        if hasattr(model, "extrapolation_mask"):
            # A representative age (1 -- always valid) for this quick
            # single-point check. Also fill in any age-dependent covariates
            # (e.g. fuel_load_estimate_kg) at that same reference age --
            # they were removed from `covariates` above specifically
            # because they're handled as functions of age, not static
            # values, so this diagnostic query needs them added back.
            diagnostic_covariates = dict(covariates)
            for key, fn in age_dependent_covariates.items():
                diagnostic_covariates[key] = fn(np.array([1.0]))[0]

            query_df = pd.DataFrame({
                **{k: [v] for k, v in diagnostic_covariates.items()},
                "tire_compound": [compound],
                "tire_age_laps": [1],
            })
            is_extrapolating = bool(model.extrapolation_mask(query_df)[0])

        estimates.append(
            CompoundLifeEstimate(
                compound=compound,
                estimated_life_laps=life,
                degradation_rate_s_per_lap=rate,
                used_global_fallback=used_fallback,
                is_extrapolating=is_extrapolating,
            )
        )

    return TireStrategyReport(
        conditions=conditions_for_display, degradation_threshold_s=degradation_threshold_s, compounds=estimates
    )