"""
Base interface for feature extractors.

Why extractors are split into small classes (TireFeatureExtractor,
DriverFeatureExtractor, VehicleFeatureExtractor, TrackFeatureExtractor,
EnvironmentFeatureExtractor, StrategyFeatureExtractor) instead of one big
"compute_all_features" function:

1. Testability — you can unit test "does braking-deceleration extraction work
   on this one lap's telemetry" without needing a full session loaded.
2. Selective computation — while iterating on the physics model you may only
   need tire + driver features; you shouldn't have to pay the cost (or the
   FastF1 API calls) of computing track/environment features every time.
3. ML readiness — when you get to feature selection / importance analysis in
   the ML course, features already being tagged by category (via
   `feature_group`) makes it trivial to ask "how much does the Vehicle group
   contribute vs. the Environment group."

Each extractor takes whatever raw data it needs (a lap, a stint, telemetry)
and returns a flat dict of {feature_name: value}. A higher-level
`FeatureBuilder` (added when we get to the features/ module itself) merges
these dicts into one row per lap or per stint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FeatureExtractor(ABC):
    """Base class for all feature extractors.

    Subclasses implement `extract` and declare `feature_group` for
    downstream bookkeeping (e.g. grouping columns in visualizations,
    or excluding a whole feature group in an ablation study).
    """

    #: Category name, used for documentation, grouping in plots, and
    #: feature-importance breakdowns during the future ML phase.
    feature_group: str = "unspecified"

    @abstractmethod
    def extract(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Compute and return {feature_name: value} for one lap/stint/session.

        Concrete subclasses should have explicit, typed signatures
        (e.g. `extract(self, lap, telemetry) -> dict[str, float]`) rather than
        *args/**kwargs — the loose signature here exists only because
        different extractor categories need different inputs.
        """
        raise NotImplementedError
