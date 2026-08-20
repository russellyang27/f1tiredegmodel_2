"""
Tire features: compound, tire age, stint number, stint length.

This extractor deliberately takes a `StintInfo` (from preprocessing) rather
than re-deriving stint membership itself — segmentation and feature
extraction are separate concerns, and this keeps the "what stint is this
lap in" logic in exactly one place.
"""

from __future__ import annotations

import pandas as pd

from f1_tire_model.features.base import FeatureExtractor
from f1_tire_model.preprocessing.segmentation import StintInfo


class TireFeatureExtractor(FeatureExtractor):
    feature_group = "tire"

    def extract(self, lap: pd.Series, stint: StintInfo) -> dict[str, float]:
        """
        Parameters
        ----------
        lap : one row of session.laps for this driver/lap (needs TyreLife).
        stint : the StintInfo this lap belongs to (from preprocessing.segmentation).
        """
        tire_age = lap.get("TyreLife")

        return {
            "tire_compound": stint.compound,
            "tire_age_laps": int(tire_age) if pd.notna(tire_age) else None,
            "stint_number": stint.stint_number,
            # LEAKAGE GUARD: stint_length_laps is hindsight information (see
            # preprocessing.segmentation module docstring). Only expose it
            # when the stint has actually finished; otherwise this must stay
            # None so it can never accidentally end up as a model input for
            # a mid-race prediction.
            "stint_length_laps": stint.stint_length_laps if stint.is_complete else None,
        }
