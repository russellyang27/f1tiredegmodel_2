"""
f1_tire_model
=============

A physics-informed, ML-ready pipeline for Formula 1 tire degradation analysis.

Package layout
--------------
data/            Raw session loading and FastF1 cache management.
telemetry/       Synchronization and cleaning of car telemetry channels.
preprocessing/   Lap/stint segmentation, outlier filtering (pit laps, SC laps, etc).
features/        Feature extraction (tire, driver, vehicle, track, environment, strategy).
models/          Prediction models. Physics/statistical baseline now; ML models later,
                 both implementing the same TireDegradationModel interface.
visualization/   Engineering and validation plots.
datasets/        Canonical structured dataset builders + storage (parquet).

Design principle: every module downstream of `features/` and `models/` is written
against small, stable interfaces (see features.base and models.base), so that
swapping the baseline physics model for a machine learning model later requires
implementing a new class, not modifying the pipeline.
"""

__version__ = "0.1.0"
