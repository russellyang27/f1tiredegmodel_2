# F1 Tire Degradation Modeling Pipeline

A physics-informed, ML-ready pipeline for predicting Formula 1 tire degradation,
built on [FastF1](https://docs.fastf1.dev/) telemetry data.

**Status:** V1 — data pipeline + baseline physics/statistical model (no ML yet, by design).

## Why this architecture

This project is built to be reused: the same feature-extraction and dataset
pipeline will later train Random Forest / SVM / neural network models, without
rewriting the data pipeline. That requirement drives three decisions:

1. **`models/base.py` defines a `TireDegradationModel` interface** (fit / predict /
   evaluate). The V1 physics baseline and every future ML model implement this
   same interface, so the rest of the pipeline (visualization, validation,
   comparison across models) is written once, against the interface.
2. **`datasets/schema.py` defines a typed `LapFeatureRecord`** — one authoritative
   list of every feature/target column the project will ever produce, with an
   explicit split between "identifier/target" columns and "candidate model
   input" columns.
3. **`features/base.py` splits feature extraction into small, single-responsibility
   extractors** (tire / driver / vehicle / track / environment / strategy), so
   feature groups can be tested, computed selectively, and later analyzed for
   importance independently.

## Project layout

```
src/f1_tire_model/
    data/            FastF1 session loading + cache management
    telemetry/        Telemetry cleaning & synchronization
    preprocessing/     Lap/stint segmentation, outlier filtering
    features/         Feature extractors (tire/driver/vehicle/track/env/strategy)
    models/           TireDegradationModel interface + physics baseline
    visualization/     Engineering + validation plots
    datasets/         Record schema + dataset builders (parquet storage)
tests/                Mirrors src/ layout
notebooks/             Exploration only — no pipeline logic lives here
data_cache/            FastF1 cache (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `-e` (editable) install means `import f1_tire_model` works from anywhere —
scripts, tests, and notebooks — without path hacks.

## Running tests

```bash
pytest
```

## Roadmap

- [x] Repository architecture + core interfaces (`TireDegradationModel`, `FeatureExtractor`, `LapFeatureRecord`)
- [ ] Telemetry loading & cleaning (`telemetry/`)
- [ ] Lap/stint segmentation & outlier filtering (`preprocessing/`)
- [ ] Feature extractors, one category at a time (`features/`)
- [ ] Dataset builder assembling `LapFeatureRecord`s into a parquet dataset (`datasets/`)
- [ ] Baseline physics/statistical degradation model (`models/`)
- [ ] Validation plots (predicted vs. actual lap time degradation) (`visualization/`)
- [ ] (Future phase, separate course project) ML models implementing `TireDegradationModel`
