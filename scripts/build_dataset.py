#!/usr/bin/env python
"""
Build the tire degradation dataset from FastF1 data.

This is a thin CLI wrapper around the pipeline modules — it deliberately
contains almost no logic of its own (just iteration + logging + manifest
checks), because every piece with actual behavior (session loading,
segmentation, feature extraction, storage) already has its own tested
module. If this script breaks, the fix belongs in this file; if the
*data* is wrong, the fix belongs in one of the modules under src/.

Usage
-----
    python scripts/build_dataset.py                       # build configured seasons, Race sessions
    python scripts/build_dataset.py --season 2024          # override seasons
    python scripts/build_dataset.py --session-name Q       # build Qualifying instead of Race
    python scripts/build_dataset.py --rounds 1 2 3          # only specific rounds
    python scripts/build_dataset.py --force                # rebuild even if manifest says done

Runs are resumable: sessions already recorded in the manifest are skipped
unless --force is passed. Logging is intentionally verbose (INFO level) —
a season build takes a long time and you want to see progress, not silence.
"""

from __future__ import annotations

import argparse
import logging
import sys

import fastf1

from f1_tire_model.config import DEFAULT_SETTINGS, Settings
from f1_tire_model.data.session_adapter import build_session_inputs
from f1_tire_model.data.session_loader import load_session
from f1_tire_model.datasets.builder import SessionDatasetBuilder
from f1_tire_model.datasets.manifest import is_built, mark_built
from f1_tire_model.datasets.storage import write_fragment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--season", type=int, nargs="+", default=list(DEFAULT_SETTINGS.seasons),
        help="Season(s) to build. Defaults to Settings.seasons.",
    )
    parser.add_argument(
        "--rounds", type=int, nargs="+", default=None,
        help="Specific round numbers to build. Defaults to the full season schedule.",
    )
    parser.add_argument(
        "--session-name", type=str, default="R",
        help="FastF1 session identifier: FP1, FP2, FP3, Q, R. Default: R (Race).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild sessions even if the manifest already marks them as built.",
    )
    return parser.parse_args()


def rounds_for_season(season: int, requested_rounds: list[int] | None) -> list[int]:
    if requested_rounds is not None:
        return requested_rounds

    schedule = fastf1.get_event_schedule(season)
    # RoundNumber 0 is the FastF1 convention for pre-season testing; skip it,
    # it has no representative race-pace tire degradation data.
    return sorted(r for r in schedule["RoundNumber"].tolist() if r > 0)


def build_one_session(
    season: int, round_number: int, session_name: str, settings: Settings, force: bool
) -> None:
    if not force and is_built(season, round_number, session_name, settings=settings):
        logger.info("Skipping %s round %s (%s) — already in manifest.", season, round_number, session_name)
        return

    logger.info("Loading %s round %s session %s ...", season, round_number, session_name)
    try:
        session = load_session(season, round_number, session_name, settings=settings)
    except Exception:
        logger.exception("Failed to load %s round %s (%s) — skipping session.", season, round_number, session_name)
        return

    inputs = build_session_inputs(session)
    builder = SessionDatasetBuilder(settings=settings)

    records = builder.build_session_records(
        inputs, season=season, round_number=round_number, session_name=session_name
    )

    if not records:
        logger.warning("No records produced for %s round %s (%s).", season, round_number, session_name)
        return

    path = write_fragment(records, season=season, round_number=round_number, session_name=session_name, settings=settings)
    mark_built(season, round_number, session_name, settings=settings)
    logger.info("Wrote %d records to %s", len(records), path)


def main() -> int:
    args = parse_args()
    settings = DEFAULT_SETTINGS

    for season in args.season:
        rounds = rounds_for_season(season, args.rounds)
        logger.info("Season %s: %d round(s) to process.", season, len(rounds))

        for round_number in rounds:
            build_one_session(season, round_number, args.session_name, settings, args.force)

    return 0


if __name__ == "__main__":
    sys.exit(main())
