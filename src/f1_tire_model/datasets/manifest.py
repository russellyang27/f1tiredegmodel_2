"""
Tracks which (season, round, session) combinations have already been built,
so a multi-season dataset build is resumable instead of re-fetching
sessions (and re-hitting the FastF1 API / cache) it already has fragments
for.

A plain JSON file is intentional here rather than a database: this is a
personal, single-writer, multi-month project, not a multi-user system —
a JSON manifest is trivially inspectable (`cat manifest.json`) and diffable
in git-adjacent tooling, which matters more than the features a database
would add.
"""

from __future__ import annotations

import json
from pathlib import Path

from f1_tire_model.config import DEFAULT_SETTINGS, Settings

_MANIFEST_FILENAME = "manifest.json"


def _manifest_path(settings: Settings) -> Path:
    return settings.dataset_output_dir / _MANIFEST_FILENAME


def _session_key(season: int, round_number: int, session_name: str) -> str:
    return f"{season}_{round_number:02d}_{session_name}"


def load_manifest(settings: Settings = DEFAULT_SETTINGS) -> set[str]:
    """Return the set of session keys already built. Empty set if no manifest yet."""
    path = _manifest_path(settings)
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()))


def is_built(season: int, round_number: int, session_name: str, *, settings: Settings = DEFAULT_SETTINGS) -> bool:
    return _session_key(season, round_number, session_name) in load_manifest(settings)


def mark_built(
    season: int, round_number: int, session_name: str, *, settings: Settings = DEFAULT_SETTINGS
) -> None:
    """Record a session as built. Safe to call more than once for the same session."""
    settings.ensure_dirs()
    manifest = load_manifest(settings)
    manifest.add(_session_key(season, round_number, session_name))

    path = _manifest_path(settings)
    path.write_text(json.dumps(sorted(manifest), indent=2))
