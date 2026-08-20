"""
Thin wrapper around FastF1 session loading.

Why wrap FastF1 instead of calling `fastf1.get_session(...)` directly
throughout the codebase:
- FastF1's cache must be enabled once, globally, before any session is
  loaded. Centralizing that call means it happens exactly once, in one
  place, driven by our own `Settings` (see config.py) instead of being
  duplicated (or forgotten) in every notebook/script.
- If FastF1's API changes in a future version, only this module needs
  updating — `telemetry/`, `features/`, etc. depend on this module's
  return types, not on `fastf1` directly.
- Makes testing possible: tests can monkeypatch `load_session` without
  needing network access or a real FastF1 cache.
"""

from __future__ import annotations

import fastf1

from f1_tire_model.config import Settings, DEFAULT_SETTINGS

_cache_enabled = False


def _ensure_cache_enabled(settings: Settings) -> None:
    global _cache_enabled
    if not _cache_enabled:
        settings.ensure_dirs()
        fastf1.Cache.enable_cache(str(settings.cache_dir))
        _cache_enabled = True


def load_session(
    season: int,
    round_number: int,
    session_name: str = "R",
    *,
    load_laps: bool = True,
    load_telemetry: bool = True,
    load_weather: bool = True,
    settings: Settings = DEFAULT_SETTINGS,
) -> fastf1.core.Session:
    """Load and return a single FastF1 session, with caching enabled.

    Parameters
    ----------
    season : e.g. 2024
    round_number : championship round, e.g. 1 for the season-opening race
    session_name : "FP1", "FP2", "FP3", "Q", "R" (see FastF1 docs)
    load_laps / load_telemetry / load_weather : passed through to FastF1's
        `session.load(...)` so callers that only need lap data (no telemetry)
        can load faster.
    settings : injected so tests / alternate environments can point at a
        different cache directory.
    """
    _ensure_cache_enabled(settings)

    session = fastf1.get_session(season, round_number, session_name)
    session.load(laps=load_laps, telemetry=load_telemetry, weather=load_weather)
    return session
