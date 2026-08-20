"""
Shared visual style for the visualization module — kept separate from the
plotting logic itself so color/size choices can be tuned in one place
without touching plot-construction code.
"""

from __future__ import annotations

# Official-style F1 tire compound colors. These are widely recognized
# conventions (used across broadcasts, timing screens, and F1 media), not
# reproduced assets — a plain color mapping, defined independently here.
COMPOUND_COLORS: dict[str, str] = {
    "SOFT": "#DA291C",
    "MEDIUM": "#FFD12E",
    "HARD": "#F0F0EC",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD",
}
DEFAULT_COMPOUND_COLOR = "#888888"  # fallback for an unrecognized compound name

INVALID_LAP_COLOR = "#BBBBBB"
INVALID_LAP_ALPHA = 0.4

DEFAULT_FIGSIZE = (8, 5)


def compound_color(compound: str) -> str:
    return COMPOUND_COLORS.get(str(compound).upper(), DEFAULT_COMPOUND_COLOR)
