"""Clinical colormaps for CARTO electro-anatomical mapping data.

Matches the standard CARTO 3 color schemes for LAT, bipolar voltage,
and unipolar voltage displays.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def lat_colormap(values: np.ndarray, clamp_range: tuple[float, float] | None = None) -> np.ndarray:
    """Map LAT values (ms) to RGBA colors.

    Red (early activation) -> yellow -> green -> cyan -> blue -> purple (late).
    Matches CARTO clinical convention.

    Args:
        values: Array of LAT values in ms. NaN values get transparent color.
        clamp_range: Optional (min, max) to clamp values before normalization.
                     If None, uses data min/max.

    Returns:
        RGBA float32 array [N, 4] with values in [0, 1].
    """
    return _apply_colormap(values, _LAT_STOPS, clamp_range)


def bipolar_colormap(values: np.ndarray, clamp_range: tuple[float, float] | None = None) -> np.ndarray:
    """Map bipolar voltage values (mV) to RGBA colors.

    Red (low/scar, <0.5mV) -> yellow -> green -> cyan -> purple (normal, >1.5mV).
    Matches CARTO clinical convention for substrate mapping.

    Default clamp range: (0.05, 1.5) mV if not specified.
    """
    if clamp_range is None:
        clamp_range = (0.05, 1.5)
    return _apply_colormap(values, _BIPOLAR_STOPS, clamp_range)


def unipolar_colormap(values: np.ndarray, clamp_range: tuple[float, float] | None = None) -> np.ndarray:
    """Map unipolar voltage values (mV) to RGBA colors.

    Red (low) -> yellow -> green -> blue (high).

    Default clamp range: (3.0, 10.0) mV if not specified.
    """
    if clamp_range is None:
        clamp_range = (3.0, 10.0)
    return _apply_colormap(values, _UNIPOLAR_STOPS, clamp_range)


# Color stop definitions: list of (normalized_position, R, G, B)
# LAT: red (early) → yellow → green → cyan → blue → purple (late)
_LAT_STOPS = [
    (0.0, 1.0, 0.0, 0.0),    # red — earliest
    (0.2, 1.0, 1.0, 0.0),    # yellow
    (0.4, 0.0, 1.0, 0.0),    # green
    (0.6, 0.0, 1.0, 1.0),    # cyan
    (0.8, 0.0, 0.0, 1.0),    # blue
    (1.0, 0.5, 0.0, 1.0),    # purple — latest
]

# Bipolar voltage: red (scar) → yellow → green → cyan → purple (healthy)
_BIPOLAR_STOPS = [
    (0.0, 1.0, 0.0, 0.0),    # red — low voltage (scar)
    (0.25, 1.0, 1.0, 0.0),   # yellow
    (0.5, 0.0, 1.0, 0.0),    # green
    (0.75, 0.0, 1.0, 1.0),   # cyan
    (1.0, 0.5, 0.0, 1.0),    # purple — normal voltage
]

# Unipolar voltage: red (low) → yellow → green → blue (high)
_UNIPOLAR_STOPS = [
    (0.0, 1.0, 0.0, 0.0),    # red
    (0.33, 1.0, 1.0, 0.0),   # yellow
    (0.66, 0.0, 1.0, 0.0),   # green
    (1.0, 0.0, 0.0, 1.0),    # blue
]

# Map of colormap name to function
COLORMAPS = {
    "lat": lat_colormap,
    "bipolar": bipolar_colormap,
    "unipolar": unipolar_colormap,
}


def _apply_colormap(
    values: np.ndarray,
    stops: list[tuple[float, float, float, float]],
    clamp_range: tuple[float, float] | None,
) -> np.ndarray:
    """Normalize values and interpolate through color stops."""
    n = len(values)
    colors = np.zeros((n, 4), dtype=np.float32)

    valid = ~np.isnan(values)
    n_nan = int(np.sum(~valid))
    if n_nan > 0:
        pct = 100.0 * n_nan / n
        logger.debug("Colormap: %d / %d vertices (%.1f%%) are NaN (unmapped)", n_nan, n, pct)
    if not np.any(valid):
        # All NaN — return fully transparent gray (invisible via MASK)
        colors[:, :3] = 0.5
        colors[:, 3] = 0.0
        return colors

    v = values.copy()
    if clamp_range is not None:
        lo, hi = clamp_range
    else:
        lo = float(np.nanmin(v))
        hi = float(np.nanmax(v))

    if hi - lo < 1e-12:
        # All same value — use midpoint color
        t = np.full(n, 0.5, dtype=np.float64)
    else:
        t = (v - lo) / (hi - lo)
        # Log clamping statistics
        valid_v = v[valid]
        n_below = int(np.sum(valid_v < lo))
        n_above = int(np.sum(valid_v > hi))
        if n_below > 0 or n_above > 0:
            n_valid = int(np.sum(valid))
            logger.debug(
                "Colormap: clamped to [%.2f, %.2f] — %d below (%.1f%%), %d above (%.1f%%)",
                lo, hi, n_below, 100.0 * n_below / n_valid,
                n_above, 100.0 * n_above / n_valid,
            )
    t = np.clip(t, 0.0, 1.0)

    # Interpolate through color stops
    positions = np.array([s[0] for s in stops])
    r_stops = np.array([s[1] for s in stops])
    g_stops = np.array([s[2] for s in stops])
    b_stops = np.array([s[3] for s in stops])

    colors[valid, 0] = np.interp(t[valid], positions, r_stops).astype(np.float32)
    colors[valid, 1] = np.interp(t[valid], positions, g_stops).astype(np.float32)
    colors[valid, 2] = np.interp(t[valid], positions, b_stops).astype(np.float32)
    colors[valid, 3] = 1.0

    # NaN values: fully transparent gray (invisible via MASK)
    colors[~valid, :3] = 0.5
    colors[~valid, 3] = 0.0

    return colors
