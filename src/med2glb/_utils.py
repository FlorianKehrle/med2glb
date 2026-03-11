"""Shared utility functions for med2glb."""

from __future__ import annotations


def fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration (e.g. '2.3s', '1m 23s', '1h 5m 12s')."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    s = int(seconds)
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"
