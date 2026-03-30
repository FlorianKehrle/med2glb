"""RF ablation lesion reader for CARTO 3 export directories.

Parses RF application files (``RF_{map_name}_{N}.txt``) and cross-references
each ablation event's position from the matching CartoPoint in the _car.txt
file (matched by point_id == N).  Only files where a matching CartoPoint exists
are included; unmatched RF files are silently skipped with a debug log.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from med2glb.core.types import CartoPoint, LesionPoint

logger = logging.getLogger("med2glb")


def find_rf_files(export_dir: Path, map_name: str) -> list[tuple[int, Path]]:
    """Find RF ablation files for a given map name.

    Looks for files matching ``RF_{map_name}_{N}.txt`` in *export_dir*.
    Returns a list of ``(point_id, path)`` tuples sorted by *point_id*.
    """
    rf_files: list[tuple[int, Path]] = []
    pattern_re = re.compile(
        r"^RF_" + re.escape(map_name) + r"_(\d+)\.txt$",
    )
    for rf_path in export_dir.iterdir():
        match = pattern_re.match(rf_path.name)
        if match:
            rf_files.append((int(match.group(1)), rf_path))
    return sorted(rf_files)


def parse_rf_file(path: Path) -> dict[str, float]:
    """Parse a single RF application file and return summary statistics.

    The file is space-separated with a header line:
        PiuTimeStamp  Irrigation  Power Mode  AblTime1  Power1
        Impedance1  DistalTemperature1  ProximalTemperature1

    Returns a dict with:
        ``max_power_w``       — peak RF power (watts)
        ``duration_s``        — total ablation duration (seconds)
        ``max_temperature_c`` — peak distal tip temperature (°C)
    Returns an empty dict if the file is unreadable or has no data rows.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("Cannot read RF file %s: %s", path, exc)
        return {}

    lines = text.strip().splitlines()
    if len(lines) < 2:
        return {}

    # Header: "PiuTimeStampIrrigationPower ModeAblTime1Power1Impedance1…"
    # (column names may be run together in the header; we rely on positional parsing)
    rows: list[list[float]] = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parts = stripped.split()
            # columns: PiuTimeStamp(0) Irrigation(1) PowerMode(2)
            #          AblTime1(3) Power1(4) Impedance1(5)
            #          DistalTemp1(6) ProximalTemp1(7)
            if len(parts) >= 7:
                rows.append([
                    float(parts[3]),  # AblTime1 (ms)
                    float(parts[4]),  # Power1 (W)
                    float(parts[6]),  # DistalTemperature1 (°C)
                ])
        except (ValueError, IndexError):
            continue

    if not rows:
        return {}

    data = np.array(rows, dtype=np.float64)
    max_power_w = float(np.max(data[:, 1]))
    duration_s = float(np.max(data[:, 0])) / 1000.0   # ms → s
    max_temp_c = float(np.max(data[:, 2]))

    return {
        "max_power_w": max_power_w,
        "duration_s": duration_s,
        "max_temperature_c": max_temp_c,
    }


def load_lesion_points(
    export_dir: Path,
    map_name: str,
    car_points: list[CartoPoint],
) -> list[LesionPoint]:
    """Load ablation lesion points for a single map.

    Discovers all ``RF_{map_name}_{N}.txt`` files in *export_dir*, reads the RF
    energy statistics from each, and cross-references the ablation position from
    *car_points* by matching ``CartoPoint.point_id == N``.

    Args:
        export_dir: CARTO export directory containing RF files.
        map_name:   Mesh/map name (stem of the ``.mesh`` file, e.g. ``"1-Map"``).
        car_points: Parsed CartoPoint list for this map (from ``parse_car_file``).

    Returns:
        List of :class:`LesionPoint` objects in ascending point_id order.
        Empty list when no RF files exist or none can be matched.
    """
    rf_files = find_rf_files(export_dir, map_name)
    if not rf_files:
        return []

    # Build a lookup from point_id to CartoPoint position for fast matching
    position_by_id: dict[int, np.ndarray] = {
        pt.point_id: pt.position for pt in car_points
    }

    lesions: list[LesionPoint] = []
    for point_id, rf_path in rf_files:
        position = position_by_id.get(point_id)
        if position is None:
            logger.debug(
                "RF file %s: no CartoPoint with point_id=%d — skipping",
                rf_path.name, point_id,
            )
            continue

        stats = parse_rf_file(rf_path)
        if not stats:
            logger.debug("RF file %s: no usable data — skipping", rf_path.name)
            continue

        lesions.append(LesionPoint(
            point_id=point_id,
            position=position.copy(),
            max_power_w=stats["max_power_w"],
            duration_s=stats["duration_s"],
            max_temperature_c=stats["max_temperature_c"],
        ))

    logger.debug(
        "Loaded %d/%d lesion points for map '%s'",
        len(lesions), len(rf_files), map_name,
    )
    return lesions
