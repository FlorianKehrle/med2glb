"""CARTO 3 electro-anatomical mapping data reader.

Parses .mesh files (INI-style surface meshes) and _car.txt files
(per-point electrical measurements) from CARTO 3 export directories.
Supports old CARTO (~2015, v4), v7.1 (v5), and v7.2+ (v6) formats.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from med2glb.core.types import (
    CARTO_INACTIVE_GROUP_ID,
    CARTO_LAT_SENTINEL,
    CartoMesh,
    CartoPoint,
    CartoStudy,
)

logger = logging.getLogger("med2glb")

# Back-compat aliases (used by other modules that imported from here)
_LAT_SENTINEL = CARTO_LAT_SENTINEL
_INACTIVE_GROUP_ID = CARTO_INACTIVE_GROUP_ID


def detect_carto_directory(path: Path) -> bool:
    """Check if a directory contains CARTO export data (.mesh files)."""
    if not path.is_dir():
        return False
    return any(path.rglob("*.mesh"))


def find_carto_subdirectories(path: Path) -> list[Path]:
    """Find all directories under *path* that directly contain .mesh files.

    Recursively scans the entire directory tree.  Each returned path is a
    self-contained CARTO export directory (has ``.mesh`` files in it).
    If the root path itself contains ``.mesh`` files, returns ``[path]``.
    Results are sorted by path for deterministic ordering.
    """
    dirs: set[Path] = set()
    for mesh_file in path.rglob("*.mesh"):
        dirs.add(mesh_file.parent)
    return sorted(dirs)


def _find_export_dir(path: Path) -> Path:
    """Find the first directory containing .mesh files under *path*.

    Users may point at any ancestor directory.  Walks down recursively
    and returns the first directory that directly contains ``.mesh`` files.
    """
    if list(path.glob("*.mesh")):
        return path
    for mesh_file in sorted(path.rglob("*.mesh")):
        return mesh_file.parent
    return path


def load_carto_study(
    path: Path,
    progress: Callable[[str, int, int], None] | None = None,
) -> CartoStudy:
    """Load a complete CARTO study from an export directory.

    Discovers and parses all .mesh + _car.txt file pairs.

    Args:
        path: Path to CARTO export directory.
        progress: Optional callback(description, current, total) for progress.
    """
    export_dir = _find_export_dir(path)
    mesh_files = sorted(export_dir.glob("*.mesh"))
    if not mesh_files:
        raise FileNotFoundError(f"No .mesh files found in {path}")

    meshes: list[CartoMesh] = []
    points: dict[str, list[CartoPoint]] = {}
    version = "unknown"
    n_files = len(mesh_files)

    for i, mesh_file in enumerate(mesh_files):
        if progress:
            progress(f"Loading {mesh_file.stem}...", i, n_files)

        mesh = parse_mesh_file(mesh_file)
        meshes.append(mesh)

        # Look for matching _car.txt
        car_file = mesh_file.with_name(mesh_file.stem + "_car.txt")
        if car_file.exists():
            ver, pts = parse_car_file(car_file)
            map_name = mesh_file.stem
            points[map_name] = pts
            if version == "unknown":
                version = ver

    if progress:
        progress(f"Loaded {n_files} mesh(es)", n_files, n_files)

    study_name = export_dir.name
    return CartoStudy(
        meshes=meshes,
        points=points,
        version=version,
        study_name=study_name,
    )


def parse_mesh_file(path: Path) -> CartoMesh:
    """Parse a CARTO .mesh file (INI-style triangulated mesh).

    Format:
        [GeneralAttributes] — MeshID, NumVertex, NumTriangle, MeshColor, ColorsNames
        [VerticesSection]   — ID = X Y Z NX NY NZ GroupID
        [TrianglesSection]  — ID = V0 V1 V2 NX NY NZ GroupID
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    # Parse GeneralAttributes
    attrs = _parse_general_attributes(text)
    mesh_id = int(attrs.get("MeshID", 0))
    num_vertex = int(attrs.get("NumVertex", 0))
    num_triangle = int(attrs.get("NumTriangle", 0))

    # Parse MeshColor (RGBA floats)
    mesh_color_str = attrs.get("MeshColor", "0.5 0.5 0.5 1.0")
    mesh_color_parts = mesh_color_str.split()
    mesh_color = tuple(float(c) for c in mesh_color_parts[:4])
    if len(mesh_color) < 4:
        mesh_color = (0.5, 0.5, 0.5, 1.0)

    # Parse ColorsNames
    color_names_str = attrs.get("ColorsNames", "")
    color_names = color_names_str.split() if color_names_str else []

    # Parse TransparentGroupsIDs (space-separated ints, e.g. "0 1 2 3 4")
    transparent_str = attrs.get("TransparentGroupsIDs", "")
    transparent_group_ids: list[int] = []
    if transparent_str.strip():
        try:
            transparent_group_ids = [int(x) for x in transparent_str.split()]
        except ValueError:
            logger.debug("Could not parse TransparentGroupsIDs: %s", transparent_str)

    # Parse vertices
    vertices, normals, group_ids = _parse_vertices_section(text, num_vertex)

    # Parse triangles
    faces, face_group_ids = _parse_triangles_section(text, num_triangle)

    # Parse per-vertex color values (LAT, bipolar, unipolar from CARTO's
    # own interpolation — used for coloring instead of re-interpolating
    # from sparse car-file points).
    vertex_color_values = _parse_vertices_colors_section(
        text, color_names, num_vertex,
    )

    structure_name = path.stem

    return CartoMesh(
        mesh_id=mesh_id,
        vertices=vertices,
        faces=faces,
        normals=normals,
        group_ids=group_ids,
        face_group_ids=face_group_ids,
        mesh_color=mesh_color,
        color_names=color_names,
        transparent_group_ids=transparent_group_ids,
        structure_name=structure_name,
        vertex_color_values=vertex_color_values,
    )


def parse_car_file(path: Path) -> tuple[str, list[CartoPoint]]:
    """Parse a CARTO _car.txt file (tab-separated point measurements).

    Returns (version_string, list_of_points).

    All versions share columns 0-12:
        P, idx, pointID, 0, X, Y, Z, orientX, orientY, orientZ,
        bipolarV, unipolarV, LAT
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.strip().splitlines()
    if not lines:
        return "unknown", []

    # Parse header: "VERSION_X_Y MapName"
    header = lines[0].strip()
    version = "unknown"
    header_match = re.match(r"VERSION_(\d+)_(\d+)", header)
    if header_match:
        version = f"{header_match.group(1)}.{header_match.group(2)}"

    # Collect valid data lines for bulk parsing
    valid_lines: list[str] = []
    for line in lines[1:]:
        line = line.strip()
        if line and line.startswith("P"):
            fields = line.split("\t")
            if len(fields) >= 13:
                valid_lines.append(line)
            else:
                logger.debug(f"Skipping car line with {len(fields)} fields: {line[:80]}")

    if not valid_lines:
        return version, []

    # Bulk parse: extract columns 2,4-12 from tab-separated lines
    try:
        # Build numeric block from fields we need
        rows: list[list[float]] = []
        for line in valid_lines:
            fields = line.split("\t")
            # point_id(2), x(4), y(5), z(6), ox(7), oy(8), oz(9),
            # bipolar(10), unipolar(11), lat(12)
            rows.append([
                float(fields[2]),
                float(fields[4]), float(fields[5]), float(fields[6]),
                float(fields[7]), float(fields[8]), float(fields[9]),
                float(fields[10]), float(fields[11]), float(fields[12]),
            ])
        data = np.array(rows, dtype=np.float64)
    except (ValueError, IndexError) as e:
        logger.debug(f"Bulk car parse failed, falling back: {e}")
        return version, []

    point_ids = data[:, 0].astype(np.int64)
    positions = data[:, 1:4]   # (N, 3)
    orientations = data[:, 4:7]  # (N, 3)
    bipolar_v = data[:, 7]
    unipolar_v = data[:, 8]
    lat_raw = data[:, 9]
    lat_values = np.where(lat_raw == _LAT_SENTINEL, np.nan, lat_raw)

    points: list[CartoPoint] = []
    for i in range(len(data)):
        points.append(CartoPoint(
            point_id=int(point_ids[i]),
            position=positions[i].copy(),
            orientation=orientations[i].copy(),
            bipolar_voltage=float(bipolar_v[i]),
            unipolar_voltage=float(unipolar_v[i]),
            lat=float(lat_values[i]),
        ))

    return version, points


def _parse_general_attributes(text: str) -> dict[str, str]:
    """Extract key-value pairs from [GeneralAttributes] section."""
    attrs: dict[str, str] = {}
    in_section = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[GeneralAttributes]":
            in_section = True
            continue
        if stripped.startswith("[") and in_section:
            break
        if not in_section:
            continue
        if stripped.startswith(";") or not stripped:
            continue

        # Parse "Key = Value" (with varying whitespace)
        match = re.match(r"(\w+)\s*=\s*(.*)", stripped)
        if match:
            attrs[match.group(1)] = match.group(2).strip()

    return attrs


def _extract_section(text: str, section_name: str) -> str:
    """Extract the text block for a given [SectionName] from INI-style text."""
    start_marker = f"[{section_name}]"
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    # Find next section or end of text
    next_section = text.find("[", start)
    if next_section == -1:
        return text[start:]
    return text[start:next_section]


# Canonical lowercase names for the coloring channels we use.
_COLOR_NAME_MAP: dict[str, str] = {
    "lat": "lat",
    "bipolar": "bipolar",
    "unipolar": "unipolar",
}


def _parse_vertices_colors_section(
    text: str,
    color_names: list[str],
    num_vertex: int,
) -> dict[str, np.ndarray]:
    """Parse [VerticesColorsSection] into per-vertex scalar arrays.

    Each row: ``ID = val0 val1 val2 ...`` with one column per color channel
    listed in *color_names* (from [GeneralAttributes] ColorsNames).

    Returns a dict mapping lowercase coloring name → float64 array [N] with
    NaN where the original value was the -10000 sentinel.  Only channels
    present in ``_COLOR_NAME_MAP`` are returned.
    """
    section = _extract_section(text, "VerticesColorsSection")
    if not section:
        return {}

    # Determine which column indices to keep
    col_map: dict[int, str] = {}  # column_index → canonical name
    for idx, name in enumerate(color_names):
        canonical = _COLOR_NAME_MAP.get(name.lower())
        if canonical is not None:
            col_map[idx] = canonical

    if not col_map:
        return {}

    # Parse data lines
    data_lines: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        parts = stripped.split("=", 1)
        if len(parts) == 2:
            data_lines.append(parts[1])

    if not data_lines:
        return {}

    from io import StringIO
    block = "\n".join(data_lines)
    data = np.genfromtxt(StringIO(block), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    result: dict[str, np.ndarray] = {}
    sentinel = float(CARTO_LAT_SENTINEL)  # -10000
    for col_idx, canonical in col_map.items():
        if col_idx >= data.shape[1]:
            continue
        values = data[:, col_idx].copy()
        values[values == sentinel] = np.nan
        result[canonical] = values

    return result


def _parse_vertices_section(
    text: str, expected_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse [VerticesSection]: ID = X Y Z NX NY NZ GroupID.

    Returns (vertices[N,3], normals[N,3], group_ids[N]).
    """
    section = _extract_section(text, "VerticesSection")
    if not section:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.int32),
        )

    # Strip "ID = " prefix from each data line, keep only numeric data
    data_lines: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        parts = stripped.split("=", 1)
        if len(parts) == 2:
            data_lines.append(parts[1])

    if not data_lines:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.int32),
        )

    from io import StringIO
    block = "\n".join(data_lines)
    data = np.genfromtxt(StringIO(block), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    count = len(data)
    # Columns: X Y Z NX NY NZ GroupID [possible extras]
    vertices = data[:, :3]
    normals = data[:, 3:6]
    group_ids = data[:, 6].astype(np.int32)

    return vertices, normals, group_ids


def _parse_triangles_section(
    text: str, expected_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Parse [TrianglesSection]: ID = V0 V1 V2 NX NY NZ GroupID.

    Returns (faces[M,3], face_group_ids[M]).
    """
    section = _extract_section(text, "TrianglesSection")
    if not section:
        return np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int32)

    data_lines: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        parts = stripped.split("=", 1)
        if len(parts) == 2:
            data_lines.append(parts[1])

    if not data_lines:
        return np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int32)

    from io import StringIO
    block = "\n".join(data_lines)
    data = np.genfromtxt(StringIO(block), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    faces = data[:, :3].astype(np.int32)
    if data.shape[1] >= 7:
        face_group_ids = data[:, 6].astype(np.int32)
    else:
        face_group_ids = np.zeros(len(data), dtype=np.int32)

    return faces, face_group_ids
