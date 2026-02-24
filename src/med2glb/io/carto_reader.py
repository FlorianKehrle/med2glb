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

from med2glb.core.types import CartoMesh, CartoPoint, CartoStudy

logger = logging.getLogger("med2glb")

# Sentinel values in CARTO data
_LAT_SENTINEL = -10000
_INACTIVE_GROUP_ID = -1000000


def detect_carto_directory(path: Path) -> bool:
    """Check if a directory contains CARTO export data (.mesh files)."""
    if not path.is_dir():
        return False
    # Look for .mesh files in the directory tree (max 2 levels deep)
    for depth_pattern in [path / "*.mesh", path / "*" / "*.mesh",
                          path / "*" / "*" / "*.mesh"]:
        if list(depth_pattern.parent.glob(depth_pattern.name)):
            return True
    return False


def find_carto_subdirectories(path: Path) -> list[Path]:
    """Find all subdirectories that each contain a self-contained CARTO export.

    Walks one level of subdirectories and returns those where
    ``detect_carto_directory()`` is True.  If the root path itself is a
    single CARTO export (has .mesh files directly), returns ``[path]``.
    """
    # Root itself is a CARTO export
    if list(path.glob("*.mesh")):
        return [path]

    results: list[Path] = []
    for sub in sorted(path.iterdir()):
        if sub.is_dir() and detect_carto_directory(sub):
            results.append(sub)
    return results


def _find_export_dir(path: Path) -> Path:
    """Find the actual export directory containing .mesh files.

    Users may point at the version directory or the export subdirectory.
    Walk down until we find .mesh files.
    """
    if list(path.glob("*.mesh")):
        return path
    # Search one level of subdirectories
    for sub in sorted(path.iterdir()):
        if sub.is_dir() and list(sub.glob("*.mesh")):
            return sub
        # Two levels deep (e.g. Version_X/Study 1/Export_Study/)
        if sub.is_dir():
            for sub2 in sorted(sub.iterdir()):
                if sub2.is_dir() and list(sub2.glob("*.mesh")):
                    return sub2
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

    # Parse vertices
    vertices, normals, group_ids = _parse_vertices_section(text, num_vertex)

    # Parse triangles
    faces, face_group_ids = _parse_triangles_section(text, num_triangle)

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
        structure_name=structure_name,
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

    points: list[CartoPoint] = []
    for line in lines[1:]:
        line = line.strip()
        if not line or not line.startswith("P"):
            continue

        fields = line.split("\t")
        if len(fields) < 13:
            logger.debug(f"Skipping car line with {len(fields)} fields: {line[:80]}")
            continue

        try:
            point_id = int(fields[2])
            x, y, z = float(fields[4]), float(fields[5]), float(fields[6])
            ox, oy, oz = float(fields[7]), float(fields[8]), float(fields[9])
            bipolar_v = float(fields[10])
            unipolar_v = float(fields[11])
            lat_raw = float(fields[12])
            lat = float("nan") if lat_raw == _LAT_SENTINEL else lat_raw

            points.append(CartoPoint(
                point_id=point_id,
                position=np.array([x, y, z], dtype=np.float64),
                orientation=np.array([ox, oy, oz], dtype=np.float64),
                bipolar_voltage=bipolar_v,
                unipolar_voltage=unipolar_v,
                lat=lat,
            ))
        except (ValueError, IndexError) as e:
            logger.debug(f"Skipping malformed car line: {e}")
            continue

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


def _parse_vertices_section(
    text: str, expected_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse [VerticesSection]: ID = X Y Z NX NY NZ GroupID.

    Returns (vertices[N,3], normals[N,3], group_ids[N]).
    """
    vertices = np.zeros((expected_count, 3), dtype=np.float64)
    normals = np.zeros((expected_count, 3), dtype=np.float64)
    group_ids = np.zeros(expected_count, dtype=np.int32)

    in_section = False
    count = 0

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[VerticesSection]":
            in_section = True
            continue
        if stripped.startswith("[") and in_section:
            break
        if not in_section:
            continue
        if stripped.startswith(";") or not stripped:
            continue

        # Format: "ID = X Y Z NX NY NZ GroupID"
        parts = stripped.split("=", 1)
        if len(parts) != 2:
            continue
        idx = int(parts[0].strip())
        values = parts[1].split()
        if len(values) < 7:
            continue

        if idx < expected_count:
            vertices[idx] = [float(values[0]), float(values[1]), float(values[2])]
            normals[idx] = [float(values[3]), float(values[4]), float(values[5])]
            group_ids[idx] = int(values[6])
            count += 1

    if count < expected_count:
        # Trim to actual count
        vertices = vertices[:count]
        normals = normals[:count]
        group_ids = group_ids[:count]

    return vertices, normals, group_ids


def _parse_triangles_section(
    text: str, expected_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Parse [TrianglesSection]: ID = V0 V1 V2 NX NY NZ GroupID.

    Returns (faces[M,3], face_group_ids[M]).
    """
    faces = np.zeros((expected_count, 3), dtype=np.int32)
    face_group_ids = np.zeros(expected_count, dtype=np.int32)

    in_section = False
    count = 0

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[TrianglesSection]":
            in_section = True
            continue
        if stripped.startswith("[") and in_section:
            break
        if not in_section:
            continue
        if stripped.startswith(";") or not stripped:
            continue

        parts = stripped.split("=", 1)
        if len(parts) != 2:
            continue
        idx = int(parts[0].strip())
        values = parts[1].split()
        if len(values) < 4:
            continue

        if idx < expected_count:
            faces[idx] = [int(values[0]), int(values[1]), int(values[2])]
            face_group_ids[idx] = int(values[6]) if len(values) >= 7 else 0
            count += 1

    if count < expected_count:
        faces = faces[:count]
        face_group_ids = face_group_ids[:count]

    return faces, face_group_ids
