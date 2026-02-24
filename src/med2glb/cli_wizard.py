"""Data-driven interactive CLI wizard for med2glb.

Analyzes the input directory first, then presents only relevant options
interactively using Rich prompts. Falls back to sensible defaults in
non-TTY environments.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from med2glb.core.types import (
    CartoConfig,
    CartoMesh,
    CartoPoint,
    CartoStudy,
    DicomConfig,
    SeriesInfo,
)

logger = logging.getLogger("med2glb")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_interactive() -> bool:
    """Return True if stdin and stdout are connected to a TTY."""
    return sys.stdin.isatty() and sys.stdout.isatty()


@dataclass
class DetectedInput:
    """Result of analyzing the input path."""
    kind: str  # "carto" or "dicom"
    # CARTO fields
    carto_study: CartoStudy | None = None
    # DICOM fields
    series_list: list[SeriesInfo] | None = None


def analyze_input(path: Path) -> DetectedInput:
    """Detect whether the input is CARTO or DICOM and load metadata."""
    if path.is_dir():
        from med2glb.io.carto_reader import detect_carto_directory
        if detect_carto_directory(path):
            from med2glb.io.carto_reader import load_carto_study
            study = load_carto_study(path)
            return DetectedInput(kind="carto", carto_study=study)

    # Try DICOM
    try:
        from med2glb.io.dicom_reader import analyze_series
        series = analyze_series(path)
        if series:
            return DetectedInput(kind="dicom", series_list=series)
    except Exception:
        pass

    raise ValueError(f"Could not detect CARTO or DICOM data in: {path}")


# ---------------------------------------------------------------------------
# CARTO wizard
# ---------------------------------------------------------------------------

def _carto_point_stats(points: list[CartoPoint]) -> dict[str, str]:
    """Compute summary statistics for CARTO measurement points."""
    if not points:
        return {"Points": "0"}
    lats = np.array([p.lat for p in points], dtype=np.float64)
    valid = lats[~np.isnan(lats)]
    stats: dict[str, str] = {
        "Points": f"{len(points):,} ({len(valid):,} with valid LAT)",
    }
    if len(valid) > 0:
        stats["LAT range"] = f"{np.min(valid):.0f} – {np.max(valid):.0f} ms"
    return stats


@dataclass
class VectorQuality:
    """Assessment of whether LAT vector data is good enough for streamlines."""
    suitable: bool
    reason: str  # human-readable explanation when not suitable
    valid_points: int = 0
    lat_range_ms: float = 0.0
    lat_iqr_ms: float = 0.0
    point_density: float = 0.0  # points per mm²


_MIN_VALID_LAT_POINTS = 50
_MIN_LAT_RANGE_MS = 30.0
_MIN_LAT_IQR_MS = 50.0
_MIN_POINT_DENSITY = 0.3  # pts/mm² — need dense sampling for meaningful gradients


def _mesh_surface_area(mesh: CartoMesh) -> float:
    """Compute total surface area of a CARTO mesh in mm²."""
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return float(0.5 * np.sum(np.linalg.norm(cross, axis=1)))


def _assess_vector_quality(
    study: CartoStudy,
    selected_indices: list[int] | None,
) -> VectorQuality:
    """Check whether the selected meshes have enough LAT data for useful vectors.

    Evaluates the *best* mesh among those selected — if any mesh is suitable
    the overall assessment is suitable (vectors will only be generated where
    the data supports it).

    Checks four criteria:
    - Enough valid LAT points (≥50)
    - Sufficient total LAT range (≥30 ms)
    - Sufficient LAT spread / IQR (≥50 ms) — a wide range with most values
      clustered together produces a nearly uniform surface with no visible
      gradient, making vectors useless.
    - Sufficient point density (≥0.3 pts/mm²) — sparse sampling produces
      over-smoothed gradients after IDW interpolation, resulting in tiny
      circling streamlines instead of coherent flow arrows.
    """
    indices = selected_indices if selected_indices is not None else list(range(len(study.meshes)))

    best_points = 0
    best_range = 0.0
    best_iqr = 0.0
    best_density = 0.0

    for idx in indices:
        mesh = study.meshes[idx]
        pts = study.points.get(mesh.structure_name, [])
        if not pts:
            continue
        lats = np.array([p.lat for p in pts], dtype=np.float64)
        valid = lats[~np.isnan(lats)]
        n_valid = len(valid)
        lat_range = float(np.ptp(valid)) if n_valid > 0 else 0.0
        lat_iqr = float(np.percentile(valid, 75) - np.percentile(valid, 25)) if n_valid > 0 else 0.0

        area = _mesh_surface_area(mesh)
        density = n_valid / area if area > 1e-6 else 0.0

        if n_valid > best_points:
            best_points = n_valid
        if lat_range > best_range:
            best_range = lat_range
        if lat_iqr > best_iqr:
            best_iqr = lat_iqr
        if density > best_density:
            best_density = density

    if best_points < _MIN_VALID_LAT_POINTS:
        return VectorQuality(
            suitable=False,
            reason=f"sparse data, {best_points} valid LAT points",
            valid_points=best_points,
            lat_range_ms=best_range,
            lat_iqr_ms=best_iqr,
            point_density=best_density,
        )
    if best_range < _MIN_LAT_RANGE_MS:
        return VectorQuality(
            suitable=False,
            reason=f"small LAT range, {best_range:.0f} ms",
            valid_points=best_points,
            lat_range_ms=best_range,
            lat_iqr_ms=best_iqr,
            point_density=best_density,
        )
    if best_iqr < _MIN_LAT_IQR_MS:
        return VectorQuality(
            suitable=False,
            reason=f"low LAT spread (IQR {best_iqr:.0f} ms), uniform activation",
            valid_points=best_points,
            lat_range_ms=best_range,
            lat_iqr_ms=best_iqr,
            point_density=best_density,
        )
    if best_density < _MIN_POINT_DENSITY:
        return VectorQuality(
            suitable=False,
            reason=f"low point density ({best_density:.2f} pts/mm²), gradients too smooth",
            valid_points=best_points,
            lat_range_ms=best_range,
            lat_iqr_ms=best_iqr,
            point_density=best_density,
        )
    return VectorQuality(
        suitable=True,
        reason="",
        valid_points=best_points,
        lat_range_ms=best_range,
        lat_iqr_ms=best_iqr,
        point_density=best_density,
    )


def run_carto_wizard(
    study: CartoStudy,
    input_path: Path,
    console: Console,
    # Presets from CLI flags — if set, skip the corresponding prompt
    preset_coloring: str | None = None,
    preset_animate: bool | None = None,
    preset_static: bool | None = None,
    preset_vectors: str | None = None,
    preset_subdivide: int | None = None,
    preset_meshes: str | None = None,
) -> CartoConfig:
    """Interactive CARTO wizard. Skips prompts when presets are provided."""
    interactive = is_interactive()

    # --- Print study summary ---
    console.print(f"\n[bold cyan]CARTO Study Detected[/bold cyan]")
    version_labels = {
        "4.0": "CARTO 3 (~2015)",
        "5.0": "CARTO 3 v7.1",
        "6.0": "CARTO 3 v7.2+",
    }
    vlabel = version_labels.get(study.version, f"CARTO (v{study.version})")
    console.print(f"  System:  {vlabel}")
    if study.study_name:
        console.print(f"  Study:   {study.study_name}")
    total_pts = sum(len(p) for p in study.points.values())
    console.print(f"  Maps:    {len(study.meshes)}")
    console.print(f"  Points:  {total_pts:,}")

    # --- Mesh table ---
    table = Table(title="Meshes")
    table.add_column("#", style="bold", justify="right")
    table.add_column("Name")
    table.add_column("Vertices", justify="right")
    table.add_column("Triangles", justify="right")
    table.add_column("Points", justify="right")
    table.add_column("LAT range")

    for i, mesh in enumerate(study.meshes, 1):
        pts = study.points.get(mesh.structure_name, [])
        n_active = int(np.sum(mesh.group_ids != -1000000))
        lat_range = ""
        valid_lats = [p.lat for p in pts if not math.isnan(p.lat)]
        if valid_lats:
            lat_range = f"{min(valid_lats):.0f} – {max(valid_lats):.0f} ms"
        table.add_row(
            str(i),
            mesh.structure_name,
            f"{n_active:,}",
            f"{len(mesh.faces):,}",
            f"{len(pts):,}",
            lat_range,
        )

    console.print(table)

    # --- Mesh selection ---
    if preset_meshes is not None:
        selected_indices = _parse_mesh_selection(preset_meshes, len(study.meshes))
    elif interactive and len(study.meshes) > 1:
        choice = Prompt.ask(
            f"Select maps (1-{len(study.meshes)}, comma-separated, or 'all')",
            default="all",
            console=console,
        )
        selected_indices = _parse_mesh_selection(choice, len(study.meshes))
    else:
        selected_indices = None  # all

    # --- Coloring ---
    if preset_coloring is not None:
        coloring = preset_coloring
    elif interactive:
        coloring = Prompt.ask(
            "Coloring",
            choices=["lat", "bipolar", "unipolar"],
            default="lat",
            console=console,
        )
    else:
        coloring = "lat"
        logger.info("Using default coloring: lat")

    # --- Output mode ---
    if preset_animate is not None and preset_static is not None:
        animate = preset_animate
        static = preset_static
    elif interactive:
        mode = Prompt.ask(
            "Output",
            choices=["static", "animated", "both"],
            default="both",
            console=console,
        )
        static = mode in ("static", "both")
        animate = mode in ("animated", "both")
    else:
        static = True
        animate = True
        logger.info("Using default output mode: both (static + animated)")

    # --- LAT vectors ---
    # Assess quality across selected meshes to set a sensible default
    vec_quality = _assess_vector_quality(study, selected_indices)

    if preset_vectors is not None:
        vectors = preset_vectors
    elif interactive and coloring == "lat":
        if vec_quality.suitable:
            default_vec = "yes"
            prompt_label = "LAT conduction vectors"
        else:
            default_vec = "no"
            prompt_label = f"LAT conduction vectors [dim]({vec_quality.reason})[/dim]"
        vec_choice = Prompt.ask(
            prompt_label,
            choices=["yes", "no", "only"],
            default=default_vec,
            console=console,
        )
        vectors = vec_choice
    elif not interactive and coloring == "lat":
        vectors = "yes" if vec_quality.suitable else "no"
        if vectors == "no":
            logger.info(f"Skipping LAT vectors: {vec_quality.reason}")
    else:
        vectors = "no"

    # --- Subdivision ---
    if preset_subdivide is not None:
        subdivide = preset_subdivide
    elif interactive:
        sub_choice = Prompt.ask(
            "Subdivision level (0-3, higher = smoother colors)",
            default="2",
            console=console,
        )
        subdivide = max(0, min(3, int(sub_choice)))
    else:
        subdivide = 2
        logger.info("Using default subdivision: 2")

    # --- Summary ---
    console.print(f"\n[dim]Configuration:[/dim]")
    sel_str = "all" if selected_indices is None else ", ".join(str(i + 1) for i in selected_indices)
    console.print(f"  Maps:       {sel_str}")
    console.print(f"  Coloring:   {coloring}")
    mode_str = ("static + animated" if static and animate
                else "animated" if animate else "static")
    console.print(f"  Output:     {mode_str}")
    console.print(f"  Vectors:    {vectors}")
    console.print(f"  Subdivide:  {subdivide}")

    return CartoConfig(
        input_path=input_path,
        selected_mesh_indices=selected_indices,
        coloring=coloring,
        subdivide=subdivide,
        animate=animate,
        static=static,
        vectors=vectors,
    )


def _parse_mesh_selection(choice: str, n_meshes: int) -> list[int] | None:
    """Parse mesh selection string. Returns None for 'all'."""
    choice = choice.strip().lower()
    if choice == "all":
        return None
    indices = []
    for part in choice.split(","):
        idx = int(part.strip()) - 1  # 1-based to 0-based
        if 0 <= idx < n_meshes:
            indices.append(idx)
    return indices if indices else None


# ---------------------------------------------------------------------------
# DICOM wizard
# ---------------------------------------------------------------------------

def run_dicom_wizard(
    series_list: list[SeriesInfo],
    input_path: Path,
    console: Console,
    # Presets from CLI flags
    preset_method: str | None = None,
    preset_animate: bool | None = None,
    preset_series: str | None = None,
    preset_quality: str | None = None,
) -> DicomConfig:
    """Interactive DICOM wizard. Skips prompts when presets are provided."""
    interactive = is_interactive()

    # --- Print detected data ---
    console.print(f"\n[bold cyan]DICOM Data Detected[/bold cyan]")
    console.print(f"  Series found: {len(series_list)}")

    table = Table(title="DICOM Series")
    table.add_column("#", style="bold", justify="right")
    table.add_column("Modality", style="green")
    table.add_column("Description", max_width=40)
    table.add_column("Type", style="cyan")
    table.add_column("Detail", justify="right")
    table.add_column("Recommended", style="magenta")

    for i, info in enumerate(series_list, 1):
        table.add_row(
            str(i),
            info.modality,
            info.description or "(no desc)",
            info.data_type,
            info.detail,
            info.recommended_method,
        )

    console.print(table)

    # --- Series selection ---
    if preset_series is not None:
        series_uid = preset_series
        selected_info = series_list[0]  # fallback
        for s in series_list:
            if preset_series in s.series_uid:
                selected_info = s
                break
    elif interactive and len(series_list) > 1:
        choice = Prompt.ask(
            f"Select series (1-{len(series_list)})",
            default="1",
            console=console,
        )
        idx = max(0, min(len(series_list) - 1, int(choice.strip()) - 1))
        selected_info = series_list[idx]
        series_uid = selected_info.series_uid
    else:
        selected_info = series_list[0]
        series_uid = selected_info.series_uid
        if len(series_list) > 1:
            logger.info(f"Auto-selected series: {selected_info.description or selected_info.series_uid}")

    # --- Method ---
    # Filter to available methods; recommend based on data
    recommended = selected_info.recommended_method
    if preset_method is not None:
        method = preset_method
    elif interactive:
        # Check which AI methods are available
        ai_available = _check_ai_available()
        choices = ["classical", "marching-cubes"]
        if ai_available:
            choices.extend(["totalseg", "medsam2"])
        rec_label = f" (recommended: {recommended})" if recommended in choices else ""
        method = Prompt.ask(
            f"Method{rec_label}",
            choices=choices,
            default=recommended if recommended in choices else "classical",
            console=console,
        )
    else:
        method = recommended
        logger.info(f"Using recommended method: {method}")

    # --- Quality ---
    quality_presets = {
        "draft": (5, 40000),
        "standard": (15, 80000),
        "high": (25, 150000),
    }
    if preset_quality is not None:
        smoothing, target_faces = quality_presets.get(preset_quality, (15, 80000))
    elif interactive:
        quality = Prompt.ask(
            "Quality",
            choices=["draft", "standard", "high"],
            default="standard",
            console=console,
        )
        smoothing, target_faces = quality_presets[quality]
    else:
        smoothing, target_faces = 15, 80000

    # --- Animate ---
    has_temporal = selected_info.data_type in ("2D cine", "3D+T volume")
    if preset_animate is not None:
        animate = preset_animate
    elif interactive and has_temporal:
        anim_choice = Prompt.ask(
            "Animate (temporal data detected)",
            choices=["yes", "no"],
            default="yes",
            console=console,
        )
        animate = anim_choice == "yes"
    else:
        animate = False

    # --- Summary ---
    console.print(f"\n[dim]Configuration:[/dim]")
    console.print(f"  Series:   {selected_info.description or series_uid}")
    console.print(f"  Method:   {method}")
    console.print(f"  Quality:  smoothing={smoothing}, faces={target_faces:,}")
    console.print(f"  Animate:  {'yes' if animate else 'no'}")

    return DicomConfig(
        input_path=input_path,
        method=method,
        animate=animate,
        smoothing=smoothing,
        target_faces=target_faces,
        series_uid=series_uid,
    )


def _check_ai_available() -> bool:
    """Check if AI segmentation dependencies (torch) are installed."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False
