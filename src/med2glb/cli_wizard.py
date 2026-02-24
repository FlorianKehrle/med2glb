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


@dataclass
class ScanEntry:
    """One convertible dataset discovered during directory scan."""
    kind: str  # "carto" or "dicom"
    path: Path
    label: str  # Human-readable short name (first subfolder under root)
    detail: str  # Summary info (meshes, series, etc.)
    location: str  # Full relative path from scan root


def _first_subfolder_label(root: Path, target: Path) -> str:
    """Return the first subfolder name under *root* on the way to *target*.

    For ``root/A/B/C/Export/``, returns ``"A"``.  If *target* is *root*
    itself, returns ``target.name``.
    """
    try:
        rel = target.relative_to(root)
    except ValueError:
        return target.name
    parts = rel.parts
    if not parts:
        return target.name
    return parts[0]


def scan_directory(path: Path, console: Console | None = None) -> list[ScanEntry]:
    """Scan a directory for all convertible data (CARTO + DICOM).

    Returns a list of :class:`ScanEntry` items, one per dataset found.
    """
    entries: list[ScanEntry] = []

    if not path.is_dir():
        return entries

    # --- CARTO exports ---
    from med2glb.io.carto_reader import find_carto_subdirectories
    carto_dirs = find_carto_subdirectories(path)
    for d in carto_dirs:
        n_mesh = len(list(d.glob("*.mesh")))
        n_car = len(list(d.glob("*_car.txt")))
        try:
            location = str(d.relative_to(path))
        except ValueError:
            location = d.name
        entries.append(ScanEntry(
            kind="carto",
            path=d,
            label=_first_subfolder_label(path, d),
            detail=f"{n_mesh} mesh(es), {n_car} car file(s)",
            location=location,
        ))

    # --- DICOM data ---
    try:
        from med2glb.io.dicom_reader import analyze_series
        series = analyze_series(path)
        if series:
            entries.append(ScanEntry(
                kind="dicom",
                path=path,
                label=path.name,
                detail=f"{len(series)} series",
                location=".",
            ))
    except Exception:
        pass

    return entries


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

@dataclass
class VectorQuality:
    """Assessment of whether LAT vector data is good enough for streamlines."""
    suitable: bool
    reason: str  # human-readable explanation when not suitable
    valid_points: int = 0
    lat_range_ms: float = 0.0
    lat_iqr_ms: float = 0.0
    point_density: float = 0.0  # points per mm²
    suitable_indices: list[int] | None = None  # mesh indices suitable for vectors


_MIN_VALID_LAT_POINTS = 30
_MIN_LAT_RANGE_MS = 20.0
_MIN_LAT_IQR_MS = 10.0
_MIN_POINT_DENSITY = 0.005  # pts/mm² — subdivision + IDW compensate for sparse sampling


def _mesh_surface_area(mesh: CartoMesh) -> float:
    """Compute total surface area of a CARTO mesh in mm²."""
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return float(0.5 * np.sum(np.linalg.norm(cross, axis=1)))


def _assess_single_mesh(
    mesh: CartoMesh,
    pts: list[CartoPoint],
) -> VectorQuality:
    """Assess vector suitability for a single mesh."""
    if not pts:
        return VectorQuality(suitable=False, reason="no measurement points")

    lats = np.array([p.lat for p in pts], dtype=np.float64)
    valid = lats[~np.isnan(lats)]
    n_valid = len(valid)
    lat_range = float(np.ptp(valid)) if n_valid > 0 else 0.0
    lat_iqr = float(np.percentile(valid, 75) - np.percentile(valid, 25)) if n_valid > 0 else 0.0
    area = _mesh_surface_area(mesh)
    density = n_valid / area if area > 1e-6 else 0.0

    if n_valid < _MIN_VALID_LAT_POINTS:
        return VectorQuality(
            suitable=False,
            reason=f"sparse data, {n_valid} valid LAT points",
            valid_points=n_valid, lat_range_ms=lat_range,
            lat_iqr_ms=lat_iqr, point_density=density,
        )
    if lat_range < _MIN_LAT_RANGE_MS:
        return VectorQuality(
            suitable=False,
            reason=f"small LAT range, {lat_range:.0f} ms",
            valid_points=n_valid, lat_range_ms=lat_range,
            lat_iqr_ms=lat_iqr, point_density=density,
        )
    if lat_iqr < _MIN_LAT_IQR_MS:
        return VectorQuality(
            suitable=False,
            reason=f"low LAT spread (IQR {lat_iqr:.0f} ms), uniform activation",
            valid_points=n_valid, lat_range_ms=lat_range,
            lat_iqr_ms=lat_iqr, point_density=density,
        )
    if density < _MIN_POINT_DENSITY:
        return VectorQuality(
            suitable=False,
            reason=f"low point density ({density:.2f} pts/mm²), gradients too smooth",
            valid_points=n_valid, lat_range_ms=lat_range,
            lat_iqr_ms=lat_iqr, point_density=density,
        )
    return VectorQuality(
        suitable=True, reason="",
        valid_points=n_valid, lat_range_ms=lat_range,
        lat_iqr_ms=lat_iqr, point_density=density,
    )


def _assess_vector_quality(
    study: CartoStudy,
    selected_indices: list[int] | None,
) -> VectorQuality:
    """Check whether the selected meshes have enough LAT data for useful vectors.

    Evaluates each mesh individually. The overall result is suitable if *any*
    mesh passes. ``suitable_indices`` lists which meshes are suitable.

    Checks four criteria per mesh:
    - Enough valid LAT points (≥50)
    - Sufficient total LAT range (≥30 ms)
    - Sufficient LAT spread / IQR (≥50 ms)
    - Sufficient point density (≥0.3 pts/mm²)
    """
    indices = selected_indices if selected_indices is not None else list(range(len(study.meshes)))

    suitable_indices: list[int] = []
    best_result: VectorQuality | None = None

    for idx in indices:
        mesh = study.meshes[idx]
        pts = study.points.get(mesh.structure_name, [])
        result = _assess_single_mesh(mesh, pts)
        if result.suitable:
            suitable_indices.append(idx)
        # Track the best result for summary (prefer a suitable one)
        if best_result is None or (result.suitable and not best_result.suitable):
            best_result = result
        elif not best_result.suitable and not result.suitable:
            # Among unsuitable, keep the one with more points (more informative reason)
            if result.valid_points > best_result.valid_points:
                best_result = result

    if best_result is None:
        return VectorQuality(suitable=False, reason="no meshes selected")

    # Override overall suitability based on per-mesh results
    if suitable_indices:
        return VectorQuality(
            suitable=True,
            reason="",
            valid_points=best_result.valid_points,
            lat_range_ms=best_result.lat_range_ms,
            lat_iqr_ms=best_result.lat_iqr_ms,
            point_density=best_result.point_density,
            suitable_indices=suitable_indices,
        )

    return VectorQuality(
        suitable=False,
        reason=best_result.reason,
        valid_points=best_result.valid_points,
        lat_range_ms=best_result.lat_range_ms,
        lat_iqr_ms=best_result.lat_iqr_ms,
        point_density=best_result.point_density,
        suitable_indices=[],
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
    # Assess quality per mesh to determine which ones are suitable
    vec_quality = _assess_vector_quality(study, selected_indices)
    vector_mesh_indices: list[int] | None = None  # None = all selected

    if preset_vectors is not None:
        vectors = preset_vectors
        # When user explicitly requests vectors, apply to all selected meshes
        if vectors in ("yes", "only"):
            vector_mesh_indices = None
    elif interactive and coloring == "lat":
        if vec_quality.suitable:
            default_vec = "yes"
            prompt_label = "LAT conduction vectors"
            # If only some meshes are suitable, note which ones
            all_indices = selected_indices if selected_indices is not None else list(range(len(study.meshes)))
            if vec_quality.suitable_indices and len(vec_quality.suitable_indices) < len(all_indices):
                names = [study.meshes[i].structure_name for i in vec_quality.suitable_indices]
                prompt_label += f" [dim](suitable: {', '.join(names)})[/dim]"
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
        if vectors in ("yes", "only") and vec_quality.suitable_indices is not None:
            vector_mesh_indices = vec_quality.suitable_indices
    elif not interactive and coloring == "lat":
        vectors = "yes" if vec_quality.suitable else "no"
        if vectors == "yes" and vec_quality.suitable_indices is not None:
            vector_mesh_indices = vec_quality.suitable_indices
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
    if vectors in ("yes", "only") and vector_mesh_indices is not None:
        vec_names = [study.meshes[i].structure_name for i in vector_mesh_indices]
        console.print(f"  Vectors:    {vectors} ({', '.join(vec_names)})")
    else:
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
        vector_mesh_indices=vector_mesh_indices,
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
# Batch CARTO wizard
# ---------------------------------------------------------------------------

def run_batch_carto_wizard(
    studies: list[tuple[Path, "CartoStudy"]],
    console: Console,
    # Presets from CLI flags — if set, skip the corresponding prompt
    preset_coloring: str | None = None,
    preset_animate: bool | None = None,
    preset_static: bool | None = None,
    preset_vectors: str | None = None,
    preset_subdivide: int | None = None,
) -> list[CartoConfig]:
    """Batch CARTO wizard: ask settings once, apply to all studies.

    Shows a summary table of all detected datasets, prompts for shared
    settings (coloring, output mode, vectors, subdivide), then returns
    one ``CartoConfig`` per study with per-study vector quality assessment.
    """
    interactive = is_interactive()

    # --- Summary table of all datasets ---
    version_labels = {
        "4.0": "CARTO 3 (~2015)",
        "5.0": "CARTO 3 v7.1",
        "6.0": "CARTO 3 v7.2+",
    }
    table = Table(title="Datasets Overview")
    table.add_column("#", style="bold", justify="right")
    table.add_column("Dataset")
    table.add_column("Version")
    table.add_column("Maps")
    table.add_column("Mesh names")
    table.add_column("Points", justify="right")
    table.add_column("LAT data")

    for i, (path, study) in enumerate(studies, 1):
        total_pts = sum(len(p) for p in study.points.values())
        mesh_names = ", ".join(m.structure_name for m in study.meshes)
        vlabel = version_labels.get(study.version, f"v{study.version}")

        # Summarise LAT availability per mesh
        lat_parts = []
        for m in study.meshes:
            pts = study.points.get(m.structure_name, [])
            valid_lats = [p.lat for p in pts if not math.isnan(p.lat)]
            if valid_lats:
                lat_parts.append(f"{m.structure_name}: {min(valid_lats):.0f}..{max(valid_lats):.0f} ms")
            else:
                lat_parts.append(f"{m.structure_name}: [dim]none[/dim]")

        table.add_row(
            str(i),
            path.name,
            vlabel,
            str(len(study.meshes)),
            mesh_names,
            f"{total_pts:,}",
            "\n".join(lat_parts),
        )

    console.print(table)

    # --- Coloring (shared) ---
    if preset_coloring is not None:
        coloring = preset_coloring
    elif interactive:
        coloring = Prompt.ask(
            "Coloring (applied to all datasets)",
            choices=["lat", "bipolar", "unipolar"],
            default="lat",
            console=console,
        )
    else:
        coloring = "lat"

    # --- Output mode (shared) ---
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

    # --- Vectors (shared choice, per-study quality) ---
    if preset_vectors is not None:
        vectors = preset_vectors
    elif interactive and coloring == "lat":
        vectors = Prompt.ask(
            "LAT conduction vectors",
            choices=["yes", "no", "only"],
            default="yes",
            console=console,
        )
    elif not interactive and coloring == "lat":
        vectors = "yes"
    else:
        vectors = "no"

    # --- Subdivision (shared) ---
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

    # --- Build per-study configs ---
    configs: list[CartoConfig] = []
    for path, study in studies:
        vector_mesh_indices: list[int] | None = None
        study_vectors = vectors

        if study_vectors in ("yes", "only") and coloring == "lat":
            vec_quality = _assess_vector_quality(study, None)
            if not vec_quality.suitable:
                console.print(
                    f"[yellow]{path.name}: skipping vectors ({vec_quality.reason})[/yellow]"
                )
                study_vectors = "no"
            elif vec_quality.suitable_indices is not None:
                vector_mesh_indices = vec_quality.suitable_indices
                if len(vector_mesh_indices) < len(study.meshes):
                    skip_names = [
                        study.meshes[i].structure_name
                        for i in range(len(study.meshes))
                        if i not in set(vector_mesh_indices)
                    ]
                    console.print(
                        f"[yellow]{path.name}: vectors skipped for "
                        f"{', '.join(skip_names)} (insufficient data)[/yellow]"
                    )

        configs.append(CartoConfig(
            input_path=path,
            selected_mesh_indices=None,  # all meshes
            coloring=coloring,
            subdivide=subdivide,
            animate=animate,
            static=static,
            vectors=study_vectors,
            vector_mesh_indices=vector_mesh_indices,
        ))

    # --- Summary ---
    console.print(f"\n[dim]Batch Configuration:[/dim]")
    console.print(f"  Datasets:   {len(configs)}")
    console.print(f"  Coloring:   {coloring}")
    mode_str = ("static + animated" if static and animate
                else "animated" if animate else "static")
    console.print(f"  Output:     {mode_str}")
    console.print(f"  Vectors:    {vectors}")
    console.print(f"  Subdivide:  {subdivide}")

    return configs


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
