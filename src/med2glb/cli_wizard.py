"""Data-driven interactive CLI wizard for med2glb.

Analyzes the input directory first, then presents only relevant options
interactively using Rich prompts. Falls back to sensible defaults in
non-TTY environments.
"""

from __future__ import annotations

import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from med2glb.core.types import (
    CARTO_INACTIVE_GROUP_ID,
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


def _mesh_bbox_mm(mesh: CartoMesh) -> str:
    """Return bounding box dimensions of active vertices as 'WxHxD mm'."""
    active = mesh.group_ids != CARTO_INACTIVE_GROUP_ID
    if not np.any(active):
        return ""
    verts = mesh.vertices[active]
    extent = verts.max(axis=0) - verts.min(axis=0)
    return f"{extent[0]:.0f}×{extent[1]:.0f}×{extent[2]:.0f} mm"


def estimate_time_details(
    n_triangles: int,
    n_points: int,
    has_lat: bool,
    subdivide: int = 2,
    n_frames: int = 30,
) -> dict[str, float]:
    """Per-step time estimates in seconds.

    Returns a dict with keys: subdivide, mapping, xatlas, rasterize, bake,
    vectors, total.  Values are wall-clock seconds (float).

    Coefficients calibrated from 24 real runs (median ratios).
    Note: 'bake' reflects a single base-color texture (animated GLBs use
    COLOR_0 morph targets — no per-frame emissive textures).
    """
    post_sub = n_triangles * (4 ** subdivide) if subdivide > 0 else n_triangles

    t_subdivide = post_sub / 100_000 * 1.1 if subdivide > 0 else 0.0
    t_mapping = n_points / 1_000 * 0.4
    # xatlas: power-law model (superlinear), calibrated from observed timings
    t_xatlas = 5.8e-08 * (post_sub ** 1.79) if has_lat else 0.0
    # Rasterization: calibrated from logs (~2.5s per 100k faces)
    t_rasterize = post_sub / 100_000 * 2.5 if has_lat else 0.0
    # Texture bake: single base-color texture (~22s per 100k faces, high variance)
    t_bake = post_sub / 100_000 * 22.0 if has_lat else 0.0
    # Vectors: scales with mesh size (fixed 30s was too high for small meshes)
    t_vectors = max(post_sub / 100_000 * 20.0, 5.0) if has_lat and n_points >= 30 else 0.0

    return {
        "subdivide": t_subdivide,
        "mapping": t_mapping,
        "xatlas": t_xatlas,
        "rasterize": t_rasterize,
        "bake": t_bake,
        "vectors": t_vectors,
        "total": t_subdivide + t_mapping + t_xatlas + t_rasterize + t_bake + t_vectors,
    }


def _format_est(seconds: float) -> str:
    """Format an estimate as a short human-readable string."""
    seconds = max(seconds, 1)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s" if s else f"{m}m"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m}m" if m else f"{h}h"


def estimate_time(
    n_triangles: int,
    n_points: int,
    has_lat: bool,
    subdivide: int = 2,
) -> str:
    """Rough total processing time estimate (formatted string)."""
    total = estimate_time_details(n_triangles, n_points, has_lat, subdivide)["total"]

    if total < 60:
        return "~{:.0f} s".format(max(total, 1))
    if total < 3600:
        m, s = divmod(int(total), 60)
        return "~{}m {}s".format(m, s) if s else "~{}m".format(m)
    h, rem = divmod(int(total), 3600)
    m = rem // 60
    return "~{}h {}m".format(h, m) if m else "~{}h".format(h)


def _auto_subdivide(n_faces: int) -> int:
    """Auto-detect Loop subdivision level from face count.

    Thresholds:
        < 5 000 faces  → level 2 (4× more triangles)
        5 000–20 000   → level 1 (smoother, moderate cost)
        > 20 000       → level 0 (already dense enough)
    """
    if n_faces < 5_000:
        return 2
    if n_faces < 20_000:
        return 1
    return 0


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
    # Try subdirectories first (like CARTO), then fall back to root
    from med2glb.io.dicom_reader import analyze_series as _analyze_series

    dicom_found = False
    # Check immediate subdirectories for DICOM data
    if path.is_dir():
        for subdir in sorted(path.iterdir()):
            if not subdir.is_dir():
                continue
            # Skip directories already claimed as CARTO exports
            if any(e.path == subdir or str(subdir).startswith(str(e.path)) for e in entries):
                continue
            try:
                series = _analyze_series(subdir)
                if series:
                    modalities = sorted({s.modality for s in series})
                    types = sorted({s.data_type for s in series})
                    detail = f"{len(series)} series ({', '.join(modalities)}; {', '.join(types)})"
                    try:
                        location = str(subdir.relative_to(path))
                    except ValueError:
                        location = subdir.name
                    entries.append(ScanEntry(
                        kind="dicom",
                        path=subdir,
                        label=_first_subfolder_label(path, subdir),
                        detail=detail,
                        location=location,
                    ))
                    dicom_found = True
            except Exception:
                pass

    # Fall back: check root directory itself
    if not dicom_found:
        try:
            series = _analyze_series(path)
            if series:
                modalities = sorted({s.modality for s in series})
                types = sorted({s.data_type for s in series})
                detail = f"{len(series)} series ({', '.join(modalities)}; {', '.join(types)})"
                entries.append(ScanEntry(
                    kind="dicom",
                    path=path,
                    label=path.name,
                    detail=detail,
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
    suitable_indices: list[int] | None = None  # mesh indices suitable for vectors


_MIN_VALID_LAT_POINTS = 30
_MIN_LAT_RANGE_MS = 20.0
_MIN_GRADIENT_COVERAGE = 0.15  # at least 15% of faces must have non-zero gradient


def _trial_gradient_coverage(mesh: CartoMesh, pts: list[CartoPoint]) -> tuple[bool, str]:
    """Quick trial: compute IDW LAT + face gradients, check coverage.

    Instead of guessing from global statistics (IQR, density) whether the
    data will produce usable streamlines, this actually computes the IDW-
    interpolated LAT field and face gradients, then checks what fraction
    of faces have non-zero gradient.  ~100 ms per mesh.
    """
    from med2glb.io.carto_mapper import map_points_to_vertices_idw
    from med2glb.mesh.lat_vectors import compute_face_gradients

    lat = map_points_to_vertices_idw(mesh, pts, field="lat", k=6, power=2.0)
    grads, _, _valid = compute_face_gradients(mesh.vertices, mesh.faces, lat)
    mag = np.linalg.norm(grads, axis=1)
    nonzero_frac = float(np.sum(mag > 1e-6)) / max(len(mag), 1)

    if nonzero_frac < _MIN_GRADIENT_COVERAGE:
        return False, f"only {nonzero_frac:.0%} of faces have gradient (need ≥{_MIN_GRADIENT_COVERAGE:.0%})"
    return True, ""


def _assess_single_mesh(
    mesh: CartoMesh,
    pts: list[CartoPoint],
) -> VectorQuality:
    """Assess vector suitability for a single mesh.

    Uses cheap early-exit checks (point count, LAT range) followed by an
    actual gradient-coverage trial that computes IDW interpolation and face
    gradients to see whether enough of the mesh has usable gradient data.
    """
    if not pts:
        return VectorQuality(suitable=False, reason="no measurement points")

    lats = np.array([p.lat for p in pts], dtype=np.float64)
    valid = lats[~np.isnan(lats)]
    n_valid = len(valid)
    lat_range = float(np.ptp(valid)) if n_valid > 0 else 0.0

    if n_valid < _MIN_VALID_LAT_POINTS:
        return VectorQuality(
            suitable=False,
            reason=f"sparse data, {n_valid} valid LAT points",
            valid_points=n_valid, lat_range_ms=lat_range,
        )
    if lat_range < _MIN_LAT_RANGE_MS:
        return VectorQuality(
            suitable=False,
            reason=f"small LAT range, {lat_range:.0f} ms",
            valid_points=n_valid, lat_range_ms=lat_range,
        )

    # Trial-based check: actually compute gradients and check coverage
    ok, reason = _trial_gradient_coverage(mesh, pts)
    if not ok:
        return VectorQuality(
            suitable=False,
            reason=reason,
            valid_points=n_valid, lat_range_ms=lat_range,
        )

    return VectorQuality(
        suitable=True, reason="",
        valid_points=n_valid, lat_range_ms=lat_range,
    )


def _assess_vector_quality(
    study: CartoStudy,
    selected_indices: list[int] | None,
) -> VectorQuality:
    """Check whether the selected meshes have enough LAT data for useful vectors.

    Evaluates each mesh individually. The overall result is suitable if *any*
    mesh passes. ``suitable_indices`` lists which meshes are suitable.

    Uses cheap early-exit checks (point count, LAT range) followed by an
    actual gradient-coverage trial on each mesh to determine suitability.
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
            suitable_indices=suitable_indices,
        )

    return VectorQuality(
        suitable=False,
        reason=best_result.reason,
        valid_points=best_result.valid_points,
        lat_range_ms=best_result.lat_range_ms,
        suitable_indices=[],
    )


def run_carto_wizard(
    study: CartoStudy,
    input_path: Path,
    console: Console,
    # Presets from CLI flags — if set, skip the corresponding prompt
    preset_colorings: list[str] | None = None,
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
    if study.system_version:
        major_minor = ".".join(study.system_version.split(".")[:2])
        vlabel = f"CARTO v{major_minor} (file format v{study.version})"
    else:
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
    table.add_column("After Subdiv", justify="right")
    table.add_column("Active", justify="right")
    table.add_column("Points", justify="right")
    table.add_column("LAT range")
    table.add_column("Density", justify="right")
    table.add_column("Volts")
    table.add_column("Dimensions")
    table.add_column("Est Time", justify="right")

    for i, mesh in enumerate(study.meshes, 1):
        pts = study.points.get(mesh.structure_name, [])
        n_active = int(np.sum(mesh.group_ids != CARTO_INACTIVE_GROUP_ID))
        n_total = len(mesh.vertices)
        n_triangles = len(mesh.faces)

        # LAT range
        lat_range = ""
        valid_lats = [p.lat for p in pts if not math.isnan(p.lat)]
        has_lat = bool(valid_lats)
        if valid_lats:
            lat_range = f"{min(valid_lats):.0f} – {max(valid_lats):.0f} ms"

        # Post-subdivision triangle estimate (default subdivide=2 → ~16x)
        post_sub = n_triangles * 16
        after_sub = f"→ {post_sub:,}"

        # Active vertex percentage
        active_pct = f"{100.0 * n_active / n_total:.0f}%" if n_total > 0 else ""

        # Point density (points / active vertices)
        density = f"{100.0 * len(pts) / n_active:.1f}%" if n_active > 0 else ""

        # Voltage data availability
        has_bipolar = any(not math.isnan(p.bipolar_voltage) for p in pts)
        has_unipolar = any(not math.isnan(p.unipolar_voltage) for p in pts)
        volts = " ".join(
            filter(None, ["B" if has_bipolar else "", "U" if has_unipolar else ""])
        )

        # Bounding box dimensions
        bbox = _mesh_bbox_mm(mesh)

        # Time estimate — use preset subdivide if known, otherwise auto-detect
        _sub = preset_subdivide if preset_subdivide is not None else _auto_subdivide(n_triangles)
        est_time = estimate_time(n_triangles, len(pts), has_lat, _sub)

        table.add_row(
            str(i),
            mesh.structure_name,
            f"{n_active:,}",
            f"{n_triangles:,}",
            after_sub,
            active_pct,
            f"{len(pts):,}",
            lat_range,
            density,
            volts,
            bbox,
            est_time,
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

    # --- Colorings ---
    # All available colorings are produced by default; pipeline skips those
    # with no valid data.  CLI --coloring can restrict to a single scheme.
    colorings = preset_colorings if preset_colorings is not None else ["lat", "bipolar", "unipolar"]

    # --- Output mode ---
    # Always produce both static + animated; respect preset_animate if provided.
    if preset_animate is not None and preset_static is not None:
        animate = preset_animate
        static = preset_static
    else:
        static = True
        animate = True
        if not interactive:
            logger.info("Output mode: static + animated (default)")

    # --- LAT vectors ---
    # Vectors are only produced for LAT coloring; assess quality to decide
    # whether to offer them.  Since LAT is always in the coloring set by
    # default, we always assess (unless LAT was explicitly excluded).
    lat_included = "lat" in colorings
    vec_quality = _assess_vector_quality(study, selected_indices)
    vector_mesh_indices: list[int] | None = None  # None = all selected

    if preset_vectors is not None:
        vectors = preset_vectors
        if vectors in ("yes", "only"):
            vector_mesh_indices = None
    elif interactive and lat_included:
        if vec_quality.suitable:
            default_vec = "yes"
            prompt_label = "LAT conduction vectors"
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
    elif not interactive and lat_included:
        vectors = "yes" if vec_quality.suitable else "no"
        if vectors == "yes" and vec_quality.suitable_indices is not None:
            vector_mesh_indices = vec_quality.suitable_indices
        if vectors == "no":
            logger.info(f"Skipping LAT vectors: {vec_quality.reason}")
    else:
        vectors = "no"

    # --- Subdivision ---
    # Auto-detect from face count; use preset if provided by CLI flag.
    if preset_subdivide is not None:
        subdivide = preset_subdivide
    else:
        sel = selected_indices if selected_indices is not None else list(range(len(study.meshes)))
        max_faces = max((len(study.meshes[i].faces) for i in sel), default=0)
        subdivide = _auto_subdivide(max_faces)
        if not interactive:
            logger.info(f"Auto-detected subdivision: {subdivide} ({max_faces:,} faces)")

    # --- Name ---
    # Auto-generate: <study_or_dir>_sub<N>
    base_name = study.study_name or input_path.name
    # Sanitise for filesystem: replace spaces/slashes with underscores
    base_name = base_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    name = f"{base_name}_sub{subdivide}"

    # --- Summary ---
    console.print(f"\n[dim]Configuration:[/dim]")
    console.print(f"  Name:       {name}")
    sel_str = "all" if selected_indices is None else ", ".join(str(i + 1) for i in selected_indices)
    console.print(f"  Maps:       {sel_str}")
    console.print(f"  Colorings:  {', '.join(colorings)}")
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
        name=name,
        selected_mesh_indices=selected_indices,
        colorings=colorings,
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
    preset_colorings: list[str] | None = None,
    preset_animate: bool | None = None,
    preset_static: bool | None = None,
    preset_vectors: str | None = None,
    preset_subdivide: int | None = None,
) -> list[CartoConfig]:
    """Batch CARTO wizard: ask settings once, apply to all studies.

    Shows a summary table of all detected datasets, prompts for shared
    settings (output mode, vectors, subdivide), then returns one
    ``CartoConfig`` per study with per-study vector quality assessment.
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
        if study.system_version:
            major_minor = ".".join(study.system_version.split(".")[:2])
            vlabel = f"v{major_minor}"
        else:
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

    # --- Colorings (always all available) ---
    colorings = preset_colorings if preset_colorings is not None else ["lat", "bipolar", "unipolar"]

    # --- Output mode (shared) ---
    # Always produce both static + animated; respect preset if provided.
    if preset_animate is not None and preset_static is not None:
        animate = preset_animate
        static = preset_static
    else:
        static = True
        animate = True

    # --- Vectors (shared choice, per-study quality) ---
    lat_included = "lat" in colorings
    if preset_vectors is not None:
        vectors = preset_vectors
    elif interactive and lat_included:
        vectors = Prompt.ask(
            "LAT conduction vectors",
            choices=["yes", "no", "only"],
            default="yes",
            console=console,
        )
    elif not interactive and lat_included:
        vectors = "yes"
    else:
        vectors = "no"

    # --- Subdivision (shared) — auto-detect from max face count across studies ---
    if preset_subdivide is not None:
        subdivide = preset_subdivide
    else:
        all_faces = [len(m.faces) for _, s in studies for m in s.meshes]
        max_faces = max(all_faces, default=0)
        subdivide = _auto_subdivide(max_faces)

    # --- Build per-study configs ---
    configs: list[CartoConfig] = []
    for path, study in studies:
        vector_mesh_indices: list[int] | None = None
        study_vectors = vectors

        if study_vectors in ("yes", "only") and lat_included:
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

        # Auto-generate name: <study_or_dir>_sub<N>
        base_name = study.study_name or path.name
        base_name = base_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        study_name = f"{base_name}_sub{subdivide}"

        configs.append(CartoConfig(
            input_path=path,
            name=study_name,
            selected_mesh_indices=None,  # all meshes
            colorings=colorings,
            subdivide=subdivide,
            animate=animate,
            static=static,
            vectors=study_vectors,
            vector_mesh_indices=vector_mesh_indices,
        ))

    # --- Summary ---
    console.print(f"\n[dim]Batch Configuration:[/dim]")
    console.print(f"  Datasets:   {len(configs)}")
    console.print(f"  Colorings:  {', '.join(colorings)}")
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
    preset_name: str | None = None,
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
    table.add_column("Dimensions")
    table.add_column("Spacing")
    table.add_column("Files", justify="right")
    table.add_column("Detail", justify="right")
    table.add_column("Recommended", style="magenta")
    table.add_column("Est Time", justify="right")

    for i, info in enumerate(series_list, 1):
        table.add_row(
            str(i),
            info.modality,
            info.description or "(no desc)",
            info.data_type,
            info.dimensions or "—",
            info.spacing or "—",
            str(info.file_count),
            info.detail,
            info.recommended_method,
            info.est_time or "—",
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
        n = len(series_list)
        while True:
            choice = Prompt.ask(
                f"Select series (1-{n})",
                default="1",
                console=console,
            )
            try:
                idx = int(choice.strip()) - 1
                if 0 <= idx < n:
                    break
                console.print(f"[red]Please enter a number between 1 and {n}.[/red]")
            except ValueError:
                console.print(f"[red]Please enter a number between 1 and {n}.[/red]")
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
        choices = ["classical", "marching-cubes", "chamber-detect"]
        if ai_available:
            choices.append("totalseg")
        choices.append("compare")
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

    # --- Name ---
    if preset_name is not None:
        name = preset_name
    else:
        # Auto-generate: <modality>_<method>_s<smooth>_<faces>k[_<detail>][_anim]
        # No input dir name — output already lives inside the source folder.
        mod = selected_info.modality.lower() if selected_info.modality else "dcm"
        faces_k = f"{target_faces // 1000}k"
        if method == "compare":
            name = f"{mod}_compare_s{smoothing}_{faces_k}"
        else:
            method_short = method.replace("marching-cubes", "mc")
            name = f"{mod}_{method_short}_s{smoothing}_{faces_k}"

        # Append a series-specific disambiguator so multiple series from the
        # same folder never overwrite each other.
        # Priority 1: sanitized series description (if meaningful).
        # Priority 2: frame count (for cine / 4D data).
        desc = (selected_info.description or "").strip()
        n_frames = getattr(selected_info, "number_of_frames", 0) or 0
        is_temporal = selected_info.data_type in ("2D cine", "3D+T volume")

        if desc and desc.lower() not in ("(no desc)",):
            safe_desc = re.sub(r"[^a-zA-Z0-9]+", "_", desc).strip("_").lower()[:20]
            if safe_desc:
                name += f"_{safe_desc}"
        elif n_frames > 1 and is_temporal:
            name += f"_{n_frames}f"

        if animate:
            name += "_anim"

    # --- Summary ---
    console.print(f"\n[dim]Configuration:[/dim]")
    console.print(f"  Name:     {name}")
    console.print(f"  Series:   {selected_info.description or series_uid}")
    console.print(f"  Method:   {method}")
    console.print(f"  Quality:  smoothing={smoothing}, faces={target_faces:,}")
    console.print(f"  Animate:  {'yes' if animate else 'no'}")

    return DicomConfig(
        input_path=input_path,
        name=name,
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


# ---------------------------------------------------------------------------
# Equivalent command builders
# ---------------------------------------------------------------------------

def _quote_path(p: str | Path) -> str:
    """Quote a path for shell usage, platform-aware."""
    import os
    s = str(p)
    if os.name == "nt":
        # Windows: double-quote if spaces or special chars
        if " " in s or "&" in s or "(" in s or ")" in s:
            return f'"{s}"'
        return s
    else:
        import shlex
        return shlex.quote(s)


def build_carto_equiv_command(
    config: CartoConfig,
    output_path: Path | None = None,
    *,
    batch: bool = False,
) -> str:
    """Build the equivalent CLI command for a CARTO wizard conversion."""
    parts = ["med2glb", _quote_path(config.input_path)]

    if batch:
        parts.append("--batch")

    # Coloring
    if config.colorings:
        for c in config.colorings:
            parts.extend(["--coloring", c])

    # Subdivision
    parts.extend(["--subdivide", str(config.subdivide)])

    # Animation flags
    if config.animate and not config.static:
        parts.append("--animate")
    elif config.static and not config.animate:
        parts.append("--no-animate")

    # Vectors
    if config.vectors in ("yes", "only"):
        parts.append("--vectors")

    # Output
    out = output_path or config.output_dir
    if out is not None:
        parts.extend(["-o", _quote_path(out)])

    return " ".join(parts)


def build_dicom_equiv_command(
    config: DicomConfig,
    output_path: Path | None = None,
) -> str:
    """Build the equivalent CLI command for a DICOM wizard conversion."""
    parts = ["med2glb", _quote_path(config.input_path)]

    # Method
    parts.extend(["--method", config.method])

    # Smoothing (only if non-default)
    if config.smoothing != 15:
        parts.extend(["--smoothing", str(config.smoothing)])

    # Faces (only if non-default)
    if config.target_faces != 80000:
        parts.extend(["--faces", str(config.target_faces)])

    # Threshold
    if config.threshold is not None:
        parts.extend(["--threshold", str(config.threshold)])

    # Alpha (only if non-default)
    if config.alpha != 1.0:
        parts.extend(["--alpha", str(config.alpha)])

    # Series
    if config.series_uid:
        parts.extend(["--series", config.series_uid])

    # Animation
    if config.animate:
        parts.append("--animate")

    # Gallery
    if config.gallery:
        parts.append("--gallery")

    # Output
    out = output_path or config.output
    if out is not None:
        parts.extend(["-o", _quote_path(out)])

    return " ".join(parts)
