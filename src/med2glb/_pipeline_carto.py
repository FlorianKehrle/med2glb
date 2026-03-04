"""CARTO pipeline: wizard-config and flag-driven conversion paths."""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from med2glb._console import console, err_console
from med2glb.core.types import CartoPoint

_CARTO_VERSION_LABELS: dict[str, str] = {
    "4.0": "CARTO 3 (~2015)",
    "5.0": "CARTO 3 v7.1",
    "6.0": "CARTO 3 v7.2+",
}


def _carto_version_label(version: str) -> str:
    """Map file-format version to a human-readable CARTO system label."""
    label = _CARTO_VERSION_LABELS.get(version)
    if label:
        return f"{label} (file format v{version})"
    return f"CARTO (file format v{version})"


def _carto_point_stats(
    points: list[CartoPoint],
) -> dict[str, str]:
    """Compute summary statistics for a set of CARTO measurement points."""
    stats: dict[str, str] = {}
    if not points:
        stats["Points"] = "0"
        return stats

    total = len(points)
    lats = np.array([p.lat for p in points], dtype=np.float64)
    bipolars = np.array([p.bipolar_voltage for p in points], dtype=np.float64)
    unipolars = np.array([p.unipolar_voltage for p in points], dtype=np.float64)

    valid_lat = lats[~np.isnan(lats)]
    valid_bip = bipolars[~np.isnan(bipolars)]
    valid_uni = unipolars[~np.isnan(unipolars)]

    stats["Points"] = f"{total:,} ({len(valid_lat):,} with valid LAT)"

    if len(valid_lat) > 0:
        stats["LAT range"] = (
            f"{np.min(valid_lat):.0f} to {np.max(valid_lat):.0f} ms "
            f"(mean {np.mean(valid_lat):.0f} ms)"
        )
    if len(valid_bip) > 0:
        stats["Bipolar V"] = (
            f"{np.min(valid_bip):.2f} – {np.max(valid_bip):.2f} mV "
            f"(mean {np.mean(valid_bip):.2f} mV)"
        )
    if len(valid_uni) > 0:
        stats["Unipolar V"] = (
            f"{np.min(valid_uni):.2f} – {np.max(valid_uni):.2f} mV "
            f"(mean {np.mean(valid_uni):.2f} mV)"
        )

    return stats


def _extract_active_lat(
    mesh: "CartoMesh",
    points: list[CartoPoint],
    mesh_data: "MeshData",
    subdivide: int,
    subdivided_mesh: "CartoMesh | None" = None,
    progress_cb: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Map LAT values from sparse points to mesh_data vertices via KDTree.

    Handles subdivision-dependent interpolation (IDW for subdivided,
    NN + linear for raw) and resamples to the active vertex set in mesh_data.

    Args:
        mesh: Original CARTO mesh (pre-subdivision).
        points: Measurement points.
        mesh_data: The MeshData produced by carto_mesh_to_mesh_data.
        subdivide: Number of Loop-subdivision iterations used.
        subdivided_mesh: Already-subdivided mesh to reuse (skips re-subdivision).
        progress_cb: Optional callback(description) for status updates.
    """
    from med2glb.io.carto_mapper import (
        map_points_to_vertices,
        map_points_to_vertices_idw,
        interpolate_sparse_values,
        subdivide_carto_mesh,
    )

    if subdivided_mesh is not None:
        anim_mesh = subdivided_mesh
    elif subdivide > 0:
        if progress_cb:
            progress_cb("Subdividing mesh for LAT extraction...")
        anim_mesh = subdivide_carto_mesh(mesh, iterations=subdivide)
    else:
        anim_mesh = mesh

    if progress_cb:
        progress_cb("Mapping LAT values...")
    if subdivide > 0:
        lat_values = map_points_to_vertices_idw(anim_mesh, points, field="lat")
    else:
        lat_values = map_points_to_vertices(anim_mesh, points, field="lat")
        lat_values = interpolate_sparse_values(anim_mesh, lat_values)

    # Resample to mesh_data vertices (fill-stripping may differ)
    from scipy.spatial import KDTree
    tree = KDTree(anim_mesh.vertices)
    _, idx = tree.query(mesh_data.vertices)
    return lat_values[idx]


def _print_carto_summary(
    study: "CartoStudy",
    mesh: "CartoMesh",
    mesh_data: "MeshData",
    points: list[CartoPoint] | None,
    coloring: str,
    subdivide: int,
    do_animate: bool,
    do_vectors: bool,
    out_path: Path,
    start_time: float,
) -> None:
    """Print a rich summary table after a CARTO mesh conversion."""
    file_size = out_path.stat().st_size / 1024
    elapsed = time.time() - start_time
    n_total_verts = len(mesh.vertices)

    clamp_info = ""
    if coloring == "bipolar":
        clamp_info = "0.05 – 1.5 mV"
    elif coloring == "unipolar":
        clamp_info = "3.0 – 10.0 mV"
    elif coloring == "lat" and points:
        valid_lats = [p.lat for p in points if not math.isnan(p.lat)]
        if valid_lats:
            clamp_info = f"{min(valid_lats):.0f} – {max(valid_lats):.0f} ms (auto)"

    console.print(f"\n[green]CARTO conversion complete![/green]")
    console.print(f"  System:     {_carto_version_label(study.version)}")
    if study.study_name:
        console.print(f"  Study:      {study.study_name}")
    console.print(f"  Map:        {mesh.structure_name}")
    console.print(f"  Coloring:   {coloring}")
    if subdivide > 0:
        console.print(f"  Subdivide:  level {subdivide} (~{4**subdivide}x face increase)")
    if clamp_info:
        console.print(f"  Color range: {clamp_info}")

    mesh_points = points or []
    point_stats = _carto_point_stats(mesh_points)
    for label, value in point_stats.items():
        console.print(f"  {label + ':':14s}{value}")

    console.print(f"  Vertices:   {len(mesh_data.vertices):,} active / {n_total_verts:,} total")
    console.print(f"  Faces:      {len(mesh_data.faces):,}")
    anim_desc = "No"
    if do_animate and points:
        anim_desc = "Yes (excitation ring)"
        if do_vectors:
            anim_desc += " + LAT vectors"
    console.print(f"  Animated:   {anim_desc}")
    console.print(f"  Output:     {out_path}")
    console.print(f"  Size:       {file_size:.1f} KB")
    console.print(f"  Time:       {elapsed:.1f}s")


def _estimate_carto_faces_for_limit(
    max_bytes: int,
    n_frames: int = 1,
    animated: bool = False,
) -> int:
    """Estimate the maximum face count for a CARTO GLB to stay under max_bytes.

    CARTO GLBs store vertex positions (12B), normals (12B), colors (16B) per
    vertex, plus 12B per face for indices.  Animated GLBs duplicate colors for
    each frame.  A ~10% overhead accounts for glTF JSON, buffer alignment, and
    animation channels.
    """
    overhead = 1.10  # 10% for JSON + headers + animation data
    # verts ≈ faces × 0.5 (shared vertices in a triangle mesh)
    vert_ratio = 0.5
    if animated:
        # Per-vertex: pos(12) + norm(12) + color(16) × n_frames
        bytes_per_vert = 12 + 12 + 16 * n_frames
    else:
        bytes_per_vert = 12 + 12 + 16  # pos + norm + color
    bytes_per_face = 12  # 3 × uint32
    bytes_per_face_total = bytes_per_face + bytes_per_vert * vert_ratio
    usable = max_bytes / overhead
    return max(1000, int(usable / bytes_per_face_total))


def _decimate_with_colors(mesh_data: "MeshData", target_faces: int) -> "MeshData":
    """Decimate a MeshData preserving vertex colors, then recompute normals."""
    from med2glb.mesh.processing import decimate, compute_normals

    result = decimate(mesh_data, target_faces=target_faces)
    return compute_normals(result)



def _build_compressed_carto_variant(
    original_path: Path,
    max_size_mb: int,
    mesh_data: "MeshData",
    is_animated: bool,
    n_frames: int,
    vectors: bool,
    active_lat: "np.ndarray | None",
    extra_meshes: "list[MeshData] | None",
    progress: Progress,
) -> None:
    """Build a _compressed variant of a CARTO GLB if it exceeds the size limit.

    The original full-quality file is kept untouched.  A second file with
    ``_compressed`` in the name is created by rebuilding the GLB with a reduced
    face count that should fit within *max_size_mb*.
    """
    max_bytes = max_size_mb * 1024 * 1024
    if not original_path.exists() or original_path.stat().st_size <= max_bytes:
        return

    original_kb = original_path.stat().st_size / 1024
    target_faces = _estimate_carto_faces_for_limit(
        max_bytes, n_frames=n_frames if is_animated else 1, animated=is_animated,
    )

    # Don't bother if the target is already close to the current face count
    if target_faces >= len(mesh_data.faces):
        return

    compressed_path = original_path.with_name(
        original_path.stem + "_compressed" + original_path.suffix
    )

    task = progress.add_task(
        f"Building compressed variant ({original_kb:.0f} KB > {max_size_mb} MB)...",
        total=None,
    )

    decimated = _decimate_with_colors(mesh_data, target_faces)

    if is_animated and active_lat is not None:
        from med2glb.glb.carto_builder import build_carto_animated_glb
        from scipy.spatial import KDTree
        # Resample LAT values to decimated vertex positions
        tree = KDTree(mesh_data.vertices)
        _, idx = tree.query(decimated.vertices)
        decimated_lat = active_lat[idx]
        written = build_carto_animated_glb(
            decimated, decimated_lat, compressed_path,
            target_faces=target_faces,
            vectors=vectors,
        )
        if not written:
            progress.remove_task(task)
            return
    else:
        from med2glb.glb.builder import build_glb
        build_glb([decimated], compressed_path, extra_meshes=extra_meshes, source_units="mm")

    new_kb = compressed_path.stat().st_size / 1024
    progress.update(
        task,
        description=f"Compressed variant: {new_kb:.0f} KB ({compressed_path.name})",
    )
    progress.remove_task(task)


def _load_carto_study(input_path: Path) -> "CartoStudy":
    """Load a CARTO study with a Rich progress bar."""
    from med2glb.io.carto_reader import load_carto_study, _find_export_dir

    _export_dir = _find_export_dir(input_path)
    _n_mesh_files = len(list(_export_dir.glob("*.mesh")))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading CARTO data...", total=_n_mesh_files)

        def _load_progress(desc: str, current: int, total: int) -> None:
            progress.update(task, description=desc, completed=current, total=total)

        study = load_carto_study(input_path, progress=_load_progress)
        progress.update(
            task,
            description=f"Loaded {_carto_version_label(study.version)}: "
            f"{len(study.meshes)} mesh(es), "
            f"{sum(len(p) for p in study.points.values())} points",
            completed=_n_mesh_files,
        )
        progress.remove_task(task)

    return study


def _convert_carto_meshes(
    config: "CartoConfig", study: "CartoStudy", start_time: float,
) -> None:
    """Core CARTO processing: build and export GLB variants for selected meshes.

    Handles mesh selection, subdivision, coloring, animation, vectors,
    compression, and summary printing for all requested variants.
    """
    from med2glb.io.carto_mapper import carto_mesh_to_mesh_data, subdivide_carto_mesh
    from med2glb.glb.builder import build_glb

    selected = config.selected_mesh_indices
    if selected is None:
        selected = list(range(len(study.meshes)))

    carto_output_dir = config.output_dir if config.output_dir is not None else config.input_path / "glb"
    carto_output_dir.mkdir(parents=True, exist_ok=True)

    # Build list of (mesh_idx, animate_flag, vectors_flag) jobs.
    # vectors="yes": produce BOTH with and without vectors (user can compare).
    # vectors="only": produce ONLY the animated+vectors variant.
    # vectors="no": no vector variants at all.
    # vector_mesh_indices limits which meshes get vectors (None = all).
    has_vectors = config.vectors in ("yes", "only")
    vectors_only = config.vectors == "only"
    vec_meshes = set(config.vector_mesh_indices) if config.vector_mesh_indices is not None else None
    jobs: list[tuple[int, bool, bool]] = []
    if vectors_only:
        # Only animated+vectors — nothing else
        for mesh_idx in selected:
            mesh_has_vec = vec_meshes is None or mesh_idx in vec_meshes
            if mesh_has_vec:
                jobs.append((mesh_idx, True, True))
            else:
                # Mesh not suitable for vectors — fall back to animated without
                jobs.append((mesh_idx, True, False))
    else:
        for mesh_idx in selected:
            mesh_has_vec = has_vectors and (vec_meshes is None or mesh_idx in vec_meshes)
            if config.static:
                jobs.append((mesh_idx, False, False))
            if config.animate:
                jobs.append((mesh_idx, True, False))
                if mesh_has_vec:
                    jobs.append((mesh_idx, True, True))

    # Group jobs by mesh_idx so shared intermediates are computed once per mesh
    grouped: dict[int, list[tuple[bool, bool]]] = {}
    for mesh_idx, do_animate, do_vectors in jobs:
        grouped.setdefault(mesh_idx, []).append((do_animate, do_vectors))

    for mesh_idx, variants in grouped.items():
        mesh = study.meshes[mesh_idx]
        points = study.points.get(mesh.structure_name)

        # === Shared intermediates — computed ONCE per mesh ===
        console.print(
            f"\n[bold]Preparing: {mesh.structure_name}[/bold] "
            f"({len(variants)} variant(s))"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Subdividing mesh...", total=None)

            # 1. Subdivide once
            subdivided = None
            if config.subdivide > 0:
                subdivided = subdivide_carto_mesh(mesh, iterations=config.subdivide)
            progress.update(task, description="Mapping vertices...")

            # 2. Mesh data (colors, active filtering) once
            mesh_data = carto_mesh_to_mesh_data(
                mesh, points, coloring=config.coloring,
                subdivide=config.subdivide, pre_subdivided=subdivided,
            )
            progress.update(
                task,
                description=f"Mapped {len(mesh_data.vertices):,} verts, "
                f"{len(mesh_data.faces):,} faces",
            )

            # 3. LAT values once (if any variant needs animation or vectors)
            needs_lat = any(
                (anim and points) or vec for anim, vec in variants
            )
            active_lat = None
            if needs_lat and points:
                progress.update(task, description="Extracting LAT values...")
                active_lat = _extract_active_lat(
                    mesh, points, mesh_data, config.subdivide,
                    subdivided_mesh=subdivided,
                )

            progress.remove_task(task)

        # === Assemble legend metadata ===
        _UNITS = {"lat": "ms", "bipolar": "mV", "unipolar": "mV"}
        _DEFAULT_RANGES: dict[str, tuple[float, float]] = {
            "bipolar": (0.05, 1.5),
            "unipolar": (3.0, 10.0),
        }
        unit = _UNITS.get(config.coloring, "")
        if config.coloring in _DEFAULT_RANGES:
            clamp_range = _DEFAULT_RANGES[config.coloring]
        elif active_lat is not None:
            clamp_range = (
                float(np.nanmin(active_lat)),
                float(np.nanmax(active_lat)),
            )
        else:
            clamp_range = (0.0, 1.0)

        from datetime import date
        legend_info: dict = {
            "coloring": config.coloring,
            "clamp_range": list(clamp_range),
            "metadata": {
                "study_name": study.study_name or "",
                "carto_version": _carto_version_label(study.version),
                "structure": mesh.structure_name,
                "coloring": config.coloring,
                "clamp_range": list(clamp_range),
                "unit": unit,
                "mapping_points": len(points) if points else 0,
                "export_date": date.today().isoformat(),
            },
        }

        # === Emit each variant from cached state ===
        _n_frames = 30
        for do_animate, do_vectors in variants:
            anim_suffix = "_animated" if (do_animate and points) else ""
            vec_suffix = "_vectors" if do_vectors else ""
            glb_name = f"{mesh.structure_name}_{config.coloring}{anim_suffix}{vec_suffix}.glb"
            out_path = carto_output_dir / glb_name

            console.print(
                f"\n[bold]  Variant: {mesh.structure_name}[/bold] "
                f"({config.coloring} coloring"
                f"{', animated' if do_animate else ', static'}"
                f"{', vectors' if do_vectors else ''})"
            )

            if do_animate and points:
                _total_steps = _n_frames + 1
            else:
                _total_steps = 1

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=20),
                MofNCompleteColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Building GLB...", total=_total_steps)

                if do_animate and points:
                    from med2glb.glb.carto_builder import build_carto_animated_glb

                    def _anim_progress(desc: str, current: int, _total: int) -> None:
                        progress.update(task, description=desc,
                                        completed=current + 1)

                    progress.update(task, description="Building excitation ring animation...")
                    written = build_carto_animated_glb(
                        mesh_data, active_lat, out_path,
                        target_faces=config.target_faces,
                        max_size_mb=config.max_size_mb,
                        vectors=do_vectors,
                        progress=_anim_progress,
                        legend_info=legend_info,
                    )
                    if not written:
                        console.print(
                            f"[yellow]  Skipped {out_path.name}: vectors not viable, "
                            f"non-vector variant already covers this.[/yellow]"
                        )
                        continue
                    progress.update(task, completed=_total_steps)
                else:
                    progress.update(task, description="Building GLB...")
                    build_glb(
                        [mesh_data], out_path,
                        source_units="mm", legend_info=legend_info,
                    )
                    progress.update(task, completed=_total_steps)

                # Produce a _compressed variant if the file exceeds the size limit
                if config.max_size_mb > 0:
                    _build_compressed_carto_variant(
                        out_path, config.max_size_mb, mesh_data,
                        do_animate and bool(points), _n_frames, do_vectors,
                        active_lat if (do_animate and points) else None,
                        None, progress,
                    )

            _print_carto_summary(
                study, mesh, mesh_data, points, config.coloring,
                config.subdivide, do_animate, do_vectors, out_path, start_time,
            )


def run_carto_from_config(config: "CartoConfig") -> None:
    """Execute the CARTO pipeline from a wizard-produced config.

    Loads the CARTO study once and produces all requested outputs (static
    and/or animated) for each selected mesh without re-prompting.
    """
    start_time = time.time()
    study = _load_carto_study(config.input_path)

    if not study.meshes:
        err_console.print("[red]No meshes found in CARTO export.[/red]")
        raise typer.Exit(code=1)

    _convert_carto_meshes(config, study, start_time)


def run_carto_pipeline(
    input_path: Path,
    output: Path,
    coloring: str,
    subdivide: int,
    animate: bool,
    vectors: str,
    max_size_mb: int,
    target_faces: int,
) -> None:
    """Execute the CARTO conversion pipeline."""
    from med2glb.core.types import CartoConfig

    start_time = time.time()
    study = _load_carto_study(input_path)

    if not study.meshes:
        err_console.print("[red]No meshes found in CARTO export.[/red]")
        raise typer.Exit(code=1)

    # Interactive mesh selection for multi-mesh studies
    if len(study.meshes) > 1 and sys.stdin.isatty():
        table = Table(title="CARTO Meshes")
        table.add_column("#", style="bold", justify="right")
        table.add_column("Name")
        table.add_column("Vertices", justify="right")
        table.add_column("Triangles", justify="right")
        table.add_column("Points", justify="right")

        for i, mesh in enumerate(study.meshes, 1):
            pts = study.points.get(mesh.structure_name, [])
            n_active = int(np.sum(mesh.group_ids != -1000000))
            table.add_row(
                str(i),
                mesh.structure_name,
                f"{n_active:,} / {len(mesh.vertices):,}",
                str(len(mesh.faces)),
                str(len(pts)),
            )

        console.print(table)
        choice = Prompt.ask(
            f"Select mesh (1-{len(study.meshes)}) or 'all'",
            default="all",
            console=console,
        )

        if choice.strip().lower() == "all":
            selected = list(range(len(study.meshes)))
        else:
            selected = []
            for part in choice.split(","):
                idx = int(part.strip()) - 1
                if 0 <= idx < len(study.meshes):
                    selected.append(idx)
    else:
        selected = list(range(len(study.meshes)))

    # Per-mesh vector quality assessment (same logic the wizard uses)
    _has_vectors = vectors in ("yes", "only")
    vec_suitable_list: list[int] | None = None
    effective_vectors = vectors
    if _has_vectors and coloring == "lat":
        from med2glb.cli_wizard import _assess_vector_quality
        vec_quality = _assess_vector_quality(study, selected)
        vec_suitable = set(vec_quality.suitable_indices) if vec_quality.suitable_indices else set()
        if not vec_quality.suitable:
            console.print(f"[yellow]Skipping vectors: {vec_quality.reason}[/yellow]")
            effective_vectors = "no"
        else:
            if len(vec_suitable) < len(selected):
                skip_names = [study.meshes[i].structure_name for i in selected if i not in vec_suitable]
                console.print(f"[yellow]Vectors skipped for: {', '.join(skip_names)} (insufficient data)[/yellow]")
            vec_suitable_list = list(vec_suitable)

    config = CartoConfig(
        input_path=input_path,
        output_dir=output.parent,
        selected_mesh_indices=selected,
        coloring=coloring,
        subdivide=subdivide,
        animate=animate,
        static=not animate,
        vectors=effective_vectors,
        vector_mesh_indices=vec_suitable_list,
        target_faces=target_faces,
        max_size_mb=max_size_mb,
    )
    _convert_carto_meshes(config, study, start_time)
