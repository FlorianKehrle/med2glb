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
        progress_cb: Optional callback(description) for status updates.
    """
    from med2glb.io.carto_mapper import (
        map_points_to_vertices,
        map_points_to_vertices_idw,
        interpolate_sparse_values,
        subdivide_carto_mesh,
    )

    anim_mesh = mesh
    if subdivide > 0:
        if progress_cb:
            progress_cb("Subdividing mesh for LAT extraction...")
        anim_mesh = subdivide_carto_mesh(mesh, iterations=subdivide)

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


def _build_static_vectors(
    mesh: "CartoMesh",
    points: list[CartoPoint],
    mesh_data: "MeshData",
    subdivide: int,
) -> list["MeshData"] | None:
    """Build static LAT vector arrow meshes for a single frame.

    Returns a list containing one MeshData (the merged arrow mesh),
    or None if no suitable streamlines were found.
    """
    from med2glb.mesh.lat_vectors import (
        trace_all_streamlines, compute_animated_dashes,
        compute_face_gradients, compute_dash_speed_factors,
    )
    from med2glb.glb.arrow_builder import build_frame_dashes, _auto_scale_params

    # Extract LAT values aligned to mesh_data vertices
    vec_lat_active = _extract_active_lat(mesh, points, mesh_data, subdivide)

    streamlines = trace_all_streamlines(
        mesh_data.vertices, mesh_data.faces, vec_lat_active,
        mesh_data.normals, target_count=300,
    )
    if not streamlines:
        return None

    dashes = compute_animated_dashes(streamlines, n_frames=1)
    if not dashes or not dashes[0]:
        return None

    bbox = mesh_data.vertices.max(axis=0) - mesh_data.vertices.min(axis=0)
    params = _auto_scale_params(float(np.linalg.norm(bbox)))
    max_r = params.max_radius if params.max_radius is not None else params.head_radius

    face_grads, face_centers, _ = compute_face_gradients(
        mesh_data.vertices, mesh_data.faces, vec_lat_active,
    )
    speed_factors = compute_dash_speed_factors(dashes, face_grads, face_centers)
    sf = speed_factors[0] if speed_factors and speed_factors[0] else None

    if sf is not None:
        keep = [s >= 0.15 for s in sf]
        frame_dashes = [d for d, k in zip(dashes[0], keep) if k]
        sf = [s for s, k in zip(sf, keep) if k]
        dash_radii: list[float] | None = [max_r * (1.1 - 0.3 * s) for s in sf]
    else:
        frame_dashes = dashes[0]
        dash_radii = None

    arrow_mesh = build_frame_dashes(
        frame_dashes, mesh_data.vertices, mesh_data.normals, params,
        dash_radii=dash_radii,
    )
    return [arrow_mesh] if arrow_mesh is not None else None


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
    elif do_vectors and points:
        anim_desc = "No (static LAT vectors)"
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


def _build_ar_variant(
    standard_path: Path,
    mesh_data: "MeshData",
    is_animated: bool,
    active_lat: "np.ndarray | None",
    extra_meshes: "list[MeshData] | None",
    target_faces: int = 80000,
    max_size_mb: int = 99,
    vectors: bool = False,
    progress_cb: "Callable[[str, int, int], None] | None" = None,
) -> Path:
    """Re-export the same mesh as an AR-optimized (unlit) GLB with ``_AR`` suffix.

    Flips ``material.unlit`` to True, writes the AR variant, then restores
    the original material state so the caller's mesh_data is unchanged.
    """
    ar_path = standard_path.with_name(
        standard_path.stem + "_AR" + standard_path.suffix
    )

    # Temporarily enable unlit
    mesh_data.material.unlit = True
    if extra_meshes:
        for em in extra_meshes:
            em.material.unlit = True

    try:
        if is_animated and active_lat is not None:
            from med2glb.glb.carto_builder import build_carto_animated_glb
            build_carto_animated_glb(
                mesh_data, active_lat, ar_path,
                target_faces=target_faces,
                max_size_mb=max_size_mb,
                vectors=vectors,
                progress=progress_cb,
            )
        else:
            from med2glb.glb.builder import build_glb
            build_glb([mesh_data], ar_path, extra_meshes=extra_meshes, source_units="mm")
    finally:
        # Restore standard (lit) material
        mesh_data.material.unlit = False
        if extra_meshes:
            for em in extra_meshes:
                em.material.unlit = False

    return ar_path


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
        build_carto_animated_glb(
            decimated, decimated_lat, compressed_path,
            target_faces=target_faces,
            vectors=vectors,
        )
    else:
        from med2glb.glb.builder import build_glb
        build_glb([decimated], compressed_path, extra_meshes=extra_meshes, source_units="mm")

    new_kb = compressed_path.stat().st_size / 1024
    progress.update(
        task,
        description=f"Compressed variant: {new_kb:.0f} KB ({compressed_path.name})",
    )
    progress.remove_task(task)


def run_carto_from_config(config: "CartoConfig") -> None:
    """Execute the CARTO pipeline from a wizard-produced config.

    Loads the CARTO study once and produces all requested outputs (static
    and/or animated) for each selected mesh without re-prompting.
    """
    from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
    from med2glb.io.carto_reader import load_carto_study, _find_export_dir
    from med2glb.glb.builder import build_glb

    start_time = time.time()

    # Pre-count mesh files for progress
    _export_dir = _find_export_dir(config.input_path)
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

        study = load_carto_study(config.input_path, progress=_load_progress)
        progress.update(
            task,
            description=f"Loaded {_carto_version_label(study.version)}: "
            f"{len(study.meshes)} mesh(es), "
            f"{sum(len(p) for p in study.points.values())} points",
            completed=_n_mesh_files,
        )
        progress.remove_task(task)

    if not study.meshes:
        err_console.print("[red]No meshes found in CARTO export.[/red]")
        raise typer.Exit(code=1)

    # Use wizard's mesh selection directly — no re-prompting
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
                if mesh_has_vec:
                    jobs.append((mesh_idx, False, True))
            if config.animate:
                jobs.append((mesh_idx, True, False))
                if mesh_has_vec:
                    jobs.append((mesh_idx, True, True))

    for mesh_idx, do_animate, do_vectors in jobs:
        mesh = study.meshes[mesh_idx]
        points = study.points.get(mesh.structure_name)

        anim_suffix = "_animated" if (do_animate and points) else ""
        vec_suffix = "_vectors" if do_vectors else ""
        glb_name = f"{mesh.structure_name}_{config.coloring}{anim_suffix}{vec_suffix}.glb"
        out_path = carto_output_dir / glb_name

        console.print(
            f"\n[bold]Converting: {mesh.structure_name}[/bold] "
            f"({config.coloring} coloring"
            f"{', animated' if do_animate else ', static'}"
            f"{', vectors' if do_vectors else ''})"
        )

        _n_frames = 30
        if do_animate and points:
            _total_steps = 3 + _n_frames + 1
        elif do_vectors and points:
            _total_steps = 3
        else:
            _total_steps = 2

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Subdividing & mapping vertices...", total=_total_steps)
            mesh_data = carto_mesh_to_mesh_data(
                mesh, points, coloring=config.coloring, subdivide=config.subdivide,
            )
            progress.update(task, advance=1,
                description=f"Mapped {len(mesh_data.vertices):,} verts, "
                f"{len(mesh_data.faces):,} faces")

            active_lat = None
            extra = None

            if do_animate and points:
                from med2glb.glb.carto_builder import build_carto_animated_glb

                def _lat_progress(desc: str) -> None:
                    progress.update(task, description=desc, advance=1)

                active_lat = _extract_active_lat(
                    mesh, points, mesh_data, config.subdivide,
                    progress_cb=_lat_progress,
                )

                def _anim_progress(desc: str, current: int, _total: int) -> None:
                    progress.update(task, description=desc,
                                    completed=3 + current + 1)

                progress.update(task, description="Building excitation ring animation...")
                build_carto_animated_glb(
                    mesh_data, active_lat, out_path,
                    target_faces=config.target_faces,
                    max_size_mb=config.max_size_mb,
                    vectors=do_vectors,
                    progress=_anim_progress,
                )
                progress.update(task, completed=_total_steps)
            else:
                if do_vectors and points:
                    progress.update(task, description="Generating static LAT vectors...")
                    extra = _build_static_vectors(
                        mesh, points, mesh_data, config.subdivide,
                    )

                progress.update(task, description="Building GLB...")
                build_glb([mesh_data], out_path, extra_meshes=extra, source_units="mm")
                progress.update(task, completed=_total_steps)

            # Produce a _compressed variant if the file exceeds the size limit
            if config.max_size_mb > 0:
                _build_compressed_carto_variant(
                    out_path, config.max_size_mb, mesh_data,
                    do_animate and bool(points), _n_frames, do_vectors,
                    active_lat if (do_animate and points) else None,
                    extra, progress,
                )

            # Produce AR-optimized (unlit) variant
            progress.add_task("Building AR variant (unlit)...", total=None)
            ar_path = _build_ar_variant(
                out_path, mesh_data,
                is_animated=do_animate and bool(points),
                active_lat=active_lat if (do_animate and points) else None,
                extra_meshes=extra,
                target_faces=config.target_faces,
                max_size_mb=config.max_size_mb,
                vectors=do_vectors,
            )
            console.print(f"  [dim]AR variant: {ar_path.name}[/dim]")

        _print_carto_summary(
            study, mesh, mesh_data, points, config.coloring,
            config.subdivide, do_animate, do_vectors, out_path, start_time,
        )


def run_carto_pipeline(
    input_path: Path,
    output: Path,
    coloring: str,
    subdivide: int,
    animate: bool,
    vectors: str,
    max_size_mb: int,
    compress_strategy: str,
    target_faces: int,
    verbose: bool,
) -> None:
    """Execute the CARTO conversion pipeline."""
    from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
    from med2glb.io.carto_reader import load_carto_study
    from med2glb.glb.builder import build_glb

    start_time = time.time()

    # Pre-count mesh files so progress bar has a total from the start
    from med2glb.io.carto_reader import _find_export_dir
    _export_dir = _find_export_dir(input_path)
    _n_mesh_files = len(list(_export_dir.glob("*.mesh")))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        # Step 1: Load CARTO study
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
    vec_suitable: set[int] | None = None
    if _has_vectors and coloring == "lat":
        from med2glb.cli_wizard import _assess_vector_quality
        vec_quality = _assess_vector_quality(study, selected)
        vec_suitable = set(vec_quality.suitable_indices) if vec_quality.suitable_indices else set()
        if not vec_quality.suitable:
            console.print(f"[yellow]Skipping vectors: {vec_quality.reason}[/yellow]")
            _has_vectors = False
        elif len(vec_suitable) < len(selected):
            skip_names = [study.meshes[i].structure_name for i in selected if i not in vec_suitable]
            console.print(f"[yellow]Vectors skipped for: {', '.join(skip_names)} (insufficient data)[/yellow]")

    # Use output's parent directory for per-mesh files
    carto_output_dir = output.parent
    carto_output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each selected mesh
    for mesh_idx in selected:
        mesh = study.meshes[mesh_idx]
        points = study.points.get(mesh.structure_name)

        # Gate vectors per mesh based on quality assessment
        mesh_has_vec = _has_vectors and (vec_suitable is None or mesh_idx in vec_suitable)

        # Build descriptive filename: <structure>_<coloring>[_animated][_vectors].glb
        anim_suffix = "_animated" if (animate and points) else ""
        vec_suffix = "_vectors" if mesh_has_vec else ""
        glb_name = f"{mesh.structure_name}_{coloring}{anim_suffix}{vec_suffix}.glb"
        out_path = carto_output_dir / glb_name

        console.print(
            f"\n[bold]Converting: {mesh.structure_name}[/bold] "
            f"({coloring} coloring)"
        )

        # Determine total steps upfront so progress bar never shows "?"
        _n_frames = 30  # must match build_carto_animated_glb default
        if animate and points:
            # Steps: map vertices + subdivide LAT + map LAT + N frames + assemble
            _total_steps = 3 + _n_frames + 1
        elif mesh_has_vec and points:
            # Steps: map vertices + vectors + build GLB
            _total_steps = 3
        else:
            # Steps: map vertices + build GLB
            _total_steps = 2

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Subdividing & mapping vertices...", total=_total_steps)
            mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring=coloring, subdivide=subdivide)
            progress.update(task, advance=1,
                description=f"Mapped {len(mesh_data.vertices):,} verts, "
                f"{len(mesh_data.faces):,} faces")

            active_lat = None
            extra = None

            if animate and points:
                from med2glb.glb.carto_builder import build_carto_animated_glb

                def _lat_progress(desc: str) -> None:
                    progress.update(task, description=desc, advance=1)

                active_lat = _extract_active_lat(
                    mesh, points, mesh_data, subdivide,
                    progress_cb=_lat_progress,
                )

                def _anim_progress(desc: str, current: int, _total: int) -> None:
                    progress.update(task, description=desc,
                                    completed=3 + current + 1)

                progress.update(task, description="Building excitation ring animation...")
                build_carto_animated_glb(
                    mesh_data, active_lat, out_path,
                    target_faces=target_faces,
                    max_size_mb=max_size_mb,
                    vectors=mesh_has_vec,
                    progress=_anim_progress,
                )
                progress.update(task, completed=_total_steps)
            else:
                if mesh_has_vec and points:
                    progress.update(task, description="Generating static LAT vectors...")
                    extra = _build_static_vectors(
                        mesh, points, mesh_data, subdivide,
                    )

                progress.update(task, description="Building GLB...")
                build_glb([mesh_data], out_path, extra_meshes=extra, source_units="mm")
                progress.update(task, completed=_total_steps)

            # Produce a _compressed variant if the file exceeds the size limit
            if max_size_mb > 0:
                _build_compressed_carto_variant(
                    out_path, max_size_mb, mesh_data,
                    animate and bool(points), _n_frames, mesh_has_vec,
                    active_lat if (animate and points) else None,
                    extra, progress,
                )

            # Produce AR-optimized (unlit) variant
            progress.add_task("Building AR variant (unlit)...", total=None)
            ar_path = _build_ar_variant(
                out_path, mesh_data,
                is_animated=animate and bool(points),
                active_lat=active_lat if (animate and points) else None,
                extra_meshes=extra,
                target_faces=target_faces,
                max_size_mb=max_size_mb,
                vectors=mesh_has_vec,
            )
            console.print(f"  [dim]AR variant: {ar_path.name}[/dim]")

        _print_carto_summary(
            study, mesh, mesh_data, points, coloring,
            subdivide, animate, mesh_has_vec, out_path, start_time,
        )
