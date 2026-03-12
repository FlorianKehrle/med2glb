"""CARTO pipeline: wizard-config and flag-driven conversion paths."""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.status import Status
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

    Prefers pre-computed vertex values from the mesh file's
    [VerticesColorsSection] when available (matches CARTO display exactly).
    Falls back to IDW/NN interpolation from car-file points.

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

    # Prefer CARTO's own pre-computed vertex LAT values
    mesh_lat = anim_mesh.vertex_color_values.get("lat")
    if mesh_lat is not None and len(mesh_lat) == len(anim_mesh.vertices):
        lat_values = mesh_lat.copy()
    else:
        # Fallback: interpolate from sparse car-file points
        if progress_cb:
            progress_cb("Mapping LAT values...")
        if subdivide > 0:
            lat_values = map_points_to_vertices_idw(anim_mesh, points, field="lat")
        else:
            lat_values = map_points_to_vertices(anim_mesh, points, field="lat")
            lat_values = interpolate_sparse_values(anim_mesh, lat_values)

        # Distance cutoff for sparse LAT data (same logic as carto_mapper)
        from med2glb.io.carto_mapper import extract_point_field
        point_positions, point_values = extract_point_field(points, "lat")
        valid_ratio = len(point_values) / len(points) if points else 1.0
        if valid_ratio < 0.4 and len(point_positions) >= 2:
            from scipy.spatial import KDTree as _KDTree
            dist_tree = _KDTree(point_positions)
            distances, _ = dist_tree.query(anim_mesh.vertices)
            cutoff_pct = min(80, max(20, valid_ratio * 150))
            max_distance = float(np.percentile(distances, cutoff_pct))
            lat_values[distances > max_distance] = np.nan

    # Resample to mesh_data vertices (fill-stripping may differ)
    from scipy.spatial import KDTree
    tree = KDTree(anim_mesh.vertices)
    _, idx = tree.query(mesh_data.vertices)
    return lat_values[idx]


from med2glb._utils import fmt_duration as _format_duration


def _print_carto_summary(
    study: "CartoStudy",
    mesh: "CartoMesh",
    mesh_data: "MeshData",
    points: list[CartoPoint] | None,
    colorings: list[str],
    subdivide: int,
    all_outputs: list[tuple[str, bool, bool, Path]],
    start_time: float,
    step_times: dict[str, float] | None = None,
) -> None:
    """Print a single summary after all colorings of a CARTO mesh are built.

    Args:
        colorings: List of coloring schemes that were produced.
        all_outputs: List of (coloring, do_animate, do_vectors, out_path) for
            each successfully written variant across all colorings.
    """
    elapsed = time.time() - start_time
    n_total_verts = len(mesh.vertices)

    console.print(f"\n[green]Done: {mesh.structure_name}[/green]")
    console.print(f"  System:     {_carto_version_label(study.version)}")
    if study.study_name:
        console.print(f"  Study:      {study.study_name}")
    console.print(f"  Colorings:  {', '.join(colorings)}")
    if subdivide > 0:
        console.print(f"  Subdivide:  level {subdivide} (~{4**subdivide}x face increase)")

    mesh_points = points or []
    point_stats = _carto_point_stats(mesh_points)
    for label, value in point_stats.items():
        console.print(f"  {label + ':':14s}{value}")

    console.print(f"  Vertices:   {len(mesh_data.vertices):,} active / {n_total_verts:,} total")
    console.print(f"  Faces:      {len(mesh_data.faces):,}")

    console.print(f"  Output:")
    for coloring, do_animate, do_vectors, out_path in all_outputs:
        file_size = out_path.stat().st_size / 1024
        label = "static"
        if do_animate:
            label = "animated"
            if do_vectors:
                label += " + vectors"
        console.print(f"    {out_path.name}  [dim]({file_size:.0f} KB, {coloring} {label})[/dim]")

    console.print(f"  Total time: {_format_duration(elapsed)}")
    if step_times:
        # Separate shared (one-time) steps from per-coloring steps
        shared_labels = ("Subdivide", "Mapping", "xatlas", "Rasterize", "Textures")
        shared_parts = []
        for label in shared_labels:
            if label in step_times and step_times[label] >= 0.5:
                shared_parts.append(f"{label} {_format_duration(step_times[label])}")
        if shared_parts:
            console.print(f"  [dim]  Shared:   {', '.join(shared_parts)}[/dim]")

        # Per-coloring recolor times
        coloring_parts = []
        for label in sorted(step_times):
            if label.startswith("Recolor:") and step_times[label] >= 0.5:
                coloring_parts.append(f"{label} {_format_duration(step_times[label])}")
        if coloring_parts:
            console.print(f"  [dim]  Recolor:  {', '.join(coloring_parts)}[/dim]")

        if "KTX2" in step_times and step_times["KTX2"] >= 0.5:
            console.print(f"  [dim]  KTX2:     {_format_duration(step_times['KTX2'])}[/dim]")


def _write_carto_log(
    output_dir: Path,
    study: "CartoStudy",
    mesh: "CartoMesh",
    mesh_data: "MeshData",
    points: list[CartoPoint] | None,
    colorings: list[str],
    subdivide: int,
    all_outputs: list[tuple[str, bool, bool, Path]],
    start_time: float,
    source_path: Path,
    step_times: dict[str, float] | None = None,
    data_coverage_pct: float | None = None,
    equivalent_command: str | None = None,
    estimated_time: str | None = None,
) -> None:
    """Append conversion metadata to the log file in the output directory."""
    from datetime import datetime
    from med2glb.io.conversion_log import append_carto_entry

    end_now = time.time()
    elapsed = end_now - start_time

    # For the log, summarize all colorings produced
    color_range = ", ".join(colorings)

    # Convert all_outputs to variant_outputs format expected by log
    variant_outputs = [
        (do_animate, do_vectors, out_path)
        for _coloring, do_animate, do_vectors, out_path in all_outputs
    ]

    append_carto_entry(
        output_dir,
        structure_name=mesh.structure_name,
        carto_version=_carto_version_label(study.version),
        study_name=study.study_name or "",
        coloring=color_range,
        color_range=color_range,
        subdivide=subdivide,
        active_vertices=len(mesh_data.vertices),
        total_vertices=len(mesh.vertices),
        face_count=len(mesh_data.faces),
        mapping_points=len(points) if points else 0,
        variant_outputs=variant_outputs,
        elapsed_seconds=elapsed,
        source_path=str(source_path),
        start_time=datetime.fromtimestamp(start_time),
        end_time=datetime.fromtimestamp(end_now),
        estimated_time=estimated_time,
        step_times=step_times,
        data_coverage_pct=data_coverage_pct,
        equivalent_command=equivalent_command,
    )


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
        TimeElapsedColumn(),
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

    Handles mesh selection, subdivision, multi-coloring, animation, vectors,
    compression, and summary printing for all requested variants.

    Expensive shared work (subdivision, xatlas UV unwrap) is computed once per
    mesh. Per-coloring work (vertex color mapping, texture baking) is cheap
    and repeated for each requested coloring that has valid data.
    """
    from med2glb.io.carto_mapper import (
        carto_mesh_to_mesh_data,
        extract_point_field,
        subdivide_carto_mesh,
    )
    from med2glb.glb.builder import build_glb

    selected = config.selected_mesh_indices
    if selected is None:
        selected = list(range(len(study.meshes)))

    carto_output_dir = config.output_dir if config.output_dir is not None else config.input_path / "glb"
    carto_output_dir.mkdir(parents=True, exist_ok=True)

    # Build list of (mesh_idx, animate_flag, vectors_flag) jobs.
    # Animation and vectors only apply to LAT coloring — they are handled
    # inside the per-coloring loop, not scheduled here.
    has_vectors = config.vectors in ("yes", "only")
    vectors_only = config.vectors == "only"
    vec_meshes = set(config.vector_mesh_indices) if config.vector_mesh_indices is not None else None

    for mesh_idx in selected:
        mesh_start_time = time.time()
        mesh = study.meshes[mesh_idx]
        points = study.points.get(mesh.structure_name)
        mesh_has_vec = has_vectors and (vec_meshes is None or mesh_idx in vec_meshes)

        # Determine which colorings have valid data for this mesh
        available_colorings: list[str] = []
        for c in config.colorings:
            if not points:
                # No points at all — skip all colorings
                break
            _, valid_values = extract_point_field(points, c)
            if len(valid_values) > 0:
                available_colorings.append(c)

        if not available_colorings:
            console.print(
                f"\n[yellow]Skipping {mesh.structure_name}: "
                f"no valid data for any coloring[/yellow]"
            )
            continue

        # Count total variants for the status message
        n_variants = 0
        for c in available_colorings:
            if c == "lat":
                # LAT gets full variant set (static/animated/vectors)
                if vectors_only:
                    n_variants += 1  # animated+vectors only
                else:
                    if config.static:
                        n_variants += 1
                    if config.animate and points:
                        n_variants += 1  # animated
                        if mesh_has_vec:
                            n_variants += 1  # animated+vectors
            else:
                # Bipolar/unipolar are static-only
                n_variants += 1

        # === Shared intermediates — computed ONCE per mesh ===
        step_times: dict[str, float] = {}

        # Compute time estimate for this mesh
        n_triangles = len(mesh.faces)
        n_points = len(points) if points else 0
        has_lat = "lat" in available_colorings
        from med2glb.cli_wizard import estimate_time, estimate_time_details
        estimated_time = estimate_time(n_triangles, n_points, has_lat)
        step_est = estimate_time_details(
            n_triangles, n_points, has_lat, config.subdivide,
        )

        # Show preparation header with estimated total time + step breakdown
        console.print(
            f"\n[bold]Preparing: {mesh.structure_name}[/bold] "
            f"({n_variants} variant(s) across {len(available_colorings)} coloring(s))"
            f" — estimated {estimated_time}"
        )
        _est_parts: list[str] = []
        for _label, _key in [
            ("subdivide", "subdivide"), ("mapping", "mapping"),
            ("xatlas", "xatlas"), ("rasterize", "rasterize"),
            ("bake textures", "bake"), ("vectors", "vectors"),
        ]:
            if step_est[_key] >= 1:
                from med2glb.cli_wizard import _format_est
                _est_parts.append(f"{_label} ~{_format_est(step_est[_key])}")
        if _est_parts:
            console.print(f"  [dim]{' → '.join(_est_parts)}[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # 1. Subdivide once (coloring-independent)
            post_sub_count = n_triangles * (4 ** config.subdivide) if config.subdivide > 0 else n_triangles
            sub_est = _format_duration(step_est["subdivide"])
            task = progress.add_task(
                f"Subdividing {n_triangles:,} → {post_sub_count:,} faces (~{sub_est})..."
                if config.subdivide > 0
                else "Preparing mesh...",
                total=None,
            )

            subdivided = None
            t_step = time.monotonic()
            if config.subdivide > 0:
                subdivided = subdivide_carto_mesh(mesh, iterations=config.subdivide)
            step_times["Subdivide"] = time.monotonic() - t_step

            # 2. Compute LAT mesh data first — needed for animation cache
            #    which is shared across LAT variants
            map_est = _format_duration(step_est["mapping"])
            progress.update(task, description=f"Mapping vertices (~{map_est})...")
            t_step = time.monotonic()
            lat_mesh_data = carto_mesh_to_mesh_data(
                mesh, points, coloring="lat",
                subdivide=config.subdivide, pre_subdivided=subdivided,
            )
            step_times["Mapping"] = time.monotonic() - t_step
            progress.update(
                task,
                description=f"Mapped {len(lat_mesh_data.vertices):,} verts, "
                f"{len(lat_mesh_data.faces):,} faces",
            )

            # 3. LAT values once (needed for animation and vectors)
            active_lat = None
            if points and "lat" in available_colorings:
                progress.update(task, description="Extracting LAT values...")
                active_lat = _extract_active_lat(
                    mesh, points, lat_mesh_data, config.subdivide,
                    subdivided_mesh=subdivided,
                )

            progress.remove_task(task)

        # === Pre-compute LAT animated cache (shared xatlas + textures) ===
        _n_frames = 30
        lat_needs_anim = (
            "lat" in available_colorings
            and active_lat is not None
            and points
            and (config.animate or vectors_only)
        )

        anim_cache = None
        if lat_needs_anim:
            from med2glb.glb.carto_builder import prepare_animated_cache

            _frame_progress: Progress | None = None
            _frame_task = None
            _status_obj: Status | None = None

            def _cache_progress(desc: str, current: int, total: int) -> None:
                nonlocal _frame_progress, _frame_task, _status_obj
                if total == 0:
                    # Blocking step — use a live spinner that updates in-place
                    if _frame_progress is not None:
                        _frame_progress.stop()
                        _frame_progress = None
                        _frame_task = None
                    if _status_obj is None:
                        _status_obj = Status(desc, spinner="dots", console=console)
                        _status_obj.start()
                    _status_obj.update(desc)
                else:
                    # Frame-based step — switch to progress bar
                    if _status_obj is not None:
                        _status_obj.stop()
                        _status_obj = None
                    if _frame_progress is None:
                        _frame_progress = Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            BarColumn(bar_width=20),
                            MofNCompleteColumn(),
                            TimeElapsedColumn(),
                            console=console,
                        )
                        _frame_progress.start()
                        _frame_task = _frame_progress.add_task(desc, total=total)
                    _frame_progress.update(_frame_task, description=desc, completed=current)

            anim_cache = prepare_animated_cache(
                lat_mesh_data, active_lat, n_frames=_n_frames,
                progress=_cache_progress,
            )
            if _status_obj is not None:
                _status_obj.stop()
                _status_obj = None
            if _frame_progress is not None:
                _frame_progress.update(_frame_task, completed=_n_frames)
                _frame_progress.stop()
                _frame_progress = None
            if anim_cache is not None and anim_cache.step_times:
                step_times.update(anim_cache.step_times)

        # === Per-coloring loop ===
        _UNITS = {"lat": "ms", "bipolar": "mV", "unipolar": "mV"}
        _DEFAULT_RANGES: dict[str, tuple[float, float]] = {
            "bipolar": (0.05, 1.5),
            "unipolar": (3.0, 10.0),
        }

        # Collect all outputs across colorings for one combined summary
        all_outputs: list[tuple[str, bool, bool, Path]] = []
        produced_colorings: list[str] = []
        last_data_coverage: float | None = None

        for coloring in available_colorings:
            is_lat = coloring == "lat"

            # Get mesh data for this coloring (reuse LAT data if already computed)
            if is_lat:
                mesh_data = lat_mesh_data
            else:
                t_step = time.monotonic()
                mesh_data = carto_mesh_to_mesh_data(
                    mesh, points, coloring=coloring,
                    subdivide=config.subdivide, pre_subdivided=subdivided,
                )
                step_times["Mapping"] = step_times.get("Mapping", 0) + (time.monotonic() - t_step)

            # Compute data coverage for this coloring
            data_coverage_pct: float | None = None
            if points and mesh_data.vertex_colors is not None:
                n_active = len(mesh_data.vertices)
                colors = mesh_data.vertex_colors
                is_gray = (
                    (np.abs(colors[:, 0] - 0.5) < 0.01)
                    & (np.abs(colors[:, 1] - 0.5) < 0.01)
                    & (np.abs(colors[:, 2] - 0.5) < 0.01)
                )
                n_colored = int(np.sum(~is_gray))
                data_coverage_pct = 100.0 * n_colored / n_active if n_active > 0 else 100.0

                if data_coverage_pct < 50.0:
                    _, valid_values = extract_point_field(points, coloring)
                    n_valid = len(valid_values)
                    console.print(
                        f"  [yellow]Warning: only {n_valid:,} of {len(points):,} points "
                        f"({100.0 * n_valid / len(points):.0f}%) have valid {coloring.upper()} data "
                        f"— {data_coverage_pct:.0f}% of mesh colored, rest shown as gray[/yellow]"
                    )

            # Assemble legend metadata for this coloring
            unit = _UNITS.get(coloring, "")
            if coloring in _DEFAULT_RANGES:
                clamp_range = _DEFAULT_RANGES[coloring]
            elif is_lat:
                # Use CARTO's own pre-computed vertex values for the range
                # (matches the CARTO viewer display exactly).
                mesh_lat = mesh.vertex_color_values.get("lat")
                if mesh_lat is not None and np.any(~np.isnan(mesh_lat)):
                    clamp_range = (
                        float(np.nanmin(mesh_lat)),
                        float(np.nanmax(mesh_lat)),
                    )
                elif active_lat is not None:
                    clamp_range = (
                        float(np.nanmin(active_lat)),
                        float(np.nanmax(active_lat)),
                    )
                else:
                    clamp_range = (0.0, 1.0)
            else:
                clamp_range = (0.0, 1.0)

            from datetime import date
            legend_metadata: dict = {
                "study_name": study.study_name or "",
                "carto_version": _carto_version_label(study.version),
                "structure": mesh.structure_name,
                "coloring": coloring,
                "clamp_range": list(clamp_range),
                "unit": unit,
                "mapping_points": len(points) if points else 0,
                "export_date": date.today().isoformat(),
            }
            if data_coverage_pct is not None and data_coverage_pct < 100.0:
                legend_metadata["data_coverage"] = f"{data_coverage_pct:.0f}% {coloring.upper()}"

            legend_info: dict = {
                "coloring": coloring,
                "clamp_range": list(clamp_range),
                "metadata": legend_metadata,
            }

            # Build variant list for this coloring
            # Animation + vectors only for LAT; other colorings are static-only
            coloring_variants: list[tuple[bool, bool]] = []
            if is_lat:
                if vectors_only:
                    if mesh_has_vec:
                        coloring_variants.append((True, True))
                    else:
                        coloring_variants.append((True, False))
                else:
                    if config.static:
                        coloring_variants.append((False, False))
                    if config.animate and points:
                        coloring_variants.append((True, False))
                        if mesh_has_vec:
                            coloring_variants.append((True, True))
            else:
                # Bipolar/unipolar: static only
                coloring_variants.append((False, False))

            # Emit GLB files for this coloring
            t_coloring_start = time.monotonic()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=20),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                for do_animate, do_vectors in coloring_variants:
                    anim_suffix = "_animated" if (do_animate and points) else ""
                    vec_suffix = "_vectors" if do_vectors else ""
                    glb_name = f"{mesh.structure_name}_{coloring}{anim_suffix}{vec_suffix}.glb"
                    out_path = carto_output_dir / glb_name

                    variant_label = "static"
                    if do_animate:
                        variant_label = "animated + vectors" if do_vectors else "animated"

                    if do_animate and points:
                        _total_steps = _n_frames + 1
                    else:
                        _total_steps = 1

                    task = progress.add_task(
                        f"Building {coloring} {variant_label}...", total=_total_steps,
                    )

                    _variant_written = True
                    if do_animate and points and is_lat:
                        from med2glb.glb.carto_builder import build_carto_animated_glb

                        def _anim_progress(desc: str, current: int, _total: int) -> None:
                            progress.update(task, description=desc,
                                            completed=current + 1)

                        written = build_carto_animated_glb(
                            mesh_data, active_lat, out_path,
                            vectors=do_vectors,
                            progress=_anim_progress,
                            legend_info=legend_info,
                            cache=anim_cache,
                        )
                        if not written:
                            _variant_written = False
                        else:
                            progress.update(task, completed=_total_steps)
                    else:
                        if is_lat and anim_cache is not None:
                            # LAT static — reuse full cache (geometry + texture)
                            from med2glb.glb.carto_builder import build_carto_static_glb
                            build_carto_static_glb(
                                anim_cache, out_path, legend_info=legend_info,
                            )
                        elif not is_lat and anim_cache is not None and anim_cache.vmapping is not None:
                            # Bipolar/unipolar — reuse cached xatlas geometry,
                            # only re-bake the texture with new vertex colors
                            from med2glb.glb.carto_builder import build_carto_recolored_static_glb
                            build_carto_recolored_static_glb(
                                anim_cache, mesh_data, out_path,
                                legend_info=legend_info,
                            )
                        else:
                            build_glb(
                                [mesh_data], out_path,
                                source_units="mm", legend_info=legend_info,
                            )
                        progress.update(task, completed=_total_steps)

                    if _variant_written:
                        from med2glb.glb.compress import optimize_textures_ktx2
                        t_ktx = time.monotonic()
                        if optimize_textures_ktx2(out_path):
                            step_times["KTX2"] = step_times.get("KTX2", 0) + (time.monotonic() - t_ktx)
                        all_outputs.append((coloring, do_animate, do_vectors, out_path))
                    else:
                        progress.update(
                            task,
                            description=f"[dim]Skipped {variant_label} (data not suitable)[/dim]",
                            completed=_total_steps,
                        )

            # Track per-coloring build time (excludes shared steps)
            if not is_lat:
                step_times[f"Recolor:{coloring}"] = time.monotonic() - t_coloring_start

            if coloring not in produced_colorings and any(
                c == coloring for c, *_ in all_outputs
            ):
                produced_colorings.append(coloring)
            last_data_coverage = data_coverage_pct

        # === One combined summary per mesh (after all colorings) ===
        if all_outputs:
            _print_carto_summary(
                study, mesh, lat_mesh_data, points,
                produced_colorings,
                config.subdivide, all_outputs, mesh_start_time,
                step_times=step_times,
            )
            from med2glb.cli_wizard import build_carto_equiv_command
            equiv_cmd = build_carto_equiv_command(config)
            _write_carto_log(
                carto_output_dir, study, mesh, lat_mesh_data, points,
                produced_colorings, config.subdivide, all_outputs, mesh_start_time,
                source_path=config.input_path,
                step_times=step_times,
                data_coverage_pct=last_data_coverage,
                equivalent_command=equiv_cmd,
                estimated_time=estimated_time,
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

    # Print equivalent command for reproducibility
    from med2glb.cli_wizard import build_carto_equiv_command
    equiv_cmd = build_carto_equiv_command(config)
    console.print(f"\n[dim]💡 Equivalent command:[/dim]")
    console.print(f"[dim]   {equiv_cmd}[/dim]")


def run_carto_pipeline(
    input_path: Path,
    output: Path,
    coloring: str,
    subdivide: int,
    animate: bool,
    vectors: str,
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
    if _has_vectors:
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

    # When --coloring is specified, restrict to that single coloring;
    # otherwise produce all available colorings.
    colorings = [coloring] if coloring != "all" else ["lat", "bipolar", "unipolar"]

    config = CartoConfig(
        input_path=input_path,
        output_dir=output.parent,
        selected_mesh_indices=selected,
        colorings=colorings,
        subdivide=subdivide,
        animate=animate,
        static=not animate,
        vectors=effective_vectors,
        vector_mesh_indices=vec_suitable_list,
        target_faces=target_faces,
    )
    _convert_carto_meshes(config, study, start_time)
