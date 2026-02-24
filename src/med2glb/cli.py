"""CLI entry point for med2glb."""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from med2glb import __version__
from med2glb.core.types import (
    CartoPoint,
    MaterialConfig,
    MethodParams,
    SeriesInfo,
    ThresholdLayer,
)

app = typer.Typer(
    name="med2glb",
    help="Convert DICOM medical imaging data to GLB 3D models.",
    add_completion=False,
)

# Reconfigure stdout/stderr to UTF-8 to avoid Windows charmap encoding errors
# with Rich's Unicode spinners.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

console = Console()
err_console = Console(stderr=True)

logger = logging.getLogger("med2glb")


def version_callback(value: bool):
    if value:
        console.print(f"med2glb {__version__}")
        raise typer.Exit()


def list_methods_callback(value: bool):
    if value:
        from med2glb.methods.registry import _ensure_methods_loaded, list_methods

        _ensure_methods_loaded()
        methods = list_methods()

        console.print("\n[bold]Available conversion methods:[/bold]\n")
        for m in methods:
            status = "[green]installed[/green]" if m["available"] else "[red]not installed[/red]"
            console.print(f"  [bold]{m['name']:<18}[/bold] {m['description']}")
            console.print(f"  {'':18} Best for: {m['recommended_for']}")
            console.print(f"  {'':18} Status: {status} — {m['dependency_message']}")
            console.print()
        raise typer.Exit()


@app.command()
def main(
    ctx: typer.Context,
    input_path: Path = typer.Argument(
        ...,
        help="Path to a DICOM file or directory containing DICOM files.",
        exists=True,
    ),
    output: Path = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file path (default: <input_name>.glb).",
    ),
    method: str = typer.Option(
        "classical",
        "-m",
        "--method",
        help="Conversion method: marching-cubes, classical, totalseg, medsam2.",
    ),
    format: str = typer.Option(
        "glb",
        "-f",
        "--format",
        help="Output format: glb, stl, obj.",
    ),
    animate: bool = typer.Option(
        False,
        "--animate",
        help="Enable animation for temporal (4D) data.",
    ),
    threshold: float = typer.Option(
        None,
        "--threshold",
        help="Intensity threshold for isosurface extraction.",
    ),
    smoothing: int = typer.Option(
        15,
        "--smoothing",
        help="Taubin smoothing iterations (0 to disable).",
    ),
    faces: int = typer.Option(
        80000,
        "--faces",
        help="Target triangle count after decimation.",
    ),
    alpha: float = typer.Option(
        1.0,
        "--alpha",
        help="Global transparency (0.0-1.0) for non-segmented output.",
    ),
    multi_threshold: str = typer.Option(
        None,
        "--multi-threshold",
        help='Multi-threshold config: "val1:label1:alpha1,val2:label2:alpha2".',
    ),
    series: str = typer.Option(
        None,
        "--series",
        help="Select specific DICOM series by UID (partial match supported).",
    ),
    do_list_methods: bool = typer.Option(
        False,
        "--list-methods",
        callback=list_methods_callback,
        is_eager=True,
        help="List available conversion methods and exit.",
    ),
    coloring: str = typer.Option(
        "lat",
        "--coloring",
        help="CARTO coloring scheme: lat, bipolar, unipolar.",
    ),
    subdivide: int = typer.Option(
        2,
        "--subdivide",
        help="CARTO mesh subdivision level (0-3). Higher = smoother color maps, more vertices.",
        min=0,
        max=3,
    ),
    vectors: bool = typer.Option(
        False,
        "--vectors",
        help="Add animated LAT streamline arrows (CARTO LAT maps only).",
    ),
    gallery: bool = typer.Option(
        False,
        "--gallery",
        help="Gallery mode: individual GLBs, lightbox grid, and spatial fan.",
    ),
    columns: int = typer.Option(
        6,
        "--columns",
        help="Number of columns in the lightbox grid (gallery mode).",
    ),
    no_animate: bool = typer.Option(
        False,
        "--no-animate",
        help="Force static output even if temporal data is detected.",
    ),
    max_size: int = typer.Option(
        99,
        "--max-size",
        help="Maximum output GLB file size in MB (0 to disable).",
    ),
    compress: str = typer.Option(
        "draco",
        "--compress",
        help="Compression strategy: draco (default), downscale, jpeg.",
    ),
    do_list_series: bool = typer.Option(
        False,
        "--list-series",
        help="List DICOM series found in input directory and exit.",
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Show detailed processing information.",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    """Convert DICOM medical imaging data to GLB 3D models."""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Detect if --method was explicitly provided
    method_explicit = _was_option_provided(ctx, "method")

    # Handle --list-series
    if do_list_series:
        from med2glb.io.dicom_reader import analyze_series

        series_list = analyze_series(input_path)
        _print_series_table(series_list, input_path)
        raise typer.Exit()

    # --- Interactive wizard ---
    # If no pipeline flags were explicitly set and we have a TTY, run the wizard
    from med2glb.cli_wizard import is_interactive as _is_interactive
    if _is_interactive() and not _has_pipeline_flags(ctx):
        try:
            from med2glb.cli_wizard import analyze_input, run_carto_wizard, run_dicom_wizard
            detected = analyze_input(input_path)

            if detected.kind == "carto" and detected.carto_study is not None:
                config = run_carto_wizard(
                    detected.carto_study, input_path, console,
                )
                if output is not None:
                    base = output if output.suffix == "" else output.parent
                    config.output_dir = base / "glb"
                _run_carto_from_config(config)
                return
            elif detected.kind == "dicom" and detected.series_list is not None:
                dicom_config = run_dicom_wizard(
                    detected.series_list, input_path, console,
                )
                stem = input_path.stem if input_path.is_file() else input_path.name
                if output is not None:
                    base = output if output.suffix == "" else output.parent
                    glb_dir = base / "glb"
                else:
                    glb_dir = (input_path if input_path.is_dir() else input_path.parent) / "glb"
                out_path = glb_dir / f"{stem}.glb"
                _run_dicom_from_config(dicom_config, out_path)
                return
        except ValueError:
            pass  # Fall through to normal pipeline
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled.[/dim]")
            raise typer.Exit()

    # Derive output path from input name when not fully specified
    stem = input_path.stem if input_path.is_file() else input_path.name
    parent = input_path.parent if input_path.is_file() else input_path.parent
    # Default output dir: "glb" subfolder keeps generated files together
    glb_dir = (input_path if input_path.is_dir() else parent) / "glb"
    if output is None or (not gallery and output.suffix == ""):
        label = _get_data_type_label(input_path, series)
        auto_stem = f"{stem}_{label}" if label else stem
        if output is None:
            if gallery:
                # Gallery: directory inside glb subfolder
                output = glb_dir / auto_stem
            else:
                # Place GLB in glb subfolder
                output = glb_dir / f"{auto_stem}.{format}"
        else:
            # -o points to a directory — put <input_name>_<type>.<format> inside it
            output = output / f"{auto_stem}.{format}"

    try:
        # Auto-detect CARTO data
        if input_path.is_dir():
            from med2glb.io.carto_reader import detect_carto_directory
            if detect_carto_directory(input_path):
                _run_carto_pipeline(
                    input_path=input_path,
                    output=output,
                    coloring=coloring,
                    subdivide=subdivide,
                    animate=animate,
                    vectors="yes" if vectors else "no",
                    max_size_mb=max_size,
                    compress_strategy=compress,
                    target_faces=faces,
                    verbose=verbose,
                )
                return

        if gallery:
            _run_gallery_mode(
                input_path=input_path,
                output=output,
                series=series,
                columns=columns,
                no_animate=no_animate,
                max_size_mb=max_size,
                compress_strategy=compress,
                verbose=verbose,
            )
        else:
            _run_pipeline(
                input_path=input_path,
                output=output,
                method_name=method,
                method_explicit=method_explicit,
                format=format,
                animate=animate,
                threshold=threshold,
                smoothing=smoothing,
                target_faces=faces,
                alpha=alpha,
                multi_threshold=multi_threshold,
                series=series,
                max_size_mb=max_size,
                compress_strategy=compress,
                verbose=verbose,
            )
    except ValueError as e:
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=4)
    except ImportError as e:
        err_console.print(
            f"[red]Missing dependency: {e}[/red]\n"
            "Install AI dependencies with: pip install med2glb[ai]"
        )
        raise typer.Exit(code=3)
    except Exception as e:
        err_console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            err_console.print(traceback.format_exc())
        raise typer.Exit(code=1)


def _was_option_provided(ctx: typer.Context, param_name: str) -> bool:
    """Check if a CLI option was explicitly provided by the user."""
    # typer/click stores the source of each parameter value
    source = ctx.get_parameter_source(param_name)
    if source is None:
        return False
    import click
    return source == click.core.ParameterSource.COMMANDLINE


def _has_pipeline_flags(ctx: typer.Context) -> bool:
    """Return True if the user explicitly set any pipeline-specific flags.

    When no pipeline flags are provided and the session is interactive,
    the wizard runs instead.
    """
    pipeline_params = [
        "method", "coloring", "animate", "threshold", "gallery",
        "no_animate", "vectors", "multi_threshold",
    ]
    return any(_was_option_provided(ctx, p) for p in pipeline_params)


def _run_pipeline(
    input_path: Path,
    output: Path,
    method_name: str,
    method_explicit: bool,
    format: str,
    animate: bool,
    threshold: float | None,
    smoothing: int,
    target_faces: int,
    alpha: float,
    multi_threshold: str | None,
    series: str | None,
    max_size_mb: int,
    compress_strategy: str,
    verbose: bool,
) -> None:
    """Execute the conversion pipeline with series selection."""
    # If --series provided or input is a single file, pass through directly
    if series or input_path.is_file():
        _convert_series(
            input_path=input_path,
            output=output,
            method_name=method_name,
            format=format,
            animate=animate,
            threshold=threshold,
            smoothing=smoothing,
            target_faces=target_faces,
            alpha=alpha,
            multi_threshold=multi_threshold,
            series_uid=series,
            max_size_mb=max_size_mb,
            compress_strategy=compress_strategy,
            verbose=verbose,
        )
        return

    # Directory input — analyze series
    from med2glb.io.dicom_reader import analyze_series

    series_list = analyze_series(input_path)

    if len(series_list) == 1:
        # Single series — proceed automatically
        info = series_list[0]
        effective_method = method_name if method_explicit else info.recommended_method
        _convert_series(
            input_path=input_path,
            output=output,
            method_name=effective_method,
            format=format,
            animate=animate,
            threshold=threshold,
            smoothing=smoothing,
            target_faces=target_faces,
            alpha=alpha,
            multi_threshold=multi_threshold,
            series_uid=info.series_uid,
            max_size_mb=max_size_mb,
            compress_strategy=compress_strategy,
            verbose=verbose,
        )
        return

    # Multiple series detected
    if sys.stdin.isatty():
        # Interactive selection
        _print_series_table(series_list, input_path)
        selected = _interactive_select_series(series_list)
    else:
        # Non-TTY: auto-select best (first after sorting)
        selected = [series_list[0]]
        logger.warning(
            f"Multiple series found, auto-selecting best: "
            f"{selected[0].description or selected[0].series_uid} "
            f"({selected[0].data_type}, {selected[0].detail})"
        )

    # Convert each selected series
    for i, info in enumerate(selected):
        out_path = _make_output_path(output, i, len(selected))
        effective_method = method_name if method_explicit else info.recommended_method
        console.print(
            f"\n[bold]Converting series {i + 1}/{len(selected)}: "
            f"{info.description or info.series_uid} ({info.data_type})[/bold]"
        )
        _convert_series(
            input_path=input_path,
            output=out_path,
            method_name=effective_method,
            format=format,
            animate=animate,
            threshold=threshold,
            smoothing=smoothing,
            target_faces=target_faces,
            alpha=alpha,
            multi_threshold=multi_threshold,
            series_uid=info.series_uid,
            max_size_mb=max_size_mb,
            compress_strategy=compress_strategy,
            verbose=verbose,
        )


def _run_carto_from_config(config: "CartoConfig") -> None:
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
                from med2glb.io.carto_mapper import (
                    map_points_to_vertices,
                    map_points_to_vertices_idw,
                    interpolate_sparse_values,
                    subdivide_carto_mesh,
                )

                anim_mesh = mesh
                if config.subdivide > 0:
                    progress.update(task, description="Subdividing mesh for LAT extraction...")
                    anim_mesh = subdivide_carto_mesh(mesh, iterations=config.subdivide)
                progress.update(task, advance=1)

                progress.update(task, description="Mapping LAT values...")
                if config.subdivide > 0:
                    lat_values = map_points_to_vertices_idw(anim_mesh, points, field="lat")
                else:
                    lat_values = map_points_to_vertices(anim_mesh, points, field="lat")
                    lat_values = interpolate_sparse_values(anim_mesh, lat_values)
                active_mask = anim_mesh.group_ids != -1000000
                active_lat = lat_values[active_mask]
                progress.update(task, advance=1)

                def _anim_progress(desc: str, current: int, _total: int) -> None:
                    progress.update(task, description=desc,
                                    completed=3 + current + 1)

                progress.update(task, description="Building excitation ring animation...")
                build_carto_animated_glb(
                    mesh_data, active_lat, out_path,
                    target_faces=config.target_faces,
                    vectors=do_vectors,
                    progress=_anim_progress,
                )
                progress.update(task, completed=_total_steps)
            else:
                if do_vectors and points:
                    progress.update(task, description="Generating static LAT vectors...")
                    from med2glb.mesh.lat_vectors import (
                        trace_all_streamlines, compute_animated_dashes,
                        compute_face_gradients, compute_dash_speed_factors,
                    )
                    from med2glb.glb.arrow_builder import build_frame_dashes, ArrowParams, _auto_scale_params
                    from med2glb.io.carto_mapper import (
                        map_points_to_vertices,
                        map_points_to_vertices_idw,
                        interpolate_sparse_values,
                        subdivide_carto_mesh,
                    )
                    vec_mesh = mesh
                    if config.subdivide > 0:
                        vec_mesh = subdivide_carto_mesh(mesh, iterations=config.subdivide)
                    if config.subdivide > 0:
                        vec_lat = map_points_to_vertices_idw(vec_mesh, points, field="lat")
                    else:
                        vec_lat = map_points_to_vertices(vec_mesh, points, field="lat")
                        vec_lat = interpolate_sparse_values(vec_mesh, vec_lat)
                    active_mask = vec_mesh.group_ids != -1000000
                    vec_lat_active = vec_lat[active_mask]

                    streamlines = trace_all_streamlines(
                        mesh_data.vertices, mesh_data.faces, vec_lat_active,
                        mesh_data.normals, target_count=300,
                    )
                    if streamlines:
                        dashes = compute_animated_dashes(streamlines, n_frames=1)
                        if dashes and dashes[0]:
                            bbox = mesh_data.vertices.max(axis=0) - mesh_data.vertices.min(axis=0)
                            params = _auto_scale_params(float(np.linalg.norm(bbox)))
                            max_r = params.max_radius if params.max_radius is not None else params.head_radius
                            face_grads, face_centers, _ = compute_face_gradients(
                                mesh_data.vertices, mesh_data.faces, vec_lat_active,
                            )
                            speed_factors = compute_dash_speed_factors(
                                dashes, face_grads, face_centers,
                            )
                            sf = speed_factors[0] if speed_factors and speed_factors[0] else None
                            if sf is not None:
                                # Cull low-gradient dashes and compute per-dash radii
                                keep = [s >= 0.15 for s in sf]
                                frame_dashes = [d for d, k in zip(dashes[0], keep) if k]
                                sf = [s for s, k in zip(sf, keep) if k]
                                dash_radii = [max_r * (1.1 - 0.3 * s) for s in sf]
                            else:
                                frame_dashes = dashes[0]
                                dash_radii = None
                            arrow_mesh = build_frame_dashes(
                                frame_dashes, mesh_data.vertices, mesh_data.normals, params,
                                dash_radii=dash_radii,
                            )
                            if arrow_mesh is not None:
                                extra = [arrow_mesh]

                progress.update(task, description="Building GLB...")
                build_glb([mesh_data], out_path, extra_meshes=extra)
                progress.update(task, completed=_total_steps)

            # Produce a _compressed variant if the file exceeds the size limit
            if config.max_size_mb > 0:
                _build_compressed_carto_variant(
                    out_path, config.max_size_mb, mesh_data,
                    do_animate and bool(points), _n_frames, do_vectors,
                    active_lat if (do_animate and points) else None,
                    extra, progress,
                )

        # Print summary
        file_size = out_path.stat().st_size / 1024
        elapsed = time.time() - start_time
        n_total_verts = len(mesh.vertices)

        clamp_info = ""
        if config.coloring == "bipolar":
            clamp_info = "0.05 – 1.5 mV"
        elif config.coloring == "unipolar":
            clamp_info = "3.0 – 10.0 mV"
        elif config.coloring == "lat" and points:
            valid_lats = [p.lat for p in points if not math.isnan(p.lat)]
            if valid_lats:
                clamp_info = f"{min(valid_lats):.0f} – {max(valid_lats):.0f} ms (auto)"

        console.print(f"\n[green]CARTO conversion complete![/green]")
        console.print(f"  System:     {_carto_version_label(study.version)}")
        if study.study_name:
            console.print(f"  Study:      {study.study_name}")
        console.print(f"  Map:        {mesh.structure_name}")
        console.print(f"  Coloring:   {config.coloring}")
        if config.subdivide > 0:
            console.print(f"  Subdivide:  level {config.subdivide} (~{4**config.subdivide}x face increase)")
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


def _run_dicom_from_config(config: "DicomConfig", output: Path) -> None:
    """Execute the DICOM pipeline from a wizard-produced config."""
    _convert_series(
        input_path=config.input_path,
        output=output,
        method_name=config.method,
        format=config.format,
        animate=config.animate,
        threshold=config.threshold,
        smoothing=config.smoothing,
        target_faces=config.target_faces,
        alpha=config.alpha,
        multi_threshold=None,
        series_uid=config.series_uid,
        max_size_mb=config.max_size_mb,
        compress_strategy=config.compress_strategy,
        verbose=config.verbose,
    )


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


def _run_carto_pipeline(
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

    # Use output's parent directory for per-mesh files
    carto_output_dir = output.parent
    carto_output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each selected mesh
    for mesh_idx in selected:
        mesh = study.meshes[mesh_idx]
        points = study.points.get(mesh.structure_name)

        # Build descriptive filename: <structure>_<coloring>[_animated][_vectors].glb
        anim_suffix = "_animated" if (animate and points) else ""
        _has_vectors = vectors in ("yes", "only")
        vec_suffix = "_vectors" if _has_vectors else ""
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
        elif _has_vectors and points:
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
                from med2glb.io.carto_mapper import (
                    map_points_to_vertices,
                    map_points_to_vertices_idw,
                    interpolate_sparse_values,
                    subdivide_carto_mesh,
                )

                # Use the same subdivided mesh for animation LAT extraction
                anim_mesh = mesh
                if subdivide > 0:
                    progress.update(task, description="Subdividing mesh for LAT extraction...")
                    anim_mesh = subdivide_carto_mesh(mesh, iterations=subdivide)
                progress.update(task, advance=1)

                progress.update(task, description="Mapping LAT values...")
                if subdivide > 0:
                    lat_values = map_points_to_vertices_idw(anim_mesh, points, field="lat")
                else:
                    lat_values = map_points_to_vertices(anim_mesh, points, field="lat")
                    lat_values = interpolate_sparse_values(anim_mesh, lat_values)
                # Filter to active vertices
                active_mask = anim_mesh.group_ids != -1000000
                active_lat = lat_values[active_mask]
                progress.update(task, advance=1)

                def _anim_progress(desc: str, current: int, _total: int) -> None:
                    # Map frame progress into our unified step counter
                    progress.update(task, description=desc,
                                    completed=3 + current + 1)

                progress.update(task, description="Building excitation ring animation...")
                build_carto_animated_glb(
                    mesh_data, active_lat, out_path,
                    target_faces=target_faces,
                    vectors=_has_vectors,
                    progress=_anim_progress,
                )
                progress.update(task, completed=_total_steps)
            else:
                # Static GLB
                if _has_vectors and points:
                    progress.update(task, description="Generating static LAT vectors...")
                    from med2glb.mesh.lat_vectors import (
                        trace_all_streamlines, compute_animated_dashes,
                        compute_face_gradients, compute_dash_speed_factors,
                    )
                    from med2glb.glb.arrow_builder import build_frame_dashes, ArrowParams, _auto_scale_params
                    from med2glb.io.carto_mapper import (
                        map_points_to_vertices,
                        map_points_to_vertices_idw,
                        interpolate_sparse_values,
                        subdivide_carto_mesh,
                    )
                    # Get LAT values for gradient computation
                    vec_mesh = mesh
                    if subdivide > 0:
                        vec_mesh = subdivide_carto_mesh(mesh, iterations=subdivide)
                    if subdivide > 0:
                        vec_lat = map_points_to_vertices_idw(vec_mesh, points, field="lat")
                    else:
                        vec_lat = map_points_to_vertices(vec_mesh, points, field="lat")
                        vec_lat = interpolate_sparse_values(vec_mesh, vec_lat)
                    active_mask = vec_mesh.group_ids != -1000000
                    vec_lat_active = vec_lat[active_mask]

                    streamlines = trace_all_streamlines(
                        mesh_data.vertices, mesh_data.faces, vec_lat_active,
                        mesh_data.normals, target_count=300,
                    )
                    if streamlines:
                        dashes = compute_animated_dashes(streamlines, n_frames=1)
                        if dashes and dashes[0]:
                            bbox = mesh_data.vertices.max(axis=0) - mesh_data.vertices.min(axis=0)
                            params = _auto_scale_params(float(np.linalg.norm(bbox)))
                            max_r = params.max_radius if params.max_radius is not None else params.head_radius
                            face_grads, face_centers, _ = compute_face_gradients(
                                mesh_data.vertices, mesh_data.faces, vec_lat_active,
                            )
                            speed_factors = compute_dash_speed_factors(
                                dashes, face_grads, face_centers,
                            )
                            sf = speed_factors[0] if speed_factors and speed_factors[0] else None
                            if sf is not None:
                                # Cull low-gradient dashes and compute per-dash radii
                                keep = [s >= 0.15 for s in sf]
                                frame_dashes = [d for d, k in zip(dashes[0], keep) if k]
                                sf = [s for s, k in zip(sf, keep) if k]
                                dash_radii = [max_r * (1.1 - 0.3 * s) for s in sf]
                            else:
                                frame_dashes = dashes[0]
                                dash_radii = None
                            arrow_mesh = build_frame_dashes(
                                frame_dashes, mesh_data.vertices, mesh_data.normals, params,
                                dash_radii=dash_radii,
                            )
                            if arrow_mesh is not None:
                                extra = [arrow_mesh]

                progress.update(task, description="Building GLB...")
                build_glb([mesh_data], out_path, extra_meshes=extra)
                progress.update(task, completed=_total_steps)

            # Produce a _compressed variant if the file exceeds the size limit
            if max_size_mb > 0:
                _build_compressed_carto_variant(
                    out_path, max_size_mb, mesh_data,
                    animate and bool(points), _n_frames, _has_vectors,
                    active_lat if (animate and points) else None,
                    extra, progress,
                )

        # Print summary
        file_size = out_path.stat().st_size / 1024
        elapsed = time.time() - start_time
        n_total_verts = len(mesh.vertices)
        n_active_verts = int(np.sum(mesh.group_ids != -1000000))

        # Colormap clamp range info
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

        # Point statistics
        mesh_points = points or []
        point_stats = _carto_point_stats(mesh_points)
        for label, value in point_stats.items():
            console.print(f"  {label + ':':14s}{value}")

        console.print(f"  Vertices:   {len(mesh_data.vertices):,} active / {n_total_verts:,} total")
        console.print(f"  Faces:      {len(mesh_data.faces):,}")
        anim_desc = "No"
        if animate and points:
            anim_desc = "Yes (excitation ring)"
            if _has_vectors:
                anim_desc += " + LAT vectors"
        elif _has_vectors and points:
            anim_desc = "No (static LAT vectors)"
        console.print(f"  Animated:   {anim_desc}")
        console.print(f"  Output:     {out_path}")
        console.print(f"  Size:       {file_size:.1f} KB")
        console.print(f"  Time:       {elapsed:.1f}s")


def _print_series_table(series_list: list[SeriesInfo], input_path: Path) -> None:
    """Display a Rich table of DICOM series with classification info."""
    table = Table(title=f"DICOM Series in {input_path}")
    table.add_column("#", style="bold", justify="right")
    table.add_column("Modality", style="green")
    table.add_column("Description", max_width=40)
    table.add_column("Data Type", style="cyan")
    table.add_column("Detail", justify="right")
    table.add_column("Animated", justify="center")
    table.add_column("Recommended Output", style="yellow")
    table.add_column("Recommended Method", style="magenta")

    for i, info in enumerate(series_list, 1):
        desc = info.description if info.description else "(no desc)"
        animated = "[green]Yes[/green]" if info.data_type in ("2D cine", "3D+T volume") else "[dim]No[/dim]"
        table.add_row(
            str(i),
            info.modality,
            desc,
            info.data_type,
            info.detail,
            animated,
            info.recommended_output,
            info.recommended_method,
        )

    console.print(table)

    # Print recommendation
    best = series_list[0]
    console.print(
        f"\nRecommendation: Series 1 ({best.data_type}, {best.detail}) "
        f"→ [magenta]{best.recommended_method}[/magenta]"
    )


def _interactive_select_series(series_list: list[SeriesInfo]) -> list[SeriesInfo]:
    """Prompt user to select series interactively. Returns selected SeriesInfo list."""
    n = len(series_list)
    choice = Prompt.ask(
        f"Enter number (1-{n}), comma-separated (1,3), or 'all'",
        default="1",
        console=console,
    )
    return _parse_selection(choice, series_list)


def _parse_selection(choice: str, series_list: list[SeriesInfo]) -> list[SeriesInfo]:
    """Parse user selection string into list of SeriesInfo.

    Accepts: "1", "1,3", "all"
    """
    choice = choice.strip().lower()
    if choice == "all":
        return list(series_list)

    selected = []
    for part in choice.split(","):
        part = part.strip()
        try:
            idx = int(part) - 1  # 1-based to 0-based
        except ValueError:
            raise ValueError(f"Invalid selection: '{part}'. Use numbers like '1', '1,3', or 'all'.")
        if idx < 0 or idx >= len(series_list):
            raise ValueError(
                f"Selection {part} out of range. Choose 1-{len(series_list)}."
            )
        selected.append(series_list[idx])
    return selected


def _make_output_path(base: Path, index: int, total: int) -> Path:
    """Generate output path for multi-series conversion.

    Single series: output.glb
    Multiple series: output_1.glb, output_2.glb, ...
    """
    if total <= 1:
        return base
    return base.parent / f"{base.stem}_{index + 1}{base.suffix}"


def _convert_series(
    input_path: Path,
    output: Path,
    method_name: str,
    format: str,
    animate: bool,
    threshold: float | None,
    smoothing: int,
    target_faces: int,
    alpha: float,
    multi_threshold: str | None,
    series_uid: str | None,
    max_size_mb: int = 0,
    compress_strategy: str = "draco",
    verbose: bool = False,
) -> None:
    """Execute the full conversion pipeline for a single series."""
    from med2glb.io.dicom_reader import InputType, load_dicom_directory
    from med2glb.methods.registry import _ensure_methods_loaded, get_method

    # Ensure output directory exists (may be auto-created "glb" subfolder)
    output.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Parse multi-threshold if provided
    mt_layers = _parse_multi_threshold(multi_threshold) if multi_threshold else None

    params = MethodParams(
        threshold=threshold,
        smoothing_iterations=smoothing,
        target_faces=target_faces,
        multi_threshold=mt_layers,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        # Step 1: Load DICOM
        task = progress.add_task("Loading DICOM data...", total=None)
        input_type, data = load_dicom_directory(input_path, series_uid=series_uid)
        progress.update(task, description=f"Loaded {input_type.value} data")
        progress.remove_task(task)

        # Step 2: Get conversion method
        _ensure_methods_loaded()
        converter = get_method(method_name)

        # Step 3: Convert

        # Detect 2D temporal data (multi-frame ultrasound cine clips).
        # Without --animate, export first frame as a static textured plane.
        # With --animate, fall through to the animated pipeline.
        if input_type == InputType.TEMPORAL and not animate:
            from med2glb.core.volume import TemporalSequence
            if isinstance(data, TemporalSequence) and data.frames[0].voxels.shape[0] == 1:
                # 2D cine without --animate — export first frame as textured plane
                task = progress.add_task("Building textured plane from 2D echo...", total=None)
                from med2glb.glb.texture import build_textured_plane_glb

                console.print(
                    f"[yellow]2D ultrasound cine detected ({data.frame_count} frames). "
                    f"Exporting first frame as textured plane. "
                    f"Use --animate for animated output.[/yellow]"
                )
                build_textured_plane_glb(data.frames[0], output)
                progress.remove_task(task)
                if max_size_mb > 0:
                    _enforce_size_limit(output, max_size_mb, compress_strategy, progress)

                file_size = output.stat().st_size / 1024
                elapsed = time.time() - start_time
                console.print(f"\n[green]Conversion complete![/green]")
                console.print(f"  Input:    2D echo cine ({data.frame_count} frames)")
                console.print(f"  Output:   {output}")
                console.print(f"  Size:     {file_size:.1f} KB")
                console.print(f"  Time:     {elapsed:.1f}s")
                return

        if input_type == InputType.SINGLE_SLICE:
            task = progress.add_task("Building textured plane...", total=None)
            from med2glb.glb.texture import build_textured_plane_glb

            build_textured_plane_glb(data, output)
            progress.remove_task(task)
            if max_size_mb > 0:
                _enforce_size_limit(output, max_size_mb, compress_strategy, progress)

            # Print summary for single slice
            file_size = output.stat().st_size / 1024
            elapsed = time.time() - start_time
            console.print(f"\n[green]Conversion complete![/green]")
            console.print(f"  Input:    single slice ({input_path})")
            console.print(f"  Output:   {output}")
            console.print(f"  Size:     {file_size:.1f} KB")
            console.print(f"  Time:     {elapsed:.1f}s")
            return

        if input_type == InputType.TEMPORAL and animate:
            from med2glb.core.volume import TemporalSequence
            if isinstance(data, TemporalSequence) and data.frames[0].voxels.shape[0] == 1:
                # 2D cine + --animate → animated height-map textured plane
                task = progress.add_task(
                    f"Building animated surface from {data.frame_count} frames...", total=None
                )
                from med2glb.glb.texture import build_animated_textured_plane_glb

                build_animated_textured_plane_glb(data, output)
                progress.remove_task(task)
                if max_size_mb > 0:
                    _enforce_size_limit(output, max_size_mb, compress_strategy, progress)

                file_size = output.stat().st_size / 1024
                elapsed = time.time() - start_time
                console.print(f"\n[green]Conversion complete![/green]")
                console.print(f"  Input:    2D echo cine ({data.frame_count} frames)")
                console.print(f"  Output:   {output} (animated)")
                console.print(f"  Size:     {file_size:.1f} KB")
                console.print(f"  Time:     {elapsed:.1f}s")
                return
            # 3D temporal + --animate → standard morph target pipeline
            result = _run_animated_pipeline(data, converter, params, alpha, progress)
        else:
            if input_type == InputType.TEMPORAL:
                from med2glb.core.volume import TemporalSequence
                if isinstance(data, TemporalSequence):
                    data = data.frames[0]
                    console.print(
                        "[yellow]Temporal data detected but --animate not set. "
                        "Using first frame only.[/yellow]"
                    )

            task = progress.add_task(f"Converting with {method_name}...", total=None)

            def on_progress(desc, current=None, total=None):
                if total is not None:
                    progress.update(task, description=desc, completed=current, total=total)
                else:
                    progress.update(task, description=desc)

            result = converter.convert(data, params, progress=on_progress)
            progress.remove_task(task)

            # Apply alpha override
            if alpha < 1.0:
                for mesh in result.meshes:
                    mesh.material = MaterialConfig(
                        base_color=mesh.material.base_color,
                        alpha=alpha,
                        metallic=mesh.material.metallic,
                        roughness=mesh.material.roughness,
                        name=mesh.material.name,
                    )

            # Step 4: Mesh processing
            task = progress.add_task("Processing meshes...", total=None)
            from med2glb.mesh.processing import process_mesh

            processed = []
            for mesh in result.meshes:
                processed.append(
                    process_mesh(mesh, smoothing_iterations=smoothing, target_faces=target_faces)
                )
            result.meshes = processed
            progress.remove_task(task)

        # Step 5: Export
        task = progress.add_task(f"Exporting {format.upper()}...", total=None)
        _export(result, output, format, animate)
        progress.remove_task(task)

        # Step 6: Constrain file size
        if max_size_mb > 0 and format == "glb":
            _enforce_size_limit(output, max_size_mb, compress_strategy, progress)

    # Print summary
    elapsed = time.time() - start_time
    total_verts = sum(len(m.vertices) for m in result.meshes)
    total_faces = sum(len(m.faces) for m in result.meshes)
    file_size = output.stat().st_size / 1024

    console.print(f"\n[green]Conversion complete![/green]")
    console.print(f"  Input:    {input_type.value} ({input_path})")
    console.print(f"  Method:   {method_name}")
    console.print(f"  Output:   {output}")
    console.print(f"  Meshes:   {len(result.meshes)}")
    console.print(f"  Vertices: {total_verts:,}")
    console.print(f"  Faces:    {total_faces:,}")
    console.print(f"  Size:     {file_size:.1f} KB")
    console.print(f"  Time:     {elapsed:.1f}s")

    for w in result.warnings:
        err_console.print(f"[yellow]Warning: {w}[/yellow]")


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
    """Decimate a MeshData while preserving vertex colors via KDTree resampling."""
    from med2glb.mesh.processing import decimate, compute_normals
    from scipy.spatial import KDTree

    orig_verts = mesh_data.vertices.copy()
    orig_colors = mesh_data.vertex_colors
    result = decimate(mesh_data, target_faces=target_faces)
    result = compute_normals(result)
    if orig_colors is not None:
        tree = KDTree(orig_verts)
        _, idx = tree.query(result.vertices)
        result.vertex_colors = orig_colors[idx]
    return result


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
        build_glb([decimated], compressed_path, extra_meshes=extra_meshes)

    new_kb = compressed_path.stat().st_size / 1024
    progress.update(
        task,
        description=f"Compressed variant: {new_kb:.0f} KB ({compressed_path.name})",
    )
    progress.remove_task(task)


def _enforce_size_limit(
    path: Path,
    max_size_mb: int,
    strategy: str,
    progress: Progress,
) -> None:
    """Compress a GLB file if it exceeds the size limit.

    Tries texture-based compression (for DICOM GLBs with textures).
    For CARTO GLBs (vertex-color only), use _build_compressed_carto_variant
    instead — this function only handles texture-based strategies.
    """
    max_bytes = max_size_mb * 1024 * 1024
    if not path.exists() or path.stat().st_size <= max_bytes:
        return

    from med2glb.glb.compress import constrain_glb_size

    original_kb = path.stat().st_size / 1024
    task = progress.add_task(
        f"Compressing GLB ({original_kb:.0f} KB > {max_size_mb} MB limit)...",
        total=None,
    )

    applied = constrain_glb_size(path, max_bytes, strategy=strategy)

    new_kb = path.stat().st_size / 1024
    if applied and new_kb < original_kb:
        progress.update(task, description=f"Compressed: {original_kb:.0f} KB → {new_kb:.0f} KB")
    else:
        progress.update(task, description=f"Size {original_kb:.0f} KB (no further compression possible)")
    progress.remove_task(task)


def _run_gallery_mode(
    input_path: Path,
    output: Path,
    series: str | None,
    columns: int,
    no_animate: bool,
    max_size_mb: int = 0,
    compress_strategy: str = "draco",
    verbose: bool = False,
) -> None:
    """Execute gallery mode: individual GLBs, lightbox grid, and spatial fan."""
    from med2glb.gallery import (
        build_individual_glbs,
        build_lightbox_glb,
        build_spatial_glb,
        load_all_slices,
    )
    from med2glb.io.dicom_reader import analyze_series

    start_time = time.time()

    # Determine which series to process
    if series:
        target_uids = [series]
    elif input_path.is_dir():
        series_list = analyze_series(input_path)
        if len(series_list) > 1 and sys.stdin.isatty():
            _print_series_table(series_list, input_path)
            selected = _interactive_select_series(series_list)
        else:
            selected = series_list
        target_uids = [s.series_uid for s in selected]
    else:
        target_uids = [None]

    # Build a lookup for series descriptions
    series_info_map: dict[str, SeriesInfo] = {}
    if input_path.is_dir():
        for info in analyze_series(input_path):
            series_info_map[info.series_uid] = info

    output_dir = Path(output) if output.suffix == "" else output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    total_slices = 0
    series_summaries: list[dict] = []

    for uid in target_uids:
        # Derive a meaningful folder name
        info = series_info_map.get(uid) if uid else None
        if info and info.description:
            series_name = _sanitize_name(info.description)
        elif info:
            series_name = _sanitize_name(f"{info.modality}_{info.data_type}")
        else:
            series_name = "series"

        # Deduplicate folder names
        series_dir = output_dir / series_name
        if series_dir.exists() and series_summaries:
            series_dir = output_dir / f"{series_name}_{len(series_summaries) + 1}"

        console.print(
            f"\n[bold]Processing series: "
            f"{info.description or info.series_uid if info else 'default'} "
            f"({info.data_type}, {info.detail})[/bold]" if info else
            f"\n[bold]Processing series...[/bold]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            # Step 1: Load all slices
            task = progress.add_task("Loading DICOM slices...", total=None)
            slices = load_all_slices(input_path, series_uid=uid)
            progress.update(task, description=f"Loaded {len(slices)} slices")
            progress.remove_task(task)

            if not slices:
                console.print(f"  [yellow]No slices loaded — skipping.[/yellow]")
                continue

            total_slices += len(slices)

            # Auto-detect temporal data
            has_temporal = any(s.temporal_index is not None for s in slices)
            animate = has_temporal and not no_animate

            # Step 2: Individual GLBs
            task = progress.add_task("Building individual GLBs...", total=None)
            individual_paths = build_individual_glbs(
                slices, series_dir, animate=animate,
            )
            progress.update(task, description=f"Built {len(individual_paths)} individual GLBs")
            progress.remove_task(task)
            if max_size_mb > 0:
                for p in individual_paths:
                    _enforce_size_limit(p, max_size_mb, compress_strategy, progress)

            # Step 3: Lightbox GLB (inside the series folder)
            lightbox_path = series_dir / "lightbox.glb"
            task = progress.add_task("Building lightbox grid...", total=None)
            build_lightbox_glb(
                slices, lightbox_path, columns=columns, animate=animate,
            )
            progress.remove_task(task)
            if max_size_mb > 0:
                _enforce_size_limit(lightbox_path, max_size_mb, compress_strategy, progress)

            # Step 4: Spatial fan GLB (inside the series folder)
            spatial_path = series_dir / "spatial.glb"
            task = progress.add_task("Building spatial fan...", total=None)
            spatial_created = build_spatial_glb(
                slices, spatial_path, animate=animate,
            )
            progress.remove_task(task)
            if max_size_mb > 0 and spatial_created:
                _enforce_size_limit(spatial_path, max_size_mb, compress_strategy, progress)

        summary = {
            "name": series_name,
            "dir": series_dir,
            "slices": len(slices),
            "individual": len(individual_paths),
            "animated": animate,
            "spatial": spatial_created,
        }
        series_summaries.append(summary)

    # Summary
    elapsed = time.time() - start_time
    console.print(f"\n[green]Gallery mode complete![/green]")
    console.print(f"  Series:     {len(series_summaries)}")
    console.print(f"  Slices:     {total_slices}")
    for s in series_summaries:
        console.print(f"\n  [bold]{s['name']}/[/bold]")
        console.print(f"    Individual: {s['individual']} files")
        console.print(f"    Lightbox:   lightbox.glb")
        if s["spatial"]:
            console.print(f"    Spatial:    spatial.glb")
        else:
            console.print(f"    Spatial:    [dim]skipped (no spatial metadata)[/dim]")
        console.print(f"    Animated:   {'Yes' if s['animated'] else 'No'}")
    console.print(f"\n  Output:     {output_dir}")
    console.print(f"  Time:       {elapsed:.1f}s")


def _sanitize_name(name: str) -> str:
    """Sanitize a string for use as a directory/file name."""
    import re
    clean = re.sub(r"[^\w\s-]", "", name).strip()
    clean = re.sub(r"[\s]+", "_", clean)
    return clean or "series"


def _data_type_label(modality: str, data_type: str) -> str:
    """Create a physician-friendly label from modality and data type."""
    modality_names = {
        "US": "Echo",
        "MR": "MRI",
        "CT": "CT",
        "XA": "Angio",
        "NM": "Nuclear",
    }
    clinical = modality_names.get(modality, modality)
    dim_label = {
        "2D cine": "2D_animated",
        "3D volume": "3D",
        "3D+T volume": "3D_animated",
        "still image": "2D",
    }
    dt = dim_label.get(data_type, data_type.replace(" ", "_"))
    return f"{clinical}_{dt}"


def _get_data_type_label(input_path: Path, series_uid: str | None) -> str:
    """Analyze input to produce a data type label for auto-naming output files."""
    try:
        if input_path.is_file():
            import pydicom

            ds = pydicom.dcmread(str(input_path), stop_before_pixels=True)
            modality = getattr(ds, "Modality", "unknown")
            n_frames = int(getattr(ds, "NumberOfFrames", 1))
            if n_frames > 1:
                return _data_type_label(modality, "2D cine")
            return _data_type_label(modality, "still image")

        from med2glb.io.dicom_reader import analyze_series

        series_list = analyze_series(input_path)
        if not series_list:
            return ""

        if series_uid:
            for info in series_list:
                if series_uid in info.series_uid:
                    return _data_type_label(info.modality, info.data_type)

        return _data_type_label(series_list[0].modality, series_list[0].data_type)
    except Exception:
        return ""


def _run_animated_pipeline(data, converter, params, alpha, progress):
    """Run the animated conversion pipeline for temporal data."""
    from med2glb.core.types import AnimatedResult
    from med2glb.core.volume import TemporalSequence
    from med2glb.mesh.processing import process_mesh
    from med2glb.mesh.temporal import build_morph_targets_from_frames

    if not isinstance(data, TemporalSequence):
        raise ValueError("Animation requires temporal data")

    task = progress.add_task(
        f"Converting {data.frame_count} frames...", total=data.frame_count
    )

    # Convert each frame
    frame_results = []
    for i, frame in enumerate(data.frames):
        progress.update(task, description=f"Converting frame {i + 1}/{data.frame_count}...")
        result = converter.convert(frame, params)
        processed = []
        for mesh in result.meshes:
            processed.append(
                process_mesh(
                    mesh,
                    smoothing_iterations=params.smoothing_iterations,
                    target_faces=params.target_faces,
                )
            )
        result.meshes = processed
        frame_results.append(result)
        progress.update(task, advance=1)
    progress.remove_task(task)

    # Build morph targets
    task = progress.add_task("Building morph targets...", total=None)
    animated = build_morph_targets_from_frames(frame_results, data.temporal_resolution)
    progress.remove_task(task)

    # Apply alpha
    if alpha < 1.0:
        for mesh in animated.base_meshes:
            mesh.material = MaterialConfig(
                base_color=mesh.material.base_color,
                alpha=alpha,
                metallic=mesh.material.metallic,
                roughness=mesh.material.roughness,
                name=mesh.material.name,
            )

    return animated


def _export(result, output: Path, format: str, animate: bool) -> None:
    """Export conversion result to file."""
    from med2glb.core.types import AnimatedResult
    from med2glb.io.exporters import export_glb, export_obj, export_stl

    if format == "glb":
        if isinstance(result, AnimatedResult) and animate:
            from med2glb.glb.animation import build_animated_glb
            build_animated_glb(result, output)
        else:
            export_glb(result.meshes, output)
    elif format == "stl":
        export_stl(result.meshes, output)
    elif format == "obj":
        export_obj(result.meshes, output)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _parse_multi_threshold(spec: str) -> list[ThresholdLayer]:
    """Parse multi-threshold CLI string: 'val:label:alpha,val:label:alpha,...'."""
    layers = []
    for part in spec.split(","):
        parts = part.strip().split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid multi-threshold format: '{part}'. Expected: 'value:label:alpha'"
            )
        val, label, a = parts
        layers.append(
            ThresholdLayer(
                threshold=float(val),
                label=label,
                material=MaterialConfig(alpha=float(a), name=label),
            )
        )
    return layers
