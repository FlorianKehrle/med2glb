"""CLI entry point for dicom2glb."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from dicom2glb import __version__
from dicom2glb.core.types import (
    MaterialConfig,
    MethodParams,
    SeriesInfo,
    ThresholdLayer,
)

app = typer.Typer(
    name="dicom2glb",
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

logger = logging.getLogger("dicom2glb")


def version_callback(value: bool):
    if value:
        console.print(f"dicom2glb {__version__}")
        raise typer.Exit()


def list_methods_callback(value: bool):
    if value:
        from dicom2glb.methods.registry import _ensure_methods_loaded, list_methods

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
        "output.glb",
        "-o",
        "--output",
        help="Output file path.",
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
        from dicom2glb.io.dicom_reader import analyze_series

        series_list = analyze_series(input_path)
        _print_series_table(series_list, input_path)
        raise typer.Exit()

    try:
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
            "Install AI dependencies with: pip install dicom2glb[ai]"
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
            verbose=verbose,
        )
        return

    # Directory input — analyze series
    from dicom2glb.io.dicom_reader import analyze_series

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
            verbose=verbose,
        )


def _print_series_table(series_list: list[SeriesInfo], input_path: Path) -> None:
    """Display a Rich table of DICOM series with classification info."""
    table = Table(title=f"DICOM Series in {input_path}")
    table.add_column("#", style="bold", justify="right")
    table.add_column("Modality", style="green")
    table.add_column("Description")
    table.add_column("Data Type", style="cyan")
    table.add_column("Detail", justify="right")
    table.add_column("Recommended", style="yellow")

    for i, info in enumerate(series_list, 1):
        desc = info.description if info.description else "(no desc)"
        # Truncate long descriptions
        if len(desc) > 20:
            desc = desc[:17] + "..."
        table.add_row(
            str(i),
            info.modality,
            desc,
            info.data_type,
            info.detail,
            info.recommended_output,
        )

    console.print(table)

    # Print recommendation
    best = series_list[0]
    console.print(
        f"\nRecommendation: Series 1 ({best.data_type}, {best.detail})"
    )


def _interactive_select_series(series_list: list[SeriesInfo]) -> list[SeriesInfo]:
    """Prompt user to select series interactively. Returns selected SeriesInfo list."""
    choice = Prompt.ask(
        "Select series to convert",
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
    verbose: bool,
) -> None:
    """Execute the full conversion pipeline for a single series."""
    from dicom2glb.io.dicom_reader import InputType, load_dicom_directory
    from dicom2glb.methods.registry import _ensure_methods_loaded, get_method

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
            from dicom2glb.core.volume import TemporalSequence
            if isinstance(data, TemporalSequence) and data.frames[0].voxels.shape[0] == 1:
                # 2D cine without --animate — export first frame as textured plane
                task = progress.add_task("Building textured plane from 2D echo...", total=None)
                from dicom2glb.glb.texture import build_textured_plane_glb

                console.print(
                    f"[yellow]2D ultrasound cine detected ({data.frame_count} frames). "
                    f"Exporting first frame as textured plane. "
                    f"Use --animate for animated output.[/yellow]"
                )
                build_textured_plane_glb(data.frames[0], output)
                progress.remove_task(task)

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
            from dicom2glb.glb.texture import build_textured_plane_glb

            build_textured_plane_glb(data, output)
            progress.remove_task(task)

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
            from dicom2glb.core.volume import TemporalSequence
            if isinstance(data, TemporalSequence) and data.frames[0].voxels.shape[0] == 1:
                # 2D cine + --animate → animated height-map textured plane
                task = progress.add_task(
                    f"Building animated surface from {data.frame_count} frames...", total=None
                )
                from dicom2glb.glb.texture import build_animated_textured_plane_glb

                build_animated_textured_plane_glb(data, output)
                progress.remove_task(task)

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
                from dicom2glb.core.volume import TemporalSequence
                if isinstance(data, TemporalSequence):
                    data = data.frames[0]
                    console.print(
                        "[yellow]Temporal data detected but --animate not set. "
                        "Using first frame only.[/yellow]"
                    )

            task = progress.add_task(f"Converting with {method_name}...", total=None)
            result = converter.convert(data, params)
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
            from dicom2glb.mesh.processing import process_mesh

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


def _run_animated_pipeline(data, converter, params, alpha, progress):
    """Run the animated conversion pipeline for temporal data."""
    from dicom2glb.core.types import AnimatedResult
    from dicom2glb.core.volume import TemporalSequence
    from dicom2glb.mesh.processing import process_mesh
    from dicom2glb.mesh.temporal import build_morph_targets_from_frames

    if not isinstance(data, TemporalSequence):
        raise ValueError("Animation requires temporal data")

    task = progress.add_task(
        f"Converting {data.frame_count} frames...", total=data.frame_count
    )

    # Convert each frame
    frame_results = []
    for i, frame in enumerate(data.frames):
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
    from dicom2glb.core.types import AnimatedResult
    from dicom2glb.io.exporters import export_glb, export_obj, export_stl

    if format == "glb":
        if isinstance(result, AnimatedResult) and animate:
            from dicom2glb.glb.animation import build_animated_glb
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
