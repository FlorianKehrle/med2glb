"""CLI entry point for dicom2glb."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from dicom2glb import __version__
from dicom2glb.core.types import MaterialConfig, MethodParams, ThresholdLayer

app = typer.Typer(
    name="dicom2glb",
    help="Convert DICOM medical imaging data to GLB 3D models.",
    add_completion=False,
)
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

    # Handle --list-series
    if do_list_series:
        from dicom2glb.io.dicom_reader import list_series

        series_list = list_series(input_path)
        table = Table(title="DICOM Series")
        table.add_column("Series UID", style="cyan")
        table.add_column("Modality", style="green")
        table.add_column("Description")
        table.add_column("Slices", justify="right")
        for s in series_list:
            table.add_row(
                s["series_uid"], s["modality"], s["description"], str(s["slice_count"])
            )
        console.print(table)
        raise typer.Exit()

    try:
        _run_pipeline(
            input_path=input_path,
            output=output,
            method_name=method,
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


def _run_pipeline(
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
    series: str | None,
    verbose: bool,
) -> None:
    """Execute the full conversion pipeline."""
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
        input_type, data = load_dicom_directory(input_path, series_uid=series)
        progress.update(task, description=f"Loaded {input_type.value} data")
        progress.remove_task(task)

        # Step 2: Get conversion method
        _ensure_methods_loaded()
        converter = get_method(method_name)

        # Step 3: Convert
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
