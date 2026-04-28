"""DICOM pipeline: series selection, conversion, and export."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.table import Table

from med2glb._console import console, err_console
from med2glb.core.types import (
    ConversionStats,
    MaterialConfig,
    MethodParams,
    SeriesInfo,
    ThresholdLayer,
)

logger = logging.getLogger("med2glb")


def print_series_table(series_list: list[SeriesInfo], input_path: Path) -> None:
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
    return parse_selection(choice, series_list)


def parse_selection(choice: str, series_list: list[SeriesInfo]) -> list[SeriesInfo]:
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


def make_output_path(base: Path, index: int, total: int) -> Path:
    """Generate output path for multi-series conversion.

    Single series: output.glb
    Multiple series: output_1.glb, output_2.glb, ...
    """
    if total <= 1:
        return base
    return base.parent / f"{base.stem}_{index + 1}{base.suffix}"


def enforce_size_limit(
    path: Path,
    max_size_mb: int,
    strategy: str,
    progress: Progress,
) -> None:
    """Compress a GLB file if it exceeds the size limit.

    Tries texture-based compression (for GLBs with textures).
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


def _export(result, output: Path, format: str, animate: bool, model_type: str = "dicom3d") -> None:
    """Export conversion result to file."""
    from med2glb.core.types import AnimatedResult
    from med2glb.io.exporters import export_glb

    if format != "glb":
        raise ValueError(f"Unsupported format: {format}")

    if isinstance(result, AnimatedResult) and animate:
        from med2glb.glb.animation import build_animated_glb
        build_animated_glb(result, output, model_type=model_type)
    else:
        export_glb(result.meshes, output, model_type=model_type)


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


def run_dicom_from_config(config: "DicomConfig", output: Path) -> None:
    """Execute the DICOM pipeline from a wizard-produced config."""
    from med2glb.cli_wizard import build_dicom_equiv_command
    equiv_cmd = build_dicom_equiv_command(config, output)

    if config.gallery:
        from med2glb._pipeline_gallery import run_gallery_mode
        # Gallery outputs a directory, not a single .glb
        gallery_dir = output.parent / config.name if output.suffix else output
        run_gallery_mode(
            input_path=config.input_path,
            output=gallery_dir,
            series=config.series_uid,
            columns=6,
            no_animate=not config.animate,
            verbose=config.verbose,
        )
        console.print(f"\n[dim]💡 Equivalent command:[/dim]")
        console.print(f"[dim]   {equiv_cmd}[/dim]")
        return

    if config.method == "compare":
        run_compare_mode(config, output.parent, output.stem)
        console.print(f"\n[dim]💡 Equivalent command:[/dim]")
        console.print(f"[dim]   {equiv_cmd}[/dim]")
        return

    convert_series(
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
        verbose=config.verbose,
        equivalent_command=equiv_cmd,
    )

    console.print(f"\n[dim]💡 Equivalent command:[/dim]")
    console.print(f"[dim]   {equiv_cmd}[/dim]")


def run_pipeline(
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
    # Compare mode: run all methods and produce a summary
    if method_name == "compare":
        from med2glb.core.types import DicomConfig

        config = DicomConfig(
            input_path=input_path,
            method="compare",
            format=format,
            animate=animate,
            threshold=threshold,
            smoothing=smoothing,
            target_faces=target_faces,
            alpha=alpha,
            series_uid=series,
            verbose=verbose,
        )
        output_dir = output.parent
        base_name = output.stem
        run_compare_mode(config, output_dir, base_name)
        return

    # If --series provided or input is a single file, pass through directly
    if series or input_path.is_file():
        convert_series(
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
    from med2glb.io.dicom_reader import analyze_series

    series_list = analyze_series(input_path)

    if len(series_list) == 1:
        # Single series — proceed automatically
        info = series_list[0]
        effective_method = method_name if method_explicit else info.recommended_method
        convert_series(
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
        print_series_table(series_list, input_path)
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
        out_path = make_output_path(output, i, len(selected))
        effective_method = method_name if method_explicit else info.recommended_method
        console.print(
            f"\n[bold]Converting series {i + 1}/{len(selected)}: "
            f"{info.description or info.series_uid} ({info.data_type})[/bold]"
        )
        convert_series(
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


def convert_series(
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
    verbose: bool = False,
    equivalent_command: str | None = None,
) -> ConversionStats | None:
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
        TimeElapsedColumn(),
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

                file_size = output.stat().st_size / 1024
                elapsed = time.time() - start_time
                console.print(f"\n[green]Conversion complete![/green]")
                console.print(f"  Input:    2D echo cine ({data.frame_count} frames)")
                console.print(f"  Output:   {output}")
                console.print(f"  Size:     {file_size:.1f} KB")
                console.print(f"  Time:     {elapsed:.1f}s")
                return None

        if input_type == InputType.SINGLE_SLICE:
            task = progress.add_task("Building textured plane...", total=None)
            from med2glb.glb.texture import build_textured_plane_glb

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
            return None

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

                file_size = output.stat().st_size / 1024
                elapsed = time.time() - start_time
                console.print(f"\n[green]Conversion complete![/green]")
                console.print(f"  Input:    2D echo cine ({data.frame_count} frames)")
                console.print(f"  Output:   {output} (animated)")
                console.print(f"  Size:     {file_size:.1f} KB")
                console.print(f"  Time:     {elapsed:.1f}s")
                return None
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

    # Append to conversion log
    from datetime import datetime
    from med2glb.io.conversion_log import append_dicom_entry
    append_dicom_entry(
        output.parent,
        output_path=output,
        method_name=method_name,
        input_path=str(input_path),
        input_type=input_type.value,
        series_uid=series_uid,
        mesh_count=len(result.meshes),
        vertex_count=total_verts,
        face_count=total_faces,
        file_size_kb=file_size,
        animated=animate,
        elapsed_seconds=elapsed,
        start_time=datetime.fromtimestamp(start_time),
        end_time=datetime.fromtimestamp(start_time + elapsed),
        warnings=result.warnings or None,
        equivalent_command=equivalent_command,
    )

    return ConversionStats(
        method_name=method_name,
        output_path=output,
        file_size_kb=file_size,
        vertex_count=total_verts,
        face_count=total_faces,
        mesh_count=len(result.meshes),
        elapsed_seconds=elapsed,
    )


def run_compare_mode(
    config: "DicomConfig",
    output_dir: Path,
    base_name: str,
) -> list[ConversionStats]:
    """Run all available conversion methods and produce a comparison summary.

    Each method writes to ``<output_dir>/<base_name>_<method_short>.glb``.
    Methods that fail are recorded with ``success=False`` and skipped.
    """
    from med2glb.methods.registry import _ensure_methods_loaded, list_methods

    _ensure_methods_loaded()
    methods = list_methods()
    available = [m for m in methods if m["available"]]

    if not available:
        err_console.print("[red]No conversion methods available.[/red]")
        return []

    console.print(f"\n[bold cyan]Compare Mode[/bold cyan]")
    console.print(f"  Methods:  {', '.join(m['name'] for m in available)}")
    console.print(f"  Output:   {output_dir}/")
    console.print()

    output_dir.mkdir(parents=True, exist_ok=True)
    stats: list[ConversionStats] = []

    for m in available:
        method_name = m["name"]
        method_short = method_name.replace("marching-cubes", "mc")
        out_path = output_dir / f"{base_name}_{method_short}.glb"

        console.print(f"\n[bold]--- {method_name} ---[/bold]")
        try:
            result = convert_series(
                input_path=config.input_path,
                output=out_path,
                method_name=method_name,
                format=config.format,
                animate=config.animate,
                threshold=config.threshold,
                smoothing=config.smoothing,
                target_faces=config.target_faces,
                alpha=config.alpha,
                multi_threshold=None,
                series_uid=config.series_uid,
                verbose=config.verbose,
            )
            if result is not None:
                stats.append(result)
            else:
                # 2D / single-slice early returns — record basic stats
                file_size = out_path.stat().st_size / 1024 if out_path.exists() else 0
                stats.append(ConversionStats(
                    method_name=method_name,
                    output_path=out_path,
                    file_size_kb=file_size,
                    vertex_count=0,
                    face_count=0,
                    mesh_count=0,
                    elapsed_seconds=0,
                ))
        except Exception as exc:
            console.print(f"[red]  {method_name} failed: {exc}[/red]")
            stats.append(ConversionStats(
                method_name=method_name,
                output_path=out_path,
                file_size_kb=0,
                vertex_count=0,
                face_count=0,
                mesh_count=0,
                elapsed_seconds=0,
                success=False,
                error=str(exc),
            ))

    _print_compare_table(stats)
    return stats


def _print_compare_table(stats: list[ConversionStats]) -> None:
    """Print a Rich comparison table of conversion results."""
    table = Table(title="Comparison Results")
    table.add_column("Method", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("File Size", justify="right")
    table.add_column("Vertices", justify="right")
    table.add_column("Faces", justify="right")
    table.add_column("Meshes", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Output")

    for s in stats:
        if s.success:
            status = "[green]OK[/green]"
            size = f"{s.file_size_kb:.0f} KB"
            verts = f"{s.vertex_count:,}" if s.vertex_count else "-"
            faces = f"{s.face_count:,}" if s.face_count else "-"
            meshes = str(s.mesh_count) if s.mesh_count else "-"
            elapsed = f"{s.elapsed_seconds:.1f}s"
            output = str(s.output_path)
        else:
            status = "[red]FAIL[/red]"
            size = "-"
            verts = "-"
            faces = "-"
            meshes = "-"
            elapsed = "-"
            output = s.error

        table.add_row(s.method_name, status, size, verts, faces, meshes, elapsed, output)

    console.print(f"\n")
    console.print(table)
