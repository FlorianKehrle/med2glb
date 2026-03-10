"""CLI entry point for med2glb."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import typer
from rich.prompt import Prompt
from rich.table import Table

from med2glb import __version__
from med2glb._console import console, err_console

# Re-export for backward compatibility (used by tests)
from med2glb._pipeline_dicom import make_output_path as _make_output_path  # noqa: F401
from med2glb._pipeline_dicom import parse_selection as _parse_selection  # noqa: F401

app = typer.Typer(
    name="med2glb",
    help="Convert DICOM medical imaging data to GLB 3D models.",
    add_completion=False,
    invoke_without_command=True,
)

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


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_path: Path = typer.Argument(
        None,
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
        help="Conversion method: marching-cubes, classical, totalseg, chamber-detect, compare.",
    ),
    format: str = typer.Option(
        "glb",
        "-f",
        "--format",
        help="Output format.",
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
        "all",
        "--coloring",
        help="CARTO coloring scheme filter: lat, bipolar, unipolar, or all (default). "
             "When 'all', every available coloring is produced.",
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
    do_list_series: bool = typer.Option(
        False,
        "--list-series",
        help="List DICOM series found in input directory and exit.",
    ),
    batch: bool = typer.Option(
        False,
        "--batch",
        help="Batch mode: find all CARTO exports in subdirectories and convert with shared settings.",
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
    # If a subcommand is being invoked, skip the main logic
    if ctx.invoked_subcommand is not None:
        return

    # input_path is required when running the main command (not subcommands)
    if input_path is None:
        err_console.print("[red]Error: Missing argument 'INPUT_PATH'.[/red]")
        raise typer.Exit(code=2)

    from med2glb._pipeline_carto import run_carto_from_config, run_carto_pipeline
    from med2glb._pipeline_dicom import run_dicom_from_config, run_pipeline, print_series_table
    from med2glb._pipeline_gallery import run_gallery_mode

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Detect if --method was explicitly provided
    method_explicit = _was_option_provided(ctx, "method")

    # Handle --list-series
    if do_list_series:
        from med2glb.io.dicom_reader import analyze_series

        series_list = analyze_series(input_path)
        print_series_table(series_list, input_path)
        raise typer.Exit()

    # --- Interactive wizard ---
    # If no pipeline flags were explicitly set and we have a TTY, run the wizard
    from med2glb.cli_wizard import is_interactive as _is_interactive
    if _is_interactive() and not _has_pipeline_flags(ctx):
        try:
            from med2glb.cli_wizard import (
                analyze_input, run_carto_wizard, run_dicom_wizard, scan_directory,
            )

            # --- Universal directory scan ---
            if input_path.is_dir():
                entries = scan_directory(input_path, console)
                n_carto = sum(1 for e in entries if e.kind == "carto")
                n_dicom = sum(1 for e in entries if e.kind == "dicom")

                if len(entries) > 1 or (n_carto >= 1 and n_dicom >= 1):
                    # Multiple datasets or mixed types — show overview
                    console.print(f"\n[bold cyan]Directory Scan: {input_path.name}[/bold cyan]")
                    if n_carto:
                        console.print(f"  CARTO:  {n_carto} export(s)")
                    if n_dicom:
                        console.print(f"  DICOM:  {n_dicom} folder(s)")
                    console.print()

                    # Show overview table
                    overview = Table(title="Detected Datasets")
                    overview.add_column("#", style="bold", justify="right")
                    overview.add_column("Type")
                    overview.add_column("Name")
                    overview.add_column("Path")
                    overview.add_column("Details")
                    for i, entry in enumerate(entries, 1):
                        type_label = "[green]CARTO[/green]" if entry.kind == "carto" else "[blue]DICOM[/blue]"
                        overview.add_row(
                            str(i), type_label, entry.label, entry.location, entry.detail,
                        )
                    console.print(overview)

                    # Let user choose what to process
                    carto_entries = [e for e in entries if e.kind == "carto"]
                    dicom_entries = [e for e in entries if e.kind == "dicom"]

                    if n_carto >= 1 or n_dicom >= 1:
                        # Build choices based on what's available
                        from med2glb.io.carto_reader import load_carto_study
                        from med2glb.cli_wizard import run_batch_carto_wizard

                        choices: list[str] = []
                        if n_carto >= 1:
                            choices.append("all-carto")
                        if n_dicom >= 1:
                            choices.append("all-dicom")
                        if n_carto >= 1 and n_dicom >= 1:
                            choices.append("all")
                        choices.append("select")
                        default = "all-carto" if n_carto >= 1 else "all-dicom"

                        choice = Prompt.ask(
                            "Process",
                            choices=choices,
                            default=default,
                            console=console,
                        )

                        if choice in ("all-carto", "all"):
                            studies = []
                            for e in carto_entries:
                                try:
                                    studies.append(
                                        (e.path, load_carto_study(e.path, progress=lambda d, c, t: None))
                                    )
                                except Exception as exc:
                                    console.print(f"[yellow]  Skipping {e.label}: {exc}[/yellow]")
                            if studies:
                                configs = run_batch_carto_wizard(studies, console)
                                for j, cfg in enumerate(configs, 1):
                                    console.print(
                                        f"\n[bold]=== Dataset {j}/{len(configs)}: "
                                        f"{cfg.name} ===[/bold]"
                                    )
                                    run_carto_from_config(cfg)

                        if choice in ("all-dicom", "all"):
                            for e in dicom_entries:
                                try:
                                    from med2glb.io.dicom_reader import analyze_series
                                    series_list = analyze_series(e.path)
                                    if series_list:
                                        dicom_cfg = run_dicom_wizard(series_list, e.path, console)
                                        console.print(
                                            f"\n[bold]=== DICOM: "
                                            f"{dicom_cfg.name} ===[/bold]"
                                        )
                                        glb_dir = e.path / "glb"
                                        out_path = glb_dir / f"{dicom_cfg.name}.glb"
                                        run_dicom_from_config(dicom_cfg, out_path)
                                except Exception as exc:
                                    console.print(f"[yellow]  Skipping DICOM {e.label}: {exc}[/yellow]")

                        if choice in ("all-carto", "all-dicom", "all"):
                            return

                        if choice == "select":
                            sel = Prompt.ask(
                                "Enter dataset numbers (comma-separated, e.g. 1,3,5)",
                                console=console,
                            )
                            indices = [int(x.strip()) - 1 for x in sel.split(",") if x.strip().isdigit()]
                            selected = [entries[i] for i in indices if 0 <= i < len(entries)]
                            sel_carto = [e for e in selected if e.kind == "carto"]
                            sel_dicom = [e for e in selected if e.kind == "dicom"]

                            if sel_carto:
                                studies = []
                                for e in sel_carto:
                                    try:
                                        studies.append(
                                            (e.path, load_carto_study(e.path, progress=lambda d, c, t: None))
                                        )
                                    except Exception as exc:
                                        console.print(f"[yellow]  Skipping {e.label}: {exc}[/yellow]")
                                if studies:
                                    configs = run_batch_carto_wizard(studies, console)
                                    for j, cfg in enumerate(configs, 1):
                                        console.print(
                                            f"\n[bold]=== Dataset {j}/{len(configs)}: "
                                            f"{cfg.name} ===[/bold]"
                                        )
                                        run_carto_from_config(cfg)

                            if sel_dicom:
                                for e in sel_dicom:
                                    try:
                                        from med2glb.io.dicom_reader import analyze_series
                                        series_list = analyze_series(e.path)
                                        if series_list:
                                            dicom_cfg = run_dicom_wizard(series_list, e.path, console)
                                            glb_dir = e.path / "glb"
                                            out_path = glb_dir / f"{dicom_cfg.name}.glb"
                                            run_dicom_from_config(dicom_cfg, out_path)
                                    except Exception as exc:
                                        console.print(f"[yellow]  Skipping DICOM {e.label}: {exc}[/yellow]")
                            return

                        return

            # --- Single dataset detection (original flow) ---
            detected = analyze_input(input_path)

            if detected.kind == "carto" and detected.carto_study is not None:
                config = run_carto_wizard(
                    detected.carto_study, input_path, console,
                )
                if output is not None:
                    base = output if output.suffix == "" else output.parent
                    config.output_dir = base / "glb"
                run_carto_from_config(config)
                return
            elif detected.kind == "dicom" and detected.series_list is not None:
                dicom_config = run_dicom_wizard(
                    detected.series_list, input_path, console,
                )
                if output is not None:
                    base = output if output.suffix == "" else output.parent
                    glb_dir = base / "glb"
                else:
                    glb_dir = (input_path if input_path.is_dir() else input_path.parent) / "glb"
                out_path = glb_dir / f"{dicom_config.name}.glb"
                run_dicom_from_config(dicom_config, out_path)
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
        # --- Batch mode ---
        if batch and input_path.is_dir():
            from med2glb.io.carto_reader import find_carto_subdirectories, load_carto_study
            subdirs = find_carto_subdirectories(input_path)
            if not subdirs:
                err_console.print("[red]No CARTO exports found in subdirectories.[/red]")
                raise typer.Exit(code=1)

            if _is_interactive():
                # Batch wizard: load all studies, ask settings once
                studies = []
                for d in subdirs:
                    try:
                        studies.append((d, load_carto_study(d)))
                    except Exception as e:
                        console.print(f"[yellow]Skipping {d.name}: {e}[/yellow]")
                if not studies:
                    err_console.print("[red]No valid CARTO studies found.[/red]")
                    raise typer.Exit(code=1)

                from med2glb.cli_wizard import run_batch_carto_wizard
                configs = run_batch_carto_wizard(
                    studies, console,
                    preset_colorings=[coloring] if _was_option_provided(ctx, "coloring") and coloring != "all" else None,
                    preset_animate=animate if _was_option_provided(ctx, "animate") else None,
                    preset_static=not no_animate if _was_option_provided(ctx, "no_animate") else None,
                    preset_vectors=("yes" if vectors else None) if _was_option_provided(ctx, "vectors") else None,
                    preset_subdivide=subdivide if _was_option_provided(ctx, "subdivide") else None,
                )
                for i, cfg in enumerate(configs, 1):
                    console.print(f"\n[bold]=== Dataset {i}/{len(configs)}: {cfg.name or cfg.input_path.name} ===[/bold]")
                    run_carto_from_config(cfg)
            else:
                # Non-interactive batch: run each subdir with CLI flags
                console.print(f"[bold]Batch mode: {len(subdirs)} CARTO export(s) found[/bold]")
                for i, subdir in enumerate(subdirs, 1):
                    console.print(f"\n[bold]=== Dataset {i}/{len(subdirs)}: {subdir.name} ===[/bold]")
                    sub_output = subdir / "glb" / "output.glb"
                    run_carto_pipeline(
                        input_path=subdir,
                        output=sub_output,
                        coloring=coloring,
                        subdivide=subdivide,
                        animate=animate,
                        vectors="yes" if vectors else "no",
                        target_faces=faces,
                    )
            return

        # Auto-detect CARTO data
        if input_path.is_dir():
            from med2glb.io.carto_reader import detect_carto_directory
            if detect_carto_directory(input_path):
                # Check for multiple exports (auto-batch)
                from med2glb.io.carto_reader import find_carto_subdirectories
                subdirs = find_carto_subdirectories(input_path)
                if len(subdirs) > 1:
                    console.print(f"[bold]Auto-batch: {len(subdirs)} CARTO export(s) found[/bold]")
                    for i, subdir in enumerate(subdirs, 1):
                        console.print(f"\n[bold]=== Dataset {i}/{len(subdirs)}: {subdir.name} ===[/bold]")
                        sub_output = subdir / "glb" / "output.glb"
                        run_carto_pipeline(
                            input_path=subdir,
                            output=sub_output,
                            coloring=coloring,
                            subdivide=subdivide,
                            animate=animate,
                            vectors="yes" if vectors else "no",
                            target_faces=faces,
                        )
                    return
                run_carto_pipeline(
                    input_path=input_path,
                    output=output,
                    coloring=coloring,
                    subdivide=subdivide,
                    animate=animate,
                    vectors="yes" if vectors else "no",
                    target_faces=faces,
                )
                return

        if gallery:
            run_gallery_mode(
                input_path=input_path,
                output=output,
                series=series,
                columns=columns,
                no_animate=no_animate,
                verbose=verbose,
            )
        else:
            run_pipeline(
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
            "Install AI dependencies with: pip install med2glb[ai]"
        )
        raise typer.Exit(code=3)
    except Exception as e:
        err_console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            err_console.print(traceback.format_exc())
        raise typer.Exit(code=1)


@app.command()
def compress(
    glb_path: Path = typer.Argument(
        ...,
        exists=True,
        help="GLB file to compress.",
    ),
    max_size: int = typer.Option(
        25,
        "--max-size",
        "-s",
        help="Target size in MB.",
    ),
    strategy: str = typer.Option(
        "ktx2",
        "--strategy",
        help="Compression strategy: ktx2, draco, downscale, jpeg.",
    ),
    output: Path = typer.Option(
        None,
        "-o",
        "--output",
        help="Output path (default: compress in-place).",
    ),
):
    """Compress a GLB file to fit a target size."""
    if glb_path.suffix.lower() != ".glb":
        err_console.print(f"[red]Error: {glb_path} is not a GLB file.[/red]")
        raise typer.Exit(code=1)

    from med2glb.glb.compress import constrain_glb_size

    target_path = output if output is not None else glb_path
    if output is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(glb_path, target_path)

    original_size = target_path.stat().st_size
    max_bytes = max_size * 1024 * 1024

    if original_size <= max_bytes:
        console.print(
            f"[green]Already within limit:[/green] "
            f"{original_size / 1024:.0f} KB <= {max_size} MB"
        )
        raise typer.Exit()

    console.print(
        f"Compressing {target_path.name} "
        f"({original_size / 1024:.0f} KB → {max_size} MB target, "
        f"strategy: {strategy})..."
    )

    applied = constrain_glb_size(target_path, max_bytes, strategy=strategy)

    new_size = target_path.stat().st_size
    if applied and new_size < original_size:
        console.print(
            f"[green]Compressed:[/green] "
            f"{original_size / 1024:.0f} KB → {new_size / 1024:.0f} KB"
        )
    else:
        console.print(
            f"[yellow]No further compression possible:[/yellow] "
            f"{new_size / 1024:.0f} KB"
        )


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
        "no_animate", "vectors", "multi_threshold", "batch",
    ]
    return any(_was_option_provided(ctx, p) for p in pipeline_params)


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
