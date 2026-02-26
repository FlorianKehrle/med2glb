"""Gallery pipeline: individual GLBs, lightbox grid, and spatial fan."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from med2glb._console import console
from med2glb._pipeline_dicom import (
    enforce_size_limit,
    print_series_table,
    _interactive_select_series,
)
from med2glb.core.types import SeriesInfo


def _sanitize_name(name: str) -> str:
    """Sanitize a string for use as a directory/file name."""
    clean = re.sub(r"[^\w\s-]", "", name).strip()
    clean = re.sub(r"[\s]+", "_", clean)
    return clean or "series"


def run_gallery_mode(
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
            print_series_table(series_list, input_path)
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
                    enforce_size_limit(p, max_size_mb, compress_strategy, progress)

            # Step 3: Lightbox GLB (inside the series folder)
            lightbox_path = series_dir / "lightbox.glb"
            task = progress.add_task("Building lightbox grid...", total=None)
            build_lightbox_glb(
                slices, lightbox_path, columns=columns, animate=animate,
            )
            progress.remove_task(task)
            if max_size_mb > 0:
                enforce_size_limit(lightbox_path, max_size_mb, compress_strategy, progress)

            # Step 4: Spatial fan GLB (inside the series folder)
            spatial_path = series_dir / "spatial.glb"
            task = progress.add_task("Building spatial fan...", total=None)
            spatial_created = build_spatial_glb(
                slices, spatial_path, animate=animate,
            )
            progress.remove_task(task)
            if max_size_mb > 0 and spatial_created:
                enforce_size_limit(spatial_path, max_size_mb, compress_strategy, progress)

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
