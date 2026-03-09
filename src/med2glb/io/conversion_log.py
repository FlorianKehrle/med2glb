"""Append-only conversion log for med2glb pipeline runs.

Each conversion appends a human-readable entry to ``med2glb_log.txt``
in the output directory.  Existing content is preserved.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    if mins < 60:
        return f"{mins}m {secs}s"
    hours = mins // 60
    mins = mins % 60
    return f"{hours}h {mins}m"


def append_carto_entry(
    output_dir: Path,
    *,
    structure_name: str,
    carto_version: str,
    study_name: str,
    coloring: str,
    color_range: str,
    subdivide: int,
    active_vertices: int,
    total_vertices: int,
    face_count: int,
    mapping_points: int,
    variant_outputs: list[tuple[bool, bool, Path]],
    elapsed_seconds: float,
    source_path: str,
) -> None:
    """Append a CARTO conversion entry to the log file."""
    lines: list[str] = []
    lines.append(f"{'=' * 60}")
    lines.append(f"CARTO Conversion — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"{'=' * 60}")
    lines.append(f"  Source:          {source_path}")
    lines.append(f"  Structure:       {structure_name}")
    lines.append(f"  System:          {carto_version}")
    if study_name:
        lines.append(f"  Study:           {study_name}")
    lines.append(f"  Coloring:        {coloring}")
    if color_range:
        lines.append(f"  Color range:     {color_range}")
    if subdivide > 0:
        lines.append(f"  Subdivision:     level {subdivide} (~{4 ** subdivide}x face increase)")
    lines.append(f"  Mapping points:  {mapping_points:,}")
    lines.append(f"  Vertices:        {active_vertices:,} active / {total_vertices:,} total")
    lines.append(f"  Faces:           {face_count:,}")
    lines.append(f"  Computing time:  {_format_duration(elapsed_seconds)}")
    lines.append(f"  Output files:")
    for do_animate, do_vectors, out_path in variant_outputs:
        label = "static"
        if do_animate:
            label = "animated + vectors" if do_vectors else "animated"
        size_kb = out_path.stat().st_size / 1024 if out_path.exists() else 0
        lines.append(f"    {out_path.name}  ({size_kb:.0f} KB, {label})")
    lines.append("")

    log_path = output_dir / "med2glb_log.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def append_dicom_entry(
    output_dir: Path,
    *,
    output_path: Path,
    method_name: str,
    input_path: str,
    input_type: str,
    series_uid: str | None,
    mesh_count: int,
    vertex_count: int,
    face_count: int,
    file_size_kb: float,
    animated: bool,
    elapsed_seconds: float,
    warnings: list[str] | None = None,
) -> None:
    """Append a DICOM conversion entry to the log file."""
    lines: list[str] = []
    lines.append(f"{'=' * 60}")
    lines.append(f"DICOM Conversion — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"{'=' * 60}")
    lines.append(f"  Source:          {input_path}")
    lines.append(f"  Input type:      {input_type}")
    lines.append(f"  Method:          {method_name}")
    if series_uid:
        lines.append(f"  Series UID:      {series_uid}")
    lines.append(f"  Animated:        {'yes' if animated else 'no'}")
    lines.append(f"  Meshes:          {mesh_count}")
    lines.append(f"  Vertices:        {vertex_count:,}")
    lines.append(f"  Faces:           {face_count:,}")
    lines.append(f"  Computing time:  {_format_duration(elapsed_seconds)}")
    lines.append(f"  Output file:")
    lines.append(f"    {output_path.name}  ({file_size_kb:.0f} KB)")
    if warnings:
        lines.append(f"  Warnings:")
        for w in warnings:
            lines.append(f"    - {w}")
    lines.append("")

    log_path = output_dir / "med2glb_log.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
