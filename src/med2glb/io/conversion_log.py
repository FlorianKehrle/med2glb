"""Append-only conversion log for med2glb pipeline runs.

Each conversion appends a human-readable entry to ``med2glb_log.txt``
in the output directory.  Existing content is preserved.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from med2glb._utils import fmt_duration as _format_duration


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
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    estimated_time: str | None = None,
    step_times: dict[str, float] | None = None,
    data_coverage_pct: float | None = None,
    equivalent_command: str | None = None,
) -> None:
    """Append a CARTO conversion entry to the log file."""
    _start = start_time or datetime.now()
    _end = end_time or datetime.now()

    lines: list[str] = []
    lines.append(f"{'=' * 60}")
    lines.append(f"CARTO Conversion — {_start.strftime('%Y-%m-%d %H:%M:%S')}")
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
    if data_coverage_pct is not None and data_coverage_pct < 100.0:
        lines.append(f"  Data coverage:   {data_coverage_pct:.0f}% (gray = no data within range)")
    lines.append(f"  Vertices:        {active_vertices:,} active / {total_vertices:,} total")
    lines.append(f"  Faces:           {face_count:,}")
    if estimated_time:
        lines.append(f"  Estimated time:  {estimated_time}")
    lines.append(f"  Started:         {_start.strftime('%H:%M:%S')}")
    lines.append(f"  Finished:        {_end.strftime('%H:%M:%S')}")
    lines.append(f"  Computing time:  {_format_duration(elapsed_seconds)}")
    if step_times:
        parts = []
        for label in ("Subdivide", "Mapping", "xatlas", "Rasterize", "Textures", "KTX2"):
            if label in step_times and step_times[label] >= 0.5:
                parts.append(f"{label} {_format_duration(step_times[label])}")
        if parts:
            lines.append(f"                   ({', '.join(parts)})")
    lines.append(f"  Output files:")
    for do_animate, do_vectors, out_path in variant_outputs:
        label = "static"
        if do_animate:
            label = "animated + vectors" if do_vectors else "animated"
        size_kb = out_path.stat().st_size / 1024 if out_path.exists() else 0
        lines.append(f"    {out_path.name}  ({size_kb:.0f} KB, {label})")
    if equivalent_command:
        lines.append(f"  Equivalent command:")
        lines.append(f"    {equivalent_command}")
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
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    warnings: list[str] | None = None,
    equivalent_command: str | None = None,
) -> None:
    """Append a DICOM conversion entry to the log file."""
    _start = start_time or datetime.now()
    _end = end_time or datetime.now()

    lines: list[str] = []
    lines.append(f"{'=' * 60}")
    lines.append(f"DICOM Conversion — {_start.strftime('%Y-%m-%d %H:%M:%S')}")
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
    lines.append(f"  Started:         {_start.strftime('%H:%M:%S')}")
    lines.append(f"  Finished:        {_end.strftime('%H:%M:%S')}")
    lines.append(f"  Computing time:  {_format_duration(elapsed_seconds)}")
    lines.append(f"  Output file:")
    lines.append(f"    {output_path.name}  ({file_size_kb:.0f} KB)")
    if warnings:
        lines.append(f"  Warnings:")
        for w in warnings:
            lines.append(f"    - {w}")
    if equivalent_command:
        lines.append(f"  Equivalent command:")
        lines.append(f"    {equivalent_command}")
    lines.append("")

    log_path = output_dir / "med2glb_log.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
