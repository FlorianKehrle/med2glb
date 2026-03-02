"""GLB exporter."""

from __future__ import annotations

from pathlib import Path

from med2glb.core.types import MeshData
from med2glb.glb.builder import build_glb


def export_glb(meshes: list[MeshData], output_path: Path) -> None:
    """Export meshes to GLB format with PBR materials."""
    build_glb(meshes, output_path)
