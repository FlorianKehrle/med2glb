"""Multi-format exporter: GLB, STL, OBJ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from dicom2glb.core.types import MeshData
from dicom2glb.glb.builder import build_glb


def export_glb(meshes: list[MeshData], output_path: Path) -> None:
    """Export meshes to GLB format with PBR materials."""
    build_glb(meshes, output_path)


def export_stl(meshes: list[MeshData], output_path: Path) -> None:
    """Export meshes to STL format (merged into single mesh)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined = _combine_meshes(meshes)
    tri = trimesh.Trimesh(vertices=combined.vertices, faces=combined.faces, process=False)
    tri.export(str(output_path), file_type="stl")


def export_obj(meshes: list[MeshData], output_path: Path) -> None:
    """Export meshes to OBJ format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(meshes) == 1:
        tri = trimesh.Trimesh(
            vertices=meshes[0].vertices, faces=meshes[0].faces, process=False
        )
        tri.export(str(output_path), file_type="obj")
    else:
        # Export as scene with named meshes
        scene = trimesh.Scene()
        for mesh in meshes:
            tri = trimesh.Trimesh(
                vertices=mesh.vertices, faces=mesh.faces, process=False
            )
            scene.add_geometry(tri, node_name=mesh.structure_name)
        scene.export(str(output_path), file_type="obj")


def _combine_meshes(meshes: list[MeshData]) -> MeshData:
    """Combine multiple meshes into one (for STL export)."""
    if len(meshes) == 1:
        return meshes[0]

    all_verts = []
    all_faces = []
    offset = 0

    for mesh in meshes:
        all_verts.append(mesh.vertices)
        all_faces.append(mesh.faces + offset)
        offset += len(mesh.vertices)

    return MeshData(
        vertices=np.concatenate(all_verts, axis=0),
        faces=np.concatenate(all_faces, axis=0),
        structure_name="combined",
    )
