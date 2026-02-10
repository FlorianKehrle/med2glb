"""Mesh processing: Taubin smoothing, decimation, hole filling, normals."""

from __future__ import annotations

import numpy as np
import trimesh

from dicom2glb.core.types import MeshData


def taubin_smooth(mesh: MeshData, iterations: int = 15) -> MeshData:
    """Apply Taubin smoothing to preserve volume.

    Taubin smoothing alternates between positive and negative Laplacian
    smoothing to prevent volume shrinkage that occurs with pure Laplacian.
    """
    if iterations <= 0:
        return mesh

    tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    trimesh.smoothing.filter_taubin(tri, iterations=iterations)

    return MeshData(
        vertices=np.array(tri.vertices, dtype=np.float32),
        faces=np.array(tri.faces, dtype=np.int32),
        normals=None,
        structure_name=mesh.structure_name,
        material=mesh.material,
    )


def decimate(mesh: MeshData, target_faces: int = 80000) -> MeshData:
    """Reduce triangle count via simplification."""
    current_faces = len(mesh.faces)
    if current_faces <= target_faces:
        return mesh

    tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)

    try:
        import fast_simplification

        verts_out, faces_out = fast_simplification.simplify(
            np.array(tri.vertices, dtype=np.float64),
            np.array(tri.faces, dtype=np.int32),
            target_count=target_faces,
        )
        decimated = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=False)
    except ImportError:
        target_reduction = 1.0 - target_faces / current_faces
        decimated = tri.simplify_quadric_decimation(target_reduction)

    return MeshData(
        vertices=np.array(decimated.vertices, dtype=np.float32),
        faces=np.array(decimated.faces, dtype=np.int32),
        normals=None,
        structure_name=mesh.structure_name,
        material=mesh.material,
    )


def fill_holes(mesh: MeshData) -> MeshData:
    """Fill holes in the mesh using trimesh."""
    tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    trimesh.repair.fill_holes(tri)

    return MeshData(
        vertices=np.array(tri.vertices, dtype=np.float32),
        faces=np.array(tri.faces, dtype=np.int32),
        normals=None,
        structure_name=mesh.structure_name,
        material=mesh.material,
    )


def compute_normals(mesh: MeshData) -> MeshData:
    """Recalculate vertex normals."""
    tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    normals = np.array(tri.vertex_normals, dtype=np.float32)

    return MeshData(
        vertices=mesh.vertices,
        faces=mesh.faces,
        normals=normals,
        structure_name=mesh.structure_name,
        material=mesh.material,
    )


def process_mesh(
    mesh: MeshData,
    smoothing_iterations: int = 15,
    target_faces: int = 80000,
    do_fill_holes: bool = True,
) -> MeshData:
    """Full mesh processing pipeline: smooth -> decimate -> fill holes -> normals."""
    result = taubin_smooth(mesh, iterations=smoothing_iterations)
    result = decimate(result, target_faces=target_faces)
    if do_fill_holes:
        result = fill_holes(result)
    result = compute_normals(result)
    return result
