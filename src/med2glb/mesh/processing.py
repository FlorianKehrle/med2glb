"""Mesh processing: Taubin smoothing, decimation, hole filling, normals."""

from __future__ import annotations

import numpy as np
import trimesh

from med2glb.core.types import MeshData


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
    """Reduce triangle count via simplification.

    When the input mesh has ``vertex_colors``, they are preserved by
    resampling from the original vertices via nearest-neighbour KDTree
    lookup (decimation algorithms don't track per-vertex attributes).
    """
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

    new_verts = np.array(decimated.vertices, dtype=np.float32)

    # Preserve vertex colors via nearest-neighbour resampling
    new_colors = None
    if mesh.vertex_colors is not None:
        from scipy.spatial import KDTree

        tree = KDTree(mesh.vertices)
        _, idx = tree.query(new_verts)
        new_colors = mesh.vertex_colors[idx]

    return MeshData(
        vertices=new_verts,
        faces=np.array(decimated.faces, dtype=np.int32),
        normals=None,
        vertex_colors=new_colors,
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


def remove_degenerate(mesh: MeshData) -> MeshData:
    """Remove needle-like faces and disconnected fragments.

    1. Drop faces whose longest edge is >10x the median edge length
       (these create visible spikes from disconnected segmentation blobs).
    2. Keep only the largest connected component to discard small fragments.
    """
    tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)

    if len(tri.faces) == 0:
        return mesh

    # --- Remove faces with extremely long edges ---
    v0 = tri.vertices[tri.faces[:, 0]]
    v1 = tri.vertices[tri.faces[:, 1]]
    v2 = tri.vertices[tri.faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    max_edge = np.maximum(e0, np.maximum(e1, e2))
    median_edge = np.median(max_edge)

    if median_edge > 0:
        keep_mask = max_edge < median_edge * 10
        if keep_mask.sum() < len(tri.faces):
            tri.update_faces(keep_mask)
            tri.remove_unreferenced_vertices()

    # --- Keep largest connected component ---
    if len(tri.faces) > 0:
        components = trimesh.graph.connected_components(tri.face_adjacency, min_len=1)
        if len(components) > 1:
            largest = max(components, key=len)
            mask = np.zeros(len(tri.faces), dtype=bool)
            mask[largest] = True
            tri.update_faces(mask)
            tri.remove_unreferenced_vertices()

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
    """Full mesh processing pipeline: smooth -> decimate -> cleanup -> fill holes -> normals."""
    result = taubin_smooth(mesh, iterations=smoothing_iterations)
    result = decimate(result, target_faces=target_faces)
    result = remove_degenerate(result)
    if do_fill_holes:
        result = fill_holes(result)
    result = compute_normals(result)
    return result
