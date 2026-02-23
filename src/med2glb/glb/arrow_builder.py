"""Arrow/dash mesh generation for LAT streamline visualization.

Generates 3D cylinder+cone geometry for each dash segment and assembles
per-frame GLB nodes with scale-based visibility animation (same pattern
as carto_builder.py wavefront animation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pygltflib
from scipy.spatial import KDTree

from med2glb.core.types import MeshData
from med2glb.glb.builder import _pad_to_4, write_accessor

logger = logging.getLogger("med2glb")


@dataclass
class ArrowParams:
    """Parameters for dash/arrow mesh generation."""
    shaft_radius: float = 0.15
    head_radius: float = 0.25
    head_length: float = 0.4
    segments: int = 6
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    normal_offset: float = 0.3


def _auto_scale_params(bbox_diagonal: float) -> ArrowParams:
    """Scale arrow geometry based on mesh bounding box."""
    scale = bbox_diagonal / 100.0
    return ArrowParams(
        shaft_radius=0.15 * scale,
        head_radius=0.25 * scale,
        head_length=0.4 * scale,
        normal_offset=0.3 * scale,
    )


def generate_dash_mesh(
    start: np.ndarray,
    end: np.ndarray,
    surface_normal: np.ndarray,
    params: ArrowParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a single dash: thin cylinder with a small cone head.

    The dash is oriented along (end - start), offset above the surface
    along the surface normal.

    Returns:
        (vertices [V, 3], faces [F, 3]) as float32 and int32.
    """
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-8:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    axis = direction / length

    # Offset start and end above the surface
    offset = surface_normal * params.normal_offset
    start_off = start + offset
    end_off = end + offset

    # Build a local coordinate frame
    # Find a vector not parallel to axis
    if abs(np.dot(axis, np.array([0, 1, 0]))) < 0.9:
        up = np.array([0, 1, 0], dtype=np.float64)
    else:
        up = np.array([1, 0, 0], dtype=np.float64)
    right = np.cross(axis, up)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, axis)
    up /= np.linalg.norm(up) + 1e-12

    seg = params.segments
    angles = np.linspace(0, 2 * np.pi, seg, endpoint=False)

    # Circle points (unit)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])  # [seg, 2]

    # --- Cylinder (shaft) ---
    shaft_end = start_off + axis * max(length - params.head_length, length * 0.7)

    # Bottom ring
    shaft_bottom = start_off + params.shaft_radius * (
        circle[:, 0:1] * right + circle[:, 1:2] * up
    )
    # Top ring
    shaft_top = shaft_end + params.shaft_radius * (
        circle[:, 0:1] * right + circle[:, 1:2] * up
    )

    # --- Cone (head) ---
    cone_base = shaft_end + params.head_radius * (
        circle[:, 0:1] * right + circle[:, 1:2] * up
    )
    cone_tip = end_off  # single point

    # Assemble vertices
    # Layout: shaft_bottom[0..seg-1], shaft_top[seg..2seg-1],
    #         cone_base[2seg..3seg-1], cone_tip[3seg]
    verts = np.vstack([
        shaft_bottom,  # 0 .. seg-1
        shaft_top,     # seg .. 2seg-1
        cone_base,     # 2seg .. 3seg-1
        cone_tip.reshape(1, 3),  # 3seg
    ]).astype(np.float32)

    faces_list = []

    # Shaft quads (as 2 triangles each)
    for i in range(seg):
        j = (i + 1) % seg
        # Bottom ring: 0+i, 0+j
        # Top ring: seg+i, seg+j
        faces_list.append([i, j, seg + j])
        faces_list.append([i, seg + j, seg + i])

    # Cone triangles
    tip_idx = 3 * seg
    for i in range(seg):
        j = (i + 1) % seg
        faces_list.append([2 * seg + i, 2 * seg + j, tip_idx])

    faces = np.array(faces_list, dtype=np.int32)
    return verts, faces


def build_frame_dashes(
    dashes: list[tuple[np.ndarray, np.ndarray]],
    mesh_vertices: np.ndarray,
    mesh_normals: np.ndarray,
    params: ArrowParams,
) -> MeshData | None:
    """Generate and merge all dash meshes for one animation frame.

    Uses a KDTree to look up surface normals at each dash position.

    Returns:
        Single MeshData with all dashes merged, or None if no dashes.
    """
    if not dashes or mesh_normals is None:
        return None

    tree = KDTree(mesh_vertices)
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vert_offset = 0

    for start, end in dashes:
        mid = (start + end) / 2.0
        _, idx = tree.query(mid)
        normal = mesh_normals[idx]
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        verts, faces = generate_dash_mesh(start, end, normal, params)
        if len(verts) == 0:
            continue

        all_verts.append(verts)
        all_faces.append(faces + vert_offset)
        vert_offset += len(verts)

    if not all_verts:
        return None

    merged_verts = np.vstack(all_verts).astype(np.float32)
    merged_faces = np.vstack(all_faces).astype(np.int32)

    # Uniform white color
    n_verts = len(merged_verts)
    colors = np.tile(
        np.array(params.color, dtype=np.float32),
        (n_verts, 1),
    )

    return MeshData(
        vertices=merged_verts,
        faces=merged_faces,
        normals=None,
        vertex_colors=colors,
        structure_name="lat_vectors",
    )


def build_animated_arrow_nodes(
    all_frame_dashes: list[list[tuple[np.ndarray, np.ndarray]]],
    mesh_vertices: np.ndarray,
    mesh_normals: np.ndarray,
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    n_frames: int,
    params: ArrowParams | None = None,
) -> list[int]:
    """Build per-frame arrow nodes in the glTF.

    Frame 0 visible (scale 1), others hidden (scale 0) — same pattern as
    wavefront animation. The caller is responsible for adding animation
    channels to synchronize visibility.

    Returns:
        List of node indices (one per frame).
    """
    if params is None:
        bbox = mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0)
        diag = float(np.linalg.norm(bbox))
        params = _auto_scale_params(diag)

    node_indices: list[int] = []

    # White material for arrows
    mat_idx = len(gltf.materials)
    gltf.materials.append(pygltflib.Material(
        name="lat_vectors",
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.5,
        ),
        doubleSided=True,
    ))

    for fi in range(n_frames):
        dashes = all_frame_dashes[fi] if fi < len(all_frame_dashes) else []
        mesh_data = build_frame_dashes(dashes, mesh_vertices, mesh_normals, params)

        if mesh_data is None or len(mesh_data.vertices) == 0:
            # Empty frame — add a minimal placeholder node
            # (tiny invisible mesh so animation channels still work)
            mesh_data = MeshData(
                vertices=np.zeros((3, 3), dtype=np.float32),
                faces=np.array([[0, 1, 2]], dtype=np.int32),
                structure_name="lat_vectors_empty",
            )

        # Write geometry
        verts = mesh_data.vertices.astype(np.float32)
        pos_acc = write_accessor(
            gltf, binary_data, verts, pygltflib.ARRAY_BUFFER,
            pygltflib.FLOAT, pygltflib.VEC3, with_minmax=True,
        )

        faces = mesh_data.faces.astype(np.uint32)
        idx_acc = write_accessor(
            gltf, binary_data, faces.ravel(), pygltflib.ELEMENT_ARRAY_BUFFER,
            pygltflib.UNSIGNED_INT, pygltflib.SCALAR, with_minmax=True,
        )

        attrs = pygltflib.Attributes(POSITION=pos_acc)

        # COLOR_0 if available
        if mesh_data.vertex_colors is not None:
            color_acc = write_accessor(
                gltf, binary_data, mesh_data.vertex_colors.astype(np.float32),
                pygltflib.ARRAY_BUFFER, pygltflib.FLOAT, pygltflib.VEC4,
            )
            attrs.COLOR_0 = color_acc

        mesh_idx = len(gltf.meshes)
        gltf.meshes.append(pygltflib.Mesh(
            name=f"lat_vectors_{fi}",
            primitives=[pygltflib.Primitive(
                attributes=attrs,
                indices=idx_acc,
                material=mat_idx,
            )],
        ))

        # Node: first frame visible, rest hidden
        scale = [1.0, 1.0, 1.0] if fi == 0 else [0.0, 0.0, 0.0]
        node_idx = len(gltf.nodes)
        gltf.nodes.append(pygltflib.Node(
            name=f"lat_vectors_{fi}",
            mesh=mesh_idx,
            scale=scale,
        ))
        node_indices.append(node_idx)

    logger.info(f"Built {len(node_indices)} arrow animation frames")
    return node_indices
