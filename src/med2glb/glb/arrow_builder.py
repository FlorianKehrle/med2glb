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
    segments: int = 12
    n_rings: int = 6
    max_radius: float | None = None
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    normal_offset: float = 0.3


def _teardrop_radius(t: float, max_r: float) -> float:
    """Compute radius at axial position *t* (0=tail, 1=head) for a teardrop profile.

    Peak radius at ~62% from the tail, smooth zero at both tips.
    """
    if t <= 0.0 or t >= 1.0:
        return 0.0
    return max_r * t ** 0.5 * (1.0 - t) ** 0.3


def _auto_scale_params(bbox_diagonal: float) -> ArrowParams:
    """Scale arrow geometry based on mesh bounding box."""
    scale = bbox_diagonal / 100.0
    return ArrowParams(
        shaft_radius=0.15 * scale,
        head_radius=0.25 * scale,
        head_length=0.4 * scale,
        max_radius=0.25 * scale,
        normal_offset=0.3 * scale,
    )


def generate_dash_mesh(
    start: np.ndarray,
    end: np.ndarray,
    surface_normal: np.ndarray,
    params: ArrowParams,
    radius_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a single dash as a smooth teardrop revolution surface.

    The dash is oriented along (end - start), offset above the surface
    along the surface normal.  The teardrop is wider near the head and
    tapers to a point at both tail (t=0) and head (t=1).

    Args:
        radius_override: If set, overrides the max teardrop radius for
            this specific dash (used for speed-dependent sizing).

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

    # Build a local coordinate frame
    if abs(np.dot(axis, np.array([0, 1, 0]))) < 0.9:
        up = np.array([0, 1, 0], dtype=np.float64)
    else:
        up = np.array([1, 0, 0], dtype=np.float64)
    right = np.cross(axis, up)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, axis)
    up /= np.linalg.norm(up) + 1e-12

    seg = params.segments
    n_rings = params.n_rings
    max_r = radius_override if radius_override is not None else (
        params.max_radius if params.max_radius is not None else params.head_radius
    )
    angles = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])  # [seg, 2]

    # Vertex layout: tail_tip, ring_1 .. ring_(n_rings-2), head_tip
    # Total: 2 + (n_rings - 2) * seg
    verts_list: list[np.ndarray] = []

    # Tail tip (t=0)
    tail_tip = start_off.copy()
    verts_list.append(tail_tip.reshape(1, 3))

    # Internal rings
    t_values = np.linspace(0, 1, n_rings)  # includes endpoints
    for ri in range(1, n_rings - 1):
        t = t_values[ri]
        r = _teardrop_radius(t, max_r)
        center = start_off + axis * (t * length)
        ring = center + r * (circle[:, 0:1] * right + circle[:, 1:2] * up)
        verts_list.append(ring)

    # Head tip (t=1)
    head_tip = start_off + axis * length
    verts_list.append(head_tip.reshape(1, 3))

    verts = np.vstack(verts_list).astype(np.float32)

    faces_list: list[list[int]] = []

    # Fan from tail tip (index 0) to first ring (indices 1 .. seg)
    for i in range(seg):
        j = (i + 1) % seg
        faces_list.append([0, 1 + j, 1 + i])

    # Triangle strips between consecutive internal rings
    n_internal = n_rings - 2
    for ri in range(n_internal - 1):
        base_a = 1 + ri * seg
        base_b = 1 + (ri + 1) * seg
        for i in range(seg):
            j = (i + 1) % seg
            faces_list.append([base_a + i, base_a + j, base_b + j])
            faces_list.append([base_a + i, base_b + j, base_b + i])

    # Fan from last ring to head tip
    head_tip_idx = len(verts) - 1
    last_ring_base = 1 + (n_internal - 1) * seg
    for i in range(seg):
        j = (i + 1) % seg
        faces_list.append([last_ring_base + i, last_ring_base + j, head_tip_idx])

    faces = np.array(faces_list, dtype=np.int32)
    return verts, faces


def build_frame_dashes(
    dashes: list[tuple[np.ndarray, np.ndarray]],
    mesh_vertices: np.ndarray,
    mesh_normals: np.ndarray,
    params: ArrowParams,
    dash_radii: list[float] | None = None,
) -> MeshData | None:
    """Generate and merge all dash meshes for one animation frame.

    Uses a KDTree to look up surface normals at each dash position.

    Args:
        dash_radii: Per-dash max radius override. If provided, must be the
            same length as *dashes*. Used for speed-dependent sizing.

    Returns:
        Single MeshData with all dashes merged, or None if no dashes.
    """
    if not dashes or mesh_normals is None:
        return None

    tree = KDTree(mesh_vertices)
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vert_offset = 0

    for di, (start, end) in enumerate(dashes):
        mid = (start + end) / 2.0
        _, idx = tree.query(mid)
        normal = mesh_normals[idx]
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        r_override = dash_radii[di] if dash_radii is not None else None
        verts, faces = generate_dash_mesh(start, end, normal, params, radius_override=r_override)
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
    speed_factors: list[list[float]] | None = None,
) -> list[int]:
    """Build per-frame arrow nodes in the glTF.

    Frame 0 visible (scale 1), others hidden (scale 0) — same pattern as
    wavefront animation. The caller is responsible for adding animation
    channels to synchronize visibility.

    Args:
        speed_factors: Per-frame, per-dash speed values in [0, 1].
            0 = slow (thick), 1 = fast (thin). If provided, must have
            the same structure as *all_frame_dashes*.

    Returns:
        List of node indices (one per frame).
    """
    if params is None:
        bbox = mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0)
        diag = float(np.linalg.norm(bbox))
        params = _auto_scale_params(diag)

    max_r = params.max_radius if params.max_radius is not None else params.head_radius

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
        # Compute per-dash radii from speed factors
        dash_radii: list[float] | None = None
        if speed_factors is not None and fi < len(speed_factors):
            sf = speed_factors[fi]
            # speed=1 (fast) → 0.5*max_r (thin), speed=0 (slow) → 1.5*max_r (thick)
            dash_radii = [max_r * (1.5 - s) for s in sf]
        mesh_data = build_frame_dashes(dashes, mesh_vertices, mesh_normals, params, dash_radii=dash_radii)

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
