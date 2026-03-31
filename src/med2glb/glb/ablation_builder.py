"""Ablation sphere nodes for CARTO GLB exports.

Embeds one sphere per ablation point as a named glTF node (``ablation_000``,
``ablation_001``, …) directly into the scene alongside the mesh, legend, and
arrow nodes.  Spheres use an unlit orange-red material so they are always
clearly visible regardless of AR lighting conditions.  The radius is chosen
so ablation points appear at a clinically appropriate size (~3 cm) under the
10× AR display scale applied by the root node (0.01 scale × 3 mm = 3 cm).
"""

from __future__ import annotations

import logging

import numpy as np
import pygltflib

from med2glb.core.types import AblationPoint
from med2glb.glb.builder import _pad_to_4, write_accessor

logger = logging.getLogger("med2glb")

# Sphere radius in CARTO mm-space.  After the root-node 0.01 scale:
#   3 mm × 0.01 = 0.03 m = 3 cm displayed in AR
_SPHERE_RADIUS_MM: float = 3.0

# Unlit orange-red (RGBA linear) — highly visible in AR against dark tissue
_ABLATION_COLOR: tuple[float, float, float, float] = (0.95, 0.30, 0.05, 1.0)

# UV-sphere tessellation — 10 lat rings × 18 lon segments is lightweight (~360 verts)
_RINGS: int = 10
_SEGMENTS: int = 18

_MATERIAL_NAME = "ablation_lesion"


def _build_uv_sphere(
    radius: float,
    rings: int,
    segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a UV sphere centred at the origin.

    Returns:
        vertices : float32 [N, 3]
        normals  : float32 [N, 3]
        faces    : uint32  [M, 3]
    """
    verts: list[list[float]] = []
    norms: list[list[float]] = []

    # Top pole
    verts.append([0.0, radius, 0.0])
    norms.append([0.0, 1.0, 0.0])

    # Ring rows (lat = π/(rings+1) … π·rings/(rings+1))
    for r in range(1, rings + 1):
        lat = np.pi * r / (rings + 1)
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        for s in range(segments):
            lon = 2.0 * np.pi * s / segments
            nx = np.cos(lon) * sin_lat
            ny = cos_lat
            nz = np.sin(lon) * sin_lat
            verts.append([nx * radius, ny * radius, nz * radius])
            norms.append([nx, ny, nz])

    # Bottom pole
    verts.append([0.0, -radius, 0.0])
    norms.append([0.0, -1.0, 0.0])

    vertices = np.array(verts, dtype=np.float32)
    normals = np.array(norms, dtype=np.float32)

    tris: list[int] = []
    n_ring_start = 1  # first ring vertex index

    # Top cap
    for s in range(segments):
        a = n_ring_start + s
        b = n_ring_start + (s + 1) % segments
        tris.extend([0, a, b])

    # Middle quads
    for r in range(rings - 1):
        row0 = n_ring_start + r * segments
        row1 = n_ring_start + (r + 1) * segments
        for s in range(segments):
            s_next = (s + 1) % segments
            tl = row0 + s
            tr = row0 + s_next
            bl = row1 + s
            br = row1 + s_next
            tris.extend([tl, bl, tr, tr, bl, br])

    # Bottom cap
    bottom_pole = len(verts) - 1
    last_ring = n_ring_start + (rings - 1) * segments
    for s in range(segments):
        a = last_ring + s
        b = last_ring + (s + 1) % segments
        tris.extend([a, bottom_pole, b])

    faces = np.array(tris, dtype=np.uint32)
    return vertices, normals, faces


def _ensure_ablation_material(
    gltf: pygltflib.GLTF2,
) -> int:
    """Return the index of the shared ablation_lesion material, creating it if absent."""
    for i, mat in enumerate(gltf.materials):
        if mat.name == _MATERIAL_NAME:
            return i

    mat_idx = len(gltf.materials)
    r, g, b, a = _ABLATION_COLOR
    gltf.materials.append(pygltflib.Material(
        name=_MATERIAL_NAME,
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[r, g, b, a],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
        doubleSided=False,
        extensions={"KHR_materials_unlit": {}},
    ))

    if not hasattr(gltf, "extensionsUsed") or gltf.extensionsUsed is None:
        gltf.extensionsUsed = []
    if "KHR_materials_unlit" not in gltf.extensionsUsed:
        gltf.extensionsUsed.append("KHR_materials_unlit")

    return mat_idx


def add_ablation_nodes(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    ablation_points: list[AblationPoint],
    centroid: list[float],
) -> list[int]:
    """Add one sphere node per ablation point to a glTF document.

    Each node is named ``ablation_000``, ``ablation_001``, … so VeldtAR can
    toggle visibility with a single ``"show/hide ablations"`` voice command that
    matches by node name prefix.

    All spheres share one material and one sphere mesh primitive (instanced via
    separate nodes with individual translations) to minimise binary blob size.

    Args:
        gltf:            The glTF document being built.
        binary_data:     Binary buffer being assembled.
        ablation_points: Ablation positions and energy stats from the RF reader.
        centroid:        The mesh centroid subtracted from vertices so that
                         ablation positions align with the centred mesh geometry.

    Returns:
        List of node indices for the added sphere nodes.
    """
    if not ablation_points:
        return []

    mat_idx = _ensure_ablation_material(gltf)

    # Build shared sphere geometry (written once into the binary buffer)
    verts, norms, faces = _build_uv_sphere(_SPHERE_RADIUS_MM, _RINGS, _SEGMENTS)

    pos_acc = write_accessor(
        gltf, binary_data, verts, pygltflib.ARRAY_BUFFER,
        pygltflib.FLOAT, pygltflib.VEC3, with_minmax=True,
    )
    norm_acc = write_accessor(
        gltf, binary_data, norms, pygltflib.ARRAY_BUFFER,
        pygltflib.FLOAT, pygltflib.VEC3,
    )
    idx_acc = write_accessor(
        gltf, binary_data, faces, pygltflib.ELEMENT_ARRAY_BUFFER,
        pygltflib.UNSIGNED_INT, pygltflib.SCALAR, with_minmax=True,
    )

    sphere_mesh_idx = len(gltf.meshes)
    gltf.meshes.append(pygltflib.Mesh(
        name="ablation_sphere",
        primitives=[pygltflib.Primitive(
            attributes=pygltflib.Attributes(
                POSITION=pos_acc,
                NORMAL=norm_acc,
            ),
            indices=idx_acc,
            material=mat_idx,
        )],
    ))

    cx, cy, cz = centroid[0], centroid[1], centroid[2]
    node_indices: list[int] = []

    for i, ablation_point in enumerate(ablation_points):
        x, y, z = ablation_point.position
        node_idx = len(gltf.nodes)
        gltf.nodes.append(pygltflib.Node(
            name=f"ablation_{i:03d}",
            mesh=sphere_mesh_idx,
            translation=[float(x - cx), float(y - cy), float(z - cz)],
        ))
        node_indices.append(node_idx)

    logger.debug("Added %d ablation sphere nodes", len(node_indices))
    return node_indices
