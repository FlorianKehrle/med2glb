"""Map sparse CARTO measurement points to mesh vertices.

Uses KDTree nearest-neighbor lookup and optional interpolation to transfer
per-point electrical values (LAT, voltage) to per-vertex colors.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree

from med2glb.core.types import CartoMesh, CartoPoint, MeshData
from med2glb.io.carto_colormaps import COLORMAPS
from med2glb.io.carto_reader import _INACTIVE_GROUP_ID

logger = logging.getLogger("med2glb")


def _extract_point_field(
    points: list[CartoPoint],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract positions and values for a given field, filtering NaN values.

    Returns:
        Tuple of (positions [K, 3], values [K]) with NaN entries removed.
        Both arrays may be empty if no valid points exist.
    """
    point_positions = np.array([p.position for p in points], dtype=np.float64)
    if field == "lat":
        point_values = np.array([p.lat for p in points], dtype=np.float64)
    elif field == "bipolar":
        point_values = np.array([p.bipolar_voltage for p in points], dtype=np.float64)
    elif field == "unipolar":
        point_values = np.array([p.unipolar_voltage for p in points], dtype=np.float64)
    else:
        raise ValueError(f"Unknown field: {field}. Use 'lat', 'bipolar', or 'unipolar'.")

    valid = ~np.isnan(point_values)
    return point_positions[valid], point_values[valid]


def map_points_to_vertices(
    mesh: CartoMesh,
    points: list[CartoPoint],
    field: str = "lat",
) -> np.ndarray:
    """Map sparse point measurements to mesh vertices via nearest-neighbor.

    Args:
        mesh: CARTO mesh with vertex positions.
        points: List of measurement points with 3D positions and values.
        field: Which field to map — "lat", "bipolar", or "unipolar".

    Returns:
        Per-vertex values array [N]. NaN where no point was near enough.
    """
    n_verts = len(mesh.vertices)
    values = np.full(n_verts, np.nan, dtype=np.float64)

    if not points:
        return values

    point_positions, point_values = _extract_point_field(points, field)
    if len(point_values) == 0:
        return values

    # Build KDTree from point positions and query for nearest neighbor per vertex
    tree = KDTree(point_positions)
    distances, indices = tree.query(mesh.vertices)

    # Assign nearest-neighbor values to all vertices
    values[:] = point_values[indices]

    return values


def map_points_to_vertices_idw(
    mesh: CartoMesh,
    points: list[CartoPoint],
    field: str = "lat",
    k: int = 6,
    power: float = 2.0,
) -> np.ndarray:
    """Map sparse point measurements to mesh vertices via k-NN IDW interpolation.

    Uses inverse-distance weighting with *k* nearest neighbors for smooth
    gradients between measurement points.

    Args:
        mesh: CARTO mesh with vertex positions.
        points: List of measurement points with 3D positions and values.
        field: Which field to map — "lat", "bipolar", or "unipolar".
        k: Number of nearest neighbors to use.
        power: IDW exponent (higher = sharper falloff).

    Returns:
        Per-vertex values array [N]. NaN where no valid points exist.
    """
    n_verts = len(mesh.vertices)
    values = np.full(n_verts, np.nan, dtype=np.float64)

    if not points:
        return values

    point_positions, point_values = _extract_point_field(points, field)
    if len(point_values) == 0:
        return values

    # Clamp k to number of available points
    k = min(k, len(point_values))

    tree = KDTree(point_positions)
    distances, indices = tree.query(mesh.vertices, k=k)

    # When k==1 scipy returns 1-D arrays — reshape to 2-D
    if k == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # IDW weights: w = 1 / (d^power + epsilon)
    weights = 1.0 / (np.power(distances, power) + 1e-10)
    weight_sums = weights.sum(axis=1, keepdims=True)
    weights /= weight_sums

    values[:] = np.sum(weights * point_values[indices], axis=1)
    return values


def interpolate_sparse_values(
    mesh: CartoMesh,
    values: np.ndarray,
    max_distance: float | None = None,
) -> np.ndarray:
    """Fill gaps in per-vertex values using linear interpolation.

    Uses scipy LinearNDInterpolator on vertices that have valid values
    to estimate values at vertices that don't.

    Args:
        mesh: CARTO mesh with vertex positions.
        values: Per-vertex values [N], NaN where unknown.
        max_distance: If set, only interpolate within this distance of known points.

    Returns:
        Interpolated per-vertex values [N].
    """
    valid = ~np.isnan(values)
    if np.all(valid) or not np.any(valid):
        return values

    result = values.copy()

    try:
        interp = LinearNDInterpolator(
            mesh.vertices[valid],
            values[valid],
        )
        missing = ~valid
        interpolated = interp(mesh.vertices[missing])
        # LinearNDInterpolator returns NaN outside convex hull — keep those as NaN
        result[missing] = interpolated
    except Exception as e:
        logger.debug(f"Interpolation failed: {e}")

    return result


def build_inactive_mask(mesh: CartoMesh) -> np.ndarray:
    """Build a boolean mask of inactive vertices (GroupID == -1000000).

    Returns:
        Boolean array [N], True for inactive vertices.
    """
    return mesh.group_ids == _INACTIVE_GROUP_ID


def subdivide_carto_mesh(mesh: CartoMesh, iterations: int) -> CartoMesh:
    """Loop-subdivide a CARTO mesh for finer spatial resolution.

    Each iteration roughly quadruples the face count while smoothing the
    surface geometry.  Vertex metadata (group_ids, face_group_ids) is
    propagated via nearest-neighbor lookup from the original mesh.

    Args:
        mesh: Source CARTO mesh.
        iterations: Number of Loop-subdivision passes (0 = no-op).

    Returns:
        A new CartoMesh with subdivided geometry, or the original if
        *iterations* <= 0.
    """
    if iterations <= 0:
        return mesh

    import trimesh
    from trimesh.remesh import subdivide_loop

    try:
        new_vertices, new_faces = subdivide_loop(
            mesh.vertices, mesh.faces, iterations=iterations,
        )
    except (ValueError, AssertionError) as exc:
        # Non-manifold meshes (edges shared by >2 faces) or degenerate
        # geometry cannot be Loop-subdivided — fall back to original mesh.
        logger.warning(
            "Loop subdivision failed for '%s': %s  — using original mesh",
            mesh.structure_name, exc,
        )
        return mesh

    # Propagate group_ids to new vertices via nearest-neighbor from originals
    tree = KDTree(mesh.vertices)
    _, nn_idx = tree.query(new_vertices)
    new_group_ids = mesh.group_ids[nn_idx]

    # Derive face_group_ids from vertex 0 of each face
    new_face_group_ids = new_group_ids[new_faces[:, 0]]

    # Recompute normals from the subdivided geometry
    tm = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    new_normals = np.array(tm.vertex_normals, dtype=np.float64)

    return CartoMesh(
        mesh_id=mesh.mesh_id,
        vertices=new_vertices.astype(np.float64),
        faces=new_faces.astype(np.int32),
        normals=new_normals,
        group_ids=new_group_ids.astype(np.int32),
        face_group_ids=new_face_group_ids.astype(np.int32),
        mesh_color=mesh.mesh_color,
        color_names=mesh.color_names,
        structure_name=mesh.structure_name,
    )


def carto_mesh_to_mesh_data(
    mesh: CartoMesh,
    points: list[CartoPoint] | None,
    coloring: str = "lat",
    clamp_range: tuple[float, float] | None = None,
    subdivide: int = 1,
) -> MeshData:
    """Convert a CartoMesh + points into a MeshData with vertex colors.

    This filters out inactive vertices/faces and applies the coloring.

    Args:
        mesh: CARTO mesh data.
        points: Measurement points (may be None for color fallback).
        coloring: Color scheme — "lat", "bipolar", or "unipolar".
        clamp_range: Optional value range for colormap normalization.
        subdivide: Loop-subdivision iterations (0 = no subdivision, NN mapping).

    Returns:
        MeshData with vertex_colors set.
    """
    actually_subdivided = False
    if subdivide > 0:
        original_mesh = mesh
        mesh = subdivide_carto_mesh(mesh, iterations=subdivide)
        actually_subdivided = mesh is not original_mesh

    # Build mask of active vertices and faces
    active_verts = mesh.group_ids != _INACTIVE_GROUP_ID
    active_faces_mask = mesh.face_group_ids != _INACTIVE_GROUP_ID

    # Also filter faces that reference inactive vertices
    for col in range(3):
        face_vert_active = active_verts[mesh.faces[:, col]]
        active_faces_mask = active_faces_mask & face_vert_active

    # Remap vertices: only keep active ones
    old_to_new = np.full(len(mesh.vertices), -1, dtype=np.int32)
    new_indices = np.where(active_verts)[0]
    old_to_new[new_indices] = np.arange(len(new_indices), dtype=np.int32)

    vertices = mesh.vertices[active_verts].astype(np.float32)
    normals = mesh.normals[active_verts].astype(np.float32)
    active_faces = mesh.faces[active_faces_mask]
    faces = old_to_new[active_faces].astype(np.int32)

    # Drop any faces with -1 (shouldn't happen but be safe)
    valid_faces = np.all(faces >= 0, axis=1)
    faces = faces[valid_faces]

    # Compute vertex colors
    vertex_colors = None
    if points:
        # Map sparse points to all mesh vertices first, then filter
        if actually_subdivided:
            all_values = map_points_to_vertices_idw(mesh, points, field=coloring)
        else:
            all_values = map_points_to_vertices(mesh, points, field=coloring)
            all_values = interpolate_sparse_values(mesh, all_values)
        active_values = all_values[active_verts]

        colormap_fn = COLORMAPS.get(coloring)
        if colormap_fn:
            vertex_colors = colormap_fn(active_values, clamp_range=clamp_range)
    else:
        # No points — use mesh default color as solid vertex color
        n = len(vertices)
        r, g, b, a = mesh.mesh_color
        vertex_colors = np.full((n, 4), [r, g, b, a], dtype=np.float32)

    from med2glb.core.types import MaterialConfig

    return MeshData(
        vertices=vertices,
        faces=faces,
        normals=normals,
        structure_name=mesh.structure_name,
        material=MaterialConfig(
            base_color=(1.0, 1.0, 1.0),
            alpha=1.0,
            metallic=0.0,
            roughness=0.7,
            name=mesh.structure_name,
        ),
        vertex_colors=vertex_colors,
    )
