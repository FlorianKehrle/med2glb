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


def _smooth_singularities(
    vertices: np.ndarray,
    faces: np.ndarray,
    label: str = "",
    threshold_factor: float = 10.0,
    max_iterations: int = 50,
) -> np.ndarray:
    """Iteratively smooth geometric singularities (spike vertices).

    Detects vertices whose Laplacian displacement (distance from the
    average of their neighbors) exceeds *threshold_factor* times the
    median displacement, and snaps them to the neighbor average.
    Repeats until no outliers remain or *max_iterations* is reached.

    Args:
        vertices: Vertex positions [N, 3] (modified in-place and returned).
        faces: Triangle indices [M, 3].
        label: Mesh name for log messages.
        threshold_factor: Displacement threshold as multiple of median.
        max_iterations: Safety limit on smoothing passes.

    Returns:
        The (possibly modified) vertices array.
    """
    import trimesh

    vertices = np.array(vertices, dtype=np.float64)
    tm_tmp = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    edges = np.asarray(tm_tmp.edges_unique)
    if len(edges) == 0:
        return vertices
    e0, e1 = edges[:, 0], edges[:, 1]
    n_verts = len(vertices)

    neighbor_count = np.zeros(n_verts, dtype=int)
    np.add.at(neighbor_count, e0, 1)
    np.add.at(neighbor_count, e1, 1)
    has_neighbors = neighbor_count > 0

    total_snapped = 0
    n_iter = 0
    for n_iter in range(max_iterations):
        neighbor_sum = np.zeros_like(vertices)
        np.add.at(neighbor_sum, e0, vertices[e1])
        np.add.at(neighbor_sum, e1, vertices[e0])

        avg_pos = np.zeros_like(vertices)
        avg_pos[has_neighbors] = (
            neighbor_sum[has_neighbors]
            / neighbor_count[has_neighbors, np.newaxis]
        )
        disp = np.linalg.norm(vertices - avg_pos, axis=1)
        disp[~has_neighbors] = 0.0
        med_disp = max(float(np.median(disp[has_neighbors])), 1e-10)

        spike_verts = has_neighbors & (disp > threshold_factor * med_disp)
        n_spikes = int(np.sum(spike_verts))
        if n_spikes == 0:
            break

        total_snapped += n_spikes
        vertices[spike_verts] = avg_pos[spike_verts]

    if total_snapped > 0:
        logger.debug(
            "Smoothed singularities for '%s': %d vertex corrections "
            "over %d passes",
            label, total_snapped, n_iter + 1,
        )
    return vertices


def subdivide_carto_mesh(mesh: CartoMesh, iterations: int) -> CartoMesh:
    """Loop-subdivide a CARTO mesh for finer spatial resolution.

    Each iteration roughly quadruples the face count while smoothing the
    surface geometry.  Vertex metadata (group_ids, face_group_ids) is
    propagated via nearest-neighbor lookup from the original mesh.

    Inactive faces (group_id == -1000000) are stripped **before**
    subdivision so that Loop smoothing cannot pull active vertices
    toward inactive geometry — which would create spike artifacts.

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

    # --- Strip inactive faces before subdivision ---
    # Faces with group_id == -1000000 are fully inactive and may bridge
    # distant regions.  Keep all other faces (including transparent fill
    # groups) so the mesh stays closed during subdivision — open boundaries
    # cause Loop subdivision to pull vertices inward, creating ragged edges.
    active_face_mask = mesh.face_group_ids != _INACTIVE_GROUP_ID
    clean_faces = mesh.faces[active_face_mask]
    clean_face_gids = mesh.face_group_ids[active_face_mask]

    # Identify visible vertex groups: those used by the dominant face group.
    # Fill-only vertex groups will be marked inactive after subdivision so
    # the overlapping fill geometry doesn't produce z-fighting artifacts.
    unique_fgids, fgid_counts = np.unique(clean_face_gids, return_counts=True)
    dominant_fgid = unique_fgids[np.argmax(fgid_counts)]
    dominant_face_verts = np.unique(clean_faces[clean_face_gids == dominant_fgid].ravel())
    visible_vert_groups = set(mesh.group_ids[dominant_face_verts].tolist())

    # Compact vertices: keep only those referenced by active faces
    used_verts = np.unique(clean_faces.ravel())
    old_to_new = np.full(len(mesh.vertices), -1, dtype=np.int32)
    old_to_new[used_verts] = np.arange(len(used_verts), dtype=np.int32)

    clean_vertices = mesh.vertices[used_verts].copy()
    clean_normals = mesh.normals[used_verts]
    clean_group_ids = mesh.group_ids[used_verts]
    clean_faces = old_to_new[clean_faces]

    # --- Pre-subdivision singularity smoothing ---
    # Fix outlier vertices BEFORE subdivision so Loop smoothing doesn't
    # amplify them (~16x vertex increase at level 2).
    clean_vertices = _smooth_singularities(
        clean_vertices, clean_faces, mesh.structure_name,
    )

    try:
        new_vertices, new_faces = subdivide_loop(
            clean_vertices, clean_faces, iterations=iterations,
        )
    except (ValueError, AssertionError) as exc:
        # Non-manifold meshes (edges shared by >2 faces) or degenerate
        # geometry cannot be Loop-subdivided — fall back to original mesh.
        logger.warning(
            "Loop subdivision failed for '%s': %s  — using original mesh",
            mesh.structure_name, exc,
        )
        return mesh

    # Propagate group_ids to new vertices via nearest-neighbor from
    # the cleaned (active-only) originals
    tree = KDTree(clean_vertices)
    _, nn_idx = tree.query(new_vertices)
    new_group_ids = clean_group_ids[nn_idx]

    # Mark fill-only vertices as inactive so carto_mesh_to_mesh_data
    # strips them.  Fill faces overlap the visible surface and cause
    # z-fighting / pinch artifacts if rendered.
    fill_mask = np.array([gid not in visible_vert_groups for gid in new_group_ids])
    n_fill = int(np.sum(fill_mask))
    if n_fill > 0:
        new_group_ids[fill_mask] = _INACTIVE_GROUP_ID
        logger.debug(
            "Marked %d fill-only vertices as inactive for '%s'",
            n_fill, mesh.structure_name,
        )

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

    # Build mask of active vertices and faces.
    # The dominant face group (most faces, excluding -1000000) is the
    # visible surface.  Transparent fill faces from other groups overlap
    # the visible surface and cause z-fighting artifacts — strip them.
    non_inactive_mask = mesh.face_group_ids != _INACTIVE_GROUP_ID
    non_inactive_gids = mesh.face_group_ids[non_inactive_mask]
    if len(non_inactive_gids) > 0:
        unique_gids, counts = np.unique(non_inactive_gids, return_counts=True)
        dominant_gid = unique_gids[np.argmax(counts)]
        dominant_verts = np.unique(
            mesh.faces[mesh.face_group_ids == dominant_gid].ravel()
        )
        visible_vert_groups = set(mesh.group_ids[dominant_verts].tolist())
    else:
        visible_vert_groups = set(mesh.group_ids.tolist())

    # Active vertices: belong to visible vertex groups
    active_verts = np.isin(mesh.group_ids, list(visible_vert_groups))

    # Active faces: not inactive AND all 3 vertices are active
    active_faces_mask = non_inactive_mask.copy()
    for col in range(3):
        active_faces_mask &= active_verts[mesh.faces[:, col]]

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

    # Post-subdivision singularity smoothing (catches residual artifacts)
    if len(faces) > 0:
        vertices = _smooth_singularities(
            vertices, faces, mesh.structure_name,
        ).astype(np.float32)

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
            unlit=True,
        ),
        vertex_colors=vertex_colors,
    )
