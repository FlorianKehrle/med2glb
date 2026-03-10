"""Map sparse CARTO measurement points to mesh vertices.

Uses KDTree nearest-neighbor lookup and optional interpolation to transfer
per-point electrical values (LAT, voltage) to per-vertex colors.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree

from med2glb.core.types import CARTO_INACTIVE_GROUP_ID as _INACTIVE_GROUP_ID
from med2glb.core.types import CartoMesh, CartoPoint, MeshData
from med2glb.io.carto_colormaps import COLORMAPS

logger = logging.getLogger("med2glb")


def extract_point_field(
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


def compute_point_spacing(
    points: list[CartoPoint],
    field: str = "lat",
) -> float:
    """Compute median nearest-neighbor distance between valid points.

    Returns:
        Median inter-point spacing in mm, or inf if fewer than 2 valid points.
    """
    positions, _ = extract_point_field(points, field)
    if len(positions) < 2:
        return float("inf")
    tree = KDTree(positions)
    dd, _ = tree.query(positions, k=2)  # k=2: closest is self
    return float(np.median(dd[:, 1]))


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

    point_positions, point_values = extract_point_field(points, field)
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

    point_positions, point_values = extract_point_field(points, field)
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


def _get_fill_face_group_ids(face_group_ids: np.ndarray) -> set[int]:
    """Identify face group IDs that represent fill/transparent geometry.

    In the CARTO .mesh format, face GroupIDs follow a clear convention
    (validated across 15 meshes from CARTO v7.1 and v7.2):

    - ``GroupID == 0``: visible surface (myocardium)
    - ``GroupID < 0`` (not -1000000): fill/transparent caps (e.g. PV ostia)
    - ``GroupID == -1000000``: inactive/degenerate (handled separately)

    Note: the `TransparentGroupsIDs` field in the .mesh header is a
    CARTO-internal ID that does **not** correspond to face GroupIDs.
    Its count matches the number of negative face groups, but the values
    are from a different namespace.  Comparing them against face GroupIDs
    is incorrect and can misidentify the visible surface as fill.

    Args:
        face_group_ids: Per-face group IDs (may include -1000000).

    Returns:
        Set of face group IDs to treat as fill (never includes -1000000).
    """
    unique_gids = set(np.unique(face_group_ids).tolist())
    unique_gids.discard(_INACTIVE_GROUP_ID)
    return {g for g in unique_gids if g < 0}


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

    # Identify visible vertex groups from surface (non-fill) face groups.
    # Negative face GroupIDs are fill/transparent caps; GroupID 0 is visible.
    fill_face_gids = _get_fill_face_group_ids(clean_face_gids)
    if fill_face_gids:
        surface_fmask = ~np.isin(clean_face_gids, list(fill_face_gids))
        surface_verts_idx = np.unique(clean_faces[surface_fmask].ravel())
        visible_vert_groups = (
            set(mesh.group_ids[surface_verts_idx].tolist())
            if len(surface_verts_idx) > 0
            else set(mesh.group_ids.tolist())
        )
    else:
        visible_vert_groups = set(mesh.group_ids.tolist())

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
    if fill_face_gids:
        fill_mask = ~np.isin(new_group_ids, list(visible_vert_groups))
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
        transparent_group_ids=mesh.transparent_group_ids,
        structure_name=mesh.structure_name,
    )


def carto_mesh_to_mesh_data(
    mesh: CartoMesh,
    points: list[CartoPoint] | None,
    coloring: str = "lat",
    clamp_range: tuple[float, float] | None = None,
    subdivide: int = 1,
    pre_subdivided: CartoMesh | None = None,
) -> MeshData:
    """Convert a CartoMesh + points into a MeshData with vertex colors.

    This filters out inactive vertices/faces and applies the coloring.

    Args:
        mesh: CARTO mesh data.
        points: Measurement points (may be None for color fallback).
        coloring: Color scheme — "lat", "bipolar", or "unipolar".
        clamp_range: Optional value range for colormap normalization.
        subdivide: Loop-subdivision iterations (0 = no subdivision, NN mapping).
        pre_subdivided: Already-subdivided mesh to reuse (skips subdivision).

    Returns:
        MeshData with vertex_colors set.
    """
    actually_subdivided = False
    if pre_subdivided is not None:
        mesh = pre_subdivided
        actually_subdivided = True
    elif subdivide > 0:
        original_mesh = mesh
        mesh = subdivide_carto_mesh(mesh, iterations=subdivide)
        actually_subdivided = mesh is not original_mesh

    # Build mask of active faces.
    # 1) Always strip truly inactive faces (face_group_ids == -1000000).
    # 2) For non-subdivided meshes, also strip fill faces (negative face
    #    GroupID).  Subdivided meshes already had fill handled in
    #    subdivide_carto_mesh() via -1000000 vertex marking.
    non_inactive_mask = mesh.face_group_ids != _INACTIVE_GROUP_ID

    # Only detect fill on original (non-subdivided) meshes.  After
    # subdivision, face_group_ids are derived from vertex group IDs
    # (which can all be negative) and don't follow the face-group convention.
    if not actually_subdivided:
        fill_face_gids = _get_fill_face_group_ids(mesh.face_group_ids)
    else:
        fill_face_gids = set()

    if fill_face_gids:
        active_faces_mask = non_inactive_mask & ~np.isin(
            mesh.face_group_ids, list(fill_face_gids),
        )
    else:
        active_faces_mask = non_inactive_mask

    # Determine which vertices are used by active faces
    active_vert_indices = np.unique(mesh.faces[active_faces_mask].ravel())
    active_verts = np.zeros(len(mesh.vertices), dtype=bool)
    active_verts[active_vert_indices] = True

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

        # Distance cutoff: blank out vertices far from any valid measurement
        # to avoid misleading extrapolation.  Use 3x the median inter-point
        # spacing of valid measurements as the hard cutoff.
        point_positions, _ = extract_point_field(points, coloring)
        if len(point_positions) >= 2:
            spacing = compute_point_spacing(points, coloring)
            max_distance = spacing * 3
            dist_tree = KDTree(point_positions)
            distances, _ = dist_tree.query(mesh.vertices)
            too_far = distances > max_distance
            n_blanked = int(np.sum(too_far))
            if n_blanked > 0:
                all_values[too_far] = np.nan
                pct = 100.0 * n_blanked / len(all_values)
                logger.info(
                    "Distance cutoff (%.1f mm): blanked %d / %d vertices (%.0f%%) "
                    "beyond 3x inter-point spacing",
                    max_distance, n_blanked, len(all_values), pct,
                )

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
    from med2glb.mesh.processing import remove_small_components

    mesh_data = MeshData(
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
            unlit=False,
        ),
        vertex_colors=vertex_colors,
    )
    return remove_small_components(mesh_data)
