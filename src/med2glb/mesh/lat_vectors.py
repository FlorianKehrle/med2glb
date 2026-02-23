"""LAT gradient computation, streamline tracing, and animated dash placement.

Computes the conduction direction from per-vertex LAT values on a triangle
mesh, traces streamlines following the gradient, and generates animated
dash positions that flow along those streamlines frame-by-frame.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree

logger = logging.getLogger("med2glb")


# ---------------------------------------------------------------------------
# Streamline path smoothing
# ---------------------------------------------------------------------------

def _smooth_streamline(path: list[np.ndarray], sigma: float = 2.0) -> list[np.ndarray]:
    """Smooth a polyline path using Gaussian filtering on XYZ coordinates.

    Sigma=2 smooths the sharp face-edge corners that occur when a streamline
    crosses triangle boundaries, without destroying the overall path direction.
    """
    if len(path) < 4:
        return path
    pts = np.array(path)  # [N, 3]
    smoothed = gaussian_filter1d(pts, sigma=sigma, axis=0)
    return [smoothed[i] for i in range(len(smoothed))]


# ---------------------------------------------------------------------------
# Per-face gradient computation
# ---------------------------------------------------------------------------

def compute_face_gradients(
    vertices: np.ndarray,
    faces: np.ndarray,
    lat_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the LAT gradient in each triangle's plane.

    For each face with vertices p0, p1, p2 and LAT values l0, l1, l2:
        e1 = p1 - p0,  e2 = p2 - p0
        dl1 = l1 - l0, dl2 = l2 - l0
        Gram matrix G = [[e1·e1, e1·e2], [e1·e2, e2·e2]]
        [a, b] = G^{-1} [dl1, dl2]
        grad = a * e1 + b * e2

    Returns:
        face_gradients: [M, 3] gradient vector per face (zero for degenerate/NaN faces).
        face_centers: [M, 3] centroid of each face.
        face_valid: [M] bool mask — True where the gradient is meaningful.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1 = v1 - v0  # [M, 3]
    e2 = v2 - v0  # [M, 3]

    l0 = lat_values[faces[:, 0]]
    l1 = lat_values[faces[:, 1]]
    l2 = lat_values[faces[:, 2]]

    dl1 = l1 - l0
    dl2 = l2 - l0

    # Gram matrix elements
    g11 = np.sum(e1 * e1, axis=1)
    g12 = np.sum(e1 * e2, axis=1)
    g22 = np.sum(e2 * e2, axis=1)

    det = g11 * g22 - g12 * g12
    eps = 1e-12

    # Validity: non-degenerate triangle AND no NaN LAT in any vertex
    any_nan = np.isnan(l0) | np.isnan(l1) | np.isnan(l2)
    face_valid = (~any_nan) & (np.abs(det) > eps)

    # Safe inverse (only where valid)
    safe_det = np.where(face_valid, det, 1.0)
    inv_det = 1.0 / safe_det

    a = inv_det * (g22 * dl1 - g12 * dl2)
    b = inv_det * (-g12 * dl1 + g11 * dl2)

    grad = a[:, None] * e1 + b[:, None] * e2
    grad[~face_valid] = 0.0

    face_centers = (v0 + v1 + v2) / 3.0

    return grad.astype(np.float64), face_centers.astype(np.float64), face_valid


def compute_vertex_gradients(
    vertices: np.ndarray,
    faces: np.ndarray,
    lat_values: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """Area-weighted accumulation of face gradients to vertices.

    Projects onto tangent plane: g -= (g·n)*n.

    Returns:
        [N, 3] per-vertex gradient vectors.
    """
    face_grads, _, face_valid = compute_face_gradients(vertices, faces, lat_values)

    # Face areas for weighting
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    weighted = face_grads * areas[:, None]
    weighted[~face_valid] = 0.0

    # Accumulate to vertices
    n_verts = len(vertices)
    vert_grad = np.zeros((n_verts, 3), dtype=np.float64)
    weight_sum = np.zeros(n_verts, dtype=np.float64)

    for k in range(3):
        np.add.at(vert_grad, faces[:, k], weighted)
        np.add.at(weight_sum, faces[:, k], areas * face_valid.astype(np.float64))

    nonzero = weight_sum > 1e-12
    vert_grad[nonzero] /= weight_sum[nonzero, None]

    # Project onto tangent plane
    if normals is not None:
        n = normals.astype(np.float64)
        dot = np.sum(vert_grad * n, axis=1, keepdims=True)
        vert_grad -= dot * n

    return vert_grad


# ---------------------------------------------------------------------------
# Face adjacency for mesh-surface tracing
# ---------------------------------------------------------------------------

def build_face_adjacency(faces: np.ndarray) -> dict[tuple[int, int], int]:
    """Build edge-to-face lookup: directed half-edge (va, vb) -> face index.

    For each face, the three half-edges are (v0,v1), (v1,v2), (v2,v0).
    The *opposite* directed edge leads to the adjacent face.
    """
    adj: dict[tuple[int, int], int] = {}
    for fi in range(len(faces)):
        v0, v1, v2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        adj[(v0, v1)] = fi
        adj[(v1, v2)] = fi
        adj[(v2, v0)] = fi
    return adj


# ---------------------------------------------------------------------------
# Streamline tracing on the triangle mesh surface
# ---------------------------------------------------------------------------

def _barycentric_to_world(bary: np.ndarray, tri_verts: np.ndarray) -> np.ndarray:
    """Convert barycentric coords [3] to world position given triangle vertices [3,3]."""
    return bary[0] * tri_verts[0] + bary[1] * tri_verts[1] + bary[2] * tri_verts[2]


def _world_to_barycentric(point: np.ndarray, tri_verts: np.ndarray) -> np.ndarray:
    """Project a 3D point onto a triangle and return barycentric coords."""
    v0, v1, v2 = tri_verts[0], tri_verts[1], tri_verts[2]
    e0 = v1 - v0
    e1 = v2 - v0
    d = point - v0
    d00 = np.dot(e0, e0)
    d01 = np.dot(e0, e1)
    d11 = np.dot(e1, e1)
    d20 = np.dot(d, e0)
    d21 = np.dot(d, e1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])


def trace_streamline(
    start_face: int,
    start_bary: np.ndarray,
    faces: np.ndarray,
    vertices: np.ndarray,
    face_gradients: np.ndarray,
    face_adjacency: dict[tuple[int, int], int],
    max_steps: int = 200,
    step_size: float = 0.5,
) -> list[np.ndarray]:
    """Trace a single streamline on the mesh surface following the gradient.

    Starting from a point in barycentric coordinates within a face, advance
    along the face gradient. When the point exits the current face (a
    barycentric coord goes negative), cross to the adjacent face and continue.

    Returns:
        List of 3D points forming the streamline polyline.
    """
    path: list[np.ndarray] = []
    cur_face = start_face
    cur_bary = start_bary.copy()

    for _ in range(max_steps):
        fi = faces[cur_face]
        tri = vertices[fi]  # [3, 3]
        pos = _barycentric_to_world(cur_bary, tri)
        path.append(pos.copy())

        grad = face_gradients[cur_face]
        grad_len = np.linalg.norm(grad)
        if grad_len < 1e-10:
            break

        direction = grad / grad_len
        new_pos = pos + direction * step_size
        new_bary = _world_to_barycentric(new_pos, tri)

        # Check if we're still inside the face
        if np.all(new_bary >= -1e-8):
            cur_bary = np.maximum(new_bary, 0.0)
            cur_bary /= cur_bary.sum()
            continue

        # Find which edge we crossed (most negative barycentric coord)
        min_idx = int(np.argmin(new_bary))

        # The edge opposite to vertex min_idx
        edge_verts = [k for k in range(3) if k != min_idx]
        va_local, vb_local = edge_verts
        va = int(fi[va_local])
        vb = int(fi[vb_local])

        # Look up adjacent face via the reversed half-edge
        adj_face = face_adjacency.get((vb, va))
        if adj_face is None:
            # Mesh boundary — stop
            path.append(new_pos)
            break

        # Project position into the new face
        adj_fi = faces[adj_face]
        adj_tri = vertices[adj_fi]
        cur_bary = _world_to_barycentric(new_pos, adj_tri)
        cur_bary = np.maximum(cur_bary, 0.0)
        s = cur_bary.sum()
        if s > 1e-12:
            cur_bary /= s
        else:
            cur_bary = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        cur_face = adj_face

    return path


# ---------------------------------------------------------------------------
# Seed point generation
# ---------------------------------------------------------------------------

def generate_streamline_seeds(
    vertices: np.ndarray,
    faces: np.ndarray,
    lat_values: np.ndarray,
    face_gradients: np.ndarray,
    face_valid: np.ndarray,
    target_count: int = 300,
) -> list[tuple[int, np.ndarray]]:
    """Distribute seed points across the mesh for streamline starts.

    Uses grid-based spatial seeding: divide the bounding box into cells,
    pick one valid face centroid per cell.

    Returns:
        List of (face_index, barycentric_coords) tuples.
    """
    # Filter to faces with valid gradient and non-tiny magnitude
    grad_mag = np.linalg.norm(face_gradients, axis=1)
    candidate_mask = face_valid & (grad_mag > 1e-6)
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        return []

    # Face centroids
    v0 = vertices[faces[candidate_indices, 0]]
    v1 = vertices[faces[candidate_indices, 1]]
    v2 = vertices[faces[candidate_indices, 2]]
    centroids = (v0 + v1 + v2) / 3.0

    # Grid-based seeding
    bbox_min = centroids.min(axis=0)
    bbox_max = centroids.max(axis=0)
    bbox_size = bbox_max - bbox_min
    bbox_diag = np.linalg.norm(bbox_size)

    if bbox_diag < 1e-6:
        return []

    # Target cell size for desired number of seeds
    n_cells_per_axis = max(int(np.cbrt(target_count)), 2)
    cell_size = bbox_diag / n_cells_per_axis

    # Assign each centroid to a grid cell
    cell_coords = ((centroids - bbox_min) / max(cell_size, 1e-6)).astype(np.int32)
    cell_keys = cell_coords[:, 0] * 10000 * 10000 + cell_coords[:, 1] * 10000 + cell_coords[:, 2]

    # Pick one face per cell (highest gradient magnitude wins)
    seeds: list[tuple[int, np.ndarray]] = []
    seen_cells: dict[int, int] = {}  # cell_key -> index in candidate_indices

    for i, key in enumerate(cell_keys):
        key_int = int(key)
        if key_int not in seen_cells:
            seen_cells[key_int] = i
        else:
            # Keep the one with larger gradient
            prev_i = seen_cells[key_int]
            if grad_mag[candidate_indices[i]] > grad_mag[candidate_indices[prev_i]]:
                seen_cells[key_int] = i

    for _, i in seen_cells.items():
        face_idx = int(candidate_indices[i])
        bary = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        seeds.append((face_idx, bary))

    # Trim to target count if we got too many
    if len(seeds) > target_count:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(seeds), target_count, replace=False)
        seeds = [seeds[i] for i in indices]

    logger.debug(f"Generated {len(seeds)} streamline seeds from {len(candidate_indices)} candidate faces")
    return seeds


# ---------------------------------------------------------------------------
# Top-level: trace all streamlines
# ---------------------------------------------------------------------------

def trace_all_streamlines(
    vertices: np.ndarray,
    faces: np.ndarray,
    lat_values: np.ndarray,
    normals: np.ndarray | None,
    target_count: int = 300,
    step_size: float | None = None,
    max_steps: int = 200,
) -> list[list[np.ndarray]]:
    """Compute gradients, seed, and trace all streamlines.

    Args:
        vertices: [N, 3] mesh vertices.
        faces: [M, 3] triangle indices.
        lat_values: [N] per-vertex LAT values (NaN for unknown).
        normals: [N, 3] vertex normals (for tangent-plane projection). May be None.
        target_count: Desired number of streamlines.
        step_size: Step size per iteration. Auto-scaled from mesh size if None.
        max_steps: Maximum tracing steps per streamline.

    Returns:
        List of polylines, each a list of 3D points.
    """
    face_grads, _, face_valid = compute_face_gradients(vertices, faces, lat_values)

    # Auto-scale step size from mesh bounding box
    if step_size is None:
        bbox = vertices.max(axis=0) - vertices.min(axis=0)
        diag = float(np.linalg.norm(bbox))
        step_size = diag / 500.0  # ~500 steps to cross the mesh
        step_size = max(step_size, 0.01)

    seeds = generate_streamline_seeds(
        vertices, faces, lat_values, face_grads, face_valid, target_count,
    )

    if not seeds:
        logger.warning("No valid streamline seeds found")
        return []

    adjacency = build_face_adjacency(faces)

    streamlines: list[list[np.ndarray]] = []
    for face_idx, bary in seeds:
        path = trace_streamline(
            face_idx, bary, faces, vertices, face_grads, adjacency,
            max_steps=max_steps, step_size=step_size,
        )
        if len(path) >= 3:  # Need at least 3 points for a meaningful streamline
            path = _smooth_streamline(path)
            streamlines.append(path)

    logger.info(f"Traced {len(streamlines)} streamlines (from {len(seeds)} seeds)")
    return streamlines


# ---------------------------------------------------------------------------
# Animated dash placement along streamlines
# ---------------------------------------------------------------------------

def _streamline_arc_lengths(points: list[np.ndarray]) -> np.ndarray:
    """Compute cumulative arc length along a polyline. Returns [len(points)] array."""
    pts = np.array(points)
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arcs = np.zeros(len(pts))
    arcs[1:] = np.cumsum(diffs)
    return arcs


def _interpolate_along(
    points: list[np.ndarray],
    arc_lengths: np.ndarray,
    s: float,
) -> np.ndarray | None:
    """Interpolate a 3D position at arc-length s along the polyline."""
    total = arc_lengths[-1]
    if total < 1e-10:
        return None
    s = max(0.0, min(s, total))  # clamp — never wrap back to start
    idx = np.searchsorted(arc_lengths, s, side="right") - 1
    idx = max(0, min(idx, len(points) - 2))
    seg_start = arc_lengths[idx]
    seg_end = arc_lengths[idx + 1]
    seg_len = seg_end - seg_start
    if seg_len < 1e-12:
        return np.array(points[idx])
    t = (s - seg_start) / seg_len
    return (1 - t) * np.array(points[idx]) + t * np.array(points[idx + 1])


def compute_animated_dashes(
    streamlines: list[list[np.ndarray]],
    n_frames: int = 30,
    dash_length: float | None = None,
    gap_length: float | None = None,
    offset_per_frame: float | None = None,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Place animated dashes along streamlines, advancing each frame.

    Each frame, dash positions shift forward by offset_per_frame along each
    streamline. Dashes wrap around when they pass the end.

    Args:
        streamlines: List of polylines (each a list of 3D points).
        n_frames: Number of animation frames.
        dash_length: Length of each dash segment. Auto-scaled if None.
        gap_length: Gap between dashes. Auto-scaled if None.
        offset_per_frame: How far dashes advance per frame. Auto-scaled if None.

    Returns:
        frames[frame_idx] = list of (start_point, end_point) dash segments.
    """
    if not streamlines:
        return [[] for _ in range(n_frames)]

    # Compute total extent for auto-scaling
    all_lengths = []
    arc_data: list[tuple[list[np.ndarray], np.ndarray]] = []
    for pts in streamlines:
        arcs = _streamline_arc_lengths(pts)
        total_len = arcs[-1]
        if total_len > 1e-6:
            all_lengths.append(total_len)
            arc_data.append((pts, arcs))

    if not all_lengths:
        return [[] for _ in range(n_frames)]

    median_len = float(np.median(all_lengths))

    # Auto-scale parameters based on streamline lengths
    if dash_length is None:
        dash_length = median_len * 0.08
    if gap_length is None:
        gap_length = dash_length * 1.5
    if offset_per_frame is None:
        offset_per_frame = (dash_length + gap_length) / n_frames * 3

    period = dash_length + gap_length

    frames: list[list[tuple[np.ndarray, np.ndarray]]] = []

    for fi in range(n_frames):
        frame_dashes: list[tuple[np.ndarray, np.ndarray]] = []
        phase = fi * offset_per_frame

        for pts, arcs in arc_data:
            total_len = arcs[-1]
            # Place dashes at regular intervals along the streamline
            s = phase % period
            while s < total_len:
                s_start = s
                s_end = s + dash_length

                if s_end > total_len:
                    # Don't wrap individual dashes — just clip
                    s_end = total_len

                p0 = _interpolate_along(pts, arcs, s_start)
                p1 = _interpolate_along(pts, arcs, s_end)

                if p0 is not None and p1 is not None:
                    seg_len = np.linalg.norm(p1 - p0)
                    if seg_len > 1e-6:
                        frame_dashes.append((p0, p1))

                s += period

        frames.append(frame_dashes)

    return frames


# ---------------------------------------------------------------------------
# Speed-dependent dash sizing
# ---------------------------------------------------------------------------

def compute_dash_speed_factors(
    frame_dashes: list[list[tuple[np.ndarray, np.ndarray]]],
    face_gradients: np.ndarray,
    face_centers: np.ndarray,
) -> list[list[float]]:
    """Compute per-dash speed factors from the local gradient magnitude.

    Uses robust percentile-based normalization (5th–95th) so that outlier
    faces with extreme gradients don't skew the entire distribution.
    Values are clamped to [0, 1] where 0 = slowest and 1 = fastest.

    Returns:
        Nested list parallel to *frame_dashes*: speed_factors[frame][dash].
    """
    grad_mag = np.linalg.norm(face_gradients, axis=1)

    # Filter to non-zero magnitudes for percentile computation
    nonzero = grad_mag[grad_mag > 1e-12]
    if len(nonzero) == 0:
        return [[] for _ in frame_dashes]

    p5 = float(np.percentile(nonzero, 5))
    p95 = float(np.percentile(nonzero, 95))
    prange = p95 - p5
    if prange < 1e-12:
        prange = 1.0

    tree = KDTree(face_centers)

    result: list[list[float]] = []
    for dashes in frame_dashes:
        if not dashes:
            result.append([])
            continue
        midpoints = np.array([(s + e) / 2.0 for s, e in dashes])
        _, indices = tree.query(midpoints)
        speeds = np.clip((grad_mag[indices] - p5) / prange, 0.0, 1.0)
        result.append(speeds.tolist())

    return result
