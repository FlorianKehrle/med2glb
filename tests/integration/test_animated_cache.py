"""Integration test: AnimatedBakeCache correctness and speedup measurement.

Builds two animated CARTO variants (animated + animated-second) both with and
without the bake cache, verifying that cached builds produce valid GLBs and
measuring the wall-clock time savings.
"""

from __future__ import annotations

import time

import numpy as np
import pygltflib
import pytest
import trimesh

from med2glb.core.types import CartoMesh, CartoPoint


@pytest.fixture
def large_carto_data():
    """Synthetic CARTO mesh (~5K faces) with dense LAT coverage."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=30.0)
    n_verts = len(sphere.vertices)

    mesh = CartoMesh(
        mesh_id=1,
        vertices=sphere.vertices.astype(np.float64),
        faces=sphere.faces.astype(np.int32),
        normals=sphere.vertex_normals.astype(np.float64),
        group_ids=np.zeros(n_verts, dtype=np.int32),
        face_group_ids=np.zeros(len(sphere.faces), dtype=np.int32),
        mesh_color=(0.0, 1.0, 0.0, 1.0),
        color_names=["Unipolar", "Bipolar", "LAT"],
        structure_name="cache_test",
    )

    # Dense points with LAT values spread over the mesh surface
    rng = np.random.RandomState(42)
    indices = rng.choice(n_verts, size=min(300, n_verts), replace=False)
    points = []
    for i, vi in enumerate(indices):
        pos = sphere.vertices[vi]
        points.append(CartoPoint(
            point_id=i,
            position=pos,
            orientation=rng.randn(3),
            bipolar_voltage=rng.uniform(0.1, 5.0),
            unipolar_voltage=rng.uniform(1.0, 15.0),
            lat=rng.uniform(-200, 100),
        ))

    return mesh, points


def _prepare_mesh_and_lat(mesh, points):
    """Convert CartoMesh to MeshData and extract LAT values."""
    from scipy.spatial import KDTree

    from med2glb.io.carto_mapper import (
        carto_mesh_to_mesh_data,
        interpolate_sparse_values,
        map_points_to_vertices,
    )

    mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat", subdivide=0)
    lat_all = map_points_to_vertices(mesh, points, field="lat")
    lat_all = interpolate_sparse_values(mesh, lat_all)
    tree = KDTree(mesh.vertices)
    _, idx = tree.query(mesh_data.vertices)
    return mesh_data, lat_all[idx]


class TestAnimatedBakeCache:
    """Verify cache correctness and measure speedup."""

    def test_cached_glb_is_valid(self, large_carto_data, tmp_path):
        """Cached build produces a structurally valid animated GLB."""
        from med2glb.glb.carto_builder import (
            build_carto_animated_glb,
            prepare_animated_cache,
        )

        mesh, points = large_carto_data
        mesh_data, active_lat = _prepare_mesh_and_lat(mesh, points)
        n_frames = 5

        cache = prepare_animated_cache(
            mesh_data, active_lat, n_frames=n_frames,
        )
        assert cache is not None
        assert cache.n_frames == n_frames
        assert cache.base_texture is not None
        assert cache.frame_colors.shape == (n_frames, len(mesh_data.vertices), 4)

        out = tmp_path / "cached.glb"
        result = build_carto_animated_glb(
            mesh_data, active_lat, out,
            n_frames=n_frames,
            cache=cache,
        )

        assert result is True
        assert out.exists()
        gltf = pygltflib.GLTF2.load(str(out))
        assert len(gltf.animations) == 1
        # Single mesh with COLOR_0 morph targets (not 5 separate meshes)
        assert len(gltf.meshes) == 1
        assert len(gltf.meshes[0].primitives[0].targets) == n_frames
        # No emissive textures — vertex color morph targets
        assert len(gltf.textures) == 0

    def test_cached_matches_uncached(self, large_carto_data, tmp_path):
        """Cached and uncached paths produce identical file sizes."""
        from med2glb.glb.carto_builder import (
            build_carto_animated_glb,
            prepare_animated_cache,
        )

        mesh, points = large_carto_data
        mesh_data, active_lat = _prepare_mesh_and_lat(mesh, points)
        n_frames = 5

        # Uncached
        out_a = tmp_path / "uncached.glb"
        build_carto_animated_glb(
            mesh_data, active_lat, out_a,
            n_frames=n_frames,
        )

        # Cached
        cache = prepare_animated_cache(
            mesh_data, active_lat, n_frames=n_frames,
        )
        out_b = tmp_path / "cached.glb"
        build_carto_animated_glb(
            mesh_data, active_lat, out_b,
            n_frames=n_frames,
            cache=cache,
        )

        gltf_a = pygltflib.GLTF2.load(str(out_a))
        gltf_b = pygltflib.GLTF2.load(str(out_b))

        assert len(gltf_a.meshes) == len(gltf_b.meshes)
        assert len(gltf_a.textures) == len(gltf_b.textures)
        assert len(gltf_a.animations) == len(gltf_b.animations)
        # File sizes should match (deterministic pipeline)
        assert out_a.stat().st_size == out_b.stat().st_size

    def test_cache_reuse_two_variants(self, large_carto_data, tmp_path):
        """Single cache serves two animated builds (the real use case)."""
        from med2glb.glb.carto_builder import (
            build_carto_animated_glb,
            prepare_animated_cache,
        )

        mesh, points = large_carto_data
        mesh_data, active_lat = _prepare_mesh_and_lat(mesh, points)
        n_frames = 5

        cache = prepare_animated_cache(
            mesh_data, active_lat, n_frames=n_frames,
        )
        assert cache is not None

        # First variant (animated)
        out1 = tmp_path / "animated.glb"
        r1 = build_carto_animated_glb(
            mesh_data, active_lat, out1,
            n_frames=n_frames,
            cache=cache,
        )

        # Second variant (animated, same params)
        out2 = tmp_path / "animated_2.glb"
        r2 = build_carto_animated_glb(
            mesh_data, active_lat, out2,
            n_frames=n_frames,
            cache=cache,
        )

        assert r1 is True and r2 is True
        assert out1.stat().st_size == out2.stat().st_size

    def test_speedup_measurement(self, large_carto_data, tmp_path):
        """Measure wall-clock speedup: uncached x 2 vs cached(prep+build x 2)."""
        from med2glb.glb.carto_builder import (
            build_carto_animated_glb,
            prepare_animated_cache,
        )

        mesh, points = large_carto_data
        mesh_data, active_lat = _prepare_mesh_and_lat(mesh, points)
        n_frames = 10

        # --- Uncached: two full builds (old behavior) ---
        t0 = time.perf_counter()
        build_carto_animated_glb(
            mesh_data, active_lat, tmp_path / "unc_1.glb",
            n_frames=n_frames,
        )
        build_carto_animated_glb(
            mesh_data, active_lat, tmp_path / "unc_2.glb",
            n_frames=n_frames,
        )
        t_uncached = time.perf_counter() - t0

        # --- Cached: one prep + two fast builds (new behavior) ---
        t0 = time.perf_counter()
        cache = prepare_animated_cache(
            mesh_data, active_lat,
            n_frames=n_frames,
        )
        build_carto_animated_glb(
            mesh_data, active_lat, tmp_path / "cac_1.glb",
            n_frames=n_frames,
            cache=cache,
        )
        build_carto_animated_glb(
            mesh_data, active_lat, tmp_path / "cac_2.glb",
            n_frames=n_frames,
            cache=cache,
        )
        t_cached = time.perf_counter() - t0

        speedup = t_uncached / t_cached if t_cached > 0 else float("inf")
        saved = t_uncached - t_cached

        print(f"\n{'=' * 60}")
        print(f"  AnimatedBakeCache Performance")
        print(f"{'=' * 60}")
        print(f"  Mesh:      {len(mesh_data.faces):,} faces, {len(mesh_data.vertices):,} verts")
        print(f"  Frames:    {n_frames}")
        print(f"  Uncached:  {t_uncached:.2f}s  (2 full builds)")
        print(f"  Cached:    {t_cached:.2f}s  (1 prep + 2 builds)")
        print(f"  Speedup:   {speedup:.2f}x")
        print(f"  Saved:     {saved:.2f}s")
        print(f"{'=' * 60}")

        assert t_cached < t_uncached, (
            f"Cached path should be faster: {t_cached:.2f}s >= {t_uncached:.2f}s"
        )
