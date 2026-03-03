"""Unit tests for vertex_color_bake: UV rasterization, gutter bleeding, texture baking."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from med2glb.glb.vertex_color_bake import (
    _bleed_gutter,
    _compute_pixel_mapping,
    apply_rasterization_map,
    bake_vertex_colors_to_texture,
    compute_texture_size,
    precompute_rasterization_map,
    rasterize_vertex_colors,
    xatlas_unwrap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_triangle_mesh():
    """A single right triangle covering the lower-left half of UV space."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    return vertices, faces, uvs


def _quad_mesh():
    """A quad (two triangles) covering full UV space."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    uvs = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1],
    ], dtype=np.float32)
    return vertices, faces, uvs


def _png_to_array(png_bytes: bytes) -> np.ndarray:
    """Decode PNG bytes to RGBA uint8 array."""
    img = Image.open(io.BytesIO(png_bytes))
    return np.array(img.convert("RGBA"))


# ---------------------------------------------------------------------------
# compute_texture_size
# ---------------------------------------------------------------------------

class TestComputeTextureSize:
    def test_small_mesh(self):
        assert compute_texture_size(100) == 512

    def test_boundary_5000(self):
        assert compute_texture_size(5000) == 512

    def test_medium_mesh(self):
        assert compute_texture_size(10000) == 1024

    def test_boundary_20000(self):
        assert compute_texture_size(20000) == 1024

    def test_large_mesh(self):
        assert compute_texture_size(50000) == 2048

    def test_boundary_80000(self):
        assert compute_texture_size(80000) == 2048

    def test_very_large_mesh(self):
        assert compute_texture_size(80001) == 4096


# ---------------------------------------------------------------------------
# _compute_pixel_mapping
# ---------------------------------------------------------------------------

class TestComputePixelMapping:
    def test_single_triangle_maps_pixels(self):
        """A triangle covering half the UV square should map ~half the pixels."""
        _, faces, uvs = _single_triangle_mesh()
        tex = 16
        py, px, v0, v1, v2, bw0, bw1, bw2 = _compute_pixel_mapping(faces, uvs, tex)

        assert len(py) > 0
        # All arrays must be the same length
        assert len(px) == len(py)
        assert len(v0) == len(py)
        assert len(bw0) == len(py)

        # Pixels should be within bounds
        assert np.all(py >= 0) and np.all(py < tex)
        assert np.all(px >= 0) and np.all(px < tex)

    def test_barycentric_weights_sum_to_one(self):
        """Barycentric weights at every mapped pixel should sum to ~1."""
        _, faces, uvs = _single_triangle_mesh()
        _, _, _, _, _, bw0, bw1, bw2 = _compute_pixel_mapping(faces, uvs, 32)

        if len(bw0) > 0:
            sums = bw0 + bw1 + bw2
            np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_barycentric_weights_non_negative(self):
        """All barycentric weights should be >= 0 after clamping."""
        _, faces, uvs = _single_triangle_mesh()
        _, _, _, _, _, bw0, bw1, bw2 = _compute_pixel_mapping(faces, uvs, 32)

        assert np.all(bw0 >= 0)
        assert np.all(bw1 >= 0)
        assert np.all(bw2 >= 0)

    def test_quad_covers_more_than_triangle(self):
        """Two triangles (quad) should map more pixels than a single triangle."""
        _, tri_faces, tri_uvs = _single_triangle_mesh()
        _, quad_faces, quad_uvs = _quad_mesh()
        tex = 32

        py_tri, *_ = _compute_pixel_mapping(tri_faces, tri_uvs, tex)
        py_quad, *_ = _compute_pixel_mapping(quad_faces, quad_uvs, tex)

        assert len(py_quad) > len(py_tri)

    def test_degenerate_triangle_produces_no_pixels(self):
        """A degenerate (zero-area) triangle should produce no mapped pixels."""
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        # All three vertices at the same UV point
        uvs = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32)

        py, px, *_ = _compute_pixel_mapping(faces, uvs, 16)
        assert len(py) == 0

    def test_empty_faces(self):
        """No faces should produce empty output arrays."""
        faces = np.zeros((0, 3), dtype=np.uint32)
        uvs = np.zeros((0, 2), dtype=np.float32)

        py, px, v0, v1, v2, bw0, bw1, bw2 = _compute_pixel_mapping(faces, uvs, 16)
        assert len(py) == 0
        assert len(bw0) == 0


# ---------------------------------------------------------------------------
# rasterize_vertex_colors
# ---------------------------------------------------------------------------

class TestRasterizeVertexColors:
    def test_solid_red_triangle(self):
        """A solid red triangle should produce red pixels in the texture."""
        _, faces, uvs = _single_triangle_mesh()
        colors = np.array([
            [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1],
        ], dtype=np.float32)

        png = rasterize_vertex_colors(faces, uvs, colors, 32)
        tex = _png_to_array(png)

        assert tex.shape == (32, 32, 4)

        # Some pixels should be red (R=255)
        red_pixels = (tex[:, :, 0] > 200) & (tex[:, :, 1] < 50) & (tex[:, :, 2] < 50)
        assert np.sum(red_pixels) > 0

    def test_gradient_interpolation(self):
        """Vertex colors R/G/B at three corners should produce interpolated interior."""
        _, faces, uvs = _single_triangle_mesh()
        colors = np.array([
            [1, 0, 0, 1],  # red at (0,0)
            [0, 1, 0, 1],  # green at (1,0)
            [0, 0, 1, 1],  # blue at (0,1)
        ], dtype=np.float32)

        png = rasterize_vertex_colors(faces, uvs, colors, 64)
        tex = _png_to_array(png)

        # Find all non-black pixels (mapped region)
        mapped = tex[:, :, 3] > 0
        if np.any(mapped):
            # Interior pixels should have mixed colors (not purely one channel)
            mapped_colors = tex[mapped]
            # At least some pixels should have multiple non-zero channels
            multi_channel = np.sum(mapped_colors[:, :3] > 30, axis=1) >= 2
            assert np.sum(multi_channel) > 0

    def test_returns_valid_png(self):
        """Output should be valid PNG bytes."""
        _, faces, uvs = _single_triangle_mesh()
        colors = np.ones((3, 4), dtype=np.float32)

        png = rasterize_vertex_colors(faces, uvs, colors, 16)

        # PNG magic bytes
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

        # Should be decodable
        img = Image.open(io.BytesIO(png))
        assert img.size == (16, 16)
        assert img.mode == "RGBA"

    def test_full_quad_covers_entire_texture(self):
        """A quad spanning [0,1]x[0,1] should fill most of the texture."""
        _, faces, uvs = _quad_mesh()
        colors = np.ones((4, 4), dtype=np.float32) * 0.5

        png = rasterize_vertex_colors(faces, uvs, colors, 32)
        tex = _png_to_array(png)

        # Most pixels should be filled (allowing gutter bleed)
        filled = tex[:, :, 3] > 0
        fill_ratio = np.sum(filled) / (32 * 32)
        assert fill_ratio > 0.8


# ---------------------------------------------------------------------------
# precompute / apply rasterization map
# ---------------------------------------------------------------------------

class TestPrecomputeApplyMap:
    def test_precompute_returns_expected_keys(self):
        _, faces, uvs = _single_triangle_mesh()
        rmap = precompute_rasterization_map(faces, uvs, 16)

        for key in ("pixel_y", "pixel_x", "v0", "v1", "v2",
                     "bw0", "bw1", "bw2", "mask", "tex_size"):
            assert key in rmap

        assert rmap["tex_size"] == 16
        assert rmap["mask"].shape == (16, 16)

    def test_apply_matches_one_shot(self):
        """precompute + apply should produce the same texture as rasterize_vertex_colors."""
        _, faces, uvs = _quad_mesh()
        colors = np.array([
            [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1],
        ], dtype=np.float32)
        tex_size = 32

        one_shot = rasterize_vertex_colors(faces, uvs, colors, tex_size)
        rmap = precompute_rasterization_map(faces, uvs, tex_size)
        two_step = apply_rasterization_map(rmap, colors)

        # Decode both
        tex_a = _png_to_array(one_shot)
        tex_b = _png_to_array(two_step)

        np.testing.assert_array_equal(tex_a, tex_b)

    def test_apply_with_different_colors(self):
        """Same map, different colors → different textures."""
        _, faces, uvs = _single_triangle_mesh()
        rmap = precompute_rasterization_map(faces, uvs, 16)

        red = np.array([[1, 0, 0, 1]] * 3, dtype=np.float32)
        blue = np.array([[0, 0, 1, 1]] * 3, dtype=np.float32)

        tex_r = _png_to_array(apply_rasterization_map(rmap, red))
        tex_b = _png_to_array(apply_rasterization_map(rmap, blue))

        # Red texture should have more red, blue texture more blue
        assert np.sum(tex_r[:, :, 0]) > np.sum(tex_r[:, :, 2])
        assert np.sum(tex_b[:, :, 2]) > np.sum(tex_b[:, :, 0])

    def test_mask_matches_mapped_pixels(self):
        """The mask should be True exactly where pixels are mapped."""
        _, faces, uvs = _single_triangle_mesh()
        rmap = precompute_rasterization_map(faces, uvs, 16)

        n_mapped = len(rmap["pixel_y"])
        n_mask = np.sum(rmap["mask"])

        # mask may have fewer entries if multiple pixels map to same location,
        # but should have at most as many as mapped pixels
        assert n_mask <= n_mapped
        assert n_mask > 0


# ---------------------------------------------------------------------------
# _bleed_gutter
# ---------------------------------------------------------------------------

class TestBleedGutter:
    def test_expands_filled_region(self):
        """Gutter bleeding should expand filled pixels into empty neighbors."""
        tex = np.zeros((8, 8, 4), dtype=np.float32)
        mask = np.zeros((8, 8), dtype=bool)

        # Place a single red pixel in the center
        tex[4, 4] = [1, 0, 0, 1]
        mask[4, 4] = True

        _bleed_gutter(tex, mask, 8, 8)

        # Neighbors should now be filled
        assert mask[3, 4]  # up
        assert mask[5, 4]  # down
        assert mask[4, 3]  # left
        assert mask[4, 5]  # right

        # Expanded pixels should be red
        np.testing.assert_allclose(tex[3, 4], [1, 0, 0, 1])

    def test_no_bleeding_when_fully_filled(self):
        """No change when entire texture is already filled."""
        tex = np.ones((4, 4, 4), dtype=np.float32) * 0.5
        mask = np.ones((4, 4), dtype=bool)
        tex_copy = tex.copy()

        _bleed_gutter(tex, mask, 4, 4)

        np.testing.assert_array_equal(tex, tex_copy)

    def test_no_bleeding_when_empty(self):
        """No change when texture is completely empty."""
        tex = np.zeros((4, 4, 4), dtype=np.float32)
        mask = np.zeros((4, 4), dtype=bool)

        _bleed_gutter(tex, mask, 4, 4)

        assert not np.any(mask)

    def test_bleeding_averages_neighbors(self):
        """When an empty pixel has two filled neighbors, it gets their average."""
        tex = np.zeros((5, 5, 4), dtype=np.float32)
        mask = np.zeros((5, 5), dtype=bool)

        # Red pixel above, blue pixel below an empty center
        tex[1, 2] = [1, 0, 0, 1]
        mask[1, 2] = True
        tex[3, 2] = [0, 0, 1, 1]
        mask[3, 2] = True

        _bleed_gutter(tex, mask, 5, 5)

        # Center pixel (2,2) should be average of red and blue
        assert mask[2, 2]
        np.testing.assert_allclose(tex[2, 2], [0.5, 0, 0.5, 1], atol=0.01)


# ---------------------------------------------------------------------------
# xatlas_unwrap
# ---------------------------------------------------------------------------

class TestXatlasUnwrap:
    def test_single_triangle(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)

        vmapping, new_faces, uvs = xatlas_unwrap(verts, faces)

        # xatlas may split vertices, so V' >= V
        assert len(vmapping) >= 3
        assert new_faces.shape[1] == 3
        assert len(new_faces) == 1  # one triangle
        assert uvs.shape == (len(vmapping), 2)

        # UVs should be in [0, 1]
        assert np.all(uvs >= 0)
        assert np.all(uvs <= 1)

        # vmapping should index into original vertices
        assert np.all(vmapping >= 0)
        assert np.all(vmapping < 3)

    def test_with_normals(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)

        vmapping, new_faces, uvs = xatlas_unwrap(verts, faces, normals)

        assert len(vmapping) >= 3
        assert uvs.shape[1] == 2

    def test_cube_mesh(self):
        """A cube should produce valid UVs with multiple charts."""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ], dtype=np.uint32)

        vmapping, new_faces, uvs = xatlas_unwrap(verts, faces)

        assert len(new_faces) == 12
        assert np.all(uvs >= 0) and np.all(uvs <= 1)


# ---------------------------------------------------------------------------
# bake_vertex_colors_to_texture (full pipeline)
# ---------------------------------------------------------------------------

class TestBakeVertexColorsToTexture:
    def test_full_pipeline_single_triangle(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        colors = np.array([
            [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1],
        ], dtype=np.float32)

        new_verts, new_faces, new_normals, uvs, png = bake_vertex_colors_to_texture(
            verts, faces, colors, texture_size=64,
        )

        # Geometry
        assert new_verts.shape[1] == 3
        assert len(new_verts) >= 3
        assert new_faces.shape == (1, 3)
        assert new_normals is None  # no normals provided
        assert uvs.shape == (len(new_verts), 2)

        # Texture
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        tex = _png_to_array(png)
        assert tex.shape == (64, 64, 4)
        # Should have some colored pixels
        assert np.any(tex[:, :, :3] > 0)

    def test_full_pipeline_with_normals(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        colors = np.ones((3, 4), dtype=np.float32)

        new_verts, new_faces, new_normals, uvs, png = bake_vertex_colors_to_texture(
            verts, faces, colors, texture_size=32, normals=normals,
        )

        assert new_normals is not None
        assert new_normals.shape == new_verts.shape

    def test_full_pipeline_cube(self):
        """Cube with per-face-ish colors should produce a textured result."""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ], dtype=np.uint32)
        colors = np.random.RandomState(42).rand(8, 4).astype(np.float32)
        colors[:, 3] = 1.0

        new_verts, new_faces, _, uvs, png = bake_vertex_colors_to_texture(
            verts, faces, colors, texture_size=128,
        )

        assert len(new_faces) == 12
        tex = _png_to_array(png)
        filled = tex[:, :, 3] > 0
        assert np.sum(filled) > 100  # significant coverage
